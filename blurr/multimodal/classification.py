"""blurr.multimodal.classification — Vision-language classification, contrastive training, and VQA."""

from __future__ import annotations

import inspect
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import Dataset, DatasetDict, load_dataset
from fastai.data.block import CategoryBlock, DataBlock, MultiCategoryBlock, TransformBlock
from fastai.data.core import DataLoaders
from fastai.data.transforms import ColSplitter, RandomSplitter
from fastai.metrics import F1Score, F1ScoreMulti, accuracy, accuracy_multi
from fastai.torch_core import TensorBase
from fastcore.foundation import L
from fasttransform import Transform
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForVisualQuestionAnswering,
    AutoProcessor,
)

from ..text._core import BlurrLearner
from ..vision._core import apply_image_augmentations, load_image


class MultimodalInput(TensorBase):
    """Marker type for multimodal pixel inputs."""


# ===================================================================
# Batch transforms
# ===================================================================

class MultimodalBatchTransform(Transform):
    """Apply a HF processor to (image, text) batches on-the-fly."""

    def __init__(
        self,
        processor,
        *,
        task: str = "classification",
        include_labels: bool = True,
        is_multilabel: bool = False,
        labels: list[str] | None = None,
        max_length: int | None = None,
        padding: bool | str = True,
        truncation: bool | str = True,
        augmentations: Any | None = None,
        processor_call_kwargs: dict[str, Any] | None = None,
    ):
        self.processor = processor
        self.task = task
        self.include_labels = include_labels
        self.is_multilabel = is_multilabel
        self.labels = labels
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.augmentations = augmentations
        self.processor_call_kwargs = processor_call_kwargs or {}

    def _normalize_sample(self, sample):
        if isinstance(sample, tuple):
            return sample
        if isinstance(sample, list):
            return tuple(sample)
        return (sample,)

    def _extract_image_text(self, x):
        if isinstance(x, dict):
            image = x.get("image", x.get("img", x.get("pixel_values")))
            text = x.get("text", x.get("question", ""))
            return image, text
        if isinstance(x, (list, tuple)) and len(x) >= 2:
            return x[0], x[1]
        raise ValueError("Expected multimodal input as (image, text) tuple or dict")

    def encodes(self, samples):
        samples = L(samples)
        if len(samples) == 0:
            return samples

        samples = [self._normalize_sample(s) for s in samples]

        pairs = [self._extract_image_text(s[0]) for s in samples]
        images = [apply_image_augmentations(load_image(img), self.augmentations) for img, _ in pairs]
        texts = [str(txt) for _, txt in pairs]

        proc_kwargs = dict(
            return_tensors="pt",
            padding=self.padding,
            truncation=self.truncation,
            **self.processor_call_kwargs,
        )
        if self.max_length is not None:
            proc_kwargs["max_length"] = self.max_length

        inputs = self.processor(images=images, text=texts, **proc_kwargs)

        updated = []
        for idx, sample in enumerate(samples):
            inp_dict = {k: v[idx] for k, v in inputs.items()}
            targets = sample[1:]

            if self.include_labels and len(targets) > 0 and self.task in {"classification", "vqa"}:
                if self.is_multilabel:
                    inp_dict["labels"] = targets[0].float()
                else:
                    inp_dict["labels"] = targets[0]

            if len(targets) > 0:
                updated.append((inp_dict, *targets))
            else:
                updated.append((inp_dict,))

        return updated


class MultimodalBatchDecodeTransform(Transform):
    def __init__(self, processor):
        self.processor = processor

    def decodes(self, x: dict) -> MultimodalInput:
        return MultimodalInput(x["pixel_values"])


def MultimodalTextImageBlock(
    processor,
    *,
    task: str = "classification",
    include_labels: bool = True,
    is_multilabel: bool = False,
    labels: list[str] | None = None,
    max_length: int | None = None,
    padding: bool | str = True,
    truncation: bool | str = True,
    augmentations: Any | None = None,
    processor_call_kwargs: dict[str, Any] | None = None,
):
    batch_tfm = MultimodalBatchTransform(
        processor,
        task=task,
        include_labels=include_labels,
        is_multilabel=is_multilabel,
        labels=labels,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        augmentations=augmentations,
        processor_call_kwargs=processor_call_kwargs,
    )
    decode_tfm = MultimodalBatchDecodeTransform(processor)
    return TransformBlock(
        dls_kwargs={"before_batch": batch_tfm},
        batch_tfms=decode_tfm,
    )


def _find_mm_batch_tfm(dls) -> MultimodalBatchTransform:
    for tfm in dls.before_batch:
        if isinstance(tfm, MultimodalBatchTransform):
            return tfm
    raise ValueError("Could not find MultimodalBatchTransform in DataLoaders")


# ===================================================================
# Models
# ===================================================================

class MultimodalClassificationModel(nn.Module):
    """Adds a classification head on top of a vision-language backbone."""

    def __init__(self, hf_model: nn.Module, num_labels: int, *, is_multilabel: bool = False, dropout: float = 0.1):
        super().__init__()
        self.hf_model = hf_model
        self.num_labels = num_labels
        self.is_multilabel = is_multilabel
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.LazyLinear(num_labels)
        self._fwd_params = set(inspect.signature(hf_model.forward).parameters.keys())

    def _extract_embeddings(self, outputs) -> torch.Tensor:
        image_embeds = getattr(outputs, "image_embeds", None)
        text_embeds = getattr(outputs, "text_embeds", None)

        if image_embeds is None and isinstance(outputs, dict):
            image_embeds = outputs.get("image_embeds")
            text_embeds = outputs.get("text_embeds")

        if image_embeds is None and hasattr(outputs, "vision_model_output"):
            vision_out = outputs.vision_model_output
            image_embeds = getattr(vision_out, "pooler_output", None)
            if image_embeds is None and hasattr(vision_out, "last_hidden_state"):
                image_embeds = vision_out.last_hidden_state[:, 0]

        if text_embeds is None and hasattr(outputs, "text_model_output"):
            text_out = outputs.text_model_output
            text_embeds = getattr(text_out, "pooler_output", None)
            if text_embeds is None and hasattr(text_out, "last_hidden_state"):
                text_embeds = text_out.last_hidden_state[:, 0]

        if image_embeds is None and hasattr(outputs, "pooler_output"):
            image_embeds = outputs.pooler_output

        if image_embeds is None and hasattr(outputs, "last_hidden_state"):
            image_embeds = outputs.last_hidden_state[:, 0]

        if image_embeds is None:
            raise ValueError("Could not extract multimodal embeddings from model outputs")

        if text_embeds is None:
            return image_embeds

        return torch.cat([image_embeds, text_embeds], dim=-1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        token_type_ids=None,
        labels=None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "token_type_ids": token_type_ids,
            "output_hidden_states": output_hidden_states,
            "output_attentions": output_attentions,
            "return_dict": return_dict,
            **kwargs,
        }
        filtered_inputs = {
            k: v
            for k, v in model_inputs.items()
            if (k in self._fwd_params and v is not None)
        }

        outputs = self.hf_model(**filtered_inputs)

        feats = self._extract_embeddings(outputs)
        logits = self.classifier(self.dropout(feats))

        out = {"logits": logits}
        if labels is not None:
            if self.is_multilabel:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            else:
                loss = F.cross_entropy(logits, labels.long())
            out["loss"] = loss

        out["backbone_outputs"] = outputs
        return out


class ContrastiveHFModel(nn.Module):
    """CLIP/SigLIP style contrastive wrapper using model-computed loss."""

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.hf_model = hf_model
        self._fwd_params = set(inspect.signature(hf_model.forward).parameters.keys())

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        token_type_ids=None,
        labels=None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "token_type_ids": token_type_ids,
            "output_hidden_states": output_hidden_states,
            "output_attentions": output_attentions,
            "return_dict": return_dict,
            "return_loss": True,
            **kwargs,
        }
        filtered_inputs = {
            k: v
            for k, v in model_inputs.items()
            if (k in self._fwd_params and v is not None)
        }

        outputs = self.hf_model(**filtered_inputs)

        logits_per_image = getattr(outputs, "logits_per_image", None)
        logits_per_text = getattr(outputs, "logits_per_text", None)

        if logits_per_image is None and isinstance(outputs, dict):
            logits_per_image = outputs.get("logits_per_image")
            logits_per_text = outputs.get("logits_per_text")

        if logits_per_image is None:
            image_embeds = getattr(outputs, "image_embeds", None)
            text_embeds = getattr(outputs, "text_embeds", None)
            if image_embeds is None and isinstance(outputs, dict):
                image_embeds = outputs.get("image_embeds")
                text_embeds = outputs.get("text_embeds")

            image_embeds = F.normalize(image_embeds, dim=-1)
            text_embeds = F.normalize(text_embeds, dim=-1)
            logits_per_image = image_embeds @ text_embeds.T
            logits_per_text = logits_per_image.T

        loss = getattr(outputs, "loss", None)
        if loss is None and isinstance(outputs, dict):
            loss = outputs.get("loss")

        if loss is None:
            target = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
            loss = 0.5 * (
                F.cross_entropy(logits_per_image, target)
                + F.cross_entropy(logits_per_text, target)
            )

        return {
            "loss": loss,
            "logits": logits_per_image,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
        }


# ===================================================================
# Learner
# ===================================================================

class MultimodalClassificationLearner(BlurrLearner):
    """Learner for multimodal classification/contrastive/VQA tasks."""

    def blurr_predict(
        self,
        items,
        *,
        threshold: float = 0.5,
        rm_type_tfms: int | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(items, tuple):
            items = [items]
        elif isinstance(items, dict):
            items = [items]
        elif not isinstance(items, list):
            raise TypeError("items must be a (image, text) tuple, dict, or list of those")

        tfm = _find_mm_batch_tfm(self.dls)
        labels = tfm.labels
        is_multilabel = tfm.is_multilabel
        task = tfm.task

        dl = self.dls.test_dl(items, rm_type_tfms=rm_type_tfms, num_workers=0)
        preds = self.get_preds(dl=dl, with_input=False, with_decoded=False)
        preds_t = preds[0]

        results = []
        for i in range(len(items)):
            probs = preds_t[i]
            res: dict[str, Any] = {"probs": probs.tolist()}

            if task == "contrastive":
                res["similarity_scores"] = probs.tolist()
                results.append(res)
                continue

            if is_multilabel:
                indices = (probs > threshold).nonzero(as_tuple=True)[0].tolist()
                res["labels"] = [labels[j] for j in indices] if labels else indices
                res["scores"] = [probs[j].item() for j in indices]
            else:
                idx = probs.argmax().item()
                res["label"] = labels[idx] if labels else idx
                res["score"] = probs[idx].item()

            res["class_labels"] = labels if labels else list(range(len(probs)))
            results.append(res)

        return results

    def blurr_zero_shot_predict(
        self,
        images,
        candidate_labels: list[str],
        *,
        prompt_template: str = "a photo of {}",
    ) -> list[dict[str, Any]]:
        """Zero-shot prediction with CLIP/SigLIP-style backbones."""
        if not isinstance(images, list):
            images = [images]

        tfm = _find_mm_batch_tfm(self.dls)
        processor = tfm.processor

        hf_model = self.model.hf_model
        if isinstance(hf_model, MultimodalClassificationModel):
            backbone = hf_model.hf_model
        elif isinstance(hf_model, ContrastiveHFModel):
            backbone = hf_model.hf_model
        else:
            backbone = hf_model

        text_prompts = [prompt_template.format(lbl) for lbl in candidate_labels]
        image_objs = [load_image(img) for img in images]

        device = next(self.model.parameters()).device
        proc_inputs = processor(
            images=image_objs,
            text=text_prompts,
            padding=True,
            return_tensors="pt",
        )
        proc_inputs = {k: v.to(device) for k, v in proc_inputs.items()}

        backbone.eval()
        with torch.no_grad():
            if hasattr(backbone, "get_image_features") and hasattr(backbone, "get_text_features"):
                image_inputs = {k: v for k, v in proc_inputs.items() if k.startswith("pixel")}
                text_inputs = {k: v for k, v in proc_inputs.items() if k.startswith("input") or k == "attention_mask"}
                image_embeds = backbone.get_image_features(**image_inputs)
                text_embeds = backbone.get_text_features(**text_inputs)
                image_embeds = F.normalize(image_embeds, dim=-1)
                text_embeds = F.normalize(text_embeds, dim=-1)
                logits_per_image = image_embeds @ text_embeds.T
            else:
                outputs = backbone(**proc_inputs, return_dict=True)
                logits_per_image = getattr(outputs, "logits_per_image", None)
                if logits_per_image is None and isinstance(outputs, dict):
                    logits_per_image = outputs.get("logits_per_image")
                if logits_per_image is None:
                    raise ValueError("Backbone does not expose zero-shot logits/features")

            probs = logits_per_image.softmax(dim=-1)

        results = []
        for i in range(len(images)):
            p = probs[i]
            idx = p.argmax().item()
            results.append(
                {
                    "label": candidate_labels[idx],
                    "score": p[idx].item(),
                    "probs": p.tolist(),
                    "class_labels": candidate_labels,
                }
            )
        return results

    # -----------------------------------------------------------------
    # Factories
    # -----------------------------------------------------------------
    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        model_name: str,
        *,
        image_col: str = "image",
        text_col: str = "text",
        label_col: str | list[str] | None = "label",
        task: str = "classification",  # classification|contrastive|vqa
        is_multilabel: bool = False,
        valid_col: str | None = None,
        valid_pct: float = 0.2,
        n_labels: int | None = None,
        max_length: int | None = None,
        bs: int = 8,
        val_bs: int | None = None,
        seed: int = 42,
        augmentations: Any | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        processor_call_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        metrics: list | None = None,
        dl_kwargs: dict[str, Any] | None = None,
        learner_kwargs: dict[str, Any] | None = None,
    ) -> "MultimodalClassificationLearner":
        processor_kwargs = processor_kwargs or {}
        processor_call_kwargs = processor_call_kwargs or {}
        config_kwargs = config_kwargs or {}
        model_kwargs = model_kwargs or {}
        model_kwargs.setdefault("ignore_mismatched_sizes", True)
        dl_kwargs = dl_kwargs or {}
        learner_kwargs = learner_kwargs or {}

        if task not in {"classification", "contrastive", "vqa"}:
            raise ValueError("task must be one of: classification, contrastive, vqa")

        labels = None
        if task == "contrastive":
            n_labels = n_labels or 2
        elif is_multilabel:
            label_cols = label_col if isinstance(label_col, list) else [label_col]
            labels = label_cols
            n_labels = n_labels or len(label_cols)
        else:
            n_labels = n_labels or df[label_col].nunique()
            labels = sorted(df[label_col].unique().tolist())

        config_kwargs = dict(config_kwargs)
        if task in {"classification", "vqa"}:
            config_kwargs["num_labels"] = n_labels
            if labels is not None and not is_multilabel:
                config_kwargs.setdefault("id2label", {i: lbl for i, lbl in enumerate(labels)})
                config_kwargs.setdefault("label2id", {lbl: i for i, lbl in enumerate(labels)})
            if is_multilabel:
                config_kwargs["problem_type"] = "multi_label_classification"

        config = AutoConfig.from_pretrained(model_name, **config_kwargs)
        processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)

        if task == "vqa":
            hf_model = AutoModelForVisualQuestionAnswering.from_pretrained(
                model_name,
                config=config,
                **model_kwargs,
            )
        else:
            backbone = AutoModel.from_pretrained(model_name, config=config, **model_kwargs)
            if task == "contrastive":
                hf_model = ContrastiveHFModel(backbone)
            else:
                hf_model = MultimodalClassificationModel(backbone, n_labels, is_multilabel=is_multilabel)

        mm_block = MultimodalTextImageBlock(
            processor,
            task=task,
            include_labels=(task in {"classification", "vqa"}),
            is_multilabel=is_multilabel,
            labels=labels,
            max_length=max_length,
            augmentations=augmentations,
            processor_call_kwargs=processor_call_kwargs,
        )

        if valid_col and valid_col in df.columns:
            splitter = ColSplitter(valid_col)
        else:
            splitter = RandomSplitter(valid_pct, seed=seed)

        if task == "contrastive":
            target_block = CategoryBlock
            get_y = lambda r: "pair"
        elif is_multilabel:
            target_block = MultiCategoryBlock(encoded=True, vocab=label_cols)
            get_y = lambda r: [r[c] for c in label_cols]
        else:
            target_block = CategoryBlock
            get_y = lambda r: r[label_col]

        dblock = DataBlock(
            blocks=(mm_block, target_block),
            get_x=lambda r: (r[image_col], r[text_col]),
            get_y=get_y,
            splitter=splitter,
        )

        dl_kwargs.setdefault("bs", bs)
        if val_bs:
            dl_kwargs["val_bs"] = val_bs
        dls = dblock.dataloaders(df, **dl_kwargs)

        if metrics is None and task != "contrastive":
            if is_multilabel:
                metrics = [F1ScoreMulti(), accuracy_multi]
            else:
                metrics = [F1Score(), accuracy]
        if metrics is not None:
            learner_kwargs["metrics"] = metrics

        return cls(dls, hf_model, **learner_kwargs)

    @classmethod
    def from_hf(
        cls,
        dataset: str | Dataset | DatasetDict,
        model_name: str,
        *,
        image_col: str = "image",
        text_col: str = "text",
        label_col: str | list[str] | None = "label",
        task: str = "classification",
        is_multilabel: bool = False,
        subset: str | None = None,
        train_split: str = "train",
        valid_split: str | None = None,
        valid_pct: float = 0.2,
        max_samples: int | None = None,
        **kwargs,
    ) -> "MultimodalClassificationLearner":
        if isinstance(dataset, str):
            ds = load_dataset(dataset, subset, trust_remote_code=True)
        else:
            ds = dataset

        if isinstance(ds, DatasetDict):
            train_ds = ds[train_split]
            valid_ds = None
            if valid_split and valid_split in ds:
                valid_ds = ds[valid_split]
            else:
                for name in ("validation", "valid", "test", "dev"):
                    if name in ds:
                        valid_ds = ds[name]
                        break
        else:
            train_ds = ds
            valid_ds = None

        if max_samples:
            train_ds = train_ds.select(range(min(max_samples, len(train_ds))))
            if valid_ds:
                valid_ds = valid_ds.select(range(min(max_samples, len(valid_ds))))

        def _to_df(hfds: Dataset) -> pd.DataFrame:
            rows = []
            for ex in hfds:
                row = {image_col: ex[image_col], text_col: ex[text_col]}
                if task == "contrastive":
                    pass
                elif is_multilabel and isinstance(label_col, list):
                    for c in label_col:
                        row[c] = ex[c]
                else:
                    row[label_col] = ex[label_col]
                rows.append(row)
            return pd.DataFrame(rows)

        train_df = _to_df(train_ds)
        if valid_ds is not None:
            valid_df = _to_df(valid_ds)
            train_df["is_valid"] = False
            valid_df["is_valid"] = True
            df = pd.concat([train_df, valid_df], ignore_index=True)
            valid_col = "is_valid"
        else:
            df = train_df
            valid_col = None

        return cls.from_df(
            df,
            model_name,
            image_col=image_col,
            text_col=text_col,
            label_col=label_col,
            task=task,
            is_multilabel=is_multilabel,
            valid_col=valid_col,
            valid_pct=valid_pct,
            **kwargs,
        )
