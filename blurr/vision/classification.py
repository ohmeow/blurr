"""blurr.vision.classification — Image classification with Hugging Face + fastai."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from fastai.data.block import CategoryBlock, DataBlock, MultiCategoryBlock, TransformBlock
from fastai.data.core import DataLoaders
from fastai.data.transforms import ColSplitter, RandomSplitter
from fastai.metrics import F1Score, F1ScoreMulti, accuracy, accuracy_multi
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

from ..text._core import BlurrLearner
from ._core import VisionBatchDecodeTransform, VisionBatchTransform


# ===================================================================
# DataBlock input block
# ===================================================================

def VisionClassificationBlock(
    image_processor,
    *,
    include_labels: bool = True,
    is_multilabel: bool = False,
    labels: list[str] | None = None,
    augmentations: Any | None = None,
    processor_call_kwargs: dict[str, Any] | None = None,
):
    """Create a fastai ``TransformBlock`` for image classification."""

    batch_tfm = VisionBatchTransform(
        image_processor,
        include_labels=include_labels,
        is_multilabel=is_multilabel,
        labels=labels,
        augmentations=augmentations,
        processor_call_kwargs=processor_call_kwargs,
    )
    decode_tfm = VisionBatchDecodeTransform(image_processor)

    return TransformBlock(
        dls_kwargs={"before_batch": batch_tfm},
        batch_tfms=decode_tfm,
    )


def _find_vision_batch_tfm(dls) -> VisionBatchTransform:
    for tfm in dls.before_batch:
        if isinstance(tfm, VisionBatchTransform):
            return tfm
    raise ValueError("Could not find VisionBatchTransform in DataLoaders")


# ===================================================================
# Learner
# ===================================================================

class VisionClassificationLearner(BlurrLearner):
    """fastai learner for vision classification models (ViT/Swin/ConvNeXt/...)."""

    def blurr_predict(
        self,
        images,
        *,
        threshold: float = 0.5,
        rm_type_tfms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Predict on image path(s)/URL(s)/PIL image(s).

        Returns per-item dicts with ``label(s)``, ``score(s)``, and ``probs``.
        """
        if not isinstance(images, list):
            images = [images]

        tfm = _find_vision_batch_tfm(self.dls)
        labels = tfm.labels
        is_multilabel = tfm.is_multilabel

        dl = self.dls.test_dl(images, rm_type_tfms=rm_type_tfms, num_workers=0)
        preds = self.get_preds(dl=dl, with_input=False, with_decoded=False)
        preds_t = preds[0]

        results = []
        for i in range(len(images)):
            probs = preds_t[i]
            res: dict[str, Any] = {}

            if is_multilabel:
                indices = (probs > threshold).nonzero(as_tuple=True)[0].tolist()
                res["labels"] = [labels[j] for j in indices] if labels else indices
                res["scores"] = [probs[j].item() for j in indices]
            else:
                idx = probs.argmax().item()
                res["label"] = labels[idx] if labels else idx
                res["score"] = probs[idx].item()

            res["probs"] = probs.tolist()
            res["class_labels"] = labels if labels else list(range(len(probs)))
            results.append(res)

        return results

    # -----------------------------------------------------------------
    # Factory: from DataFrame
    # -----------------------------------------------------------------
    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        model_name: str,
        *,
        image_col: str = "image",
        label_col: str | list[str] = "label",
        is_multilabel: bool = False,
        valid_col: str | None = None,
        valid_pct: float = 0.2,
        n_labels: int | None = None,
        bs: int = 16,
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
    ) -> "VisionClassificationLearner":
        """Create a learner from a DataFrame of image paths + labels."""
        processor_kwargs = processor_kwargs or {}
        processor_call_kwargs = processor_call_kwargs or {}
        config_kwargs = config_kwargs or {}
        model_kwargs = model_kwargs or {}
        model_kwargs.setdefault("ignore_mismatched_sizes", True)
        dl_kwargs = dl_kwargs or {}
        learner_kwargs = learner_kwargs or {}

        if is_multilabel:
            label_cols = label_col if isinstance(label_col, list) else [label_col]
            labels = label_cols
            if n_labels is None:
                n_labels = len(label_cols)
        else:
            if n_labels is None:
                n_labels = df[label_col].nunique()
            labels = sorted(df[label_col].unique().tolist())

        config_kwargs = dict(config_kwargs)
        config_kwargs["num_labels"] = n_labels
        if is_multilabel:
            config_kwargs["problem_type"] = "multi_label_classification"
        else:
            config_kwargs.setdefault("id2label", {i: lbl for i, lbl in enumerate(labels)})
            config_kwargs.setdefault("label2id", {lbl: i for i, lbl in enumerate(labels)})

        config = AutoConfig.from_pretrained(model_name, **config_kwargs)
        image_processor = AutoImageProcessor.from_pretrained(model_name, **processor_kwargs)
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            config=config,
            **model_kwargs,
        )

        if valid_col and valid_col in df.columns:
            splitter = ColSplitter(valid_col)
        else:
            splitter = RandomSplitter(valid_pct, seed=seed)

        image_block = VisionClassificationBlock(
            image_processor,
            include_labels=True,
            is_multilabel=is_multilabel,
            labels=labels,
            augmentations=augmentations,
            processor_call_kwargs=processor_call_kwargs,
        )

        if is_multilabel:
            target_block = MultiCategoryBlock(encoded=True, vocab=label_cols)
            get_y = lambda r: [r[c] for c in label_cols]
        else:
            target_block = CategoryBlock
            get_y = lambda r: r[label_col]

        dblock = DataBlock(
            blocks=(image_block, target_block),
            get_x=lambda r: r[image_col],
            get_y=get_y,
            splitter=splitter,
        )

        dl_kwargs.setdefault("bs", bs)
        if val_bs:
            dl_kwargs["val_bs"] = val_bs
        dls = dblock.dataloaders(df, **dl_kwargs)

        if metrics is None:
            if is_multilabel:
                metrics = [F1ScoreMulti(), accuracy_multi]
            else:
                metrics = [F1Score(), accuracy]
        learner_kwargs["metrics"] = metrics

        return cls(dls, model, **learner_kwargs)

    # -----------------------------------------------------------------
    # Factory: from HuggingFace datasets
    # -----------------------------------------------------------------
    @classmethod
    def from_hf(
        cls,
        dataset: str | Dataset | DatasetDict,
        model_name: str,
        *,
        image_col: str = "image",
        label_col: str | list[str] = "label",
        is_multilabel: bool = False,
        subset: str | None = None,
        train_split: str = "train",
        valid_split: str | None = None,
        valid_pct: float = 0.2,
        max_samples: int | None = None,
        **kwargs,
    ) -> "VisionClassificationLearner":
        """Create a learner from a HuggingFace dataset (name or object)."""
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
            records = []
            for ex in hfds:
                row = {image_col: ex[image_col]}
                if is_multilabel and isinstance(label_col, list):
                    for c in label_col:
                        row[c] = ex[c]
                else:
                    row[label_col] = ex[label_col]
                records.append(row)
            return pd.DataFrame(records)

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
            label_col=label_col,
            is_multilabel=is_multilabel,
            valid_col=valid_col,
            valid_pct=valid_pct,
            **kwargs,
        )

    # -----------------------------------------------------------------
    # Factory: from pre-built DataLoaders
    # -----------------------------------------------------------------
    @classmethod
    def from_data(
        cls,
        dls: DataLoaders,
        model_name: str,
        *,
        n_labels: int | None = None,
        is_multilabel: bool = False,
        config_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        learner_kwargs: dict[str, Any] | None = None,
    ) -> "VisionClassificationLearner":
        """Create a learner from pre-built ``DataLoaders``."""
        config_kwargs = config_kwargs or {}
        model_kwargs = model_kwargs or {}
        model_kwargs.setdefault("ignore_mismatched_sizes", True)
        learner_kwargs = learner_kwargs or {}

        if n_labels is None:
            if hasattr(dls, "vocab") and dls.vocab is not None:
                try:
                    n_labels = len(dls.vocab)
                except Exception:
                    n_labels = None
            if n_labels is None and hasattr(dls, "c"):
                n_labels = dls.c
            if n_labels is None:
                raise ValueError("Could not infer n_labels from DataLoaders; pass n_labels explicitly")

        config_kwargs = dict(config_kwargs)
        config_kwargs["num_labels"] = n_labels
        if is_multilabel:
            config_kwargs["problem_type"] = "multi_label_classification"

        config = AutoConfig.from_pretrained(model_name, **config_kwargs)
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            config=config,
            **model_kwargs,
        )

        return cls(dls, model, **learner_kwargs)

    # -----------------------------------------------------------------
    # Compatibility factory: from_data for generic data sources
    # -----------------------------------------------------------------
    @classmethod
    def from_data_source(
        cls,
        data: pd.DataFrame | Dataset | DatasetDict,
        model_name: str,
        **kwargs,
    ) -> "VisionClassificationLearner":
        """Dispatch from DataFrame/HF dataset sources.

        Note: ``from_data`` is reserved for pre-built DataLoaders.
        """
        if isinstance(data, pd.DataFrame):
            return cls.from_df(data, model_name, **kwargs)
        if isinstance(data, (Dataset, DatasetDict)):
            return cls.from_hf(data, model_name, **kwargs)
        raise TypeError(f"Unsupported data type: {type(data)}")
