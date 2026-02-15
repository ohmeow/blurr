"""blurr.text.token_classification — Token classification (NER, POS tagging, etc.).

Usage::

    from blurr.text.token_classification import TokenClassificationLearner

    learn = TokenClassificationLearner.from_hf(
        "conll2003", model_name="bert-base-uncased",
        tokens_col="tokens", labels_col="ner_tags",
    )
    learn.fit_one_cycle(3, 2e-5)
    learn.blurr_predict_tokens("Hugging Face is based in New York City.")
"""

from __future__ import annotations

import os
from functools import partial
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset
from fastai.callback.core import Callback
from fastai.data.block import CategoryBlock, DataBlock, TransformBlock
from fastai.data.core import DataLoaders
from fastai.data.transforms import ColSplitter, RandomSplitter
from fastai.losses import CrossEntropyLossFlat
from fastai.metrics import accuracy
from fastai.text.data import SortedDL
from fastai.torch_core import TensorBase, tensor
from fasttransform import Transform
from fastcore.basics import store_attr
from fastcore.foundation import L

try:
    from seqeval import metrics as seq_metrics
    HAS_SEQEVAL = True
except ImportError:
    HAS_SEQEVAL = False

from transformers import (
    AutoModelForTokenClassification,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ._core import (
    BlurrLearner,
    HFModelCallback,
    PreCalculatedCrossEntropyLoss,
    PreCalculatedLoss,
    ensure_pad_token,
    hf_splitter,
    load_hf_objects,
)


# ===================================================================
# Types
# ===================================================================

class TokenClassTextInput(TensorBase):
    """Marker type for token classification inputs."""
    pass


class TokenTensorCategory(TensorBase):
    """Marker type for token-level labels."""
    pass


# ===================================================================
# Batch Transforms
# ===================================================================

IGNORE_TOKEN_ID = CrossEntropyLossFlat().ignore_index  # -100


class TokenClassBatchTokenizeTransform(Transform):
    """On-the-fly batch tokenization for token classification.

    Handles subword token alignment: each word may be split into multiple
    subword tokens, and we need to assign labels to the first subtoken only
    (other subtokens get ``ignore_token_id = -100``).
    """

    def __init__(
        self,
        hf_tokenizer: PreTrainedTokenizerBase,
        label2id: dict[str, int],
        *,
        max_length: int | None = None,
        padding: bool | str = True,
        truncation: bool | str = True,
        include_labels: bool = True,
        label_all_tokens: bool = False,
        ignore_token_id: int = IGNORE_TOKEN_ID,
    ):
        self.hf_tokenizer = hf_tokenizer
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.include_labels = include_labels
        self.label_all_tokens = label_all_tokens
        self.ignore_token_id = ignore_token_id

    def encodes(self, samples):
        samples = L(samples)
        if len(samples) == 0:
            return samples

        # samples: [(tokens_list, labels_list), ...]
        all_tokens = [s[0] for s in samples]
        all_labels = [s[1] for s in samples]

        # Tokenize as pre-split words
        inputs = self.hf_tokenizer(
            all_tokens,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            is_split_into_words=True,
            return_tensors="pt",
        )

        # Align labels with subword tokens
        aligned_labels_batch = []
        for idx, label_seq in enumerate(all_labels):
            word_ids = inputs.word_ids(batch_index=idx)
            aligned = []
            prev_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    aligned.append(self.ignore_token_id)
                elif word_id != prev_word_id:
                    # First token of a word → use the word's label
                    if word_id < len(label_seq):
                        lbl = label_seq[word_id]
                        aligned.append(lbl if isinstance(lbl, int) else self.label2id.get(lbl, 0))
                    else:
                        aligned.append(self.ignore_token_id)
                else:
                    # Subsequent subtoken
                    if self.label_all_tokens and word_id < len(label_seq):
                        lbl = label_seq[word_id]
                        aligned.append(lbl if isinstance(lbl, int) else self.label2id.get(lbl, 0))
                    else:
                        aligned.append(self.ignore_token_id)
                prev_word_id = word_id
            aligned_labels_batch.append(aligned)

        labels_tensor = torch.tensor(aligned_labels_batch, dtype=torch.long)

        # Build samples
        updated = []
        for idx, sample in enumerate(samples):
            inp_dict = {k: inputs[k][idx] for k in inputs.keys()}
            if self.include_labels:
                inp_dict["labels"] = labels_tensor[idx]
            updated.append((inp_dict, labels_tensor[idx]))

        return updated


class TokenClassBatchDecodeTransform(Transform):
    """Cast input_ids to TokenClassTextInput for fastai show methods."""

    def __init__(self, hf_tokenizer: PreTrainedTokenizerBase):
        self.hf_tokenizer = hf_tokenizer

    def decodes(self, x: dict) -> TokenClassTextInput:
        return TokenClassTextInput(x["input_ids"])


# ===================================================================
# Token Classification TextBlock
# ===================================================================

def TokenClassTextBlock(
    hf_tokenizer: PreTrainedTokenizerBase,
    label2id: dict[str, int],
    *,
    max_length: int | None = None,
    label_all_tokens: bool = False,
):
    """TransformBlock for token classification."""

    batch_tfm = TokenClassBatchTokenizeTransform(
        hf_tokenizer, label2id, max_length=max_length, label_all_tokens=label_all_tokens,
    )
    decode_tfm = TokenClassBatchDecodeTransform(hf_tokenizer)

    def sort_func(example, hf_tokenizer=hf_tokenizer):
        return len(example[0])  # Sort by number of tokens (pre-split)

    return TransformBlock(
        dl_type=partial(SortedDL, sort_func=sort_func),
        dls_kwargs={"before_batch": batch_tfm},
        batch_tfms=decode_tfm,
    )


# ===================================================================
# Entity-level Metrics Callback (seqeval)
# ===================================================================

class TokenClassMetricsCallback(Callback):
    """Callback that computes entity-level metrics using seqeval.

    Adds accuracy, precision, recall, and F1 as custom metrics.
    Also stores a full classification report after each validation epoch.
    """

    order = 9  # Before Recorder

    def __init__(
        self,
        id2label: dict[int, str],
        ignore_token_id: int = IGNORE_TOKEN_ID,
        metric_names: list[str] | None = None,
    ):
        self.id2label = id2label
        self.ignore_token_id = ignore_token_id
        self.metric_names = metric_names or ["accuracy", "precision", "recall", "f1"]
        self._metric_vals = {k: 0.0 for k in self.metric_names}
        self._setup_done = False

    def before_fit(self):
        if self._setup_done:
            return
        from fastai.metrics import ValueMetric
        custom_metrics = L([
            ValueMetric(partial(self._get_metric, k=k), k)
            for k in self.metric_names
        ])
        self.learn.metrics = self.learn.metrics + custom_metrics
        self.learn.token_classification_report = None
        self._setup_done = True

    def before_validate(self):
        self._results = []

    def after_batch(self):
        if self.training or self.learn.y is None:
            return

        preds = self.pred.argmax(dim=-1)
        targs = self.yb[0]

        for i in range(targs.shape[0]):
            pred_labels, targ_labels = [], []
            for j in range(targs.shape[1]):
                if targs[i, j] != self.ignore_token_id:
                    pred_labels.append(self.id2label.get(preds[i, j].item(), "O"))
                    targ_labels.append(self.id2label.get(targs[i, j].item(), "O"))
            self._results.append((pred_labels, targ_labels))

    def after_validate(self):
        if not self._results or not HAS_SEQEVAL:
            return

        all_preds, all_targs = zip(*self._results)
        all_preds, all_targs = list(all_preds), list(all_targs)

        if "accuracy" in self._metric_vals:
            self._metric_vals["accuracy"] = seq_metrics.accuracy_score(all_targs, all_preds)
        if "precision" in self._metric_vals:
            self._metric_vals["precision"] = seq_metrics.precision_score(all_targs, all_preds)
        if "recall" in self._metric_vals:
            self._metric_vals["recall"] = seq_metrics.recall_score(all_targs, all_preds)
        if "f1" in self._metric_vals:
            self._metric_vals["f1"] = seq_metrics.f1_score(all_targs, all_preds)

        try:
            self.learn.token_classification_report = seq_metrics.classification_report(all_targs, all_preds)
        except Exception:
            pass

    def _get_metric(self, k: str) -> float:
        return self._metric_vals.get(k, 0.0)


# ===================================================================
# Entity Aggregation for Predictions
# ===================================================================

def _aggregate_entities(
    tokens: list[str],
    preds: list[int],
    probs: torch.Tensor,
    offsets: list[tuple[int, int]],
    id2label: dict[int, str],
    *,
    strategy: str = "simple",
) -> list[dict[str, Any]]:
    """Aggregate subword predictions into entity spans.

    Args:
        strategy: One of 'none' (token-level), 'simple', 'first', 'max', 'average'.
    """
    if strategy == "none":
        results = []
        for i, (tok, pred_id) in enumerate(zip(tokens, preds)):
            label = id2label.get(pred_id, "O")
            if label == "O":
                continue
            results.append({
                "entity": label,
                "score": probs[i][pred_id].item(),
                "word": tok,
                "start": offsets[i][0] if offsets else None,
                "end": offsets[i][1] if offsets else None,
            })
        return results

    # Grouped strategies
    results = []
    i = 0
    while i < len(preds):
        label = id2label.get(preds[i], "O")
        if label == "O":
            i += 1
            continue

        # Strip B-/I- prefix
        entity_type = label.split("-", 1)[-1] if "-" in label else label
        start = offsets[i][0] if offsets else None
        end = offsets[i][1] if offsets else None
        scores = [probs[i][preds[i]].item()]
        word_tokens = [tokens[i]]

        # Collect continuation tokens
        j = i + 1
        while j < len(preds):
            next_label = id2label.get(preds[j], "O")
            if next_label == "O":
                break
            next_type = next_label.split("-", 1)[-1] if "-" in next_label else next_label
            if next_type != entity_type:
                break
            if next_label.startswith("B-"):
                break  # New entity of same type
            end = offsets[j][1] if offsets else None
            if strategy != "first":
                scores.append(probs[j][preds[j]].item())
            word_tokens.append(tokens[j])
            j += 1

        # Score aggregation
        if strategy == "max":
            score = max(scores)
        else:  # simple, first, average
            score = sum(scores) / len(scores)

        word = " ".join(word_tokens).replace(" ##", "").replace("##", "")
        results.append({
            "entity_group": entity_type,
            "score": score,
            "word": word,
            "start": start,
            "end": end,
        })
        i = j

    return results


# ===================================================================
# Learner
# ===================================================================

class TokenClassificationLearner(BlurrLearner):
    """A fastai Learner for token classification (NER, POS, chunking, etc.)."""

    def __init__(
        self,
        dls: DataLoaders,
        hf_model: PreTrainedModel,
        *,
        id2label: dict[int, str] | None = None,
        **kwargs,
    ):
        self._id2label = id2label or {}
        super().__init__(dls, hf_model, **kwargs)

    def blurr_predict_tokens(
        self,
        texts: str | list[str],
        *,
        aggregation_strategy: str = "simple",
    ) -> list[list[dict[str, Any]]]:
        """Predict named entities in raw text(s).

        Args:
            texts: One or more raw text strings.
            aggregation_strategy: How to aggregate subword predictions.
                One of: ``'none'``, ``'simple'``, ``'first'``, ``'max'``, ``'average'``.

        Returns:
            List of entity lists, one per input text.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Find tokenizer
        tfm = self._find_tok_tfm()
        hf_tokenizer = tfm.hf_tokenizer
        id2label = tfm.id2label

        # Tokenize
        inputs = hf_tokenizer(
            texts,
            return_offsets_mapping=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        offset_mapping = inputs.pop("offset_mapping")

        # Forward pass
        device = next(self.model.hf_model.parameters()).device
        model_inputs = {k: v.to(device) for k, v in inputs.items()}
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(model_inputs)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)

        # Build results
        results = []
        for idx in range(len(texts)):
            tokens = inputs.tokens(idx) if hasattr(inputs, "tokens") else hf_tokenizer.convert_ids_to_tokens(inputs["input_ids"][idx])
            offsets = offset_mapping[idx].tolist()
            preds_i = predictions[idx].tolist()
            probs_i = probs[idx]

            entities = _aggregate_entities(
                tokens, preds_i, probs_i, offsets, id2label,
                strategy=aggregation_strategy,
            )
            results.append(entities)

        return results

    def _find_tok_tfm(self) -> TokenClassBatchTokenizeTransform:
        for tfm in self.dls.before_batch:
            if isinstance(tfm, TokenClassBatchTokenizeTransform):
                return tfm
        raise ValueError("Could not find TokenClassBatchTokenizeTransform")

    # -----------------------------------------------------------------
    # Factory: from HuggingFace Dataset
    # -----------------------------------------------------------------
    @classmethod
    def from_hf(
        cls,
        dataset: str | DatasetDict | Dataset,
        model_name: str,
        *,
        tokens_col: str = "tokens",
        labels_col: str = "ner_tags",
        label_names: list[str] | None = None,
        subset: str | None = None,
        train_split: str = "train",
        valid_split: str | None = None,
        valid_pct: float = 0.2,
        max_length: int | None = None,
        max_samples: int | None = None,
        label_all_tokens: bool = False,
        bs: int = 16,
        val_bs: int | None = None,
        seed: int = 42,
        metrics: list | None = None,
        config_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
        dl_kwargs: dict | None = None,
        learner_kwargs: dict | None = None,
    ) -> "TokenClassificationLearner":
        """Create a learner from a HuggingFace dataset.

        Args:
            dataset: HF dataset name or loaded Dataset/DatasetDict.
            model_name: HuggingFace model name.
            tokens_col: Column with pre-tokenized word lists.
            labels_col: Column with label IDs or strings per token.
            label_names: Explicit list of label names. If None, auto-detected
                from the HF dataset features.
            subset: Dataset config name.
            train_split: Name of training split.
            valid_split: Name of validation split.
            valid_pct: Random validation fraction.
            max_length: Max tokenization length.
            max_samples: Limit dataset size.
            label_all_tokens: Whether to label all subword tokens (vs. first only).
            bs: Batch size.
            val_bs: Validation batch size.
            seed: Random seed.
            metrics: fastai metrics.
            config_kwargs: Extra kwargs for HF config.
            tokenizer_kwargs: Extra kwargs for HF tokenizer.
            model_kwargs: Extra kwargs for HF model.
            dl_kwargs: Extra kwargs for DataLoaders.
            learner_kwargs: Extra kwargs for Learner.
        """
        config_kwargs = config_kwargs or {}
        tokenizer_kwargs = tokenizer_kwargs or {}
        model_kwargs = model_kwargs or {}
        dl_kwargs = dl_kwargs or {}
        learner_kwargs = learner_kwargs or {}

        # Load dataset
        if isinstance(dataset, str):
            ds = load_dataset(dataset, subset, trust_remote_code=True)
        else:
            ds = dataset

        # Get splits
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

        # Limit samples
        if max_samples:
            train_ds = train_ds.select(range(min(max_samples, len(train_ds))))
            if valid_ds:
                valid_ds = valid_ds.select(range(min(max_samples, len(valid_ds))))

        # Detect label names from features
        if label_names is None:
            feat = train_ds.features.get(labels_col)
            if feat is not None and hasattr(feat, "feature") and hasattr(feat.feature, "names"):
                label_names = feat.feature.names
            else:
                # Fallback: collect all unique labels
                all_labels = set()
                for example in train_ds:
                    all_labels.update(example[labels_col])
                label_names = sorted(all_labels)

        label2id = {name: i for i, name in enumerate(label_names)}
        id2label = {i: name for i, name in enumerate(label_names)}
        n_labels = len(label_names)

        # Load HF objects
        config_kwargs["num_labels"] = n_labels
        config_kwargs["id2label"] = id2label
        config_kwargs["label2id"] = label2id

        config, tokenizer, model = load_hf_objects(
            model_name,
            AutoModelForTokenClassification,
            config_kwargs=config_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            model_kwargs=model_kwargs,
        )
        ensure_pad_token(config, tokenizer, model)

        # Build DataFrame for DataBlock
        all_tokens = [ex[tokens_col] for ex in train_ds]
        all_labels = [ex[labels_col] for ex in train_ds]
        all_valid = [False] * len(train_ds)

        if valid_ds:
            all_tokens += [ex[tokens_col] for ex in valid_ds]
            all_labels += [ex[labels_col] for ex in valid_ds]
            all_valid += [True] * len(valid_ds)

        data_df = pd.DataFrame({
            "tokens": all_tokens,
            "labels": all_labels,
            "is_valid": all_valid,
        })

        # Splitter
        if valid_ds:
            splitter = ColSplitter("is_valid")
        else:
            splitter = RandomSplitter(valid_pct, seed=seed)

        # Blocks
        text_block = TokenClassTextBlock(
            tokenizer, label2id, max_length=max_length, label_all_tokens=label_all_tokens,
        )
        # Target block is noop — labels are handled inside batch tokenize transform
        target_block = TransformBlock()

        dblock = DataBlock(
            blocks=(text_block, target_block),
            get_x=lambda r: r["tokens"],
            get_y=lambda r: r["labels"],
            splitter=splitter,
        )

        dl_kwargs.setdefault("bs", bs)
        if val_bs:
            dl_kwargs["val_bs"] = val_bs
        dls = dblock.dataloaders(data_df, **dl_kwargs)

        # Metrics callback
        if HAS_SEQEVAL:
            metrics_cb = TokenClassMetricsCallback(id2label)
            cbs = learner_kwargs.pop("cbs", [])
            cbs.append(metrics_cb)
            learner_kwargs["cbs"] = cbs

        return cls(dls, model, id2label=id2label, **learner_kwargs)

    # -----------------------------------------------------------------
    # Factory: from DataFrame
    # -----------------------------------------------------------------
    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        model_name: str,
        *,
        tokens_col: str = "tokens",
        labels_col: str = "labels",
        label_names: list[str] | None = None,
        valid_col: str | None = None,
        valid_pct: float = 0.2,
        max_length: int | None = None,
        label_all_tokens: bool = False,
        bs: int = 16,
        val_bs: int | None = None,
        seed: int = 42,
        config_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
        dl_kwargs: dict | None = None,
        learner_kwargs: dict | None = None,
    ) -> "TokenClassificationLearner":
        """Create a learner from a DataFrame.

        The DataFrame should have columns with lists of tokens and
        corresponding lists of labels (either string labels or integer IDs).
        """
        config_kwargs = config_kwargs or {}
        tokenizer_kwargs = tokenizer_kwargs or {}
        model_kwargs = model_kwargs or {}
        dl_kwargs = dl_kwargs or {}
        learner_kwargs = learner_kwargs or {}

        # Detect label names
        if label_names is None:
            all_labels: set = set()
            for labels_list in df[labels_col]:
                all_labels.update(labels_list)
            label_names = sorted(all_labels, key=str)

        label2id = {name: i for i, name in enumerate(label_names)}
        id2label = {i: name for i, name in enumerate(label_names)}
        n_labels = len(label_names)

        # Load HF objects
        config_kwargs["num_labels"] = n_labels
        config_kwargs["id2label"] = id2label
        config_kwargs["label2id"] = label2id

        config, tokenizer, model = load_hf_objects(
            model_name,
            AutoModelForTokenClassification,
            config_kwargs=config_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            model_kwargs=model_kwargs,
        )
        ensure_pad_token(config, tokenizer, model)

        # Build a DataFrame with tokens/labels columns (keep list data as-is)
        data_df = pd.DataFrame({
            "tokens": df[tokens_col].tolist(),
            "labels": df[labels_col].tolist(),
        })
        if valid_col and valid_col in df.columns:
            data_df["is_valid"] = df[valid_col].tolist()

        # Splitter
        if valid_col and valid_col in df.columns:
            splitter = ColSplitter("is_valid")
        else:
            splitter = RandomSplitter(valid_pct, seed=seed)

        # Blocks
        text_block = TokenClassTextBlock(
            tokenizer, label2id, max_length=max_length, label_all_tokens=label_all_tokens,
        )
        target_block = TransformBlock()

        dblock = DataBlock(
            blocks=(text_block, target_block),
            get_x=lambda r: r["tokens"],
            get_y=lambda r: r["labels"],
            splitter=splitter,
        )

        dl_kwargs.setdefault("bs", bs)
        if val_bs:
            dl_kwargs["val_bs"] = val_bs
        dls = dblock.dataloaders(data_df, **dl_kwargs)

        # Metrics
        if HAS_SEQEVAL:
            metrics_cb = TokenClassMetricsCallback(id2label)
            cbs = learner_kwargs.pop("cbs", [])
            cbs.append(metrics_cb)
            learner_kwargs["cbs"] = cbs

        return cls(dls, model, id2label=id2label, **learner_kwargs)
