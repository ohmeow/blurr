"""blurr.text.classification — Sequence classification (single-label & multi-label).

Usage::

    from blurr.text.classification import TextClassificationLearner

    # From a HuggingFace dataset
    learn = TextClassificationLearner.from_hf(
        "imdb", model_name="bert-base-uncased",
        text_col="text", label_col="label",
    )
    learn.fit_one_cycle(3, 2e-5)
    learn.blurr_predict("This movie was great!")

    # From a DataFrame
    learn = TextClassificationLearner.from_df(
        df, model_name="bert-base-uncased",
        text_col="text", label_col="label",
    )

    # Multi-label
    learn = TextClassificationLearner.from_df(
        df, model_name="bert-base-uncased",
        text_col="text", label_col=["toxic", "severe_toxic", "obscene"],
        is_multilabel=True,
    )
"""

from __future__ import annotations

import os
from functools import partial
from typing import Any, Callable

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset
from fastai.callback.core import Callback
from fastai.data.block import CategoryBlock, DataBlock, MultiCategoryBlock
from fastai.data.core import DataLoaders
from fastai.data.transforms import ColSplitter, RandomSplitter
from fastai.losses import BCEWithLogitsLossFlat, CrossEntropyLossFlat
from fastai.metrics import F1Score, F1ScoreMulti, accuracy, accuracy_multi
from fastai.text.data import SortedDL
from fastai.torch_core import TensorBase, tensor
from fasttransform import Transform
from fastcore.basics import store_attr
from fastcore.foundation import L
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ._core import (
    BlurrLearner,
    HFModelCallback,
    HFModelWrapper,
    PreCalculatedBCELoss,
    PreCalculatedCrossEntropyLoss,
    PreCalculatedLoss,
    ensure_pad_token,
    hf_splitter,
    load_hf_objects,
)


# ===================================================================
# Types for fastai show methods
# ===================================================================

class TextInput(TensorBase):
    """Marker type for decoded text inputs — used by fastai's type-dispatched show."""
    pass


# ===================================================================
# Batch Transforms
# ===================================================================

class ClassificationBatchTokenizeTransform(Transform):
    """On-the-fly batch tokenization for classification tasks.

    This transform sits in ``before_batch`` and tokenizes raw text strings
    into tensors right before they enter the model. This avoids needing to
    pre-tokenize the entire dataset.
    """

    def __init__(
        self,
        hf_tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int | None = None,
        padding: bool | str = True,
        truncation: bool | str = True,
        include_labels: bool = True,
        is_multilabel: bool = False,
        labels: list[str] | None = None,
    ):
        self.hf_tokenizer = hf_tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.include_labels = include_labels
        self.is_multilabel = is_multilabel
        self.labels = labels

    def encodes(self, samples):
        samples = L(samples)
        if len(samples) == 0:
            return samples

        # Extract text inputs
        texts = [s[0] for s in samples]
        # Handle text pairs: if x is a tuple/list of 2 strings
        text_pairs = None
        if isinstance(texts[0], (list, tuple)) and len(texts[0]) == 2:
            text_pairs = [t[1] for t in texts]
            texts = [t[0] for t in texts]

        # Tokenize
        inputs = self.hf_tokenizer(
            texts,
            text_pairs,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )

        # Build updated samples: ({input_dict}, target)
        updated = []
        for idx, sample in enumerate(samples):
            inp_dict = {k: inputs[k][idx] for k in inputs.keys()}
            targets = sample[1:]
            if self.include_labels and len(targets) > 0:
                if self.is_multilabel:
                    inp_dict["labels"] = targets[0].float()
                else:
                    inp_dict["labels"] = targets[0]
            updated.append((inp_dict, *targets))

        return updated


class ClassificationBatchDecodeTransform(Transform):
    """Cast input_ids to TextInput for fastai's show methods."""

    def __init__(self, hf_tokenizer: PreTrainedTokenizerBase):
        self.hf_tokenizer = hf_tokenizer

    def decodes(self, x: dict) -> TextInput:
        return TextInput(x["input_ids"])


# ===================================================================
# Sort function for SortedDL
# ===================================================================

def _sort_by_length(example, hf_tokenizer: PreTrainedTokenizerBase):
    """Sort by tokenized length for efficient batching."""
    text = example[0]
    if isinstance(text, (list, tuple)):
        text = text[0]
    return len(hf_tokenizer.tokenize(str(text)))


# ===================================================================
# TextBlock for DataBlock API
# ===================================================================

def ClassificationTextBlock(
    hf_tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int | None = None,
    padding: bool | str = True,
    truncation: bool | str = True,
    include_labels: bool = True,
    is_multilabel: bool = False,
    labels: list[str] | None = None,
):
    """Create a fastai TransformBlock for text classification.

    Returns a block suitable for use in a DataBlock.
    """
    from fastai.data.block import TransformBlock

    batch_tfm = ClassificationBatchTokenizeTransform(
        hf_tokenizer,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        include_labels=include_labels,
        is_multilabel=is_multilabel,
        labels=labels,
    )
    decode_tfm = ClassificationBatchDecodeTransform(hf_tokenizer)
    sort_func = partial(_sort_by_length, hf_tokenizer=hf_tokenizer)

    return TransformBlock(
        dl_type=partial(SortedDL, sort_func=sort_func),
        dls_kwargs={"before_batch": batch_tfm},
        batch_tfms=decode_tfm,
    )


# ===================================================================
# Show helpers
# ===================================================================

def _find_batch_tfm(dls) -> ClassificationBatchTokenizeTransform:
    """Find our batch tokenize transform in the DataLoaders pipeline."""
    for tfm in dls.before_batch:
        if isinstance(tfm, ClassificationBatchTokenizeTransform):
            return tfm
    for tfm in dls.after_batch:
        if isinstance(tfm, ClassificationBatchDecodeTransform):
            return tfm
    raise ValueError("Could not find ClassificationBatchTokenizeTransform in DataLoaders")


# ===================================================================
# Learner
# ===================================================================

class TextClassificationLearner(BlurrLearner):
    """A fastai Learner for text classification (single-label or multi-label).

    Typically constructed via the factory methods :meth:`from_df`,
    :meth:`from_hf`, or :meth:`from_data`.
    """

    def __init__(self, dls: DataLoaders, hf_model: PreTrainedModel, **kwargs):
        super().__init__(dls, hf_model, **kwargs)
        # Store references for prediction
        self._hf_tokenizer = None
        self._labels = None
        self._is_multilabel = False

    def blurr_predict(
        self,
        texts: str | list[str],
        *,
        rm_type_tfms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Predict on raw text(s).

        Returns a list of dicts with keys:
          - ``label`` / ``labels``: predicted label(s)
          - ``score`` / ``scores``: confidence(s)
          - ``probs``: full probability distribution
          - ``class_labels``: list of all class names
        """
        if isinstance(texts, str):
            texts = [texts]

        # Get tokenizer + labels from dataloader transforms
        tfm = _find_batch_tfm(self.dls)
        hf_tokenizer = tfm.hf_tokenizer
        labels = tfm.labels
        is_multilabel = tfm.is_multilabel

        dl = self.dls.test_dl(texts, rm_type_tfms=rm_type_tfms, num_workers=0)
        preds_t, _, decoded = self.get_preds(dl=dl, with_input=False, with_decoded=True)

        results = []
        for i in range(len(texts)):
            probs = preds_t[i]
            res: dict[str, Any] = {}

            if is_multilabel:
                # Sigmoid was applied by BCEWithLogitsLossFlat activation
                indices = (probs > 0.5).nonzero(as_tuple=True)[0].tolist()
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
        text_col: str = "text",
        label_col: str | list[str] = "label",
        is_multilabel: bool = False,
        valid_col: str | None = None,
        valid_pct: float = 0.2,
        n_labels: int | None = None,
        max_length: int | None = None,
        bs: int = 16,
        val_bs: int | None = None,
        seed: int = 42,
        metrics: list | None = None,
        config_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
        dl_kwargs: dict | None = None,
        learner_kwargs: dict | None = None,
    ) -> "TextClassificationLearner":
        """Create a learner from a pandas DataFrame.

        Args:
            df: DataFrame with text and label columns.
            model_name: HuggingFace model name or path.
            text_col: Column(s) containing text. Can be a single string or
                a list of two strings for text-pair tasks.
            label_col: Column(s) containing labels. For multi-label, pass a
                list of binary column names.
            is_multilabel: Whether this is a multi-label task.
            valid_col: Column name for train/valid split (True = valid).
            valid_pct: Fraction for random validation split (if valid_col is None).
            n_labels: Number of labels (inferred if None).
            max_length: Max tokenization length.
            bs: Training batch size.
            val_bs: Validation batch size (defaults to 2*bs).
            seed: Random seed.
            metrics: fastai metrics (inferred if None).
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

        # Infer n_labels
        if is_multilabel:
            label_cols = label_col if isinstance(label_col, list) else [label_col]
            if n_labels is None:
                n_labels = len(label_cols)
            labels = label_cols
        else:
            if n_labels is None:
                n_labels = df[label_col].nunique()
            labels = sorted(df[label_col].unique().tolist())

        # Load HF objects
        config_kwargs["num_labels"] = n_labels
        if is_multilabel:
            config_kwargs["problem_type"] = "multi_label_classification"

        config, tokenizer, model = load_hf_objects(
            model_name,
            AutoModelForSequenceClassification,
            config_kwargs=config_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            model_kwargs=model_kwargs,
        )
        ensure_pad_token(config, tokenizer, model)

        # Splitter
        if valid_col and valid_col in df.columns:
            splitter = ColSplitter(valid_col)
        else:
            splitter = RandomSplitter(valid_pct, seed=seed)

        # Blocks
        text_block = ClassificationTextBlock(
            tokenizer,
            max_length=max_length,
            is_multilabel=is_multilabel,
            labels=labels,
        )

        if is_multilabel:
            target_block = MultiCategoryBlock(encoded=True, vocab=label_cols)
        else:
            target_block = CategoryBlock

        # Getters
        if isinstance(text_col, list) and len(text_col) == 2:
            get_x = lambda r: (r[text_col[0]], r[text_col[1]])
        else:
            get_x = lambda r: r[text_col]

        if is_multilabel:
            get_y = lambda r: [r[c] for c in label_cols]
        else:
            get_y = lambda r: r[label_col]

        dblock = DataBlock(
            blocks=(text_block, target_block),
            get_x=get_x,
            get_y=get_y,
            splitter=splitter,
        )

        dl_kwargs.setdefault("bs", bs)
        if val_bs:
            dl_kwargs["val_bs"] = val_bs
        dls = dblock.dataloaders(df, **dl_kwargs)

        # Metrics
        if metrics is None:
            if is_multilabel:
                metrics = [F1ScoreMulti(), accuracy_multi]
            else:
                metrics = [F1Score(), accuracy]
        learner_kwargs["metrics"] = metrics

        return cls(dls, model, **learner_kwargs)

    # -----------------------------------------------------------------
    # Factory: from HuggingFace Dataset
    # -----------------------------------------------------------------
    @classmethod
    def from_hf(
        cls,
        dataset: str | DatasetDict | Dataset,
        model_name: str,
        *,
        text_col: str = "text",
        label_col: str | list[str] = "label",
        is_multilabel: bool = False,
        subset: str | None = None,
        train_split: str = "train",
        valid_split: str | None = None,
        valid_pct: float = 0.2,
        n_labels: int | None = None,
        max_length: int | None = None,
        max_samples: int | None = None,
        bs: int = 16,
        val_bs: int | None = None,
        seed: int = 42,
        metrics: list | None = None,
        config_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
        dl_kwargs: dict | None = None,
        learner_kwargs: dict | None = None,
    ) -> "TextClassificationLearner":
        """Create a learner from a HuggingFace dataset (name or object).

        Args:
            dataset: HF dataset name (e.g., ``"imdb"``) or a loaded Dataset/DatasetDict.
            model_name: HuggingFace model name or path.
            text_col: Column containing text.
            label_col: Column(s) containing labels.
            is_multilabel: Whether multi-label.
            subset: Dataset subset/config name (e.g., ``"sst2"`` for ``"glue"``).
            train_split: Name of train split.
            valid_split: Name of validation split (auto-detected if None).
            valid_pct: Fraction for random split if no valid split exists.
            n_labels: Number of labels (inferred if None).
            max_length: Max tokenization length.
            max_samples: Limit dataset size (useful for debugging).
            bs: Training batch size.
            val_bs: Validation batch size.
            seed: Random seed.
            metrics: fastai metrics.
            config_kwargs: Extra kwargs for HF config.
            tokenizer_kwargs: Extra kwargs for HF tokenizer.
            model_kwargs: Extra kwargs for HF model.
            dl_kwargs: Extra kwargs for DataLoaders.
            learner_kwargs: Extra kwargs for Learner.
        """
        # Load dataset
        if isinstance(dataset, str):
            ds = load_dataset(dataset, subset, trust_remote_code=True)
        else:
            ds = dataset

        # Get splits
        if isinstance(ds, DatasetDict):
            train_ds = ds[train_split]
            if valid_split and valid_split in ds:
                valid_ds = ds[valid_split]
            else:
                # Try common names
                for name in ("validation", "valid", "test", "dev"):
                    if name in ds:
                        valid_ds = ds[name]
                        break
                else:
                    valid_ds = None
        else:
            train_ds = ds
            valid_ds = None

        # Limit samples
        if max_samples:
            train_ds = train_ds.select(range(min(max_samples, len(train_ds))))
            if valid_ds:
                valid_ds = valid_ds.select(range(min(max_samples, len(valid_ds))))

        # Convert to DataFrame and mark valid
        train_df = train_ds.to_pandas()
        if valid_ds is not None:
            valid_df = valid_ds.to_pandas()
            train_df["is_valid"] = False
            valid_df["is_valid"] = True
            df = pd.concat([train_df, valid_df], ignore_index=True)
        else:
            df = train_df

        return cls.from_df(
            df,
            model_name,
            text_col=text_col,
            label_col=label_col,
            is_multilabel=is_multilabel,
            valid_col="is_valid" if valid_ds is not None else None,
            valid_pct=valid_pct,
            n_labels=n_labels,
            max_length=max_length,
            bs=bs,
            val_bs=val_bs,
            seed=seed,
            metrics=metrics,
            config_kwargs=config_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            model_kwargs=model_kwargs,
            dl_kwargs=dl_kwargs,
            learner_kwargs=learner_kwargs,
        )

    # -----------------------------------------------------------------
    # Factory: from_data (generic — accepts DF, HF dataset, or file path)
    # -----------------------------------------------------------------
    @classmethod
    def from_data(
        cls,
        data: pd.DataFrame | str | os.PathLike | Dataset | DatasetDict,
        model_name: str,
        **kwargs,
    ) -> "TextClassificationLearner":
        """Universal factory that dispatches to from_df or from_hf based on data type."""
        if isinstance(data, pd.DataFrame):
            return cls.from_df(data, model_name, **kwargs)
        elif isinstance(data, (Dataset, DatasetDict)):
            return cls.from_hf(data, model_name, **kwargs)
        elif isinstance(data, (str, os.PathLike)):
            path = str(data)
            # Check if it's a HF dataset name or a file path
            if os.path.exists(path):
                if path.endswith(".csv"):
                    df = pd.read_csv(path)
                elif path.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(path)
                elif path.endswith((".json", ".jsonl")):
                    df = pd.read_json(path, orient="records", lines=path.endswith(".jsonl"))
                else:
                    raise ValueError(f"Unsupported file format: {path}")
                return cls.from_df(df, model_name, **kwargs)
            else:
                # Assume HF dataset name
                return cls.from_hf(path, model_name, **kwargs)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
