"""blurr.text.seq2seq — Sequence-to-sequence tasks (summarization, translation).

Usage::

    from blurr.text.seq2seq import Seq2SeqLearner

    # Summarization
    learn = Seq2SeqLearner.from_hf(
        "cnn_dailymail", "3.0.0",
        model_name="t5-small",
        input_col="article", target_col="highlights",
        prefix="summarize: ",
    )
    learn.fit_one_cycle(3, 2e-5)
    learn.blurr_generate("The quick brown fox jumped over the lazy dog.")

    # Translation
    learn = Seq2SeqLearner.from_hf(
        "wmt16", "de-en",
        model_name="Helsinki-NLP/opus-mt-de-en",
        input_col="de", target_col="en",
    )
"""

from __future__ import annotations

import os
from functools import partial
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from fastai.callback.core import Callback
from fastai.data.block import DataBlock, TransformBlock
from fastai.data.core import DataLoaders
from fastai.data.transforms import ColSplitter, RandomSplitter
from fastai.losses import CrossEntropyLossFlat
from fastai.metrics import ValueMetric
from fastai.text.data import SortedDL
from fastai.torch_core import TensorBase, tensor
from fasttransform import Transform
from fastcore.basics import store_attr
from fastcore.foundation import L
from transformers import (
    AutoModelForSeq2SeqLM,
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

class Seq2SeqTextInput(TensorBase):
    """Marker type for seq2seq inputs."""
    pass


# ===================================================================
# Batch Transforms
# ===================================================================

IGNORE_TOKEN_ID = CrossEntropyLossFlat().ignore_index  # -100


class Seq2SeqBatchTokenizeTransform(Transform):
    """On-the-fly batch tokenization for seq2seq tasks.

    Tokenizes both source and target texts, setting up decoder input IDs
    and proper label masking (pad tokens → -100).
    """

    def __init__(
        self,
        hf_tokenizer: PreTrainedTokenizerBase,
        *,
        max_source_length: int = 512,
        max_target_length: int = 128,
        padding: bool | str = True,
        truncation: bool | str = True,
        include_labels: bool = True,
        prefix: str = "",
    ):
        self.hf_tokenizer = hf_tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.padding = padding
        self.truncation = truncation
        self.include_labels = include_labels
        self.prefix = prefix

    def encodes(self, samples):
        samples = L(samples)
        if len(samples) == 0:
            return samples

        # Extract source and target texts
        sources = [self.prefix + str(s[0]) for s in samples]
        targets = [str(s[1]) for s in samples]

        # Tokenize inputs
        model_inputs = self.hf_tokenizer(
            sources,
            max_length=self.max_source_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )

        # Tokenize targets
        labels = self.hf_tokenizer(
            text_target=targets,
            max_length=self.max_target_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )

        # Replace pad tokens in labels with -100 (ignore index)
        label_ids = labels["input_ids"]
        if self.hf_tokenizer.pad_token_id is not None:
            label_ids = label_ids.masked_fill(
                label_ids == self.hf_tokenizer.pad_token_id,
                IGNORE_TOKEN_ID,
            )

        # Build samples
        updated = []
        for idx in range(len(samples)):
            inp_dict = {k: model_inputs[k][idx] for k in model_inputs.keys()}
            if self.include_labels:
                inp_dict["labels"] = label_ids[idx]
            updated.append((inp_dict, label_ids[idx]))

        return updated


class Seq2SeqBatchDecodeTransform(Transform):
    """Decode transform for seq2seq show methods."""

    def __init__(self, hf_tokenizer: PreTrainedTokenizerBase):
        self.hf_tokenizer = hf_tokenizer

    def decodes(self, x: dict) -> Seq2SeqTextInput:
        return Seq2SeqTextInput(x["input_ids"])


# ===================================================================
# Seq2Seq TextBlock
# ===================================================================

def Seq2SeqTextBlock(
    hf_tokenizer: PreTrainedTokenizerBase,
    *,
    max_source_length: int = 512,
    max_target_length: int = 128,
    prefix: str = "",
):
    """TransformBlock for seq2seq tasks."""

    batch_tfm = Seq2SeqBatchTokenizeTransform(
        hf_tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        prefix=prefix,
    )
    decode_tfm = Seq2SeqBatchDecodeTransform(hf_tokenizer)

    def sort_func(example, hf_tokenizer=hf_tokenizer):
        text = example[0] if isinstance(example[0], str) else str(example[0])
        return len(hf_tokenizer.tokenize(text))

    return TransformBlock(
        dl_type=partial(SortedDL, sort_func=sort_func),
        dls_kwargs={"before_batch": batch_tfm},
        batch_tfms=decode_tfm,
    )


# ===================================================================
# Metrics Callback (ROUGE, BLEU, etc.)
# ===================================================================

class Seq2SeqMetricsCallback(Callback):
    """Callback that computes generation-based metrics (ROUGE, BLEU, etc.).

    Because text generation is expensive, metrics are computed on the
    full validation set after each epoch (or as configured).
    """

    order = 9

    def __init__(
        self,
        hf_tokenizer: PreTrainedTokenizerBase,
        *,
        metric_names: list[str] | None = None,
        text_gen_kwargs: dict | None = None,
        calc_every: str = "epoch",
    ):
        self.hf_tokenizer = hf_tokenizer
        self.metric_names = metric_names or ["rouge1", "rouge2", "rougeL"]
        self.text_gen_kwargs = text_gen_kwargs or {"max_new_tokens": 128, "num_beams": 4}
        self.calc_every = calc_every
        self._metric_vals = {k: 0.0 for k in self.metric_names}
        self._setup_done = False

    def before_fit(self):
        if self._setup_done:
            return
        custom_metrics = L([
            ValueMetric(partial(self._get_metric, k=k), k)
            for k in self.metric_names
        ])
        self.learn.metrics = self.learn.metrics + custom_metrics
        self._setup_done = True

    def before_epoch(self):
        epoch_n = self.epoch + 1
        if self.calc_every == "epoch":
            self._do_calc = True
        elif self.calc_every == "last_epoch":
            self._do_calc = epoch_n == self.n_epoch
        elif self.calc_every == "other_epoch":
            self._do_calc = epoch_n % 2 == 0 or epoch_n == self.n_epoch
        else:
            self._do_calc = True

    def before_validate(self):
        self._gen_texts = []
        self._ref_texts = []

    def after_batch(self):
        if self.training or self.learn.y is None or not self._do_calc:
            return

        input_ids = self.xb[0]["input_ids"]
        attention_mask = self.xb[0].get("attention_mask")

        gen_kwargs = {**self.text_gen_kwargs}
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        gen_ids = self.learn.model.hf_model.generate(
            input_ids=input_ids, **gen_kwargs,
        )

        # Decode generated and reference texts
        gen_texts = self.hf_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        ref_ids = self.yb[0]
        ref_ids = ref_ids.masked_fill(ref_ids == IGNORE_TOKEN_ID, self.hf_tokenizer.pad_token_id or 0)
        ref_texts = self.hf_tokenizer.batch_decode(ref_ids, skip_special_tokens=True)

        self._gen_texts.extend(gen_texts)
        self._ref_texts.extend(ref_texts)

    def after_validate(self):
        if not self._gen_texts or not self._do_calc:
            return

        try:
            # Try rouge_score library
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(
                [k for k in self.metric_names if k.startswith("rouge")],
                use_stemmer=True,
            )
            scores = {k: [] for k in self.metric_names if k.startswith("rouge")}
            for gen, ref in zip(self._gen_texts, self._ref_texts):
                result = scorer.score(ref, gen)
                for k in scores:
                    scores[k].append(result[k].fmeasure)
            for k, v in scores.items():
                self._metric_vals[k] = np.mean(v)
        except ImportError:
            pass

    def _get_metric(self, k: str) -> float:
        return self._metric_vals.get(k, 0.0)


# ===================================================================
# Seq2Seq Splitter
# ===================================================================

def seq2seq_splitter(model) -> list[list[torch.nn.Parameter]]:
    """Split encoder-decoder model into groups for discriminative LRs.

    Groups: [embeddings, encoder, decoder, head]
    """
    from fastai.optimizer import params

    hf_model = model.hf_model if hasattr(model, "hf_model") else model

    # Try to find encoder/decoder structure
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "encoder"):
        # BART, Pegasus, Marian, mBART, etc.
        m = hf_model.model
        children = list(hf_model.named_children())
        encoder = m.encoder
        decoder = m.decoder
        heads = [child for name, child in children if name != "model"]
        groups = L([encoder, decoder] + heads)
    elif hasattr(hf_model, "encoder") and hasattr(hf_model, "decoder"):
        # T5, etc.
        children = list(hf_model.named_children())
        encoder = hf_model.encoder
        decoder = hf_model.decoder
        heads = [child for name, child in children if name not in ("encoder", "decoder", "shared")]
        embeds = [child for name, child in children if name == "shared"]
        groups = L(embeds + [encoder, decoder] + heads)
    else:
        # Fallback to generic splitter
        return hf_splitter(model)

    return groups.map(params).filter(lambda el: len(el) > 0)


# ===================================================================
# Learner
# ===================================================================

class Seq2SeqLearner(BlurrLearner):
    """A fastai Learner for sequence-to-sequence tasks (summarization, translation)."""

    def __init__(
        self,
        dls: DataLoaders,
        hf_model: PreTrainedModel,
        *,
        splitter=None,
        freeze: bool = False,  # Seq2seq models often have tied weights, don't auto-freeze
        **kwargs,
    ):
        super().__init__(dls, hf_model, splitter=splitter or seq2seq_splitter, freeze=freeze, **kwargs)

    def blurr_generate(
        self,
        texts: str | list[str],
        *,
        prefix: str = "",
        max_new_tokens: int = 128,
        num_beams: int = 4,
        **gen_kwargs,
    ) -> list[dict[str, str]]:
        """Generate text from input(s).

        Args:
            texts: One or more source texts.
            prefix: Optional prefix to prepend (e.g., ``"summarize: "``).
            max_new_tokens: Maximum tokens to generate.
            num_beams: Beam search width.
            **gen_kwargs: Additional generation kwargs.

        Returns:
            List of dicts with ``generated_text`` key.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Find tokenizer
        tfm = self._find_s2s_tfm()
        hf_tokenizer = tfm.hf_tokenizer

        device = next(self.model.hf_model.parameters()).device
        self.model.eval()

        results = []
        for text in texts:
            input_text = prefix + text
            inputs = hf_tokenizer(
                input_text, return_tensors="pt", truncation=True, padding=True,
            ).to(device)

            with torch.no_grad():
                gen_ids = self.model.hf_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    **gen_kwargs,
                )
            decoded = hf_tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            results.append({"generated_text": decoded})

        return results

    def _find_s2s_tfm(self) -> Seq2SeqBatchTokenizeTransform:
        for tfm in self.dls.before_batch:
            if isinstance(tfm, Seq2SeqBatchTokenizeTransform):
                return tfm
        raise ValueError("Could not find Seq2SeqBatchTokenizeTransform")

    # -----------------------------------------------------------------
    # Factory: from HuggingFace Dataset
    # -----------------------------------------------------------------
    @classmethod
    def from_hf(
        cls,
        dataset: str | DatasetDict | Dataset,
        subset: str | None = None,
        *,
        model_name: str,
        input_col: str = "text",
        target_col: str = "summary",
        prefix: str = "",
        train_split: str = "train",
        valid_split: str | None = None,
        valid_pct: float = 0.2,
        max_source_length: int = 512,
        max_target_length: int = 128,
        max_samples: int | None = None,
        bs: int = 8,
        val_bs: int | None = None,
        seed: int = 42,
        text_gen_kwargs: dict | None = None,
        metric_names: list[str] | None = None,
        config_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
        dl_kwargs: dict | None = None,
        learner_kwargs: dict | None = None,
    ) -> "Seq2SeqLearner":
        """Create a seq2seq learner from a HuggingFace dataset.

        Args:
            dataset: HF dataset name or loaded object.
            subset: Dataset config (e.g., ``"3.0.0"`` for cnn_dailymail).
            model_name: HF model name (e.g., ``"t5-small"``).
            input_col: Source text column.
            target_col: Target text column.
            prefix: Prefix for source (e.g., ``"summarize: "`` for T5).
            train_split: Training split name.
            valid_split: Validation split name.
            valid_pct: Random validation fraction.
            max_source_length: Max source tokenization length.
            max_target_length: Max target tokenization length.
            max_samples: Limit dataset size.
            bs: Batch size.
            val_bs: Validation batch size.
            seed: Random seed.
            text_gen_kwargs: kwargs for ``model.generate()``.
            metric_names: ROUGE metric names.
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

        if max_samples:
            train_ds = train_ds.select(range(min(max_samples, len(train_ds))))
            if valid_ds:
                valid_ds = valid_ds.select(range(min(max_samples, len(valid_ds))))

        # Build DataFrame
        train_df = train_ds.to_pandas()

        # Handle nested columns (e.g., wmt16 has {"de": ..., "en": ...} in "translation")
        if input_col not in train_df.columns:
            # Check for nested dict column (common in translation datasets)
            for col in train_df.columns:
                if isinstance(train_df[col].iloc[0], dict):
                    nested = pd.json_normalize(train_df[col])
                    if input_col in nested.columns:
                        train_df[input_col] = nested[input_col].values
                    if target_col in nested.columns:
                        train_df[target_col] = nested[target_col].values

        if valid_ds is not None:
            valid_df = valid_ds.to_pandas()
            if input_col not in valid_df.columns:
                for col in valid_df.columns:
                    if isinstance(valid_df[col].iloc[0], dict):
                        nested = pd.json_normalize(valid_df[col])
                        if input_col in nested.columns:
                            valid_df[input_col] = nested[input_col].values
                        if target_col in nested.columns:
                            valid_df[target_col] = nested[target_col].values

            train_df["is_valid"] = False
            valid_df["is_valid"] = True
            df = pd.concat([train_df, valid_df], ignore_index=True)
        else:
            df = train_df

        return cls.from_df(
            df,
            model_name,
            input_col=input_col,
            target_col=target_col,
            prefix=prefix,
            valid_col="is_valid" if valid_ds is not None else None,
            valid_pct=valid_pct,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            bs=bs,
            val_bs=val_bs,
            seed=seed,
            text_gen_kwargs=text_gen_kwargs,
            metric_names=metric_names,
            config_kwargs=config_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            model_kwargs=model_kwargs,
            dl_kwargs=dl_kwargs,
            learner_kwargs=learner_kwargs,
        )

    # -----------------------------------------------------------------
    # Factory: from DataFrame
    # -----------------------------------------------------------------
    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        model_name: str,
        *,
        input_col: str = "text",
        target_col: str = "summary",
        prefix: str = "",
        valid_col: str | None = None,
        valid_pct: float = 0.2,
        max_source_length: int = 512,
        max_target_length: int = 128,
        bs: int = 8,
        val_bs: int | None = None,
        seed: int = 42,
        text_gen_kwargs: dict | None = None,
        metric_names: list[str] | None = None,
        config_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
        dl_kwargs: dict | None = None,
        learner_kwargs: dict | None = None,
    ) -> "Seq2SeqLearner":
        """Create a seq2seq learner from a DataFrame."""
        config_kwargs = config_kwargs or {}
        tokenizer_kwargs = tokenizer_kwargs or {}
        model_kwargs = model_kwargs or {}
        dl_kwargs = dl_kwargs or {}
        learner_kwargs = learner_kwargs or {}

        # Load HF objects
        config, tokenizer, model = load_hf_objects(
            model_name,
            AutoModelForSeq2SeqLM,
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
        text_block = Seq2SeqTextBlock(
            tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            prefix=prefix,
        )
        target_block = TransformBlock()

        dblock = DataBlock(
            blocks=(text_block, target_block),
            get_x=lambda r: r[input_col],
            get_y=lambda r: r[target_col],
            splitter=splitter,
        )

        dl_kwargs.setdefault("bs", bs)
        if val_bs:
            dl_kwargs["val_bs"] = val_bs
        dls = dblock.dataloaders(df, **dl_kwargs)

        # Metrics callback
        metrics_cb = Seq2SeqMetricsCallback(
            tokenizer,
            metric_names=metric_names,
            text_gen_kwargs=text_gen_kwargs,
        )
        cbs = learner_kwargs.pop("cbs", [])
        cbs.append(metrics_cb)
        learner_kwargs["cbs"] = cbs
        learner_kwargs.setdefault("loss_func", PreCalculatedCrossEntropyLoss())

        return cls(dls, model, **learner_kwargs)
