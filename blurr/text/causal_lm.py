"""blurr.text.causal_lm — Causal language modeling & instruction tuning with PEFT.

Usage::

    from blurr.text.causal_lm import CausalLMLearner

    # Basic causal LM fine-tuning
    learn = CausalLMLearner.from_hf(
        "wikitext", "wikitext-2-raw-v1",
        model_name="gpt2",
    )
    learn.fit_one_cycle(3, 2e-5)
    learn.blurr_generate("Once upon a time")

    # Instruction tuning with LoRA
    from peft import LoraConfig
    learn = CausalLMLearner.from_df(
        df, model_name="meta-llama/Llama-2-7b-hf",
        text_col="text",
        peft_config=LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"]),
    )

    # Chat/instruction format
    learn = CausalLMLearner.from_df(
        df, model_name="gpt2",
        instruction_col="instruction",
        response_col="response",
        prompt_template="### Instruction:\\n{instruction}\\n\\n### Response:\\n{response}",
    )
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
from fastai.data.block import DataBlock, TransformBlock
from fastai.data.core import DataLoaders
from fastai.data.transforms import ColSplitter, RandomSplitter
from fastai.losses import CrossEntropyLossFlat
from fastai.metrics import ValueMetric, perplexity
from fastai.text.data import SortedDL
from fastai.torch_core import TensorBase, tensor
from fasttransform import Transform
from fastcore.basics import store_attr
from fastcore.foundation import L
from transformers import (
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ._core import (
    BlurrLearner,
    HFModelCallback,
    HFModelWrapper,
    PreCalculatedCrossEntropyLoss,
    PreCalculatedLoss,
    ensure_pad_token,
    hf_splitter,
    load_hf_objects,
)


# Optional PEFT support
try:
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

# Optional TRL support
try:
    from trl import SFTConfig
    HAS_TRL = True
except ImportError:
    HAS_TRL = False


# ===================================================================
# Types
# ===================================================================

class CausalLMTextInput(TensorBase):
    """Marker type for causal LM inputs."""
    pass


# ===================================================================
# Batch Transforms
# ===================================================================

IGNORE_TOKEN_ID = -100


class CausalLMBatchTokenizeTransform(Transform):
    """On-the-fly batch tokenization for causal language modeling.

    For standard LM: input = text, labels = same as input_ids (shifted inside model).
    For instruction tuning: can mask the prompt portion of labels so loss is only
    computed on the response.
    """

    def __init__(
        self,
        hf_tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int = 512,
        padding: bool | str = True,
        truncation: bool | str = True,
        include_labels: bool = True,
        mask_prompt: bool = False,
        prompt_length_key: str | None = None,
    ):
        self.hf_tokenizer = hf_tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.include_labels = include_labels
        self.mask_prompt = mask_prompt
        self.prompt_length_key = prompt_length_key

    def encodes(self, samples):
        samples = L(samples)
        if len(samples) == 0:
            return samples

        texts = [str(s[0]) for s in samples]

        # Extract prompt lengths if doing instruction tuning with prompt masking
        prompt_lengths = None
        if self.mask_prompt and len(samples[0]) > 1:
            # Second element can be prompt length (int) or the target part
            try:
                prompt_lengths = [int(s[1]) for s in samples]
            except (TypeError, ValueError):
                prompt_lengths = None

        inputs = self.hf_tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )

        updated = []
        for idx in range(len(samples)):
            inp_dict = {k: inputs[k][idx] for k in inputs.keys()}

            if self.include_labels:
                labels = inputs["input_ids"][idx].clone()
                # Mask padding tokens
                if self.hf_tokenizer.pad_token_id is not None:
                    labels[labels == self.hf_tokenizer.pad_token_id] = IGNORE_TOKEN_ID

                # Mask prompt portion for instruction tuning
                if self.mask_prompt and prompt_lengths is not None:
                    labels[:prompt_lengths[idx]] = IGNORE_TOKEN_ID

                inp_dict["labels"] = labels

            # Target for fastai = labels (or dummy)
            target = inp_dict.get("labels", inputs["input_ids"][idx])
            updated.append((inp_dict, target))

        return updated


class CausalLMBatchDecodeTransform(Transform):
    """Decode transform for causal LM show methods."""

    def __init__(self, hf_tokenizer: PreTrainedTokenizerBase):
        self.hf_tokenizer = hf_tokenizer

    def decodes(self, x: dict) -> CausalLMTextInput:
        return CausalLMTextInput(x["input_ids"])


# ===================================================================
# CausalLM TextBlock
# ===================================================================

def CausalLMTextBlock(
    hf_tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int = 512,
    mask_prompt: bool = False,
):
    """TransformBlock for causal language modeling."""

    batch_tfm = CausalLMBatchTokenizeTransform(
        hf_tokenizer, max_length=max_length, mask_prompt=mask_prompt,
    )
    decode_tfm = CausalLMBatchDecodeTransform(hf_tokenizer)

    def sort_func(example, hf_tokenizer=hf_tokenizer):
        text = example[0] if isinstance(example[0], str) else str(example[0])
        return len(text)  # Approximate sort by length

    return TransformBlock(
        dl_type=partial(SortedDL, sort_func=sort_func),
        dls_kwargs={"before_batch": batch_tfm},
        batch_tfms=decode_tfm,
    )


# ===================================================================
# LM Accuracy Callback
# ===================================================================

class CausalLMMetricsCallback(Callback):
    """Callback that computes token-level accuracy for causal LM,
    ignoring tokens marked with ignore_token_id (-100)."""

    order = 9

    def __init__(self):
        self._metric_vals = {"lm_accuracy": 0.0}
        self._setup_done = False

    def before_fit(self):
        if self._setup_done:
            return
        custom_metrics = L([
            ValueMetric(partial(self._get_metric, k=k), k)
            for k in self._metric_vals
        ])
        self.learn.metrics = self.learn.metrics + custom_metrics
        self._setup_done = True

    def before_validate(self):
        self._correct = 0
        self._total = 0

    def after_batch(self):
        if self.training or self.learn.y is None:
            return

        preds = self.pred.argmax(dim=-1)
        targs = self.yb[0]

        mask = targs != IGNORE_TOKEN_ID
        if mask.any():
            self._correct += (preds[mask] == targs[mask]).sum().item()
            self._total += mask.sum().item()

    def after_validate(self):
        if self._total > 0:
            self._metric_vals["lm_accuracy"] = self._correct / self._total

    def _get_metric(self, k: str) -> float:
        return self._metric_vals.get(k, 0.0)


# ===================================================================
# PEFT Wrapper
# ===================================================================

class PEFTModelWrapper(HFModelWrapper):
    """Model wrapper that supports PEFT (LoRA, QLoRA, etc.).

    The PEFT model wraps the base model, and this wrapper ensures
    compatibility with fastai's training loop.
    """

    def __init__(self, hf_model: PreTrainedModel, **kwargs):
        super().__init__(hf_model, **kwargs)
        # PEFT models may have different forward signatures
        self._fwd_params = set(["input_ids", "attention_mask", "labels", "position_ids", "past_key_values"])


def apply_peft(
    model: PreTrainedModel,
    peft_config: Any,
    *,
    prepare_for_kbit: bool = False,
) -> PreTrainedModel:
    """Apply a PEFT configuration (LoRA, etc.) to a model.

    Args:
        model: The base HuggingFace model.
        peft_config: A PEFT config object (e.g., ``LoraConfig``).
        prepare_for_kbit: Whether to prepare for k-bit training (QLoRA).

    Returns:
        The PEFT-wrapped model.
    """
    if not HAS_PEFT:
        raise ImportError("peft is required for PEFT support. Install with: pip install peft")

    if prepare_for_kbit:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


# ===================================================================
# Instruction Tuning Helpers
# ===================================================================

def format_instruction_data(
    df: pd.DataFrame,
    *,
    instruction_col: str = "instruction",
    input_col: str | None = None,
    response_col: str = "response",
    prompt_template: str | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> pd.DataFrame:
    """Format instruction tuning data into a single text column with prompt masking info.

    Args:
        df: DataFrame with instruction/response columns.
        instruction_col: Column with instructions.
        input_col: Optional column with additional input context.
        response_col: Column with responses.
        prompt_template: Custom template string with ``{instruction}``, ``{input}``,
            ``{response}`` placeholders. Default format is used if None.
        tokenizer: If provided, computes prompt token lengths for masking.

    Returns:
        DataFrame with ``text`` and optionally ``prompt_length`` columns.
    """
    if prompt_template is None:
        if input_col:
            prompt_template = (
                "### Instruction:\n{instruction}\n\n"
                "### Input:\n{input}\n\n"
                "### Response:\n{response}"
            )
        else:
            prompt_template = (
                "### Instruction:\n{instruction}\n\n"
                "### Response:\n{response}"
            )

    texts = []
    prompt_lengths = []

    for _, row in df.iterrows():
        kwargs = {
            "instruction": str(row[instruction_col]),
            "response": str(row[response_col]),
        }
        if input_col and input_col in df.columns:
            kwargs["input"] = str(row[input_col])

        full_text = prompt_template.format(**kwargs)
        texts.append(full_text)

        # Compute prompt length (everything before the response)
        if tokenizer is not None:
            prompt_part = prompt_template.format(**{**kwargs, "response": ""})
            prompt_tokens = tokenizer(prompt_part, truncation=False)["input_ids"]
            prompt_lengths.append(len(prompt_tokens))

    result = df.copy()
    result["text"] = texts
    if prompt_lengths:
        result["prompt_length"] = prompt_lengths
    return result


# ===================================================================
# Learner
# ===================================================================

class CausalLMLearner(BlurrLearner):
    """A fastai Learner for causal language modeling and instruction tuning.

    Supports:
    - Standard causal LM fine-tuning (GPT-2, LLaMA, etc.)
    - Instruction tuning with prompt masking
    - PEFT (LoRA, QLoRA) via ``peft_config`` parameter
    """

    def __init__(
        self,
        dls: DataLoaders,
        hf_model: PreTrainedModel,
        *,
        is_peft: bool = False,
        freeze: bool = False,  # Causal LM models often have tied weights, don't auto-freeze
        **kwargs,
    ):
        self._is_peft = is_peft
        super().__init__(dls, hf_model, freeze=freeze, **kwargs)

    def blurr_generate(
        self,
        texts: str | list[str],
        *,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        **gen_kwargs,
    ) -> list[dict[str, str]]:
        """Generate text continuations.

        Args:
            texts: One or more prompt texts.
            max_new_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling.
            do_sample: Whether to sample (vs. greedy/beam).
            num_beams: Beam search width.
            repetition_penalty: Penalty for repetition.
            **gen_kwargs: Additional generation kwargs.

        Returns:
            List of dicts with ``generated_text``, ``prompt``, and ``completion`` keys.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Find tokenizer
        tfm = self._find_clm_tfm()
        hf_tokenizer = tfm.hf_tokenizer

        hf_model = self.model.hf_model if hasattr(self.model, "hf_model") else self.model
        device = next(hf_model.parameters()).device
        self.model.eval()

        results = []
        for text in texts:
            inputs = hf_tokenizer(text, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                gen_ids = hf_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=hf_tokenizer.pad_token_id or hf_tokenizer.eos_token_id,
                    **gen_kwargs,
                )

            full_text = hf_tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            completion = hf_tokenizer.decode(gen_ids[0][input_len:], skip_special_tokens=True)

            results.append({
                "generated_text": full_text,
                "prompt": text,
                "completion": completion,
            })

        return results

    def _find_clm_tfm(self) -> CausalLMBatchTokenizeTransform:
        for tfm in self.dls.before_batch:
            if isinstance(tfm, CausalLMBatchTokenizeTransform):
                return tfm
        raise ValueError("Could not find CausalLMBatchTokenizeTransform")

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
        instruction_col: str | None = None,
        response_col: str | None = None,
        input_col: str | None = None,
        prompt_template: str | None = None,
        mask_prompt: bool = True,
        valid_col: str | None = None,
        valid_pct: float = 0.1,
        max_length: int = 512,
        peft_config: Any | None = None,
        prepare_for_kbit: bool = False,
        bs: int = 4,
        val_bs: int | None = None,
        seed: int = 42,
        metrics: list | None = None,
        config_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
        dl_kwargs: dict | None = None,
        learner_kwargs: dict | None = None,
    ) -> "CausalLMLearner":
        """Create a causal LM learner from a DataFrame.

        For instruction tuning, provide ``instruction_col`` and ``response_col``
        instead of ``text_col``. The data will be formatted with a template and
        prompt masking will be applied so the loss is computed only on responses.

        Args:
            df: DataFrame with text or instruction/response columns.
            model_name: HuggingFace model name.
            text_col: Column with raw text (for standard LM).
            instruction_col: Column with instructions (for instruction tuning).
            response_col: Column with responses (for instruction tuning).
            input_col: Optional column with additional input context.
            prompt_template: Custom template for instruction formatting.
            mask_prompt: Whether to mask prompt tokens from loss.
            valid_col: Validation split column.
            valid_pct: Random validation fraction.
            max_length: Max sequence length.
            peft_config: PEFT config (e.g., LoraConfig) for parameter-efficient fine-tuning.
            prepare_for_kbit: Whether to prepare model for quantized training.
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

        # Load HF objects
        config, tokenizer, model = load_hf_objects(
            model_name,
            AutoModelForCausalLM,
            config_kwargs=config_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            model_kwargs=model_kwargs,
        )
        ensure_pad_token(config, tokenizer, model)

        # Set padding side for causal LM
        tokenizer.padding_side = "left"

        # Apply PEFT if configured
        is_peft = False
        if peft_config is not None:
            model = apply_peft(model, peft_config, prepare_for_kbit=prepare_for_kbit)
            is_peft = True

        # Format instruction data if needed
        is_instruction = instruction_col is not None and response_col is not None
        if is_instruction:
            df = format_instruction_data(
                df,
                instruction_col=instruction_col,
                input_col=input_col,
                response_col=response_col,
                prompt_template=prompt_template,
                tokenizer=tokenizer if mask_prompt else None,
            )
            text_col = "text"

        # Splitter
        if valid_col and valid_col in df.columns:
            splitter = ColSplitter(valid_col)
        else:
            splitter = RandomSplitter(valid_pct, seed=seed)

        # Determine if we should mask prompts
        do_mask = is_instruction and mask_prompt and "prompt_length" in df.columns

        # Blocks
        text_block = CausalLMTextBlock(tokenizer, max_length=max_length, mask_prompt=do_mask)
        target_block = TransformBlock()

        if do_mask:
            dblock = DataBlock(
                blocks=(text_block, target_block),
                get_x=lambda r: r[text_col],
                get_y=lambda r: r.get("prompt_length", 0),
                splitter=splitter,
            )
        else:
            dblock = DataBlock(
                blocks=(text_block, target_block),
                get_x=lambda r: r[text_col],
                get_y=lambda r: r[text_col],  # dummy target
                splitter=splitter,
            )

        dl_kwargs.setdefault("bs", bs)
        if val_bs:
            dl_kwargs["val_bs"] = val_bs
        dls = dblock.dataloaders(df, **dl_kwargs)

        # Metrics
        metrics_cb = CausalLMMetricsCallback()
        cbs = learner_kwargs.pop("cbs", [])
        cbs.append(metrics_cb)
        learner_kwargs["cbs"] = cbs

        if metrics is None:
            metrics = [perplexity]
        learner_kwargs["metrics"] = metrics
        learner_kwargs.setdefault("loss_func", PreCalculatedCrossEntropyLoss())

        return cls(dls, model, is_peft=is_peft, **learner_kwargs)

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
        text_col: str = "text",
        instruction_col: str | None = None,
        response_col: str | None = None,
        input_col: str | None = None,
        prompt_template: str | None = None,
        mask_prompt: bool = True,
        train_split: str = "train",
        valid_split: str | None = None,
        valid_pct: float = 0.1,
        max_length: int = 512,
        max_samples: int | None = None,
        peft_config: Any | None = None,
        prepare_for_kbit: bool = False,
        bs: int = 4,
        val_bs: int | None = None,
        seed: int = 42,
        metrics: list | None = None,
        config_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
        dl_kwargs: dict | None = None,
        learner_kwargs: dict | None = None,
    ) -> "CausalLMLearner":
        """Create a causal LM learner from a HuggingFace dataset."""
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

        # Convert to DataFrame
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
            instruction_col=instruction_col,
            response_col=response_col,
            input_col=input_col,
            prompt_template=prompt_template,
            mask_prompt=mask_prompt,
            valid_col="is_valid" if valid_ds is not None else None,
            valid_pct=valid_pct,
            max_length=max_length,
            peft_config=peft_config,
            prepare_for_kbit=prepare_for_kbit,
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
    # Save / Load with PEFT
    # -----------------------------------------------------------------
    def save_peft(self, path: str | os.PathLike) -> None:
        """Save a PEFT model's adapter weights."""
        if not self._is_peft:
            raise ValueError("This learner was not created with a PEFT config.")
        hf_model = self.model.hf_model if hasattr(self.model, "hf_model") else self.model
        hf_model.save_pretrained(path)

    @classmethod
    def load_peft(
        cls,
        base_model_name: str,
        adapter_path: str | os.PathLike,
        *,
        tokenizer_name: str | None = None,
    ) -> PreTrainedModel:
        """Load a PEFT adapter on top of a base model.

        Args:
            base_model_name: Name of the base HuggingFace model.
            adapter_path: Path to the saved adapter weights.
            tokenizer_name: Tokenizer name (defaults to base_model_name).

        Returns:
            The PEFT-wrapped model ready for inference.
        """
        if not HAS_PEFT:
            raise ImportError("peft is required. Install with: pip install peft")

        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, adapter_path)
        return model
