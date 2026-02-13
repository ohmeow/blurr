"""blurr.text._core — Shared foundation for all text tasks.

This module provides the common building blocks that every text task
(classification, token classification, seq2seq, causal LM) shares:

- HF object loading helpers
- fastai-compatible model wrapper & callback
- Model splitter for discriminative LR
- Pre-calculated loss functions
- Base learner class
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
from fastai.callback.core import Callback
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.losses import (
    BaseLoss,
    BCEWithLogitsLossFlat,
    CrossEntropyLossFlat,
    MSELossFlat,
)
from fastai.optimizer import params
from fastai.torch_core import Module, to_float, tensor
from fastcore.foundation import L
from fasttransform import Transform
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.utils import logging as hf_logging

# ---------------------------------------------------------------------------
# Silence HF warnings globally for cleaner training output
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()


# ===================================================================
# HF Object Loading
# ===================================================================

def load_hf_objects(
    model_name: str,
    model_cls: type[PreTrainedModel],
    *,
    config: PretrainedConfig | None = None,
    tokenizer_cls: type[PreTrainedTokenizerBase] | None = None,
    config_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    cache_dir: str | None = None,
) -> tuple[PretrainedConfig, PreTrainedTokenizerBase, PreTrainedModel]:
    """Load HF config, tokenizer, and model from a pretrained checkpoint.

    Returns:
        (config, tokenizer, model)
    """
    config_kwargs = config_kwargs or {}
    tokenizer_kwargs = tokenizer_kwargs or {}
    model_kwargs = model_kwargs or {}

    # Config
    if config is None:
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, **config_kwargs)

    # Tokenizer — some architectures need prefix space
    if any(s in model_name for s in ("gpt2", "roberta", "bart", "longformer")):
        tokenizer_kwargs.setdefault("add_prefix_space", True)

    tok_cls = tokenizer_cls or AutoTokenizer
    tokenizer = tok_cls.from_pretrained(model_name, cache_dir=cache_dir, **tokenizer_kwargs)

    # Model
    model = model_cls.from_pretrained(model_name, config=config, cache_dir=cache_dir, **model_kwargs)

    return config, tokenizer, model


def ensure_pad_token(
    config: PretrainedConfig,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
) -> None:
    """Add a pad token to architectures that don't have one (GPT-2, etc.)."""
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))


# ===================================================================
# Pre-Calculated Loss Functions
# ===================================================================

class PreCalculatedLoss(BaseLoss):
    """Dummy loss — the real loss is calculated inside the HF model's forward().

    Returns a zero tensor that requires grad so fastai's backward pass
    doesn't error before our HFModelCallback replaces it with the real loss.
    """

    def __call__(self, inp, targ, **kwargs):
        if torch.is_tensor(inp) and inp.requires_grad:
            return (inp * 0).sum()  # Preserves grad chain
        return tensor(0.0, requires_grad=True)


class PreCalculatedCrossEntropyLoss(PreCalculatedLoss, CrossEntropyLossFlat):
    pass


class PreCalculatedBCELoss(PreCalculatedLoss, BCEWithLogitsLossFlat):
    pass


class PreCalculatedMSELoss(PreCalculatedLoss):
    def __init__(self, *args, axis: int = -1, floatify: bool = True, **kwargs):
        super().__init__(nn.MSELoss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)


# ===================================================================
# Model Wrapper
# ===================================================================

class HFModelWrapper(Module):
    """Wraps a HuggingFace model so its forward() accepts a dict of inputs
    (as produced by our data pipeline) and returns HF model outputs."""

    def __init__(
        self,
        hf_model: PreTrainedModel,
        *,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ):
        super().__init__()
        self.hf_model = hf_model
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self._fwd_params = set(inspect.signature(hf_model.forward).parameters.keys())

    def forward(self, x: dict) -> Any:
        # Filter to only the keys the model's forward() accepts
        filtered = {k: v for k, v in x.items() if k in self._fwd_params}
        return self.hf_model(
            **filtered,
            output_hidden_states=self.output_hidden_states,
            output_attentions=self.output_attentions,
            return_dict=True,
        )


# ===================================================================
# Model Callback
# ===================================================================

class HFModelCallback(Callback):
    """Integrates HF model outputs with fastai's training loop.

    - Extracts logits from model output → sets as `learn.pred`
    - If labels were included and a PreCalculatedLoss is used,
      grabs the model-computed loss → sets as `learn.loss_grad`
    - Stores other model outputs (hidden_states, attentions) in
      `learn.blurr_model_outputs`
    """

    order = 10  # Run after most other callbacks

    def before_batch(self):
        self._hf_loss = None

    def after_pred(self):
        outputs = self.pred
        self.learn.blurr_model_outputs = {}

        for k, v in outputs.items():
            if k == "loss" and isinstance(self.learn.loss_func, PreCalculatedLoss):
                self._hf_loss = to_float(v)
            elif k == "logits":
                self.learn.pred = v
            else:
                self.learn.blurr_model_outputs[k] = v

    def after_loss(self):
        if self._hf_loss is not None:
            self.learn.loss_grad = self._hf_loss
            self.learn.loss = self._hf_loss.clone()


# ===================================================================
# Model Splitter for Discriminative Learning Rates
# ===================================================================

def hf_splitter(model: Module) -> list[list[nn.Parameter]]:
    """Split a HuggingFace model into parameter groups for discriminative LRs.

    Group 0: Embeddings / bottom of backbone
    Group 1..N-1: Encoder/decoder layers
    Group N: Classification/LM head
    """
    hf_model = model.hf_model if hasattr(model, "hf_model") else model
    children = list(hf_model.named_children())

    if not children:
        return [list(hf_model.parameters())]

    # First child is usually the backbone (e.g., BertModel inside BertForSequenceClassification)
    backbone_name, backbone = children[0]
    backbone_children = list(backbone.named_children())

    groups = L([m for _, m in backbone_children])
    groups += L([m for _, m in children[1:]])

    return groups.map(params).filter(lambda el: len(el) > 0)


# ===================================================================
# Base Learner
# ===================================================================

class BlurrLearner(Learner):
    """Base Learner for all blurr tasks. Sets up:

    - HFModelWrapper around the HF model
    - HFModelCallback for loss/pred extraction
    - hf_splitter for discriminative LR
    - Correct PreCalculated* loss function
    """

    def __init__(
        self,
        dls: DataLoaders,
        hf_model: PreTrainedModel,
        *,
        loss_func=None,
        splitter=None,
        cbs=None,
        freeze: bool = True,
        **kwargs,
    ):
        # Determine appropriate loss function
        if loss_func is None:
            loss_func = self._infer_loss_func(dls)

        # Wrap model
        model = HFModelWrapper(hf_model)

        # Prepend our callback
        cbs = list(cbs or [])
        cbs.insert(0, HFModelCallback())

        super().__init__(
            dls,
            model=model,
            loss_func=loss_func,
            splitter=splitter or hf_splitter,
            cbs=cbs,
            **kwargs,
        )
        if freeze:
            try:
                self.freeze()
            except Exception:
                # Some model architectures (e.g., tied weights in seq2seq)
                # may not freeze cleanly — just train fully
                pass

    @staticmethod
    def _infer_loss_func(dls: DataLoaders):
        """Pick the right pre-calculated loss based on what the DataLoaders use."""
        if hasattr(dls, "loss_func"):
            lf = dls.loss_func
            if isinstance(lf, BCEWithLogitsLossFlat):
                return PreCalculatedBCELoss()
            if isinstance(lf, CrossEntropyLossFlat):
                return PreCalculatedCrossEntropyLoss()
        return PreCalculatedCrossEntropyLoss()


# ===================================================================
# Utilities
# ===================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across numpy, torch, and python."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
