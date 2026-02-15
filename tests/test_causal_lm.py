"""Tests for blurr.text.causal_lm — runs on CPU with tiny models."""

import pytest
import pandas as pd
import torch

from blurr.text.causal_lm import (
    CausalLMLearner,
    CausalLMBatchTokenizeTransform,
    CausalLMBatchDecodeTransform,
    CausalLMTextBlock,
    CausalLMMetricsCallback,
    format_instruction_data,
    IGNORE_TOKEN_ID,
)
from blurr.text._core import load_hf_objects, ensure_pad_token

from .conftest import TINY_GPT2


# ===================================================================
# Unit tests for transforms
# ===================================================================

class TestCausalLMBatchTokenizeTransform:
    """Tests for causal LM batch tokenize transform."""

    def test_basic_tokenization(self):
        """Test causal LM tokenization (labels = input_ids)."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TINY_GPT2)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tfm = CausalLMBatchTokenizeTransform(tokenizer, max_length=32)

        samples = [
            ("The quick brown fox.", "The quick brown fox."),
        ]
        result = tfm.encodes(samples)

        assert len(result) == 1
        inp_dict, target = result[0]
        assert "input_ids" in inp_dict
        assert "labels" in inp_dict

        # Labels should match input_ids (except padding → -100)
        input_ids = inp_dict["input_ids"]
        labels = inp_dict["labels"]
        # Non-pad positions should match
        mask = labels != IGNORE_TOKEN_ID
        if mask.any():
            assert torch.equal(input_ids[mask], labels[mask])

    def test_prompt_masking(self):
        """Test that prompt masking sets early tokens to -100."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TINY_GPT2)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tfm = CausalLMBatchTokenizeTransform(
            tokenizer, max_length=32, mask_prompt=True,
        )

        # Second element is the prompt length
        samples = [
            ("Instruction: Hello Response: World", 3),
        ]
        result = tfm.encodes(samples)

        inp_dict, target = result[0]
        labels = inp_dict["labels"]

        # First 3 tokens should be masked
        assert all(l == IGNORE_TOKEN_ID for l in labels[:3].tolist())

    def test_empty_batch(self):
        """Test handling of empty batch."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TINY_GPT2)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tfm = CausalLMBatchTokenizeTransform(tokenizer)
        result = tfm.encodes([])
        assert len(result) == 0


# ===================================================================
# Unit tests for instruction formatting
# ===================================================================

class TestFormatInstructionData:
    """Tests for instruction data formatting."""

    def test_basic_formatting(self, instruction_df):
        """Test that instruction data is formatted correctly."""
        result = format_instruction_data(
            instruction_df,
            instruction_col="instruction",
            response_col="response",
        )
        assert "text" in result.columns
        # Each text should contain both instruction and response
        for _, row in result.iterrows():
            assert row["instruction"] in row["text"]
            assert row["response"] in row["text"]

    def test_custom_template(self, instruction_df):
        """Test custom prompt template."""
        template = "Q: {instruction}\nA: {response}"
        result = format_instruction_data(
            instruction_df,
            instruction_col="instruction",
            response_col="response",
            prompt_template=template,
        )
        assert result["text"].iloc[0].startswith("Q: ")

    def test_with_tokenizer_for_lengths(self, instruction_df):
        """Test that prompt lengths are computed when tokenizer is provided."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TINY_GPT2)

        result = format_instruction_data(
            instruction_df,
            instruction_col="instruction",
            response_col="response",
            tokenizer=tokenizer,
        )
        assert "prompt_length" in result.columns
        # All prompt lengths should be positive integers
        for pl in result["prompt_length"]:
            assert isinstance(pl, int)
            assert pl > 0


# ===================================================================
# Integration tests for Learner
# ===================================================================

class TestCausalLMLearner:
    """Integration tests for CausalLMLearner."""

    def test_from_df_creates_learner(self, causal_lm_df):
        """Test learner creation from DataFrame."""
        learn = CausalLMLearner.from_df(
            causal_lm_df,
            TINY_GPT2,
            text_col="text",
            bs=2,
            max_length=32,
        )
        assert learn is not None
        assert hasattr(learn, "model")
        assert hasattr(learn, "dls")

    def test_training_step(self, causal_lm_df):
        """Test that a single training step works."""
        learn = CausalLMLearner.from_df(
            causal_lm_df,
            TINY_GPT2,
            text_col="text",
            bs=2,
            max_length=32,
        )
        learn.fit(1, lr=1e-3)

    def test_generate(self, causal_lm_df):
        """Test text generation."""
        learn = CausalLMLearner.from_df(
            causal_lm_df,
            TINY_GPT2,
            text_col="text",
            bs=2,
            max_length=32,
        )
        results = learn.blurr_generate(
            "Once upon a time",
            max_new_tokens=10,
            do_sample=False,
        )
        assert len(results) == 1
        assert "generated_text" in results[0]
        assert "prompt" in results[0]
        assert "completion" in results[0]
        assert "Once upon a time" in results[0]["generated_text"]

    def test_instruction_tuning(self, instruction_df):
        """Test instruction tuning mode."""
        learn = CausalLMLearner.from_df(
            instruction_df,
            TINY_GPT2,
            instruction_col="instruction",
            response_col="response",
            mask_prompt=True,
            bs=2,
            max_length=64,
        )
        assert learn is not None
        # Should be able to train
        learn.fit(1, lr=1e-3)

    def test_metrics_callback(self, causal_lm_df):
        """Test that LM metrics callback is attached."""
        learn = CausalLMLearner.from_df(
            causal_lm_df,
            TINY_GPT2,
            text_col="text",
            bs=2,
            max_length=32,
        )
        has_metrics_cb = any(
            isinstance(cb, CausalLMMetricsCallback)
            for cb in learn.cbs
        )
        assert has_metrics_cb


class TestPEFTIntegration:
    """Tests for PEFT integration (only if peft is installed)."""

    @pytest.mark.skipif(
        not pytest.importorskip("peft", reason="peft not installed"),
        reason="peft not installed",
    )
    def test_lora_config(self, causal_lm_df):
        """Test creating a learner with LoRA config."""
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["c_attn"],  # GPT-2 attention
            lora_dropout=0.1,
        )

        learn = CausalLMLearner.from_df(
            causal_lm_df,
            TINY_GPT2,
            text_col="text",
            peft_config=peft_config,
            bs=2,
            max_length=32,
        )
        assert learn._is_peft
        # Trainable params should be much less than total
        total = sum(p.numel() for p in learn.model.parameters())
        trainable = sum(p.numel() for p in learn.model.parameters() if p.requires_grad)
        assert trainable < total
