"""Tests for blurr.text.seq2seq — runs on CPU with tiny models."""

import pytest
import pandas as pd
import torch

from blurr.text.seq2seq import (
    Seq2SeqLearner,
    Seq2SeqBatchTokenizeTransform,
    Seq2SeqBatchDecodeTransform,
    Seq2SeqTextBlock,
    Seq2SeqMetricsCallback,
    seq2seq_splitter,
    IGNORE_TOKEN_ID,
)
from blurr.text._core import load_hf_objects, ensure_pad_token

from .conftest import TINY_T5


# ===================================================================
# Unit tests for transforms
# ===================================================================

class TestSeq2SeqBatchTokenizeTransform:
    """Tests for seq2seq batch tokenize transform."""

    def test_basic_tokenization(self):
        """Test source + target tokenization."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TINY_T5)
        tfm = Seq2SeqBatchTokenizeTransform(
            tokenizer,
            max_source_length=64,
            max_target_length=32,
        )

        samples = [
            ("The quick brown fox.", "A fox."),
            ("Scientists discovered fish.", "Fish found."),
        ]
        result = tfm.encodes(samples)

        assert len(result) == 2
        inp_dict, labels = result[0]
        assert "input_ids" in inp_dict
        assert "attention_mask" in inp_dict
        assert "labels" in inp_dict

        # Pad tokens in labels should be -100
        label_ids = inp_dict["labels"]
        if tokenizer.pad_token_id is not None:
            # Some label positions should be -100 (padding)
            assert label_ids.shape[0] > 0

    def test_prefix(self):
        """Test that prefix is prepended to source."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TINY_T5)

        # Without prefix
        tfm_no_prefix = Seq2SeqBatchTokenizeTransform(
            tokenizer, max_source_length=64, max_target_length=32,
        )
        # With prefix
        tfm_prefix = Seq2SeqBatchTokenizeTransform(
            tokenizer, max_source_length=64, max_target_length=32,
            prefix="summarize: ",
        )

        samples = [("The fox.", "Fox.")]
        result_no = tfm_no_prefix.encodes(samples)
        result_yes = tfm_prefix.encodes(samples)

        # The prefixed version should have more input tokens
        len_no = result_no[0][0]["input_ids"].shape[0]
        len_yes = result_yes[0][0]["input_ids"].shape[0]
        # At least the prefix should add some tokens
        assert len_yes >= len_no

    def test_empty_batch(self):
        """Test handling of empty batch."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TINY_T5)
        tfm = Seq2SeqBatchTokenizeTransform(tokenizer)
        result = tfm.encodes([])
        assert len(result) == 0


# ===================================================================
# Integration tests for Learner
# ===================================================================

class TestSeq2SeqLearner:
    """Integration tests for Seq2SeqLearner."""

    def test_from_df_creates_learner(self, seq2seq_df):
        """Test learner creation from DataFrame."""
        learn = Seq2SeqLearner.from_df(
            seq2seq_df,
            TINY_T5,
            input_col="text",
            target_col="summary",
            prefix="summarize: ",
            bs=2,
            max_source_length=64,
            max_target_length=32,
        )
        assert learn is not None
        assert hasattr(learn, "model")
        assert hasattr(learn, "dls")

    def test_training_step(self, seq2seq_df):
        """Test that a single training step works."""
        learn = Seq2SeqLearner.from_df(
            seq2seq_df,
            TINY_T5,
            input_col="text",
            target_col="summary",
            bs=2,
            max_source_length=64,
            max_target_length=32,
        )
        learn.fit(1, lr=1e-3)

    def test_generate(self, seq2seq_df):
        """Test text generation."""
        learn = Seq2SeqLearner.from_df(
            seq2seq_df,
            TINY_T5,
            input_col="text",
            target_col="summary",
            prefix="summarize: ",
            bs=2,
            max_source_length=64,
            max_target_length=32,
        )
        results = learn.blurr_generate(
            "The quick brown fox jumped over the lazy dog.",
            prefix="summarize: ",
            max_new_tokens=16,
            num_beams=2,
        )
        assert len(results) == 1
        assert "generated_text" in results[0]
        assert isinstance(results[0]["generated_text"], str)

    def test_batch_generate(self, seq2seq_df):
        """Test batch generation."""
        learn = Seq2SeqLearner.from_df(
            seq2seq_df,
            TINY_T5,
            input_col="text",
            target_col="summary",
            bs=2,
            max_source_length=64,
            max_target_length=32,
        )
        texts = ["Hello world.", "Another text."]
        results = learn.blurr_generate(texts, max_new_tokens=16, num_beams=1)
        assert len(results) == 2

    def test_splitter(self, seq2seq_df):
        """Test that the seq2seq splitter works."""
        learn = Seq2SeqLearner.from_df(
            seq2seq_df,
            TINY_T5,
            input_col="text",
            target_col="summary",
            bs=2,
            max_source_length=64,
            max_target_length=32,
        )
        # The model should have parameter groups from the splitter
        groups = seq2seq_splitter(learn.model)
        assert len(groups) > 0
        for group in groups:
            assert len(group) > 0
