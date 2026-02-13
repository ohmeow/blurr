"""Tests for blurr.text.classification — runs on CPU with tiny models."""

import pytest
import pandas as pd
import torch

from blurr.text.classification import (
    TextClassificationLearner,
    ClassificationBatchTokenizeTransform,
    ClassificationBatchDecodeTransform,
    ClassificationTextBlock,
    TextInput,
)
from blurr.text._core import load_hf_objects, ensure_pad_token

from .conftest import TINY_BERT


# ===================================================================
# Unit tests for transforms
# ===================================================================

class TestClassificationBatchTokenizeTransform:
    """Tests for the batch tokenize transform."""

    def test_single_label_tokenization(self):
        """Test that tokenization produces correct dict structure."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TINY_BERT)
        tfm = ClassificationBatchTokenizeTransform(
            tokenizer, max_length=32, labels=["negative", "positive"],
        )

        samples = [
            ("This is great!", torch.tensor(1)),
            ("This is bad!", torch.tensor(0)),
        ]
        result = tfm.encodes(samples)

        assert len(result) == 2
        # Each sample should be (dict, tensor)
        inp_dict, target = result[0]
        assert isinstance(inp_dict, dict)
        assert "input_ids" in inp_dict
        assert "attention_mask" in inp_dict
        assert "labels" in inp_dict

    def test_multilabel_tokenization(self):
        """Test multi-label tokenization produces float labels."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TINY_BERT)
        tfm = ClassificationBatchTokenizeTransform(
            tokenizer, max_length=32, is_multilabel=True,
        )

        samples = [
            ("This is toxic!", torch.tensor([1.0, 0.0, 1.0])),
        ]
        result = tfm.encodes(samples)

        inp_dict, target = result[0]
        assert inp_dict["labels"].dtype == torch.float32

    def test_truncation(self):
        """Test that max_length truncation works."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TINY_BERT)
        tfm = ClassificationBatchTokenizeTransform(tokenizer, max_length=8)

        long_text = "This is a very long text that should definitely be truncated by the tokenizer"
        samples = [(long_text, torch.tensor(0))]
        result = tfm.encodes(samples)

        inp_dict, _ = result[0]
        assert inp_dict["input_ids"].shape[0] == 8

    def test_empty_batch(self):
        """Test handling of empty batch."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TINY_BERT)
        tfm = ClassificationBatchTokenizeTransform(tokenizer)

        result = tfm.encodes([])
        assert len(result) == 0


class TestClassificationBatchDecodeTransform:
    """Tests for the decode transform."""

    def test_decode_returns_text_input(self):
        """Test that decodes returns TextInput type."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TINY_BERT)
        tfm = ClassificationBatchDecodeTransform(tokenizer)

        fake_dict = {"input_ids": torch.tensor([101, 2023, 2003, 2307, 102])}
        result = tfm.decodes(fake_dict)
        assert isinstance(result, TextInput)


# ===================================================================
# Integration tests for Learner
# ===================================================================

class TestTextClassificationLearnerFromDF:
    """Integration tests for creating a learner from DataFrame."""

    def test_from_df_creates_learner(self, classification_df):
        """Test basic learner creation from DataFrame."""
        learn = TextClassificationLearner.from_df(
            classification_df,
            TINY_BERT,
            text_col="text",
            label_col="label",
            bs=4,
            max_length=32,
        )
        assert learn is not None
        assert hasattr(learn, "model")
        assert hasattr(learn, "dls")
        assert len(learn.dls.train_ds) > 0
        assert len(learn.dls.valid_ds) > 0

    def test_from_df_custom_valid_pct(self, classification_df):
        """Test custom validation percentage."""
        learn = TextClassificationLearner.from_df(
            classification_df,
            TINY_BERT,
            text_col="text",
            label_col="label",
            valid_pct=0.5,
            bs=4,
            max_length=32,
        )
        total = len(classification_df)
        # With 50% validation, sizes should be roughly equal
        assert len(learn.dls.valid_ds) > 0
        assert len(learn.dls.train_ds) > 0

    def test_from_df_multilabel(self, multilabel_df):
        """Test multi-label classification learner creation."""
        learn = TextClassificationLearner.from_df(
            multilabel_df,
            TINY_BERT,
            text_col="text",
            label_col=["toxic", "obscene", "insult"],
            is_multilabel=True,
            bs=4,
            max_length=32,
        )
        assert learn is not None

    def test_from_df_training_step(self, classification_df):
        """Test that a single training step works without error."""
        learn = TextClassificationLearner.from_df(
            classification_df,
            TINY_BERT,
            text_col="text",
            label_col="label",
            bs=4,
            max_length=32,
        )
        # Run one batch
        learn.fit(1, lr=1e-3)

    def test_from_df_prediction(self, classification_df):
        """Test prediction returns expected structure."""
        learn = TextClassificationLearner.from_df(
            classification_df,
            TINY_BERT,
            text_col="text",
            label_col="label",
            bs=4,
            max_length=32,
        )
        results = learn.blurr_predict("This is a test sentence.")
        assert len(results) == 1
        result = results[0]
        assert "label" in result
        assert "score" in result
        assert "probs" in result
        assert "class_labels" in result
        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 1.0

    def test_from_df_batch_prediction(self, classification_df):
        """Test batch prediction."""
        learn = TextClassificationLearner.from_df(
            classification_df,
            TINY_BERT,
            text_col="text",
            label_col="label",
            bs=4,
            max_length=32,
        )
        texts = ["This is great!", "This is terrible."]
        results = learn.blurr_predict(texts)
        assert len(results) == 2

    def test_model_is_frozen(self, classification_df):
        """Test that model is frozen after creation (ready for fine-tuning)."""
        learn = TextClassificationLearner.from_df(
            classification_df,
            TINY_BERT,
            text_col="text",
            label_col="label",
            bs=4,
            max_length=32,
        )
        # The model should be partially frozen
        frozen_params = sum(1 for p in learn.model.parameters() if not p.requires_grad)
        total_params = sum(1 for p in learn.model.parameters())
        assert frozen_params > 0
        assert frozen_params < total_params  # Head should be unfrozen


class TestTextClassificationLearnerFromData:
    """Test the universal from_data factory."""

    def test_from_data_with_df(self, classification_df):
        """Test from_data with a DataFrame."""
        learn = TextClassificationLearner.from_data(
            classification_df,
            TINY_BERT,
            text_col="text",
            label_col="label",
            bs=4,
            max_length=32,
        )
        assert learn is not None


# ===================================================================
# Core module tests
# ===================================================================

class TestLoadHFObjects:
    """Tests for HF object loading."""

    def test_load_bert_objects(self):
        """Test loading BERT config, tokenizer, model."""
        from transformers import AutoModelForSequenceClassification

        config, tokenizer, model = load_hf_objects(
            TINY_BERT,
            AutoModelForSequenceClassification,
            config_kwargs={"num_labels": 2},
        )
        assert config is not None
        assert tokenizer is not None
        assert model is not None
        assert config.num_labels == 2

    def test_ensure_pad_token_on_bert(self):
        """Test that ensure_pad_token is a no-op for BERT (already has pad token)."""
        from transformers import AutoModelForSequenceClassification

        config, tokenizer, model = load_hf_objects(
            TINY_BERT,
            AutoModelForSequenceClassification,
        )
        original_vocab = len(tokenizer)
        ensure_pad_token(config, tokenizer, model)
        # BERT already has a pad token, so vocab shouldn't change
        assert len(tokenizer) == original_vocab
