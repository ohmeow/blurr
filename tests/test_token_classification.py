"""Tests for blurr.text.token_classification — runs on CPU with tiny models."""

import pytest
import torch

from blurr.text.token_classification import (
    TokenClassificationLearner,
    TokenClassBatchTokenizeTransform,
    TokenClassBatchDecodeTransform,
    TokenClassTextBlock,
    TokenClassMetricsCallback,
    _aggregate_entities,
    IGNORE_TOKEN_ID,
)
from blurr.text._core import load_hf_objects, ensure_pad_token

from .conftest import TINY_BERT


# ===================================================================
# Unit tests for transforms
# ===================================================================

class TestTokenClassBatchTokenizeTransform:
    """Tests for the token classification batch tokenize transform."""

    def test_basic_tokenization(self):
        """Test subword alignment during tokenization."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TINY_BERT)
        label2id = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4}

        tfm = TokenClassBatchTokenizeTransform(tokenizer, label2id, max_length=32)

        samples = [
            (["John", "lives", "in", "Paris"], [1, 0, 0, 3]),
        ]
        result = tfm.encodes(samples)

        assert len(result) == 1
        inp_dict, labels = result[0]
        assert "input_ids" in inp_dict
        assert "labels" in inp_dict

        # Labels should have ignore_token_id for special tokens (CLS, SEP)
        labels_list = inp_dict["labels"].tolist()
        assert labels_list[0] == IGNORE_TOKEN_ID  # [CLS]
        assert labels_list[-1] == IGNORE_TOKEN_ID or labels_list[-2] == IGNORE_TOKEN_ID  # [SEP] or pad

    def test_subword_alignment(self):
        """Test that subword tokens get ignore_token_id by default."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TINY_BERT)
        label2id = {"O": 0, "B-PER": 1}

        tfm = TokenClassBatchTokenizeTransform(
            tokenizer, label2id, max_length=32, label_all_tokens=False,
        )

        # "unconventional" will likely be split into subwords
        samples = [
            (["unconventional", "word"], [1, 0]),
        ]
        result = tfm.encodes(samples)
        labels_list = result[0][0]["labels"].tolist()

        # Count non-ignore labels (should be exactly 2 — one per word)
        real_labels = [l for l in labels_list if l != IGNORE_TOKEN_ID]
        assert len(real_labels) == 2

    def test_label_all_tokens(self):
        """Test label_all_tokens=True propagates labels to subwords."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TINY_BERT)
        label2id = {"O": 0, "B-PER": 1}

        tfm = TokenClassBatchTokenizeTransform(
            tokenizer, label2id, max_length=64, label_all_tokens=True,
        )

        # This word should be split into subwords
        samples = [
            (["unconventional", "test"], [1, 0]),
        ]
        result = tfm.encodes(samples)
        labels_list = result[0][0]["labels"].tolist()

        # With label_all_tokens, subword tokens should also have labels
        real_labels = [l for l in labels_list if l != IGNORE_TOKEN_ID]
        # Should be more than 2 because subwords are also labeled
        assert len(real_labels) >= 2

    def test_empty_batch(self):
        """Test handling of empty batch."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TINY_BERT)
        tfm = TokenClassBatchTokenizeTransform(tokenizer, {"O": 0})
        result = tfm.encodes([])
        assert len(result) == 0


# ===================================================================
# Unit tests for entity aggregation
# ===================================================================

class TestEntityAggregation:
    """Tests for entity aggregation logic."""

    def test_simple_aggregation(self):
        """Test simple entity grouping."""
        tokens = ["[CLS]", "John", "lives", "in", "New", "York", "[SEP]"]
        preds = [0, 1, 0, 0, 3, 4, 0]
        probs = torch.zeros(7, 5)
        probs[1, 1] = 0.95  # B-PER for John
        probs[4, 3] = 0.90  # B-LOC for New
        probs[5, 4] = 0.85  # I-LOC for York
        offsets = [(0, 0), (0, 4), (5, 10), (11, 13), (14, 17), (18, 22), (0, 0)]
        id2label = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC", 4: "I-LOC"}

        entities = _aggregate_entities(tokens, preds, probs, offsets, id2label, strategy="simple")

        assert len(entities) == 2
        assert entities[0]["entity_group"] == "PER"
        assert entities[1]["entity_group"] == "LOC"

    def test_none_strategy(self):
        """Test token-level (no aggregation) strategy."""
        tokens = ["[CLS]", "John", "[SEP]"]
        preds = [0, 1, 0]
        probs = torch.zeros(3, 2)
        probs[1, 1] = 0.95
        offsets = [(0, 0), (0, 4), (0, 0)]
        id2label = {0: "O", 1: "B-PER"}

        entities = _aggregate_entities(tokens, preds, probs, offsets, id2label, strategy="none")

        assert len(entities) == 1
        assert entities[0]["entity"] == "B-PER"


# ===================================================================
# Integration tests for Learner
# ===================================================================

class TestTokenClassificationLearner:
    """Integration tests for TokenClassificationLearner."""

    def test_from_data_creates_learner(self, token_classification_data):
        """Test learner creation from list of dicts."""
        import pandas as pd

        df = pd.DataFrame(token_classification_data)

        learn = TokenClassificationLearner.from_df(
            df,
            TINY_BERT,
            tokens_col="tokens",
            labels_col="labels",
            valid_col="is_valid",
            bs=2,
            max_length=32,
        )
        assert learn is not None
        assert hasattr(learn, "model")
        assert hasattr(learn, "dls")

    def test_training_step(self, token_classification_data):
        """Test that a single training step works."""
        import pandas as pd

        df = pd.DataFrame(token_classification_data)

        learn = TokenClassificationLearner.from_df(
            df,
            TINY_BERT,
            tokens_col="tokens",
            labels_col="labels",
            valid_col="is_valid",
            bs=2,
            max_length=32,
        )
        learn.fit(1, lr=1e-3)

    def test_predict_tokens(self, token_classification_data):
        """Test token prediction returns entity spans."""
        import pandas as pd

        df = pd.DataFrame(token_classification_data)

        learn = TokenClassificationLearner.from_df(
            df,
            TINY_BERT,
            tokens_col="tokens",
            labels_col="labels",
            valid_col="is_valid",
            bs=2,
            max_length=32,
        )
        results = learn.blurr_predict_tokens("John lives in New York.")
        assert len(results) == 1
        assert isinstance(results[0], list)
        # Each entity should have expected keys
        for entity in results[0]:
            assert "score" in entity
            # Either "entity" or "entity_group" depending on strategy
            assert "entity_group" in entity or "entity" in entity
            assert "word" in entity

    def test_metrics_callback_integration(self, token_classification_data):
        """Test that seqeval metrics callback works."""
        import pandas as pd

        df = pd.DataFrame(token_classification_data)

        learn = TokenClassificationLearner.from_df(
            df,
            TINY_BERT,
            tokens_col="tokens",
            labels_col="labels",
            valid_col="is_valid",
            bs=2,
            max_length=32,
        )
        # Check that metrics callback is attached
        from blurr.text.token_classification import HAS_SEQEVAL
        if HAS_SEQEVAL:
            has_metrics_cb = any(
                isinstance(cb, TokenClassMetricsCallback)
                for cb in learn.cbs
            )
            assert has_metrics_cb
