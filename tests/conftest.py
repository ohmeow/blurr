"""Shared fixtures for blurr tests.

All tests use tiny models and small datasets so they run WITHOUT GPU
in reasonable time (< 30s per test).
"""

import os
import warnings

import pandas as pd
import pytest
import torch

# Suppress noisy HF warnings during tests
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Tiny model names — small enough to run on CPU
# ---------------------------------------------------------------------------

TINY_BERT = "prajjwal1/bert-tiny"  # 4.4M params, 2 layers
TINY_GPT2 = "sshleifer/tiny-gpt2"  # Tiny GPT-2
TINY_T5 = "google/t5-efficient-tiny"  # Tiny T5


@pytest.fixture
def tiny_bert_name():
    return TINY_BERT


@pytest.fixture
def tiny_gpt2_name():
    return TINY_GPT2


@pytest.fixture
def tiny_t5_name():
    return TINY_T5


# ---------------------------------------------------------------------------
# Sample classification data
# ---------------------------------------------------------------------------

@pytest.fixture
def classification_df():
    """Small DataFrame for classification tests."""
    data = {
        "text": [
            "This movie is great!",
            "Terrible waste of time.",
            "I loved every minute of it.",
            "Boring and dull movie.",
            "Absolutely fantastic film!",
            "Would not recommend to anyone.",
            "A masterpiece of cinema.",
            "One of the worst movies ever.",
            "Surprisingly good for a sequel.",
            "Not worth watching at all.",
            "Great performances by the cast.",
            "The plot was confusing and bad.",
            "Really enjoyed this one.",
            "Waste of money and time.",
            "Excellent direction and writing.",
            "Poorly made in every way.",
        ],
        "label": [
            "positive", "negative", "positive", "negative",
            "positive", "negative", "positive", "negative",
            "positive", "negative", "positive", "negative",
            "positive", "negative", "positive", "negative",
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def multilabel_df():
    """Small DataFrame for multi-label classification tests."""
    data = {
        "text": [
            "This is toxic and obscene content.",
            "What a beautiful day outside!",
            "You are an idiot, go away!",
            "I love reading books.",
            "Die you fool, I hate you!",
            "The weather is nice today.",
            "Shut up, you moron!",
            "Great job on the presentation.",
        ],
        "toxic": [1, 0, 1, 0, 1, 0, 1, 0],
        "obscene": [1, 0, 0, 0, 0, 0, 1, 0],
        "insult": [0, 0, 1, 0, 1, 0, 1, 0],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Sample token classification data
# ---------------------------------------------------------------------------

@pytest.fixture
def token_classification_data():
    """List of dicts for token classification tests."""
    data = [
        {
            "tokens": ["John", "lives", "in", "New", "York", "."],
            "labels": ["B-PER", "O", "O", "B-LOC", "I-LOC", "O"],
            "is_valid": False,
        },
        {
            "tokens": ["Apple", "is", "a", "company", "."],
            "labels": ["B-ORG", "O", "O", "O", "O"],
            "is_valid": False,
        },
        {
            "tokens": ["Paris", "is", "the", "capital", "of", "France", "."],
            "labels": ["B-LOC", "O", "O", "O", "O", "B-LOC", "O"],
            "is_valid": False,
        },
        {
            "tokens": ["Mary", "works", "at", "Google", "."],
            "labels": ["B-PER", "O", "O", "B-ORG", "O"],
            "is_valid": True,
        },
        {
            "tokens": ["Berlin", "is", "in", "Germany", "."],
            "labels": ["B-LOC", "O", "O", "B-LOC", "O"],
            "is_valid": True,
        },
    ]
    return data


# ---------------------------------------------------------------------------
# Sample seq2seq data
# ---------------------------------------------------------------------------

@pytest.fixture
def seq2seq_df():
    """Small DataFrame for seq2seq tests."""
    data = {
        "text": [
            "The quick brown fox jumps over the lazy dog. The fox was very fast and the dog was very slow.",
            "Scientists have discovered a new species of deep-sea fish in the Pacific Ocean.",
            "The stock market rose sharply today after the Federal Reserve announced new policies.",
            "A new study shows that regular exercise can improve mental health significantly.",
            "The city council approved a plan to build new parks and recreational facilities.",
            "Researchers found that coffee consumption may reduce the risk of certain diseases.",
        ],
        "summary": [
            "A fox jumps over a dog.",
            "New deep-sea fish species found.",
            "Stock market rises after Fed announcement.",
            "Exercise improves mental health.",
            "City council approves new parks.",
            "Coffee may reduce disease risk.",
        ],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Sample causal LM data
# ---------------------------------------------------------------------------

@pytest.fixture
def causal_lm_df():
    """Small DataFrame for causal LM tests."""
    data = {
        "text": [
            "The quick brown fox jumps over the lazy dog.",
            "Once upon a time in a faraway land, there lived a princess.",
            "The weather today is sunny and warm with a gentle breeze.",
            "In the beginning, there was nothing but darkness and silence.",
            "The scientist made a groundbreaking discovery in her laboratory.",
            "A long time ago, in a galaxy far far away.",
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def instruction_df():
    """Small DataFrame for instruction tuning tests."""
    data = {
        "instruction": [
            "Summarize the following text.",
            "Translate to French.",
            "What is the capital of France?",
            "Write a haiku about nature.",
        ],
        "response": [
            "The text discusses recent advances in AI technology.",
            "Bonjour, comment allez-vous?",
            "The capital of France is Paris.",
            "Cherry blossoms fall / Gentle breeze through bamboo grove / Nature's song endures",
        ],
    }
    return pd.DataFrame(data)
