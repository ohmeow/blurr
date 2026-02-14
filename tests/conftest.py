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
TINY_VIT = "hf-internal-testing/tiny-random-ViTForImageClassification"
TINY_CLIP = "hf-internal-testing/tiny-random-CLIPModel"
TINY_BLIP_VQA = "hf-internal-testing/tiny-random-BlipForQuestionAnswering"
TINY_SIGLIP = "hf-internal-testing/tiny-random-SiglipModel"


@pytest.fixture
def tiny_bert_name():
    return TINY_BERT


@pytest.fixture
def tiny_gpt2_name():
    return TINY_GPT2


@pytest.fixture
def tiny_t5_name():
    return TINY_T5


@pytest.fixture
def tiny_vit_name():
    return TINY_VIT


@pytest.fixture
def tiny_clip_name():
    return TINY_CLIP


@pytest.fixture
def tiny_blip_vqa_name():
    return TINY_BLIP_VQA


@pytest.fixture
def tiny_siglip_name():
    return TINY_SIGLIP


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
# Sample vision + multimodal data
# ---------------------------------------------------------------------------

@pytest.fixture
def vision_df(tmp_path):
    """Small image classification dataframe using generated RGB squares."""
    from PIL import Image
    import numpy as np

    rows = []
    for i in range(12):
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        if i % 2 == 0:
            arr[..., 0] = 255  # red-ish
            label = "red"
        else:
            arr[..., 2] = 255  # blue-ish
            label = "blue"

        path = tmp_path / f"img_{i}.png"
        Image.fromarray(arr).save(path)
        rows.append({"image": str(path), "label": label, "is_valid": i % 4 == 0})

    return pd.DataFrame(rows)


@pytest.fixture
def vision_multilabel_df(tmp_path):
    """Small multi-label image dataframe."""
    from PIL import Image
    import numpy as np

    rows = []
    for i in range(10):
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        arr[..., 0] = 255 if i % 2 == 0 else 20
        arr[..., 1] = 255 if i % 3 == 0 else 20
        arr[..., 2] = 255 if i % 4 == 0 else 20
        path = tmp_path / f"ml_img_{i}.png"
        Image.fromarray(arr).save(path)
        rows.append(
            {
                "image": str(path),
                "red": 1 if i % 2 == 0 else 0,
                "green": 1 if i % 3 == 0 else 0,
                "blue": 1 if i % 4 == 0 else 0,
                "is_valid": i % 5 == 0,
            }
        )

    return pd.DataFrame(rows)


@pytest.fixture
def multimodal_df(tmp_path):
    """Small multimodal dataframe: image path + text + class label."""
    from PIL import Image
    import numpy as np

    rows = []
    for i in range(10):
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        if i % 2 == 0:
            arr[..., 0] = 255
            text = "a red square"
            label = "warm"
        else:
            arr[..., 2] = 255
            text = "a blue square"
            label = "cool"

        path = tmp_path / f"mm_img_{i}.png"
        Image.fromarray(arr).save(path)
        rows.append({"image": str(path), "text": text, "label": label, "is_valid": i % 4 == 0})

    return pd.DataFrame(rows)


@pytest.fixture
def vqa_df(tmp_path):
    """Small VQA-style dataframe with categorical answer labels."""
    from PIL import Image
    import numpy as np

    rows = []
    answers = ["red", "blue"]
    for i in range(8):
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        answer = answers[i % 2]
        if answer == "red":
            arr[..., 0] = 255
        else:
            arr[..., 2] = 255

        path = tmp_path / f"vqa_img_{i}.png"
        Image.fromarray(arr).save(path)
        rows.append(
            {
                "image": str(path),
                "text": "what color is the square?",
                "label": answer,
                "is_valid": i % 4 == 0,
            }
        )

    return pd.DataFrame(rows)


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
