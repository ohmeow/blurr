# blurr v2.0.0a0

[![CI](https://github.com/ohmeow/blurr/actions/workflows/test.yml/badge.svg)](https://github.com/ohmeow/blurr/actions/workflows/test.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)

`blurr` is a notebook-first library that integrates **Hugging Face Transformers** with **fastai** for practical NLP, vision, and multimodal fine-tuning.

Version `2.0.0a0` focuses on a cleaner high-level learner API, modern dependency support, and explicit task modules for vNext.

---

## Overview

blurr v2 gives you task-specific learners that feel like fastai while exposing Hugging Face flexibility when you need it:

- Text classification (single-label and multi-label)
- Token classification / NER
- Seq2Seq generation (summarization/translation)
- Causal language modeling + instruction tuning
- Vision classification
- Multimodal classification (CLIP/BLIP/SigLIP style), contrastive mode, and VQA

> **Notebook-first docs:** the canonical docs and examples live in `nbs/` and are designed for nbdev workflows.

---

## Installation

### From PyPI

```bash
pip install ohmeow-blurr
```

### Editable install (recommended for contributors)

```bash
git clone https://github.com/ohmeow/blurr.git
cd blurr
pip install -e ".[dev]"
```

---

## Quick Start by Task

### 1) Text Classification

```python
from blurr.text.classification import TextClassificationLearner

learn = TextClassificationLearner.from_hf(
    "imdb",
    model_name="prajjwal1/bert-tiny",
    text_col="text",
    label_col="label",
    max_samples=200,
)
learn.fit_one_cycle(1, 2e-5)
learn.blurr_predict("This movie was excellent!")
```

### 2) Token Classification (NER)

```python
from blurr.text.token_classification import TokenClassificationLearner

learn = TokenClassificationLearner.from_hf(
    "conll2003",
    model_name="prajjwal1/bert-tiny",
    tokens_col="tokens",
    labels_col="ner_tags",
    max_samples=200,
)
learn.fit_one_cycle(1, 2e-5)
learn.blurr_predict_tokens("Hugging Face is based in New York City.")
```

### 3) Seq2Seq (Summarization / Translation)

```python
from blurr.text.seq2seq import Seq2SeqLearner

learn = Seq2SeqLearner.from_df(
    df,
    model_name="t5-small",
    input_col="article",
    target_col="summary",
    prefix="summarize: ",
)
learn.fit_one_cycle(1, 2e-4)
learn.blurr_generate("Long article text...", prefix="summarize: ")
```

### 4) Causal LM + Instruction Tuning

```python
from blurr.text.causal_lm import CausalLMLearner

learn = CausalLMLearner.from_df(
    df,
    model_name="sshleifer/tiny-gpt2",
    text_col="text",
)
learn.fit_one_cycle(1, 2e-4)
learn.blurr_generate("Once upon a time", max_new_tokens=64)
```

### 5) Vision Classification

```python
from blurr.vision.classification import VisionClassificationLearner

learn = VisionClassificationLearner.from_hf(
    "beans",
    model_name="google/vit-base-patch16-224",
    image_col="image",
    label_col="labels",
    max_samples=200,
)
learn.fit_one_cycle(1, 2e-4)
```

### 6) Multimodal Classification / Zero-shot

```python
from blurr.multimodal.classification import MultimodalClassificationLearner

learn = MultimodalClassificationLearner.from_df(
    df,
    model_name="openai/clip-vit-base-patch32",
    image_col="image",
    text_col="text",
    label_col="label",
    task="classification",
)
learn.blurr_zero_shot_predict(image, ["cat", "dog", "car"])
```

---

## Feature Matrix

| Area | Status in v2.0.0a0 | Notes |
|---|---|---|
| Text Classification | ✅ | Single-label + multi-label |
| Token Classification (NER) | ✅ | Subword label alignment + aggregation strategies |
| Seq2Seq (Summarization/Translation) | ✅ | Generation helpers + ROUGE callback support |
| Causal LM | ✅ | Generation + instruction formatting + prompt masking |
| PEFT / LoRA | ✅ | Adapter-based fine-tuning for causal LM |
| Vision Classification | ✅ | HF image processors + fastai training loops |
| Multimodal Classification | ✅ | Classification + contrastive + VQA task modes |
| Zero-shot multimodal inference | ✅ | CLIP/SigLIP-style candidate scoring |

---

## Migration Guide (v1 → v2)

v2 introduces a clearer, task-focused API and modernized dependencies.

High-level changes:

- Prefer task learners from dedicated modules:
  - `blurr.text.classification.TextClassificationLearner`
  - `blurr.text.token_classification.TokenClassificationLearner`
  - `blurr.text.seq2seq.Seq2SeqLearner`
  - `blurr.text.causal_lm.CausalLMLearner`
  - `blurr.vision.classification.VisionClassificationLearner`
  - `blurr.multimodal.classification.MultimodalClassificationLearner`
- Data source constructors are now more explicit (`from_df`, `from_hf`, `from_data`/`from_data_source` depending on task).
- Several older 1.x abstractions are replaced with focused v2 learner APIs.

For full migration details and rationale, see **[`MIGRATION_PLAN.md`](./MIGRATION_PLAN.md)**.

---

## Contributing

Contributions are welcome.

1. Create a feature branch from `master` (or your target branch)
2. Make changes in notebooks under `nbs/` when updating docs/API stories
3. Run tests:
   ```bash
   .venv/bin/python -m pytest tests/ -v
   ```
4. Keep notebooks clean for nbdev workflows (`nbdev_clean` when needed)
5. Open a PR with a concise summary of changes

See **[`CONTRIBUTING.md`](./CONTRIBUTING.md)** for project conventions.

---

## License

Apache-2.0
