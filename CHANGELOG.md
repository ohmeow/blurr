# Changelog

All notable changes to this project are documented in this file.

## [2.0.0a0] - 2026-02-14

### Added
- New v2 task-first learner APIs:
  - `TextClassificationLearner`
  - `TokenClassificationLearner`
  - `Seq2SeqLearner`
  - `CausalLMLearner`
  - `VisionClassificationLearner`
  - `MultimodalClassificationLearner`
- PEFT/LoRA integration path for causal LM workflows.
- Vision and multimodal support across classification/contrastive/VQA patterns.
- Expanded pytest coverage for text, vision, and multimodal modules (61 passing tests in current vNext phase).

### Changed
- Modernized dependency baseline and CI for Python 3.11/3.12.
- Refined package-level public API exports in:
  - `blurr/__init__.py`
  - `blurr/text/__init__.py`
  - `blurr/vision/__init__.py`
  - `blurr/multimodal/__init__.py`
- Reworked documentation for notebook-first v2 workflows.

### Documentation
- Polished vNext notebooks:
  - `nbs/50_text_classification.ipynb`
  - `nbs/51_token_classification.ipynb`
  - `nbs/52_seq2seq.ipynb`
  - `nbs/53_causal_lm.ipynb`
  - `nbs/54_vision_classification.ipynb`
  - `nbs/55_multimodal_classification.ipynb`
- Replaced README with v2-focused overview, install guidance, quick starts, feature matrix, migration summary, and contributing notes.

### Migration
- v1 users should follow `MIGRATION_PLAN.md` for detailed upgrade guidance and API mapping.
