# TODO (vNext Rewrite)

## Phase 1 - Foundation (done)
- [x] Move metadata baseline to Python >=3.11.
- [x] Refresh dependency definitions for modern HF/fastai stack.
- [x] Add build-system metadata via `pyproject.toml`.
- [x] Modernize CI for py3.11/3.12 + notebook checks.
- [x] Add migration planning docs.

## Phase 2 - Text tasks first
- [ ] Sequence classification pipeline audit + fixes.
- [ ] Token classification (NER) audit + fixes.
- [ ] Question answering audit + fixes.
- [ ] Summarization + translation audit + fixes.
- [ ] Causal + masked LM audit + fixes.
- [ ] Add pytest + notebook smoke tests per text task.
- [ ] Refresh README/docs with a single modern text quickstart.

## Phase 3 - Vision and audio
- [ ] Define explicit supported vision tasks.
- [ ] Define explicit supported audio tasks.
- [ ] Implement minimal end-to-end training/inference flow per supported task.
- [ ] Add CI smoke checks for vision/audio notebooks.

## Phase 4 - Multimodal + adapters
- [ ] Integrate Accelerate/PEFT/TRL paths where API-stable.
- [ ] Add multimodal task abstractions incrementally.
- [ ] Publish compatibility matrix (models/tasks/features).
- [ ] Write migration guide: 1.x -> 2.x API changes.
