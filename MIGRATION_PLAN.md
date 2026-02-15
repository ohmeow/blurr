# BLURR vNext Migration Plan (Python >=3.11)

## Scope
This plan tracks the migration from legacy `blurr` (Python 3.7-era stack) to a modern, maintainable vNext on top of current fastai + Hugging Face tooling.

## Principles
- Start from `master` (not `dev-3.0.0`) and keep history linear via PRs.
- Preserve nbdev notebook-first workflow (`settings.ini`, export/docs/test tooling).
- Prefer small, testable increments over large rewrites.
- Allow temporary compatibility shims only when they reduce migration risk.

## Phase 1 (completed in this branch)
- Refresh metadata/dependencies for Python >=3.11.
- Add build-system metadata (`pyproject.toml`) while keeping `settings.ini` + `setup.py` flow.
- Modernize CI for Python 3.11/3.12 with:
  - dependency install,
  - pytest run,
  - notebook cleanliness check,
  - notebook execution tests (`nbdev_test`).
- Create migration roadmap artifacts (`MIGRATION_PLAN.md`, `TODO.md`).

## Phase 2: Text stack first (priority)
1. Revalidate and update core text pipelines (classification, token classification, QA, summarization, translation, LM).
2. Resolve API breaks introduced by modern `transformers`/`datasets`/`fastai`.
3. Add/expand task-focused tests (unit + notebook smoke) for each text task.
4. Update README + docs examples to one canonical modern text workflow.

## Phase 3: Vision + audio
1. Audit current placeholders and legacy assumptions for vision/audio.
2. Implement minimal supported task path(s) with clear "supported vs experimental" boundaries.
3. Add CI-friendly smoke tests/notebooks for each supported modality.

## Phase 4: Multimodal and advanced adapters
1. Add PEFT/TRL/Accelerate-first integration points.
2. Add multimodal training/inference abstractions with explicit model support matrix.
3. Provide migration notes for users moving from 1.x APIs.

## Release strategy
- Keep vNext work on `ship/vnext-rewrite` branch via incremental PRs into `master`.
- Tag first stable modern release as `2.0.0` after Phase 2 is green and documented.
