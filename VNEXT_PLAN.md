# ohmeow/blurr — vNext (Major Rewrite) Plan

Last updated: 2026-02-13

## 1) Confirmed scope decisions (from maintainer)

This plan reflects the clarified constraints:

- **Repository:** `ohmeow/blurr`
- **Versioning:** **new MAJOR version**, clean break is allowed
- **Backward compatibility:** **not required**
- **Minimum Python:** **3.11**
- **Stack target:** latest stable
  - PyTorch
  - fastai
  - transformers
  - datasets
  - accelerate
  - peft
  - trl
- **Product goal:** practical developer workflows for **text now**, then **vision/audio/multimodal** (phased)
- **Delivery expectation:** branch + PR workflow, polished GitHub Actions CI/CD
- **Success bar:** users can actually use it; task tests exist; README + docs + code are easy to maintain

---

## 2) Recommended base branch + branch strategy

## Recommended base branch

**Use `master` as the integration base**, then create a long-lived vNext branch:

- Create: `vnext/4.x` from `master`
- All feature work branches from `vnext/4.x`
- All PRs target `vnext/4.x`
- Merge `vnext/4.x` to `master` only when release-ready (major cut)

### Why this recommendation

- `master` is the canonical default branch in GitHub workflows/release expectations.
- Clean break means we do **not** need to preserve old API behavior, so we can design forward.
- `dev-3.0.0` can be treated as a **reference/cherry-pick source**, not a merge target.

## PR workflow (required)

- Branch naming: `feat/*`, `fix/*`, `docs/*`, `ci/*`, `refactor/*`
- Conventional commits in PRs (`feat:`, `fix:`, `docs:`, etc.)
- PR template with:
  - problem statement
  - scope
  - test evidence
  - docs impact
  - breaking-change note
- CODEOWNERS review for core areas
- Squash merge into `vnext/4.x`

---

## 3) Architecture decision (recommended)

## Decision: **src-first architecture** (not notebook-first)

Use standard Python project layout:

```text
src/blurr/
  core/
  data/
  tasks/
    text/
    vision/
    audio/
    multimodal/
  training/
  integrations/
    hf/
    fastai/
    peft/
    trl/
tests/
docs/
examples/
```

### Rationale

- Better long-term maintainability and contributor onboarding.
- Cleaner CI, typing, linting, packaging, and unit-test flow.
- Better fit for multi-modality growth than notebook-export-first code ownership.

### Notebook role

- Keep notebooks as **examples/tutorials/research artifacts only**.
- Canonical source code remains in `src/`.

---

## 4) Dependency and runtime policy

- **Python:** `>=3.11`
- Package metadata via `pyproject.toml`
- Use latest stable ranges and lock tested builds (uv or pip-tools lock)
- Recommended extras:
  - `blurr[text]`
  - `blurr[vision]`
  - `blurr[audio]`
  - `blurr[multimodal]`
  - `blurr[peft]`
  - `blurr[trl]`
  - `blurr[dev]`

### Compatibility policy

- Keep CI pinned to a tested lockfile for reproducibility
- Run scheduled dependency upgrade CI weekly
- Publish compatibility matrix in docs

---

## 5) CI/CD design (GitHub Actions)

## CI workflows

1. **`ci.yml`** (PR + push to `vnext/4.x`)
   - lint (ruff)
   - format check (black)
   - type checks (pyright/mypy)
   - unit tests (pytest)
   - matrix: Python 3.11 + 3.12
   - coverage upload and minimum threshold gate

2. **`task-tests.yml`**
   - fast task smoke/integration tests with tiny models/datasets:
     - text classification
     - token classification
     - seq2seq/qa baseline (if in scope)
     - vision classification (phase gate)
     - audio classification/ASR smoke (phase gate)
     - multimodal smoke (phase gate)

3. **`docs.yml`**
   - build docs on PR
   - fail PR if docs break
   - deploy docs on merge to `vnext/4.x`/`master`

4. **`release.yml`**
   - trigger on version tag (`v4.*`)
   - build sdist + wheels
   - publish to TestPyPI then PyPI (manual approval step)
   - create GitHub Release + changelog notes

## Branch protections

- Require all CI checks green before merge
- Require 1–2 approvals
- Require up-to-date branch before merge
- Disallow direct pushes to `master` and `vnext/4.x`

---

## 6) Phased implementation roadmap

## Phase 0 — Foundation (1–2 weeks)

- Scaffold `src/` architecture
- pyproject + lockfile + tooling
- CI/CD baseline workflows + branch protections
- initial docs skeleton

**Exit:** green CI with quality gates and packaging working.

## Phase 1 — Text MVP (2–4 weeks)

- Sequence classification + token classification first
- fastai + HF training ergonomics
- reliable data pipelines/collators/metrics
- practical README quickstart

**Exit:** text users can train/eval/infer in documented flows.

## Phase 2 — Advanced training integrations (2–3 weeks)

- accelerate strategy integration
- PEFT integration (LoRA/adapter path)
- TRL-ready training hooks/wrappers

**Exit:** documented adapter fine-tuning path + tests.

## Phase 3 — Vision and audio support (3–5 weeks)

- vision classification task path
- audio starter task path (classification or ASR baseline)
- unified training API conventions across modalities

**Exit:** each modality has at least one working, tested task.

## Phase 4 — Multimodal path (3–6 weeks)

- text+vision and/or text+audio starter pipelines
- multimodal collators and inference examples

**Exit:** at least one end-to-end multimodal workflow is stable and documented.

---

## 7) Testing strategy and coverage expectations

## Unit tests (core)

- data transforms/collators
- model wrappers/callbacks
- metrics and decoding
- config and integration boundaries

## Integration tests (task-level)

- run tiny end-to-end train/eval/infer loops for each supported task
- ensure no regression in API examples from README

## Coverage targets

- Core package coverage target: **>=85%**
- Critical path tests (data→train→infer) required for every supported task

---

## 8) Documentation and usability plan

- README optimized for "time-to-first-success" (<10 min path)
- docs include:
  - install matrix
  - task tutorials (text/vision/audio/multimodal)
  - API reference from `src/`
  - migration notes from legacy major versions (high-level only, no compatibility promises)
- examples kept runnable via CI smoke checks

---

## 9) Explicit release success criteria

vNext major is successful only when all are true:

1. **Usable by developers**
   - Fresh user can install and run at least one text workflow quickly.
2. **Task test coverage exists**
   - Unit + integration tests pass for supported tasks.
3. **CI/CD is production-grade**
   - Quality gates, docs build, release workflow, branch protections are active.
4. **Maintainable code/docs**
   - Clear module boundaries; README and docs match real API.
5. **Phased modality support is real**
   - Vision/audio/multimodal either implemented or clearly marked with milestone status.

---

## 10) Immediate next actions

1. Create `vnext/4.x` from `master`.
2. Land scaffolding PR: `pyproject + src skeleton + CI + docs skeleton`.
3. Land Text MVP PRs (sequence + token classification).
4. Land integrations PRs (accelerate/peft/trl).
5. Cut `v4.0.0-alpha.1` when Phase 1 exits are met.
