# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`score-analysis` is a Python library for evaluating ML model results in binary classification scenarios. It provides vectorized implementations of common metrics (TPR, FPR, FNR, EER, AUC), threshold-setting methods, ROC curves with confidence bands, bootstrapped confidence intervals, and bias analysis across demographic groups.

The library uses biometric/security terminology (genuine/fraud, accept/reject) alongside standard ML terminology (positive/negative class) and is agnostic about score direction, controlled via a `score_class` parameter.

## Common Commands

```bash
# Install dev dependencies
uv sync --locked --dev

# Run full test suite with coverage
make test

# Run tests directly (faster, no coverage)
uv run pytest -sv tests

# Run a specific test file
uv run pytest -sv tests/test_scores.py

# Run a specific test
uv run pytest -sv tests/test_scores.py::test_function_name

# Check style (no modifications)
uv run task check-style

# Auto-format and fix
uv run task format

# Build docs
uv run task build-docs

# Bump version (bumps, commits, tags, pushes)
uv run task bumpversion
```

## Linting and Formatting

Uses **ruff** for both linting and formatting (configured in `pyproject.toml`). Target: `py312`, line length: 88. Lint rules: `E` (pycodestyle), `F` (pyflakes), `I` (isort).

Suppression: `# fmt: skip` for a line, `# fmt: off`/`# fmt: on` for blocks, `# noqa: F401` for unused imports.

## Architecture

### Core: `score_analysis.scores`

The `Scores` class is the central object. It holds sorted arrays of positive (`pos`) and negative (`neg`) scores and provides metric computation at thresholds, threshold-setting at operating points (with linear interpolation), EER, AUC, ROC curves, and bootstrapped confidence intervals. Constructed via `Scores(pos, neg)` or `Scores.from_labels(labels, scores, pos_label=1)`.

### Confusion Matrix: `score_analysis.cm`

`ConfusionMatrix` supports N-class and binary confusion matrices with vectorized shape `(X, N, N)`. Binary mode has explicit pos/neg classes. `one_vs_all()` converts N-class to N binary matrices.

### Metrics: `score_analysis.metrics`

Pure numpy functions operating on `(..., 2, 2)` arrays for all standard binary classification metrics.

### Group Analysis: `score_analysis.group_scores` and `score_analysis.showbias`

`GroupScores` extends `Scores` with group membership tracking. `groupwise()` decorator turns any metric into a per-group function. `showbias()` measures metric variation across groups in a DataFrame.

### Applications: `score_analysis.applications`

Domain-specific subclasses, e.g., `FraudScores` with genuine/fraud terminology.

### Experimental: `score_analysis.experimental`

Research-quality code not part of the stable API (synthetic datasets, alternative ROC CI methods).

## Vectorization Conventions

All operations are vectorized. Confusion matrices have shape `(X, N, N)` where `X` is arbitrary. Metrics yield `(X, Y)`. `one_vs_all()` produces `(X, N, 2, 2)`. Scalar results are returned as Python scalars when `X=Y=()`.

## CI/CD

- **Tests** (`.github/workflows/tests.yml`): Python 3.9–3.14 on ubuntu-latest
- **Publish** (`.github/workflows/publish.yml`): builds with `uv build`, publishes to PyPI via trusted publishing
