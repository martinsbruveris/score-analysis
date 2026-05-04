# Copilot Instructions

## Project Overview

`score-analysis` is a Python library for evaluating ML model results in binary classification scenarios. It provides vectorized metrics (TPR, FPR, FNR, EER, AUC), threshold-setting, ROC curves with confidence bands, bootstrapped CIs, and bias analysis across demographic groups.

The library uses biometric/security terminology (genuine/fraud, accept/reject) alongside standard ML terminology and is agnostic about score direction via a `score_class` parameter.

## Commands

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
```

## Linting and Formatting

Uses **ruff** for linting and formatting (`pyproject.toml`). Target: `py312`, line length: 88. Lint rules: `E`, `F`, `I`.

- `# fmt: skip` — suppress formatting for a line
- `# fmt: off` / `# fmt: on` — suppress formatting for a block
- `# noqa: F401` — suppress unused import warnings

## Architecture

### `score_analysis.scores` — Core

`Scores` is the central class. Stores sorted numpy arrays `pos` and `neg`. Constructed via:
- `Scores(pos, neg)` — direct construction
- `Scores.from_labels(labels, scores, pos_label=1)` — from label/score arrays

Key parameters:
- `score_class` — whether high scores indicate the positive (`"pos"`) or negative (`"neg"`) class
- `equal_class` — which class samples at the threshold boundary belong to
- `nb_easy_pos` / `nb_easy_neg` — counts of trivially-correct samples excluded from stored arrays (used to speed up evaluation of highly accurate classifiers)

`BootstrapConfig` is a frozen dataclass controlling bootstrap CI sampling (`nb_samples`, `bootstrap_method`, `sampling_method`, `stratified_sampling`).

### `score_analysis.cm` — Confusion Matrix

`ConfusionMatrix` supports N-class and binary matrices with vectorized shape `(X, N, N)`. Binary mode uses explicit pos/neg classes. `one_vs_all()` converts an N-class matrix to N binary `(X, N, 2, 2)` matrices.

The `@cm_class_metric` decorator handles the binary vs. multi-class dispatch and the `as_dict` parameter uniformly.

### `score_analysis.metrics` — Pure Metric Functions

Pure numpy functions on `(..., 2, 2)` arrays. Convention: `matrix[..., 0, 0]` = TP, `matrix[..., 1, 1]` = TN, `matrix[..., 0, 1]` = FN, `matrix[..., 1, 0]` = FP.

### `score_analysis.group_scores` and `score_analysis.showbias` — Group/Bias Analysis

`GroupScores` extends `Scores` with group membership. `@groupwise` decorator converts any metric function into a per-group function. `showbias()` / `BiasFrame` measures metric variation across groups in a DataFrame.

### `score_analysis.applications` — Domain Subclasses

Domain-specific subclasses, e.g., `FraudScores` with genuine/fraud terminology wrapping the same underlying logic.

### `score_analysis.experimental` — Research Code

Not part of the stable API. Contains synthetic datasets and alternative ROC CI methods. Changes here do not require backward compatibility.

## Vectorization Conventions

- All operations are fully vectorized using numpy broadcasting.
- Confusion matrices have shape `(X, N, N)` where `X` is arbitrary batch dimensions.
- `one_vs_all()` produces `(X, N, 2, 2)`.
- Scalar results are returned as Python scalars when batch dimensions are `()`.
- `pos` and `neg` score arrays are always kept sorted (ascending) internally.

## CI/CD

- **Tests** (`.github/workflows/tests.yml`): Python 3.9–3.14 on ubuntu-latest
- **Publish** (`.github/workflows/publish.yml`): builds with `uv build`, publishes to PyPI via trusted publishing
