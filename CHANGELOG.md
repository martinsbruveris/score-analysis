# Change Log

## v0.1.6 - 2022-11-08

- Added threshold setting at arbitrary metrics.

## v0.1.2 - 2022-07-04

- Fixing bug with bootstrapping CIs when using virtual datasets.
- Add specific doc fraud module.
- Add FAR, FRR, etc. aliases.

## v0.1.1 - 2022-05-16

- Added support for virtual datasets, i.e., `nb_easy_pos`, `nb_easy_neg` parameters
  in `Scores`.

## v0.1.0 - 2022-05-11

- Improved ROC curve CI calculation.
- Added parameter `bootstrap_method` in `Scores.bootstrap_ci` allowing calculation
  of bootstrapped CIs using the percentile, bias-corrected and accelerated methods.
- Renamed parameter in `Scores.bootsrap_ci` from `method` to `sampling_method`.
- Added `pointwise_cm` method to compute sample-wise membership for confusion matrices.
- Removed `BinaryConfusionMatrix` and absorbed the functionality in `ConfusionMatrix` 
  using the parameter `binary=True`.
- Reordered parameters in `Scores.from_labels` for consistency.

## v0.0.5 - 2022-05-02

- Fixed bug in bootstrapping code

## v0.0.4 - 2022-04-08

- Add bootstrapping to compute confidence interval for Scores
- Add notebook to plot ROC curves with confidence intervals

## v0.0.3 - 2022-04-07

- Add confidence intervals for TPR, TNR, FPR, FNR.
