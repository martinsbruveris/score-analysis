# Change Log

## Unpublished

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
