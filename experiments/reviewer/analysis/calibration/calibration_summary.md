# F06 Calibration Analysis

## Scope

Model:

`F06 — ConvNeXt-Small HR384 + Gentle Class-Balanced Focal Loss`

Evaluation set:

`AQUA20 official unchanged test split`

Samples:

`1612`

Classes:

`20`

No training, checkpoint selection, probability tuning, or post-hoc
calibration was performed. The analysis uses the frozen F06 probability
outputs preserved during the completed F08 inference workflow.

## Input verification

Recomputed Top-1 accuracy:

`92.1216%`

Frozen reported F06 Top-1:

`92.1216%`

Status:

`MATCHED`

Source archive SHA-256:

`30d28b7a7d1d79be9e48a571207607815ab5b50e7751cf80631ffa75bc7a505f`

## Calibration metrics

Expected Calibration Error (ECE; 15 equal-width bins):

`0.017507`

Negative Log-Likelihood (NLL):

`0.251138`

Multiclass Brier Score:

`0.119981`

Mean Top-1 confidence:

`0.936777`

Top-1 accuracy:

`0.921216`

Mean confidence minus accuracy:

`+0.015561`

## Metric definitions

ECE is the standard top-label expected calibration error using
15 equal-width confidence bins over [0, 1].

NLL is the mean negative logarithm of the predicted probability assigned
to the true class.

The multiclass Brier Score is calculated as:

`mean_i sum_k (p_ik - y_ik)^2`

and is not divided by the number of classes.

## Interpretation rule

These values describe the probability calibration of the frozen F06
model on the official test set.

They must not be interpreted as training-seed variability, bootstrap
uncertainty, or evidence from a fitted post-hoc calibration method.

The official test set was used only for final descriptive calibration
evaluation and was not used to tune the model or calibration parameters.
