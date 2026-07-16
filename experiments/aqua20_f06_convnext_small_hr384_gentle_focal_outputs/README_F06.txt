AQUA20 F06 OFFICIAL EXPERIMENT PACKAGE

Campaign:
04-07-2026 AQUA20 FINAL BEAT BASE PLAN

Experiment:
F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL

Scientific role:
High-resolution fine-tuning.

Source branch:
F03_CONVNEXT_SMALL_GENTLE_CB_FOCAL

Method:
ConvNeXt-Small
+ original AQUA20 training distribution
+ Gentle Class-Balanced Focal Loss
+ 384 x 384 high-resolution fine-tuning

Source checkpoint SHA-256:
928c8af60aef70537263e2c8cc1cf13782747bc5974f86ff054fbe352065c26f

F06 best epoch:
2

F06 validation Macro F1:
88.5122%

F06 checkpoint SHA-256:
d0c225c28b25f09822bc99518a37f9b1fe642b2d01213da7f68e2d76a255832a

Training termination:
9 fully completed epochs.
Epoch 10 was manually interrupted and excluded.
Training was manually terminated following a sustained validation Macro F1 plateau.
The official test split was not used for stopping or checkpoint selection.

Official test split:
1,612 images
20 classes

Official F06 test results:
Top-1: 92.1216%
Top-2: 97.9529%
Top-3: 99.2556%
Top-5: 99.7519%
Macro Precision: 90.4312%
Macro Recall: 87.4362%
Macro F1: 88.1699%
Weighted F1: 92.0810%

Scientific finding:
High-resolution fine-tuning improved the isolated F03 Gentle CB Focal branch.
F06 exceeded F03 in official-test Top-1, Macro Recall, and Macro F1.

F06 passed the campaign Macro Recall, Top-2, and Top-3 targets.

F06 did not exceed the campaign Top-1, Macro Precision, or Macro F1 targets.

The result supports high-resolution fine-tuning as a positive but insufficient
standalone intervention for fully beating the campaign target.
