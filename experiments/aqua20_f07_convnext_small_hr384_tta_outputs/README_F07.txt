AQUA20 F07 — OFFICIAL FINAL BEAT BASE CAMPAIGN

Experiment:
F07_CONVNEXT_SMALL_HR384_TTA

Scientific role:
Deterministic test-time augmentation on the locked F06 HR384 model.

Source:
F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL
Best epoch: 2

TTA protocol:
1. Original 384 x 384 image
2. Horizontal-flipped 384 x 384 image
3. Softmax each view
4. Arithmetic mean of probabilities
5. Prediction from averaged probabilities

Validation:
Single-view Macro F1: 88.5122%
F07 TTA Macro F1:    88.9836%
Delta:                +0.4714 pp

Official test:
Top-1:           92.1216%
Top-2:           97.8908%
Top-3:           99.3176%
Top-5:           99.6898%
Macro Precision: 89.8765%
Macro Recall:    86.5324%
Macro F1:        87.4026%
Weighted F1:     92.0771%

Scientific conclusion:
The fixed two-view TTA protocol improved validation Macro F1 but did not
generalize to the unchanged official test split. Test Top-1 was unchanged,
while balanced-class metrics declined. F07 therefore does not replace F06
as the stronger completed branch.

Important loss note:
The F07 NLL reported during Step 4 is standard softmax negative
log-likelihood. It is not directly comparable with the previously reported
F06 Gentle Class-Balanced Focal test loss.

F07 is inference-only and produces no new checkpoint.
The verified F06 source checkpoint is recorded by SHA-256.
