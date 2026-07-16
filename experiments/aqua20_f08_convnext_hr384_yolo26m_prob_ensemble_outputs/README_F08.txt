AQUA20 F08 OFFICIAL FINAL EXPERIMENT

Campaign:
04-07-2026 AQUA20 FINAL BEAT BASE PLAN

Experiment:
F08_CONVNEXT_SMALL_HR384_YOLO26M_PROB_ENSEMBLE

STATUS:
COMPLETED
PACKAGED
CLOSED

SCIENTIFIC ROLE
Probability-level ensemble of two completed model families.

TRAINING
None.
F08 was inference-only.
No new checkpoint was produced.

LOCKED SOURCES

F06:
F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL
Checkpoint: best_model.pth
SHA-256:
d0c225c28b25f09822bc99518a37f9b1fe642b2d01213da7f68e2d76a255832a

F01:
F01_YOLO26M_CLS_AUG200
Checkpoint: best.pt
SHA-256:
aa489def1443a6cdab42a0a20a530b9e341d574dcaaf234bfe6c7983113e8e5a

CLASS ORDER
20 AQUA20 classes.
Identical class-name mapping verified across both source models.

ENSEMBLE METHOD
Weighted arithmetic mean of class probabilities.

PREDEFINED VALIDATION-ONLY WEIGHT GRID
F06 0.50 + F01 0.50
F06 0.60 + F01 0.40
F06 0.70 + F01 0.30
F06 0.80 + F01 0.20

SELECTION METRIC
Validation Macro F1.

LOCKED TIE-BREAK
1. Higher validation Macro F1
2. Higher validation Macro Recall
3. Higher validation Top-1
4. Prefer equal weighting if still tied

SELECTED RULE
F06 ConvNeXt weight: 0.70
F01 YOLO weight: 0.30

SELECTED VALIDATION RESULTS
Top-1:
93.4451%

Macro Recall:
93.0674%

Macro F1:
90.7359%

The final ensemble weight was locked before official-test evaluation.

OFFICIAL UNCHANGED TEST
Images: 1,612
Classes: 20

F08 OFFICIAL TEST RESULTS
Top-1: 92.2457%
Top-2: 98.0769%
Top-3: 99.3797%
Top-5: 99.8759%
Macro Precision: 90.6725%
Macro Recall: 87.0170%
Macro F1: 87.9406%
Weighted F1: 92.1750%

F08 VERSUS F06
Top-1 delta: +0.1241 percentage points
Top-5 delta: +0.1241 percentage points
Macro Recall delta: -0.4192 percentage points
Macro F1 delta: -0.2293 percentage points

SCIENTIFIC FINDING
The validation-selected 0.70 F06 + 0.30 F01 probability ensemble
improved official-test Top-1, Top-2, Top-3, Top-5,
Macro Precision, and Weighted F1 relative to F06.

However, the strong validation Macro Recall and Macro F1 gains
did not reproduce on the official unchanged test split.
Official-test Macro Recall and Macro F1 declined slightly.

Therefore, F08 demonstrates useful complementary ranking behaviour
between the F06 ConvNeXt and F01 YOLO model families, but it does
not replace F06 as the stronger balanced-class branch.

F08 is retained as a positive ranking-ensemble result.

BEAT-BASE TARGET
Reference Top-1 target: 92.6800%
F08 Top-1: 92.2457%
Difference: -0.4343 percentage points

The 92.68% Top-1 target was not beaten by F08.

TEST-GOVERNANCE NOTE
The official test split was not used to select source checkpoints,
candidate ensemble weights, or the final F08 ensemble rule.

No additional weights were tested after official-test evaluation.