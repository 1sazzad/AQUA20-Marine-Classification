AQUA20 F09 FINAL METHOD FREEZE AND CAMPAIGN CONSOLIDATION
======================================================================

Official campaign:
04-07-2026 AQUA20 FINAL BEAT BASE PLAN

Experiment:
F09_FINAL_METHOD_FREEZE_AND_CAMPAIGN_COMPARISON

F09 mode:
ANALYSIS
COMPARISON
FINAL METHOD SELECTION
CAMPAIGN FREEZE

Training performed:
NONE

New checkpoint produced:
NONE


FINAL REPORTING HIERARCHY
======================================================================

Primary proposed method:
F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL

Primary scientific role:
Strongest completed balanced-class method.

F06 official test:
Top-1            : 92.1216%
Top-2            : 97.9529%
Top-3            : 99.2556%
Top-5            : 99.7519%
Macro Precision  : 90.4312%
Macro Recall     : 87.4362%
Macro F1         : 88.1699%
Weighted F1      : 92.0810%


Complementary final ensemble:
F08_CONVNEXT_SMALL_HR384_YOLO26M_PROB_ENSEMBLE

Frozen ensemble rule:
0.70 F06 + 0.30 F01

F08 official test:
Top-1            : 92.2457%
Top-2            : 98.0769%
Top-3            : 99.3797%
Top-5            : 99.8759%
Macro Precision  : 90.6725%
Macro Recall     : 87.0170%
Macro F1         : 87.9406%
Weighted F1      : 92.1750%


FINAL SCIENTIFIC DECISIONS
======================================================================

Primary proposed method:
F06

Complementary final ensemble:
F08

Strongest balanced-class model:
F06

Strongest ranking model:
F08

Best isolated imbalance intervention:
F03 — Gentle Class-Balanced Focal Loss


INTERVENTION FINDINGS
======================================================================

Gentle CB Focal:
Stronger than Aug-200 as an isolated imbalance intervention.

Aug-200:
Weaker than the Gentle CB Focal branch for balanced-class performance.

Aug-200 + Gentle CB Focal:
The two interventions were not complementary.

HR384:
Improved the F03 branch and produced the strongest balanced-class model.

Horizontal-flip TTA:
Did not improve the official-test balanced-class metrics.

Cross-family probability ensemble:
Improved Top-1, Top-2, Top-3, Top-5, Macro Precision, and Weighted F1
over F06, but reduced Macro Recall and Macro F1.


SOURCE CHECKPOINT FREEZE
======================================================================

F06 checkpoint:
best_model.pth

F06 best epoch:
2

F06 SHA-256:
d0c225c28b25f09822bc99518a37f9b1fe642b2d01213da7f68e2d76a255832a


F01 checkpoint:
best.pt

F01 best epoch:
2

F01 SHA-256:
aa489def1443a6cdab42a0a20a530b9e341d574dcaaf234bfe6c7983113e8e5a


BEAT-BASE TARGET STATUS
======================================================================

Reference Top-1 target:
92.6800%

Best final-campaign Top-1:
92.2457%

Best Top-1 experiment:
F08

Gap:
-0.4343 percentage points

Target status:
NOT BEATEN

The 92.68% Top-1 target must not be reported as beaten.


FINAL CAMPAIGN INTERPRETATION
======================================================================

The completed controlled campaign shows that Gentle Class-Balanced Focal
Loss was the strongest isolated imbalance intervention.

High-resolution 384 x 384 fine-tuning positively improved this branch and
produced F06, the strongest completed balanced-class model.

Deterministic horizontal-flip TTA did not improve the official-test
balanced metrics.

The validation-governed F06 + F01 probability ensemble improved overall
ranking and top-k performance, producing the strongest campaign ranking
result, but did not improve official-test Macro Recall or Macro F1 over F06.

Therefore F06 is frozen as the primary proposed AQUA20 method, while F08
is retained as a complementary ranking ensemble result.

F09 is closed.
