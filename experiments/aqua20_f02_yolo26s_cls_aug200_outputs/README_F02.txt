AQUA20 F02 OFFICIAL EXPERIMENT PACKAGE

Official project:
04-07-2026 AQUA20 FINAL BEAT BASE PLAN

Experiment:
F02_YOLO26S_CLS_AUG200

Model:
YOLO26s-cls

Scientific role:
G2 — Data-level balancing experiment

Method:
YOLO26s-cls + Aug-200 train-only balancing

DATASET PROTOCOL
Train : 7,284
Val   : 1,312
Test  : 1,612
Classes: 20

Validation and test were unchanged.

TRAINING CONFIGURATION
Model          : yolo26s-cls.pt
Image size     : 224
Batch size     : 64
Maximum epochs : 200
Patience       : 30
Seed           : 42
Deterministic  : True
Pretrained     : True
Optimizer      : auto
AMP            : True
Custom weights : No
Custom focal   : No

TRAINING RESULT
Epochs completed         : 68
First best-fitness epoch : 38
Locked checkpoint epoch  : 49
Equal-fitness tie        : Yes

The first best-fitness epoch and locked checkpoint epoch differ because
Ultralytics encountered an equal-fitness tie. The preserved best.pt
corresponds to the later tied checkpoint represented by epoch 49 metrics.

LOCKED VALIDATION RESULT
Val Top-1 : 86.0518%
Val Top-5 : 98.7043%

OFFICIAL TEST RESULT
Test Top-1      : 83.4367%
Test Top-5      : 97.9529%
Macro Precision : 80.9103%
Macro Recall    : 65.2565%
Macro F1        : 70.3909%
Weighted F1     : 82.8376%
Support         : 1,612

F01 COMPARISON
F01 YOLO26m-cls + Aug-200 Top-1 : 84.68%
F02 YOLO26s-cls + Aug-200 Top-1 : 83.44%

F02 Top-1 difference vs F01:
-1.24 percentage points

BASE-PAPER TARGET
Reported YOLO26m Top-1 : 92.68%
F02 Top-1              : 83.44%
Difference             : -9.24 percentage points
Beat base              : No

SCIENTIFIC INTERPRETATION
F02 confirms that Aug-200 train-only balancing alone is insufficient to
solve AQUA20 minority-class recognition imbalance.

The smaller YOLO26s-cls model performed worse than YOLO26m-cls under the
same Aug-200 protocol, particularly in macro recall and macro F1.

Important weak minority-class recall included:
marine_dolphin : 30.00%
seaCucumber    : 30.00%
octopus        : 40.00%
eel            : 48.78%
flatworm       : 23.08%

F02 is a valid negative architecture-size/data-balancing ablation result.

Do not tune F02 using the official test result.

LOCKED BEST.PT SHA-256
38a4948f97f8ed3194d4c3ce7d56132180f8a1b3b84a4c575fba45d1380ab302

PACKAGE CONTENTS
weights/
    best.pt

training_run/
    results.csv
    results.png
    args.yaml

tables/
    training history
    best epoch summary
    native test metrics
    test predictions
    classification report
    confusion matrices
    test metrics summary
    comparison-ready summary

figures/
    confusion matrix
    normalized confusion matrix

Excluded:
dataset
last.pt
