AQUA20 FINAL BEAT BASE CAMPAIGN
F05 OFFICIAL EXPERIMENT PACKAGE
================================================================================

EXPERIMENT
----------
ID:
F05_CONVNEXT_SMALL_AUG200_GENTLE_CB_FOCAL

Scientific role:
Combined imbalance intervention

METHOD
------
ConvNeXt-Small
+
Aug-200 train-only balancing
+
Gentle Class-Balanced Focal Loss

LOSS
----
Beta  : 0.999
Gamma : 1.0

Class weights:
Effective-number weighting calculated from the Aug-200 training distribution
and normalized to mean 1.

TRAINING
--------
Training images    : 7,284
Validation images  : 1,312
Official test      : 1,612
Classes            : 20

Image size         : 224 x 224
Batch size         : 16
Optimizer          : AdamW
Learning rate      : 1e-5
Weight decay       : 1e-4
Scheduler          : CosineAnnealingLR
Maximum epochs     : 25
Seed               : 42

Checkpoint rule:
Validation Macro F1

LOCKED BEST CHECKPOINT
----------------------
Best epoch          : 20
Validation Macro F1 : 89.9326%

SHA-256:
548e7c7d925f0272d06a5e57b682364b2affe302ff5c7c5071461ebde0b2e3f3

OFFICIAL TEST RESULTS
---------------------
Top-1               : 91.1290%
Top-2               : 97.4566%
Top-3               : 98.9454%
Top-5               : 99.7519%
Macro Precision     : 89.7496%
Macro Recall        : 82.6547%
Macro F1            : 84.3487%
Weighted F1         : 90.9759%

SCIENTIFIC COMPARISON
---------------------
F03:
Original train + Gentle CB Focal
Top-1       : 91.6253%
Macro Recall: 86.9302%
Macro F1    : 87.9645%

F04:
Aug-200 + Cross-Entropy
Top-1       : 91.3772%
Macro Recall: 84.3236%
Macro F1    : 86.0035%

F05:
Aug-200 + Gentle CB Focal
Top-1       : 91.1290%
Macro Recall: 82.6547%
Macro F1    : 84.3487%

OFFICIAL SCIENTIFIC FINDING
---------------------------
The data-level and loss-level imbalance interventions were not complementary
under the tested F05 configuration.

Gentle Class-Balanced Focal Loss was stronger when applied to the original
training distribution in F03.

Combining Aug-200 with Gentle CB Focal reduced official-test Macro Recall and
Macro F1 relative to both F03 and F04.

F05 therefore does not replace F03 as the strongest isolated imbalance
intervention result.

The higher F05 validation Macro F1 did not translate into stronger official
test balanced metrics.

PACKAGE CONTENT
---------------
README_F05.txt

figures/
    F05_confusion_matrix.png
    F05_confusion_matrix_normalized.png

tables/
    F05_best_epoch_summary.csv
    F05_training_history.csv
    F05_test_metrics_summary.csv
    F05_target_comparison.csv
    F05_scientific_comparison.csv
    F05_classification_report.csv
    F05_test_predictions.csv
    F05_confusion_matrix.csv
    F05_confusion_matrix_normalized.csv

weights/
    best_model.pth