AQUA20 F03 OFFICIAL EXPERIMENT PACKAGE
======================================

Experiment
----------
F03_CONVNEXT_SMALL_GENTLE_CB_FOCAL

Scientific group
----------------
G3 — Loss-level imbalance intervention

Method
------
ConvNeXt-Small
ImageNet pretrained weights
Original AQUA20 train split
Gentle Class-Balanced Focal Loss

Training data
-------------
Train : 5,247
Val   : 1,312
Test  : 1,612
Classes: 20

Loss configuration
------------------
Beta  : 0.999
Gamma : 1.0
Class weighting: Effective-number class-balanced weighting
Weight normalization: Mean = 1

Checkpoint selection
--------------------
Validation Macro F1 only

Locked best checkpoint
----------------------
Epoch : 13
SHA-256:
928c8af60aef70537263e2c8cc1cf13782747bc5974f86ff054fbe352065c26f

Locked validation result
------------------------
Top-1          : 92.5305%
Top-2          : 97.6372%
Top-3          : 99.4665%
Macro Precision: 87.9929%
Macro Recall   : 92.6189%
Macro F1       : 89.3301%

Official unchanged test result
------------------------------
Top-1             : 91.6253%
Top-2             : 97.7047%
Top-3             : 99.1315%
Top-5             : 99.6898%

Macro Precision   : 90.7850%
Macro Recall      : 86.9302%
Macro F1          : 87.9645%

Weighted Precision: 91.8777%
Weighted Recall   : 91.6253%
Weighted F1       : 91.5656%

Support            : 1,612

Base-paper numeric comparison
-----------------------------
Top-1 : exceeded
Top-2 : exceeded
Top-3 : exceeded

Precision numeric target : not exceeded
Recall numeric target    : not exceeded
F1 numeric target        : not exceeded

Campaign Top-1 target
---------------------
Target : 92.68%
F03    : 91.6253%
Status : not exceeded

Important minority-class observation
------------------------------------
Flatworm test recall:

F01 Aug-200 YOLO26m : 15.38%
F02 Aug-200 YOLO26s : 23.08%
F03 Gentle CB Focal : 76.92%

Interpretation
--------------
Gentle Class-Balanced Focal Loss substantially reduced the severe
minority-class under-recognition observed in the data-level Aug-200
experiments.

F03 exceeded the base-paper ConvNeXt numeric Top-1, Top-2 and Top-3
results, but did not exceed all precision, recall and F1 numeric targets.

F03 is therefore retained as the official G3 loss-level intervention,
not as the final campaign-winning model.

Official test protocol
----------------------
The unchanged 1,612-image AQUA20 test split was evaluated only after
the best validation Macro-F1 checkpoint was locked and preserved.