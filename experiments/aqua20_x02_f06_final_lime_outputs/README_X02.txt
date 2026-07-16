AQUA20 X02 — F06 FINAL LIME
============================================================

OFFICIAL CAMPAIGN
------------------------------------------------------------
04-07-2026 AQUA20 FINAL BEAT BASE PLAN

OFFICIAL STAGE
------------------------------------------------------------
X02_F06_FINAL_LIME

SCIENTIFIC ROLE
------------------------------------------------------------
XAI / INTERPRETABILITY / MODEL-AGNOSTIC LOCAL EXPLANATION

PRIMARY MODEL
------------------------------------------------------------
F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL

F06 was the frozen primary proposed method.

MODEL METHOD
------------------------------------------------------------
Torchvision ConvNeXt-Small
Original AQUA20 training distribution
Gentle Class-Balanced Focal Loss
384 x 384 high-resolution fine-tuning

FROZEN CHECKPOINT
------------------------------------------------------------
Checkpoint      : best_model.pth
Best epoch      : 2
SHA-256         : d0c225c28b25f09822bc99518a37f9b1fe642b2d01213da7f68e2d76a255832a

The F06 checkpoint was unchanged.

X01 REGISTRY REUSE
------------------------------------------------------------
Frozen samples  : 20
Groups           : 4
Samples/group    : 5
Publication set  : 12

X02 reused the X01 frozen 20-image XAI registry.

No new XAI sample selection was performed.

The copied X01 registry was verified as content-identical to the
original closed X01 registry.

GROUP DEFINITIONS
------------------------------------------------------------
Group A:
Correct high-confidence predictions

Group B:
Correct difficult / lower-confidence predictions

Group C:
High-confidence misclassified predictions

Group D:
Minority / weak-performing class examples

PUBLICATION SUBSET
------------------------------------------------------------
The publication subset was unchanged.

The first 3 frozen samples from every group remained the
publication subset.

4 groups x 3 samples = 12 publication examples.

IMAGE REPRESENTATION
------------------------------------------------------------
Source           : taufiktrf/AQUA20 official test split
Color mode       : RGB
Export format    : JPEG
JPEG quality     : 95
Reload           : Yes
Input resolution : 384 x 384

The exact JPEG quality=95 image representation was reused.

MODEL REPRODUCTION
------------------------------------------------------------
Frozen samples verified : 20 / 20
Prediction agreement     : 20 / 20
Maximum confidence delta : 0.00033830

The F06 predictions were reproduced before LIME generation.

LIME PROTOCOL LOCK
------------------------------------------------------------
Random seed              : 42
Perturbation samples     : 1000
Segmentation method      : SLIC
Requested SLIC segments  : 100
SLIC compactness         : 10
SLIC sigma               : 1
SLIC start label         : 0
Positive features        : 10
Positive-only            : True
Hide-rest explanation    : True
Distance metric          : cosine
Kernel width             : 0.25
Feature selection        : auto

The LIME protocol was frozen before final explanation generation.

No LIME parameters were changed according to explanation appearance.

EXPLANATION TARGET
------------------------------------------------------------
Target model      : Frozen F06
Target class rule : F06 predicted class

LIME targeted the F06 predicted class.

The same target rule was used for correct and incorrect predictions.

For misclassified samples, LIME explained the predicted class rather
than switching to the true class.

This preserves direct scientific comparability with X01 Grad-CAM.

OFFICIAL OUTPUTS
------------------------------------------------------------
Frozen X01 registry       : 20 rows
LIME protocol lock        : 1
LIME metadata             : 20 rows
Original images           : 20
Positive explanations     : 20
Binary positive masks     : 20
LIME boundary overlays    : 20

Figures:
1. X02_lime_publication_panel.png
2. X02_lime_group_comparison.png

BINARY MASK RESERVATION
------------------------------------------------------------
The 20 LIME binary positive-region masks are retained for X03.

X03 will compare:
- the same image
- the same F06 prediction
- the same explanation target class
- Grad-CAM
- LIME

Any quantitative spatial-comparison threshold and mask protocol must
be frozen in X03 before reading final similarity results.

X03 RESERVATION
------------------------------------------------------------
The X02 outputs are reserved for direct X03 comparison with X01
Grad-CAM.

X03 must use:
- X01 Grad-CAM package
- X02 LIME package
- the same frozen 20-image registry
- the same sample order
- the same target classes

X02 FINAL STATUS
------------------------------------------------------------
Training       : NONE
Fine-tuning    : NONE
New checkpoint : NONE

X02 performed no training and produced no checkpoint.

STATUS
------------------------------------------------------------
X02 COMPLETED
X02 PACKAGED
X02 CLOSED

NEXT OFFICIAL STAGE
------------------------------------------------------------
X03_GRADCAM_VS_LIME_COMPARISON