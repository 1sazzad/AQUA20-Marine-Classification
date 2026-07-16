AQUA20 X01 — F06 FINAL GRAD-CAM
============================================================

OFFICIAL CAMPAIGN
------------------------------------------------------------
04-07-2026 AQUA20 FINAL BEAT BASE PLAN

OFFICIAL STAGE
------------------------------------------------------------
X01_F06_FINAL_GRADCAM

SCIENTIFIC ROLE
------------------------------------------------------------
XAI / INTERPRETABILITY / VISUAL EXPLANATION

PRIMARY MODEL
------------------------------------------------------------
F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL

F06 was the frozen primary proposed method and the strongest
balanced-class model selected by the completed F09 campaign.

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

XAI SAMPLE PROTOCOL
------------------------------------------------------------
Frozen samples  : 20
Groups           : 4
Samples/group    : 5
Publication set  : 12

Group A:
Correct high-confidence predictions

Group B:
Correct difficult / lower-confidence predictions

Group C:
High-confidence misclassified predictions

Group D:
Minority / weak-performing class examples

The XAI image registry was selected before Grad-CAM generation.

No sample was selected according to heatmap appearance.

No frozen image was replaced after inspecting Grad-CAM outputs.

IMAGE REPRESENTATION
------------------------------------------------------------
Source           : taufiktrf/AQUA20 official test split
Color mode       : RGB
Export format    : JPEG
JPEG quality     : 95
Reload           : Yes
Input resolution : 384 x 384

The exact JPEG quality=95 representation reproduced all 20
official F06 predictions before Grad-CAM generation.

MODEL REPRODUCTION
------------------------------------------------------------
Frozen samples verified : 20 / 20
Prediction agreement     : 20 / 20
Maximum confidence delta : 0.00033830

GRAD-CAM PROTOCOL
------------------------------------------------------------
Target layer      : features.7.2
Target model      : Frozen F06
Target class rule : F06 predicted class

Grad-CAM targeted the F06 predicted class.

This target rule was used for both correct and incorrect
predictions so the explanations represent evidence associated
with the decision actually made by F06.

GRAD-CAM FLOW
------------------------------------------------------------
F06 forward pass
Predicted-class logit
Final locked ConvNeXt feature block
Gradient extraction
Channel-wise gradient averaging
Weighted activation aggregation
ReLU
Heatmap normalization
Resize to image dimensions
Overlay generation

OFFICIAL OUTPUTS
------------------------------------------------------------
Frozen XAI registry : 20 rows
Grad-CAM metadata    : 20 rows
Original images      : 20
Raw heatmaps         : 20
Grad-CAM overlays    : 20

Figures:
1. X01_gradcam_publication_panel.png
2. X01_gradcam_group_comparison.png

PUBLICATION FIGURE RULE
------------------------------------------------------------
The first 3 frozen samples from every group were pre-locked
for the publication subset.

4 groups x 3 samples = 12 publication examples.

The publication subset was not selected after viewing the
Grad-CAM heatmaps.

X02 RESERVATION
------------------------------------------------------------
The same frozen image registry is reserved for X02 LIME.

X02 must use:
- the same frozen F06 model
- the same F06 Epoch 2 checkpoint
- the same 20 image IDs
- the same JPEG quality=95 image representation
- the same F06 predicted-class explanation target

X01 FINAL STATUS
------------------------------------------------------------
Training       : NONE
Fine-tuning    : NONE
New checkpoint : NONE

X01 performed no training and produced no checkpoint.

STATUS
------------------------------------------------------------
X01 COMPLETED
X01 PACKAGED
X01 CLOSED

NEXT OFFICIAL STAGE
------------------------------------------------------------
X02_F06_FINAL_LIME