AQUA20 F00 DATA AUDIT + AUG-200 HANDOVER
================================================

Official plan:
04-07-2026 AQUA20 FINAL BEAT BASE PLAN

F00 STATUS:
COMPLETED

SOURCE:
Hugging Face dataset: taufiktrf/AQUA20

ORIGINAL HF SPLITS:
train = 6559
test  = 1612

OFFICIAL RECONSTRUCTED SPLIT:
train = 5247
val   = 1312
test  = 1612

Split method:
Stratified train/validation reconstruction from the Hugging Face training
split using random seed 42.

The Hugging Face test split was preserved unchanged.

AUG-200 TRAIN-ONLY SPLIT:
train = 7284
val   = 1312
test  = 1612

Original train images:
5247

Synthetic augmented train images added:
2037

Aug-200 rule:
Classes with fewer than 200 training images were augmented until reaching
200 images. Classes already above 200 were retained without downsampling.

Validation and test were not augmented.

FINAL F00 CHECKS:
- Official split counts passed
- Aug-200 split counts passed
- 20 classes confirmed
- Class order consistency passed
- Validation unchanged passed
- Test unchanged passed
- PyTorch ImageFolder loading passed
- One-batch tensor sanity test passed

IMPORTANT:
This package intentionally excludes:
- official_imagefolder/
- balanced_train/aug200_imagefolder/

These image datasets are large and should be recreated from the source or
stored separately as a Kaggle dataset when needed.

NEXT EXPERIMENT:
F01 YOLO26m-cls + Aug-200
