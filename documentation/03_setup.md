# D03 — AQUA20 Experimental Environment, Reproducibility Controls, and Common Training/Evaluation Protocol

## Document status

**Phase:** 3 — Experimental Environment, Reproducibility Controls, and Common Training/Evaluation Protocol  
**Status:** Completed and locked  
**Canonical manuscript:** *Improving Underwater Marine Species Classification through Imbalance-Aware Learning and Cross-Dataset Evaluation*

---

## 1. Source-control decision for this phase

This phase follows the locked evidence hierarchy:

1. latest final manuscript;
2. FINAL evidence-freeze and frozen-terminology records;
3. preserved experiment packages, configuration files, READMEs, histories, and the campaign notebook;
4. older drafts and supervisor reports for historical context only.

Run-specific configuration evidence takes priority over generic setup cells. A generic notebook cell must not be used to overwrite a later locked run configuration.

The experiment identity correction remains mandatory:

> **F02 is `F02_YOLO26S_CLS_AUG200`. It is not a ConvNeXt-Small cross-entropy baseline.**

F04 is the preserved ConvNeXt-Small + Aug-200 + standard cross-entropy configuration. No architecture-matched ConvNeXt-Small + standard cross-entropy experiment on the original 5,247-image training distribution was found in the final F01–F09 campaign.

---

## 2. Evidence-status vocabulary

Environment and reproducibility records are classified as follows.

| Status | Meaning |
|---|---|
| **Direct runtime output** | Printed by an executed notebook cell in the preserved campaign notebook. |
| **Run-specific locked configuration** | Stored in a run configuration, checkpoint metadata, README, args file, or verified history. |
| **Campaign-level compatible environment** | Verified in the preserved campaign notebook, but not independently printed inside every run package. |
| **Restored compatibility environment** | Explicitly reinstalled or restored to load a frozen source model; this does not automatically prove the original training run used the identical package state. |
| **To be verified** | Not supported by the reviewed evidence or not sufficiently run specific. |

This distinction prevents a later environment-restoration cell from being misrepresented as an original training-time manifest for every experiment.

---

## 3. Preserved campaign runtime environment

### 3.1 Directly observed core environment

The preserved notebook contains a successful P100 CUDA test and direct runtime outputs with the following values.

| Component | Verified value | Scope and interpretation |
|---|---|---|
| Execution platform | Kaggle-hosted notebook workflow | Verified from the preserved notebook context and `/kaggle/working/...` paths. |
| Operating-system string | `Linux-6.12.90+-x86_64-with-glibc2.35` | Direct runtime output from the repaired campaign environment. |
| Python | `3.12.13` | Direct runtime output in the campaign notebook, including the F04 environment check. |
| PyTorch | `2.7.1+cu118` | Direct runtime output. |
| torchvision | `0.22.1+cu118` | Direct runtime output. |
| CUDA exposed by PyTorch | `11.8` | Direct `torch.version.cuda` output. |
| CUDA availability | `True` | Direct runtime output. |
| GPU count | `1` | Direct runtime output. |
| GPU | `Tesla P100-PCIE-16GB` | Direct runtime output and run configuration evidence. |
| GPU memory | `15.89 GB` | Direct device-property output. |
| Compute capability | `6.0` | Direct device-property output. |
| CUDA tensor test | Passed | A CUDA matrix multiplication completed successfully. |
| Ultralytics | `8.4.89` | Explicitly installed, imported, printed, and asserted in the preserved notebook as the locked/restored F01-compatible version. |

### 3.2 cuDNN record boundary

The environment installation output records:

```text
nvidia-cudnn-cu11==9.1.0.70
```

This is the installed Python wheel record. A direct archived output of:

```python
torch.backends.cudnn.version()
```

was not found. Therefore:

- **installed cuDNN wheel:** `9.1.0.70` — verified for the repaired environment;
- **native runtime cuDNN version reported by PyTorch:** **To be verified**.

The two values must not be treated as interchangeable without the missing runtime query.

### 3.3 Other package-version boundaries

| Component | Status |
|---|---|
| scikit-learn exact version | **To be verified** |
| pandas exact run-specific version | **To be verified** |
| NumPy original training-run version | **To be verified**; `2.4.4` appears in a later repaired-environment install output, not as a complete original run manifest. |
| Pillow original training-run version | **To be verified**; `12.2.0` appears in the later repaired environment. |
| NVIDIA driver version | **To be verified** |
| Complete `nvidia-smi` record | **Not found** |
| Complete `pip freeze` for each run | **Not found** |
| Kaggle image/container identifier | **To be verified** |
| CPU model and host RAM | **To be verified** |

The campaign can therefore be reproduced at the level of the verified core GPU/PyTorch stack and stored run configurations, but it does not possess a complete immutable software-container manifest for every run.

---

## 4. Shared dataset-loader and class-order controls

### 4.1 ImageFolder structures

The campaign used two frozen ImageFolder-compatible dataset roots:

```text
official_imagefolder/
├── train/   5,247
├── val/     1,312
└── test/    1,612

balanced_train/aug200_imagefolder/
├── train/   7,284
├── val/     1,312
└── test/    1,612
```

F01, F02, F04, and F05 used the Aug-200 training structure. F03 and F06 used the original 5,247-image training distribution. F07 and F08 performed no new training.

### 4.2 Required loader assertions

The preserved notebook and packages support the following common checks:

- expected train, validation, and test counts;
- exactly 20 class folders;
- identical class order across train, validation, and test;
- unchanged validation and test relative file lists between the official and Aug-200 structures;
- successful ImageFolder construction;
- one-batch tensor and label sanity checks;
- 20-output classifier verification for ConvNeXt;
- identical F06 and F01 class-name mapping before F08 probability fusion.

The class order is frozen as:

```text
00_coral
01_crab
02_diver
03_eel
04_fish
05_fishInGroups
06_flatworm
07_jellyfish
08_marine_dolphin
09_octopus
10_rayfish
11_seaAnemone
12_seaCucumber
13_seaSlug
14_seaUrchin
15_shark
16_shrimp
17_squid
18_starfish
19_turtle
```

### 4.3 Loader reproducibility boundary

Class-order and count checks prove structural consistency. They do not replace:

- an immutable dataset revision;
- a complete content checksum;
- a full duplicate/near-duplicate audit; or
- a complete all-file corrupt-image decode report.

Those Phase 2 boundaries remain open.

---

## 5. Seed handling and deterministic controls

### 5.1 Campaign seed

The campaign seed was:

```text
SEED = 42
```

Seed 42 is verified in the dataset split, Aug-200 generation, YOLO arguments, ConvNeXt configurations, F07/F08 protocols, external evaluation, XAI, and later statistical analysis.

### 5.2 F01 pre-import CUDA determinism setup

The preserved F01 environment-verification cell checked that PyTorch had not already been imported and then set:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONHASHSEED=42
```

before importing PyTorch and Ultralytics. The cell required a clean kernel if PyTorch had already been loaded.

The saved F01/F02 Ultralytics arguments also record:

```text
seed=42
deterministic=True
```

The exact internal library calls triggered by the version-specific `deterministic=True` implementation were not separately exported as a run-local trace.

### 5.3 ConvNeXt backend controls

The locked F03 configuration records:

```text
cudnn_deterministic=True
cudnn_benchmark=False
```

F03 also records seed 42, AMP enabled, four workers, and a maximum gradient norm of 1.0.

A separate earlier generic setup cell used faster, non-deterministic-style backend flags. That generic cell is historical context and does not override the later run-specific F03 configuration.

For F04, F05, and F06, seed 42 is verified. Complete run-local backend-flag traces are not uniformly archived. They must therefore not all be described as bitwise deterministic.

### 5.4 Determinism claim boundary

The defensible statement is:

> The campaign used fixed data partitions, seed 42, preserved checkpoint-selection rules, and several explicit deterministic controls. However, complete bitwise reproducibility cannot be guaranteed for every run because full run-local package manifests, all backend flags, worker-seeding traces, and immutable container records were not archived uniformly.

Random online augmentation remained active during training. A fixed seed governs the pseudorandom process, but the project did not preserve every sampled crop, rotation, RandAugment operation, or worker-level random state.

### 5.5 Single-run limitation

F01–F06 are preserved completed runs, not repeated multi-seed training studies. Test-sample bootstrap analyses quantify uncertainty conditional on the frozen predictions; they do not estimate optimisation variability across independent seeds.

---

## 6. Offline Aug-200 versus online training augmentation

The two augmentation mechanisms are separate and must remain separately documented.

### Offline Aug-200

- executed before model training;
- created 2,037 additional JPEG files;
- affected only the training partition;
- raised every class below 200 images to 200;
- used the Phase 2 locked crop, flip, rotation, colour, sharpness, and JPEG-95 generator.

### Online augmentation

- applied dynamically when a training sample was loaded;
- differed by model family and experiment;
- did not create new permanent dataset files;
- was disabled for deterministic validation and test preprocessing, except for the explicitly defined F07 second inference view.

Using Aug-200 does not imply that a run used the same online transformations as another Aug-200 run.

---

## 7. Experiment-by-experiment environment and control registry

| ID | Training / role | Batch | Workers | Precision | Seed and determinism evidence | Hardware/environment evidence | Selection rule | Main unresolved fields |
|---|---|---:|---:|---|---|---|---|---|
| **F01** | YOLO26m-cls, Aug-200, 224×224 | 64 | 8 | AMP enabled | Seed 42; `deterministic=True`; F01 pre-import `CUBLAS_WORKSPACE_CONFIG` and `PYTHONHASHSEED` cell | Single P100/device 0 in notebook; campaign PyTorch/CUDA stack verified; Ultralytics 8.4.89 restored/locked | Native Ultralytics classification fitness | Original run-local full package manifest; resolved optimiser under `optimizer=auto`; exact version-specific native loss implementation |
| **F02** | YOLO26s-cls, Aug-200, 224×224 | 64 | 8 | AMP enabled | Seed 42; `deterministic=True` in saved arguments | Same preserved campaign branch; no independent full F02 environment manifest | Native Ultralytics classification fitness | Original run-local software manifest; resolved optimiser; exact native loss implementation; direct stage-local hardware printout |
| **F03** | ConvNeXt-Small, original train, Gentle CB Focal, 224×224 | 16 | 4; prefetch 2 | AMP enabled | Seed 42; cuDNN deterministic true; benchmark false | P100 recorded directly in locked config; campaign core stack compatible | Validation Macro F1 | Complete run-local `pip freeze`; scheduler `eta_min`; default AdamW betas/epsilon if not explicit |
| **F04** | ConvNeXt-Small, Aug-200, cross-entropy, 224×224 | 16 | 4 | AMP enabled | Seed 42; full run-local backend flags not frozen | Direct F04 output: Python 3.12.13, torch 2.7.1+cu118, torchvision 0.22.1+cu118, P100 | Validation Macro F1 | Run-local backend flags; gradient-clipping status; complete package manifest/original ZIP hash |
| **F05** | ConvNeXt-Small, Aug-200, Gentle CB Focal, 224×224 | 16 | **To be verified** | **To be verified** | Seed 42; complete backend/worker controls not archived | Executed in the preserved campaign notebook context; no standalone full environment manifest | Validation Macro F1 | Workers, AMP, patience, scheduler parameters, gradient clipping, pretrained-weight identifier, full package versions |
| **F06** | F03 checkpoint continuation, original train, Gentle CB Focal, 384×384 | 8 | **To be verified** | AMP supported by campaign registry | Seed 42; full stage-local backend trace not uniformly archived | P100 campaign runtime; F03 source and F06 checkpoint hashes verified | Validation Macro F1 | Worker count, scheduler parameters, runtime cuDNN, complete manifest, controlled efficiency measurements |
| **F07** | Frozen F06; deterministic original + horizontal-flip two-view inference | No training batch | Inference batch **To be verified** | Inference precision **To be verified** | Deterministic view rule; no new training seed dependency | Uses frozen F06 in preserved campaign environment | No checkpoint; fixed inference protocol | Inference batch, precision/autocast state, controlled latency/throughput |
| **F08** | Frozen F06 + frozen F01 probability ensemble | No training batch | Source-inference batch **To be verified** | Source-inference precision **To be verified** | Predefined validation grid and deterministic arithmetic rule | Ultralytics 8.4.89 restored for F01 compatibility; F06/F01 source hashes verified | Validation Macro F1 with frozen tie-breaks | Raw source-probability environment manifests; inference batch/precision; controlled ensemble cost |

---

## 8. Run-specific training and stopping controls

### 8.1 F01 — `F01_YOLO26M_CLS_AUG200`

- model: `yolo26m-cls.pt`;
- training data: 7,284 Aug-200 images;
- validation/test: unchanged 1,312 / 1,612;
- image size: 224;
- batch: 64;
- workers: 8;
- maximum/completed epochs: 200 / 32;
- patience: 30;
- optimiser request: `optimizer=auto`;
- initial learning-rate argument: `lr0=0.01`;
- final LR fraction: `lrf=0.01`;
- warm-up: 3 epochs;
- momentum argument: 0.937;
- weight decay: 0.0005;
- seed: 42;
- deterministic argument: true;
- AMP: enabled;
- best epoch: 2;
- checkpoint: `best.pt`;
- SHA-256: `aa489def1443a6cdab42a0a20a530b9e341d574dcaaf234bfe6c7983113e8e5a`.

The requested optimiser was automatic. The final evidence does not freeze the resolved runtime optimiser and parameter groups. It must remain **To be verified** rather than being relabelled as SGD, MuSGD, AdamW, or another optimiser from memory.

### 8.2 F02 — `F02_YOLO26S_CLS_AUG200`

- model: `yolo26s-cls.pt`;
- training data: 7,284 Aug-200 images;
- validation/test: unchanged 1,312 / 1,612;
- image size: 224;
- batch: 64;
- workers: 8;
- maximum/completed epochs: 200 / 68;
- patience: 30;
- optimiser request: `optimizer=auto`;
- seed: 42;
- deterministic argument: true;
- AMP: enabled;
- tied maximum native-fitness epochs: 38 and 49;
- preserved locked checkpoint epoch: 49;
- checkpoint SHA-256: `38a4948f97f8ed3194d4c3ce7d56132180f8a1b3b84a4c575fba45d1380ab302`.

F02 is a smaller-YOLO model-scale control under the same Aug-200 branch. It is not the ConvNeXt cross-entropy experiment.

### 8.3 F03 — `F03_CONVNEXT_SMALL_GENTLE_CB_FOCAL`

- model: `torchvision.models.convnext_small`;
- pretrained weights: `ConvNeXt_Small_Weights.IMAGENET1K_V1`;
- training data: original 5,247 images;
- image size: 224;
- batch: 16;
- workers: 4;
- prefetch factor: 2;
- maximum/completed epochs: 40 / 21;
- patience: 8;
- optimiser: AdamW;
- learning rate: `1e-5`;
- weight decay: `1e-4`;
- scheduler: CosineAnnealingLR;
- scheduler `T_max`: 40;
- scheduler `eta_min`: **To be verified**;
- loss: Gentle Class-Balanced Focal Loss;
- beta: 0.999;
- gamma: 1.0;
- effective-number weights normalised to mean 1;
- gradient clipping: maximum norm 1.0;
- AMP: enabled;
- recovery checkpoint: every 5 epochs;
- seed: 42;
- cuDNN deterministic: true;
- cuDNN benchmark: false;
- best epoch: 13;
- SHA-256: `928c8af60aef70537263e2c8cc1cf13782747bc5974f86ff054fbe352065c26f`.

### 8.4 F04 — `F04_CONVNEXT_SMALL_AUG200`

- model: ConvNeXt-Small;
- pretrained weights: `ConvNeXt_Small_Weights.DEFAULT`;
- training data: 7,284 Aug-200 images;
- image size: 224;
- batch: 16;
- workers: 4;
- maximum/completed epochs: 40 / 28;
- patience: 8;
- optimiser: AdamW;
- learning rate: `1e-5`;
- weight decay: `1e-4`;
- scheduler: CosineAnnealingLR;
- `T_max=40`;
- `eta_min=5e-7`;
- loss: standard CrossEntropyLoss;
- class weights: none;
- focal term: none;
- AMP: enabled;
- seed: 42;
- best epoch: 20;
- SHA-256: `4e2c4abddb528e52d9721c6e1f85ea389788495f315d2891989e4ff2e403a1a7`.

### 8.5 F05 — `F05_CONVNEXT_SMALL_AUG200_GENTLE_CB_FOCAL`

- model: ConvNeXt-Small;
- exact pretrained-weight identifier: **To be verified**;
- training data: 7,284 Aug-200 images;
- image size: 224;
- batch: 16;
- maximum/completed epochs: 25 / 25;
- optimiser: AdamW;
- learning rate: `1e-5`;
- weight decay: `1e-4`;
- scheduler: CosineAnnealingLR;
- scheduler parameters: **To be verified**;
- patience: **To be verified**;
- loss: Gentle Class-Balanced Focal Loss;
- beta: 0.999;
- gamma: 1.0;
- effective-number weights computed from the Aug-200 distribution and normalised to mean 1;
- seed: 42;
- best epoch: 20;
- SHA-256: `548e7c7d925f0272d06a5e57b682364b2affe302ff5c7c5071461ebde0b2e3f3`.

F04 and F05 are the closest loss-focused comparison in the Aug-200 ConvNeXt branch, but they used different maximum epoch budgets, 40 and 25. Their difference must not be attributed exclusively to the loss function.

### 8.6 F06 — `F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL`

- source: frozen F03 validation-selected checkpoint;
- source F03 SHA-256: `928c8af60aef70537263e2c8cc1cf13782747bc5974f86ff054fbe352065c26f`;
- training data: original 5,247 images;
- Aug-200: disabled;
- image size: 384;
- batch: 8;
- maximum epochs: 15;
- completed epochs retained: 9;
- interrupted epoch 10: excluded;
- optimiser: AdamW;
- learning rate: `5e-6`;
- weight decay: `1e-4`;
- scheduler: CosineAnnealingLR;
- exact scheduler parameters: **To be verified**;
- loss: Gentle Class-Balanced Focal Loss;
- beta: 0.999;
- gamma: 1.0;
- seed: 42;
- stopping: manual termination after the retained history plateau; no test-based stopping;
- best epoch: 2;
- SHA-256: `d0c225c28b25f09822bc99518a37f9b1fe642b2d01213da7f68e2d76a255832a`.

F06 changes resolution, continuation training, learning rate, batch size, and training budget relative to F03. It is a continuation-stage complete-configuration comparison, not an isolated resolution ablation.

### 8.7 F07 — `F07_CONVNEXT_SMALL_HR384_TTA`

F07 performed no training and produced no checkpoint.

For each frozen F06 input, it computed:

\[
p_{\mathrm{F07}}
=
\frac{p(x)+p(\operatorname{flip}_{h}(x))}{2}.
\]

The views were:

1. original image;
2. horizontal reflection.

Softmax probabilities were calculated for each view and averaged arithmetically. The TTA protocol was fixed before official-test evaluation. F07 used the unchanged F06 checkpoint and class space.

### 8.8 F08 — `F08_CONVNEXT_SMALL_HR384_YOLO26M_PROB_ENSEMBLE`

F08 performed no training and produced no checkpoint.

Frozen sources:

- F06 SHA-256: `d0c225c28b25f09822bc99518a37f9b1fe642b2d01213da7f68e2d76a255832a`;
- F01 SHA-256: `aa489def1443a6cdab42a0a20a530b9e341d574dcaaf234bfe6c7983113e8e5a`.

Predefined validation-only grid:

```text
0.50 F06 + 0.50 F01
0.60 F06 + 0.40 F01
0.70 F06 + 0.30 F01
0.80 F06 + 0.20 F01
```

Selection order:

1. higher validation Macro F1;
2. higher validation Macro Recall;
3. higher validation Top-1;
4. prefer equal weighting if still tied.

The selected rule was:

\[
p_{\mathrm{F08}}
=
0.70p_{\mathrm{F06}}
+
0.30p_{\mathrm{F01}}.
\]

The source class-name mapping was verified before fusion, and the rule was locked before official-test evaluation.

---

## 9. Effective online preprocessing and augmentation

### 9.1 F01 and F02 — YOLO classification pipeline

F01 and F02 used the same effective classification pipeline; only YOLO model scale differed.

#### Training

1. `RandomResizedCrop(224)`
   - crop-area scale: 0.50–1.00;
   - aspect ratio: 0.75–1.3333;
   - bilinear interpolation;
   - antialiasing enabled.
2. `RandomHorizontalFlip(p=0.5)`.
3. No vertical flip.
4. RandAugment:
   - two sequentially sampled operations;
   - magnitude 9;
   - 31 magnitude bins;
   - operation space: Identity, ShearX/Y, TranslateX/Y, Rotate, Brightness, Color, Contrast, Sharpness, Posterize, Solarize, AutoContrast, Equalize.
5. Tensor conversion to `[0,1]`.
6. Identity normalisation:
   - mean `[0,0,0]`;
   - standard deviation `[1,1,1]`.
7. Random erasing:
   - probability 0.40;
   - erased area 2%–33%;
   - aspect ratio 0.3–3.3;
   - fill 0;
   - in-place.

The saved `augment: false` field concerns inference augmentation and does not disable Ultralytics classification training transforms.

Detection-oriented fields such as mosaic, copy-paste, perspective, and MixUp were not part of the effective classification dataset transform.

#### Validation and test

```text
Resize shorter edge to 224 while preserving aspect ratio
→ CenterCrop(224×224)
→ ToTensor
→ identity normalisation
```

No random validation/test augmentation was used.

### 9.2 F03 — ConvNeXt original + Gentle CB Focal

#### Training

```text
Resize(256×256)
→ RandomResizedCrop(224, scale=0.80–1.00, ratio=0.75–1.3333)
→ RandomHorizontalFlip(p=0.5)
→ RandomRotation(−10° to +10°)
→ ColorJitter
→ ToTensor
→ ImageNet normalisation
```

ColorJitter ranges:

- brightness factor 0.88–1.12;
- contrast factor 0.88–1.12;
- saturation factor 0.88–1.12;
- hue −0.03 to +0.03.

Resize/crop used bilinear interpolation with antialiasing. Rotation used nearest-neighbour interpolation, `expand=False`, fill 0.

#### Validation and test

```text
Resize(224×224)
→ ToTensor
→ ImageNet normalisation
```

This was a direct tuple resize, not a shorter-edge resize plus CenterCrop.

### 9.3 F04 — ConvNeXt Aug-200 + cross-entropy

#### Training

```text
Resize(224×224)
→ RandomHorizontalFlip(p=0.5)
→ RandomRotation(−10° to +10°)
→ ToTensor
→ ImageNet normalisation
```

No RandomResizedCrop, ColorJitter, affine/shear, vertical flip, MixUp, or CutMix was used.

#### Validation and test

```text
Resize(224×224)
→ ToTensor
→ ImageNet normalisation
```

### 9.4 F05 — ConvNeXt Aug-200 + Gentle CB Focal

F05 used the same effective geometry as F04, implemented with `torchvision.transforms.v2`.

#### Training

```text
Resize(224×224)
→ RandomHorizontalFlip(p=0.5)
→ RandomRotation(−10° to +10°)
→ ToImage
→ ToDtype(float32, scale=True)
→ ImageNet normalisation
```

#### Validation and test

```text
Resize(224×224)
→ ToImage
→ ToDtype(float32, scale=True)
→ ImageNet normalisation
```

### 9.5 F06 — high-resolution continuation

#### Training

```text
Resize(440×440)
→ RandomResizedCrop(384, scale=0.80–1.00, ratio=0.75–1.3333)
→ RandomHorizontalFlip(p=0.5)
→ RandomRotation(−10° to +10°)
→ ColorJitter
→ ToTensor
→ ImageNet normalisation
```

ColorJitter ranges match F03. Resize/crop used bilinear interpolation with antialiasing; rotation used nearest-neighbour interpolation with `expand=False` and fill 0.

#### Validation and test

```text
Resize(384×384)
→ ToTensor
→ ImageNet normalisation
```

No CenterCrop or random evaluation augmentation was used.

### 9.6 ImageNet normalisation

F03–F06 used:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### 9.7 F07 and F08 preprocessing inheritance

- F07 inherited the deterministic F06 384×384 evaluation preprocessing for each of its two views.
- F08 retained each source model's own evaluation pipeline: F06 at 384×384 with ImageNet normalisation and F01 at 224×224 with the Ultralytics classification evaluation transform.

The source probabilities were aligned by the frozen 20-class order before averaging.

---

## 10. Common checkpoint-selection and test-isolation protocol

### 10.1 F01–F02

F01 and F02 used native Ultralytics classification fitness derived from validation Top-1 and Top-5 performance. They were not selected by validation Macro F1.

### 10.2 F03–F06

F03, F04, F05, and F06 selected checkpoints using validation Macro F1 only.

### 10.3 F07

F07 produced no checkpoint. It used a pre-specified deterministic two-view inference rule on the frozen F06 checkpoint.

### 10.4 F08

F08 produced no checkpoint. The ensemble weight was selected on validation Macro F1 using the frozen tie-break sequence and locked before test evaluation.

### 10.5 Official-test exclusion

The unchanged 1,612-image official test split was excluded from:

- model training;
- early stopping;
- checkpoint selection;
- F07 protocol definition; and
- F08 weight selection.

The test split was evaluated only after the relevant checkpoint or inference rule had been fixed. Preserved prediction files, classification reports, confusion matrices, and checkpoint hashes support the frozen post-selection evaluation.

---

## 11. Reproducibility boundaries and unresolved fields

### 11.1 Environment records still To be verified

1. complete run-local `pip freeze` or Conda/package manifest for F01–F08;
2. immutable Kaggle image/container identifier;
3. NVIDIA driver and complete `nvidia-smi` output;
4. direct PyTorch runtime cuDNN version;
5. exact scikit-learn version used for the official metric calculations;
6. CPU model, host RAM, and storage configuration;
7. original training-run-local Ultralytics version proof for F01/F02 separate from the later 8.4.89 restoration record;
8. complete worker-seeding and generator-state traces;
9. per-epoch or per-batch random augmentation draws;
10. deterministic-algorithm warnings or unsupported-kernel logs, if any.

### 11.2 Run-specific fields still To be verified

- F01/F02 resolved optimiser and exact version-specific native classification-loss implementation;
- F03 scheduler `eta_min` and non-explicit AdamW defaults;
- F04 exact gradient-clipping and backend-flag record;
- F05 pretrained-weight identifier, workers, AMP, patience, scheduler parameters, and clipping;
- F06 workers, scheduler parameters, and full backend trace;
- F07/F08 inference batch size, autocast/precision state, latency, throughput, and energy/compute cost.

### 11.3 Dataset-level boundaries carried from Phase 2

- immutable Hugging Face revision/fingerprint/content hash;
- split-manifest checksum and public location;
- complete duplicate/near-duplicate audit;
- exhaustive internal corrupt-image decode report;
- per-image Aug-200 random-draw log.

### 11.4 Claim restrictions

Do not write:

- “Every experiment was fully bitwise deterministic.”
- “The exact environment was completely preserved for all runs.”
- “Ultralytics 8.4.89 is independently proven as the original training version for every F01/F02 step.”
- “F01/F02 used a specific resolved optimiser” unless the missing runtime evidence is located.
- “F05 used the same patience, workers, AMP, or scheduler parameters as F03/F04.”
- “The test set was used to choose a checkpoint or ensemble weight.”
- “F02 is a ConvNeXt-Small cross-entropy baseline.”

The approved general statement is:

> Complete run-specific hardware and software-version records were not preserved for every archived experiment and are therefore not reconstructed retrospectively.

---

## 12. Reviewer and viva question bank

### Q1. What execution environment was directly observed?

A Kaggle-hosted Linux notebook using Python 3.12.13, PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118, CUDA 11.8, and one Tesla P100-PCIE-16GB GPU.

### Q2. Was GPU execution verified rather than assumed?

Yes. CUDA availability, GPU count/name, device properties, and a CUDA matrix-multiplication test were printed successfully.

### Q3. What Ultralytics version is preserved?

Ultralytics 8.4.89 was explicitly installed, printed, and asserted as the locked/restored F01-compatible version. A separate complete original training-run manifest for every F01/F02 stage was not archived.

### Q4. What cuDNN version was used?

The repaired environment installed `nvidia-cudnn-cu11==9.1.0.70`. A direct `torch.backends.cudnn.version()` output was not found, so the native runtime value remains To be verified.

### Q5. What was the campaign seed?

Seed 42.

### Q6. Were all runs bitwise deterministic?

That cannot be claimed. Several deterministic controls were used, but complete backend, worker, package, and random-state traces were not uniformly archived.

### Q7. Which explicit deterministic controls were recorded for F01?

`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=42`, seed 42, and the saved Ultralytics `deterministic=True` argument.

### Q8. Which explicit backend controls were recorded for F03?

`cudnn_deterministic=True` and `cudnn_benchmark=False`.

### Q9. Why does an earlier generic notebook setup not control the final record?

Because run-specific locked configurations take priority over generic or historical setup cells.

### Q10. Were multiple training seeds evaluated?

No. The final configurations are preserved single completed runs.

### Q11. Does the bootstrap analysis measure training-seed variability?

No. It measures uncertainty from the fixed test sample conditional on the frozen predictions.

### Q12. How was class order protected?

ImageFolder class lists were checked across train, validation, and test, and F08 independently verified identical class-name mapping between F06 and F01.

### Q13. Was the test set used during training?

No. The official test split was excluded from training, early stopping, checkpoint selection, and F08 weight selection.

### Q14. How were F01 and F02 checkpoints selected?

Using native Ultralytics validation classification fitness based on validation Top-1 and Top-5.

### Q15. How were F03–F06 checkpoints selected?

Using validation Macro F1.

### Q16. Did F07 create a new checkpoint?

No. It was inference only and used the frozen F06 checkpoint.

### Q17. How was the F08 weight chosen?

A predefined four-weight grid was evaluated on validation Macro F1, followed by Macro Recall, Top-1, and equal-weight preference tie-breaks.

### Q18. Is Aug-200 the same as online augmentation?

No. Aug-200 created permanent training files offline; online augmentation transformed samples dynamically during training.

### Q19. Did all Aug-200 runs use identical online transforms?

No. F01/F02 used the Ultralytics classification pipeline, while F04 and F05 used simpler ConvNeXt pipelines.

### Q20. What is the main difference between YOLO and ConvNeXt evaluation preprocessing?

YOLO resized the shorter edge to 224 and then centre-cropped; the ConvNeXt evaluation pipelines directly resized to the target square size.

### Q21. What normalisation did F01/F02 use?

Identity normalisation: mean `[0,0,0]`, standard deviation `[1,1,1]` after tensor scaling.

### Q22. What normalisation did F03–F06 use?

ImageNet mean `[0.485,0.456,0.406]` and standard deviation `[0.229,0.224,0.225]`.

### Q23. Why is F04–F05 not a strict loss-only ablation?

Their maximum epoch budgets differed, 40 versus 25, and several F05 controls were not completely archived.

### Q24. Why is F03–F06 not an isolated resolution ablation?

F06 also introduced continuation training, a lower learning rate, a smaller batch size, and a new training budget.

### Q25. What is the locked F02 identity?

`F02_YOLO26S_CLS_AUG200`, a YOLO26s-cls Aug-200 model-scale control.

### Q26. What environment details must remain undisclosed as exact values?

Any unverified scikit-learn version, driver version, CPU/RAM record, original run-local package freeze, or unresolved optimiser/runtime setting.

### Q27. Can another researcher reproduce the campaign exactly from the current archive?

The principal split, class order, transformations, core GPU/PyTorch stack, run configurations, checkpoints, selection rules, and predictions are substantially documented. Exact bitwise reproduction is not guaranteed because immutable source/container and complete per-run environment records are incomplete.

---

## 13. Evidence ledger

The following sources controlled Phase 3:

- latest final manuscript: *Improving Underwater Marine Species Classification through Imbalance-Aware Learning and Cross-Dataset Evaluation*;
- `P00_AQUA20_Master_Evidence_Freeze_FINAL.xlsx`;
- `P00_AQUA20_Evidence_Freeze_Report_FINAL.md`;
- `04-07-2026-aqua20-final-beat-base-plan.ipynb`;
- F01/F02 saved Ultralytics arguments and histories represented in the final evidence register;
- `README_F01_RECOVERY.txt`;
- F03 official README and locked training configuration;
- F04 training configuration, environment output, history, and final summary;
- `README_F05.txt`;
- F06 official README, locked source/configuration output, and history;
- F07 locked protocol and README;
- `README_F08.txt`;
- `Pasted markdown.md` containing the verified effective preprocessing and transform extraction;
- `D02_AQUA20_Dataset_and_Aug200_Documentation.md`;
- `H02_Phase2_to_Phase3_Handover.md`.

Older evidence-freeze rows that marked all transforms as permanently missing are superseded by the later preserved notebook and verified transform extraction. Older reports that assigned F02 another identity are superseded by the FINAL evidence freeze.

---

## 14. Phase 3 completion decision

Phase 3 is complete because the following are now locked:

- the directly observed core Python/PyTorch/torchvision/CUDA/P100 environment;
- the scope and limitation of the Ultralytics 8.4.89 restoration record;
- the cuDNN wheel/runtime distinction;
- the shared ImageFolder and class-order verification process;
- campaign seed and recorded deterministic controls;
- experiment-by-experiment batch, worker, precision, environment, and unresolved-field registry;
- offline Aug-200 versus online augmentation distinction;
- effective F01–F06 training and evaluation transformations;
- F07 two-view and F08 probability-fusion protocols;
- checkpoint-selection rules for F01–F08;
- official-test isolation from model and rule selection;
- reproducibility limitations and prohibited overclaims; and
- an environment/reproducibility reviewer question bank.

### Remaining To be verified items carried forward

1. complete run-specific package and hardware manifests;
2. original F01/F02 training-time Ultralytics environment proof and resolved optimiser/loss behaviour;
3. missing F05/F06 worker, AMP/backend, scheduler, patience, and clipping fields identified above;
4. controlled latency, throughput, FLOPs, parameter-count, memory, and ensemble-cost evidence;
5. immutable dataset/container/source checksums and Phase 2 integrity gaps.

These unresolved values limit claims of complete bitwise reproducibility, but they do not alter the frozen experiment identities, checkpoints, data partitions, preprocessing protocols, model-selection rules, or official test results.
