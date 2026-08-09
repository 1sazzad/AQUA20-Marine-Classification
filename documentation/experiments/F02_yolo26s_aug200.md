# F02 — YOLO26s-cls + Aug-200

## 1. Official identity

```text
F02_YOLO26S_CLS_AUG200
```

**Repository file:** `documentation/experiments/F02_yolo26s_aug200.md`

**Scientific role:** smaller-model YOLO scale ablation under the same Aug-200 protocol as F01.

### Mandatory identity correction

F02 must never be relabelled as a ConvNeXt-Small cross-entropy baseline.

The ConvNeXt-Small + Aug-200 + standard cross-entropy experiment is:

```text
F04_CONVNEXT_SMALL_AUG200
```

Earlier historical audit material used a conflicting F02 interpretation. That older interpretation is superseded by the latest final manuscript, FINAL evidence freeze, final experiment index, and locked Phase 4 documentation.

## Shared YOLO Aug-200 protocol

This experiment belongs to the first model-training branch of the final AQUA20 campaign. F01 and F02 used the same train-only Aug-200 protocol; the principal controlled difference was YOLO classifier scale.

### Data

| Split | Images |
|---|---:|
| Aug-200 training | 7,284 |
| Validation | 1,312 |
| Official test | 1,612 |
| Classes | 20 |
| Input | 224×224 |

Aug-200 raised classes below 200 training samples to a minimum of 200 without downsampling larger classes. Validation and test distributions were unchanged.

The official test split was excluded from training, early stopping, checkpoint selection, and protocol tuning.

### Effective online training transform

```text
RandomResizedCrop 224
  scale 0.50–1.00
  ratio 0.75–1.3333
→ RandomHorizontalFlip p=0.5
→ RandAugment
   2 operations
   magnitude 9
   31 magnitude bins
→ tensor [0,1]
→ identity normalisation
   mean [0,0,0]
   std  [1,1,1]
→ RandomErasing p=0.4
```

### Validation / official-test transform

```text
shorter-edge resize to 224
→ CenterCrop 224
→ tensor [0,1]
→ identity normalisation
```

No random evaluation augmentation was used.

### Shared saved training controls

| Field | Value |
|---|---|
| Maximum epochs | 200 |
| Batch | 64 |
| Workers | 8 |
| Patience | 30 |
| Optimiser request | `auto` |
| `lr0` | `0.01` |
| `lrf` | `0.01` |
| Momentum argument | `0.937` |
| Weight decay | `0.0005` |
| Warm-up | 3 epochs |
| AMP | enabled |
| Seed | 42 |
| Deterministic request | true |
| Pretrained | true |
| Custom class weights | none |
| Custom focal loss | none |

The resolved runtime optimiser and exact version-specific native Ultralytics classification-loss implementation remain **To be verified**. They must not be reconstructed from generic defaults.

### Checkpoint selection

F01 and F02 used native Ultralytics validation classification fitness:

```text
fitness = (validation Top-1 + validation Top-5) / 2
```

They were **not** selected by validation Macro F1.

A broad cross-task Ultralytics `args.yaml` was saved. Detection/segmentation/pose/export/tracking fields in that schema must not be interpreted as active classification operations unless supported by the classification pipeline. In particular:

- `augment: false` is an inference option and does not disable training augmentation;
- `cls=0.5` is not sufficient evidence to reconstruct the native classification objective.


## 2. Pretrained source

```text
starting checkpoint identifier = yolo26s-cls.pt
pretrained = true
```

The original upstream pretrained-file SHA-256 and immutable source provenance remain **To be verified**.

## 3. Training chronology and tied best fitness

| Item | Locked value |
|---|---:|
| Maximum epochs | 200 |
| Completed epochs | 68 |
| Patience | 30 |
| First maximum-fitness epoch | 38 |
| Equal maximum-fitness epoch | 49 |
| Preserved `best.pt` epoch | 49 |
| Selection | Ultralytics validation classification fitness |

The equal-fitness tie between epochs 38 and 49 is verified. The exact version-specific tie handling and early-stopping counter/reset semantics remain **To be verified**, so they must not be reconstructed beyond the preserved chronology.

### Locked validation result

| Metric | Value |
|---|---:|
| Validation Top-1 | 86.0518% |
| Validation Top-5 | 98.7043% |
| Reconstructed native fitness | 92.378050% |

## 4. Official unchanged test result

| Metric | Value |
|---|---:|
| Top-1 | 83.436725% |
| Top-2 | 92.307692% |
| Top-3 | 95.905707% |
| Top-5 | 97.952854% |
| Macro Precision | 80.910310% |
| Macro Recall | 65.256457% |
| Macro F1 | 70.390895% |
| Weighted F1 | 82.837646% |
| Support | 1,612 |

The saved 1,612 predictions, confusion matrix, Top-k values, and Macro/Weighted metrics were independently verified.

Weighted F1 exceeded Macro F1 by approximately **12.4468 percentage points**, larger than the corresponding F01 gap.

## 5. Weak-class evidence

The official F02 record preserves particularly weak recalls:

| Class | Recall |
|---|---:|
| `flatworm` | 23.08% |
| `marine_dolphin` | 30.00% |
| `seaCucumber` | 30.00% |
| `octopus` | 40.00% |
| `eel` | 48.78% |

This supports a bounded configuration-level conclusion: under the same Aug-200 YOLO setup, the smaller YOLO26s configuration lost more balanced minority-class recognition than headline accuracy alone suggests.

It does not prove that small classifiers are unsuitable for marine recognition in general.

## 6. F02 versus F01

F02 was weaker than F01 across all frozen metrics. Expressed as F01 minus F02:

| Metric | Difference |
|---|---:|
| Top-1 | +1.240694 pp |
| Top-5 | +0.992555 pp |
| Macro Precision | +2.395048 pp |
| Macro Recall | +8.733076 pp |
| Macro F1 | +5.599874 pp |
| Weighted F1 | +1.587226 pp |

The largest degradation was in Macro Recall, followed by Macro F1.

## 7. Checkpoint and package integrity

### Best checkpoint

```text
weights/best.pt
locked epoch = 49
SHA-256 =
38a4948f97f8ed3194d4c3ce7d56132180f8a1b3b84a4c575fba45d1380ab302

bytes = 11,078,338
```

### Official package

```text
aqua20_f02_yolo26s_cls_aug200_outputs.zip
```

| Integrity field | Value |
|---|---|
| ZIP SHA-256 | `121bc5cf88ce4cfbb9eedeade74524c2c1f48c5684c41cbbffe5be4e748a2fc7` |
| ZIP bytes | 10,909,233 |
| Extracted files | 16 |
| Extracted-tree bytes | 12,318,987 |
| Tree SHA-256 | `727a7e1f93aec1eecf51fc7ab08e65cac6d6aaa6b610320fa8f233b8acfbed16` |
| Prediction rows | 1,612 |
| Confusion matrix | verified |
| Metrics | independently recomputed |

The preserved package excludes the dataset and `last.pt`.

## 8. Environment boundary

The shared campaign environment evidence includes a Kaggle-hosted Linux environment with Python 3.12.13, PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118, CUDA 11.8, and a Tesla P100-PCIE-16GB.

However, a complete original F02 run-local environment manifest was not preserved. Therefore exact bitwise reproduction is not claimed.

## 9. Allowed and prohibited claims

### Allowed

- F02 is `F02_YOLO26S_CLS_AUG200`.
- F02 is a YOLO26s + Aug-200 scale ablation.
- F02 completed 68 epochs.
- epochs 38 and 49 tied for maximum validation fitness;
- the preserved `best.pt` is epoch 49;
- F02 achieved 83.436725% Top-1 and 70.390895% Macro F1;
- F02 was weaker than F01 on all frozen metrics;
- several minority classes had low recall.

### Do not write

- F02 was a ConvNeXt-Small cross-entropy baseline.
- F04 metrics belong to F02.
- model scale universally caused the observed result.
- the exact runtime optimiser is known.
- the exact native loss formula is known.
- equal-fitness tie mechanics are fully reconstructed.
- official-test results selected the checkpoint.

## 10. Still To be verified

- original `yolo26s-cls.pt` upstream SHA/provenance;
- exact original run-local Ultralytics version;
- resolved optimiser and parameter groups under `optimizer=auto`;
- exact native classification-loss implementation;
- exact equal-fitness/early-stopping counter semantics;
- complete immutable run-local software/container/driver/native-cuDNN/CPU/RAM manifest;
- controlled efficiency measurements;
- repeated-seed variance;
- immutable dataset revision and full duplicate/decode boundaries.

## 11. Reviewer / viva quick questions

**What is F02?**  
YOLO26s-cls trained on the same 7,284-image Aug-200 protocol as F01.

**Is F02 the ConvNeXt CE baseline?**  
No. That configuration is F04.

**What was F02's locked best checkpoint?**  
Epoch 49, tied in maximum fitness with epoch 38.

**How many epochs completed?**  
68 of a maximum 200.

**What was the official-test Macro F1?**  
70.390895%.

**Which classes were especially weak?**  
Flatworm, marine dolphin, sea cucumber, octopus, and eel.

**What is the key F01/F02 finding?**  
The smaller model's balanced-class recall/F1 dropped much more than its Top-1 accuracy.
