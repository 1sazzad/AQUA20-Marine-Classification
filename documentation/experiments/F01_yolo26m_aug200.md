# F01 — YOLO26m-cls + Aug-200

## 1. Official identity

```text
F01_YOLO26M_CLS_AUG200
```

**Repository file:** `documentation/experiments/F01_yolo26m_aug200.md`

**Scientific role:** medium-scale YOLO Aug-200 reference and the later YOLO probability source used by F08.

F01 is not the final proposed model. It is the stronger of the two YOLO Aug-200 scale configurations evaluated in the final campaign.

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
starting checkpoint identifier = yolo26m-cls.pt
pretrained = true
```

The original upstream pretrained-file SHA-256 and immutable download/repository provenance were not archived and remain **To be verified**.

## 3. Training chronology

| Item | Locked value |
|---|---:|
| Maximum epochs | 200 |
| Completed epochs | 32 |
| Best epoch | 2 |
| Patience | 30 |
| Selection | Ultralytics validation classification fitness |

The run ended before the 200-epoch maximum. The preserved `best.pt` is the validation-selected checkpoint from epoch 2.

### Locked validation result

| Metric | Value |
|---|---:|
| Validation Top-1 | 87.8811% |
| Validation Top-5 | 99.390244% |
| Reconstructed native fitness | 93.635672% |

## 4. Official unchanged test result

| Metric | Value |
|---|---:|
| Top-1 | 84.677419% |
| Top-2 | 94.230769% |
| Top-3 | 97.332506% |
| Top-5 | 98.945409% |
| Macro Precision | 83.305358% |
| Macro Recall | 73.989533% |
| Macro F1 | 75.990769% |
| Weighted F1 | 84.424872% |
| Support | 1,612 |

The saved 1,612 predictions, classification metrics, and confusion-matrix evidence were independently checked in the final campaign audit.

Weighted F1 exceeded Macro F1 by approximately **8.4341 percentage points**, showing that equal-class performance remained substantially weaker than support-weighted performance despite Aug-200.

## 5. Role relative to F02

Under the same Aug-200 YOLO protocol, F01 exceeded F02 by:

| Metric | F01 − F02 |
|---|---:|
| Top-1 | +1.240694 pp |
| Top-5 | +0.992555 pp |
| Macro Precision | +2.395048 pp |
| Macro Recall | +8.733076 pp |
| Macro F1 | +5.599874 pp |
| Weighted F1 | +1.587226 pp |

The important configuration-level observation is that the scale reduction from YOLO26m to YOLO26s was associated with a much larger loss in Macro Recall and Macro F1 than in Top-1 accuracy.

This does not establish a universal causal law about model capacity because only one completed run per configuration was frozen and no repeated-seed capacity study was performed.

## 6. Later campaign role

F01 is the frozen YOLO source used by the F08 heterogeneous probability ensemble.

F08 therefore depends on the exact frozen F01 checkpoint, class ordering, and F01 evaluation preprocessing.

## 7. Checkpoint and package integrity

### Best checkpoint

```text
weights/best.pt
epoch = 2
SHA-256 =
aa489def1443a6cdab42a0a20a530b9e341d574dcaaf234bfe6c7983113e8e5a

bytes = 20,926,870
```

`results.csv` SHA-256:

```text
e4ca9bb5cc021c4f793b797c7f496e2515243ea973bea951f8396e32de63bcb3
```

### Official package

```text
aqua20_f01_yolo26m_cls_aug200_outputs.zip
```

| Integrity field | Value |
|---|---|
| ZIP SHA-256 | `c0ac776d43a1bd7ab9c673baa855c04d009afd71f0796a4d5438ba7ea154e2b1` |
| ZIP bytes | 19,971,213 |
| Extracted files | 27 |
| Extracted-tree bytes | 22,154,877 |
| Tree SHA-256 | `397344dcea8eaea56e0ce936f2f37109ba36cd0d9fb6b0e82a1adbdb45a45a2a` |
| Prediction rows | 1,612 |
| Confusion matrix | verified |
| Metrics | independently recomputed |

The preserved release package excludes the dataset and `last.pt`.

## 8. Environment boundary

A campaign environment record preserves a Kaggle-hosted Linux notebook with:

```text
Python 3.12.13
PyTorch 2.7.1+cu118
torchvision 0.22.1+cu118
CUDA 11.8
Tesla P100-PCIE-16GB
```

Ultralytics `8.4.89` was explicitly restored and asserted as an F01-compatible environment record. A complete immutable original training-run environment manifest was not preserved, so bitwise reproduction is not claimed.

## 9. Allowed and prohibited claims

### Allowed

- F01 is `F01_YOLO26M_CLS_AUG200`.
- F01 used the 7,284-image Aug-200 training distribution.
- F01 used native Ultralytics classification training rather than custom focal/class weighting.
- F01 checkpoint selection used validation Top-1/Top-5 fitness.
- F01 achieved 84.677419% Top-1 and 75.990769% Macro F1 on the unchanged 1,612-image official test.
- F01 was stronger than F02 across all frozen test metrics.
- F01 later served as the YOLO source for F08.

### Do not write

- F01 proves Aug-200 is generally effective or ineffective.
- F01 used a specific resolved optimiser unless new direct evidence is found.
- F01 used a reconstructed native loss formula.
- all generic `args.yaml` augmentation fields were active in classification;
- official-test metrics selected the checkpoint;
- the run is fully bitwise reproducible.

## 10. Still To be verified

- original `yolo26m-cls.pt` upstream SHA/provenance;
- original run-local Ultralytics version proof separate from the restored environment record;
- resolved optimiser and exact parameter groups under `optimizer=auto`;
- exact native classification-loss implementation;
- complete internal early-stopping counter semantics;
- complete run-local package/container/driver/native-cuDNN/CPU/RAM manifest;
- controlled parameters/FLOPs/memory/latency/throughput/energy;
- repeated-seed training variance;
- immutable dataset revision and full duplicate/decode boundaries.

## 11. Reviewer / viva quick questions

**What is F01?**  
YOLO26m-cls trained at 224×224 on the 7,284-image train-only Aug-200 distribution.

**How was its checkpoint selected?**  
Using native Ultralytics validation classification fitness based on validation Top-1 and Top-5.

**Was Macro F1 used to select it?**  
No.

**What was the best epoch?**  
Epoch 2.

**What was the official-test result?**  
84.677419% Top-1 and 75.990769% Macro F1.

**Why is the Macro/Weighted gap important?**  
It shows that high-support classes contributed more strongly to aggregate performance than minority classes.

**What later experiment used F01?**  
F08 used the frozen F01 probability output as the YOLO component of its heterogeneous ensemble.
