# AQUA20 Phase 4 — F01 and F02 YOLO26 Aug-200 Experiment Documentation

## 1. Scope and completion boundary

This document completes **Phase 4 only** of the AQUA20 documentation roadmap. It freezes the first model-training branch of the final campaign:

- `F01_YOLO26M_CLS_AUG200`
- `F02_YOLO26S_CLS_AUG200`

The two experiments are interpreted only as a **YOLO model-scale comparison under the same Aug-200 protocol**. F02 is not a ConvNeXt-Small cross-entropy baseline. The ConvNeXt-Small + Aug-200 + standard cross-entropy configuration is F04.

No new experiment, checkpoint, metric, or retrospective reconstruction was created in this phase.

---

## 2. Controlling source hierarchy

The evidence was interpreted in this order:

1. latest final manuscript, `Improving Underwater Marine Species Classification through Imbalance-Aware Learning and Cross-Dataset Evaluation.pdf`;
2. `P00_AQUA20_Master_Evidence_Freeze_FINAL.xlsx` and frozen terminology/index records;
3. official F01/F02 experiment packages, saved `args.yaml`, `results.csv`, READMEs, checkpoint metadata, prediction evidence, classification reports, confusion matrices, and integrity manifests;
4. preserved campaign notebook `04-07-2026-aqua20-final-beat-base-plan.ipynb`;
5. earlier drafts and audits for historical context only.

### Superseded conflict

Some earlier audit material described the YOLO26s package as a legacy numbering conflict and assigned F02 to a ConvNeXt control. That interpretation is superseded by the latest final manuscript, FINAL evidence freeze, final README index, and Phase 3 lock. The official identity is now permanently:

> `F02_YOLO26S_CLS_AUG200`

---

## 3. Scientific role of the branch

Both experiments used:

- the same AQUA20 20-class task;
- the same train-only Aug-200 dataset containing 7,284 training images;
- the same unchanged validation set of 1,312 images;
- the same unchanged official test set of 1,612 images;
- the same 224×224 input size;
- the same Ultralytics classification training protocol;
- the same validation-fitness checkpoint-selection principle;
- no custom class weights;
- no custom focal loss.

The principal controlled change was model scale:

| Experiment | Model | Scientific role |
|---|---|---|
| F01 | YOLO26m-cls | Medium-scale YOLO Aug-200 reference; later YOLO source for F08 |
| F02 | YOLO26s-cls | Smaller-model scale ablation under the same Aug-200 protocol |

This comparison does not isolate augmentation effectiveness because both models use Aug-200. It asks whether reducing YOLO model scale changes performance while the data and major training protocol remain fixed.

---

## 4. Dataset and leakage boundary

### 4.1 Frozen data counts

| Split | Images | Intervention status |
|---|---:|---|
| Aug-200 training | 7,284 | Training-only offline expansion |
| Validation | 1,312 | Unchanged |
| Official test | 1,612 | Unchanged |
| Classes | 20 | Frozen order |

Aug-200 imposed a floor of 200 training images for classes below that count. It did not downsample the large classes and therefore did not produce perfect balance.

### 4.2 Governance

The official test set was excluded from:

- training;
- early stopping;
- checkpoint selection;
- protocol tuning.

The preserved checkpoint was fixed using validation evidence before official-test evaluation.

---

## 5. Pretrained checkpoint provenance

### F01

- starting identifier: `yolo26m-cls.pt`;
- saved setting: `pretrained: true`;
- loaded as the starting pretrained Ultralytics classification checkpoint.

### F02

- starting identifier: `yolo26s-cls.pt`;
- saved setting: `pretrained: true`;
- loaded as the starting pretrained Ultralytics classification checkpoint.

### Claim boundary

The archive preserves the checkpoint identifiers, not an immutable copy/hash and complete upstream download provenance for the original pretrained files. Therefore the following remain **To be verified**:

- original pretrained file SHA-256 values;
- exact upstream asset URL/release record used at first download;
- immutable upstream model-repository revision.

The final trained `best.pt` files are separately hash-locked below.

---

## 6. Saved `args.yaml` record

The two saved argument files are materially identical except for model and output paths.

### 6.1 Run-specific differences

| Field | F01 | F02 |
|---|---|---|
| `model` | `yolo26m-cls.pt` | `yolo26s-cls.pt` |
| `project` | `.../experiments/F01_YOLO26M_CLS_AUG200/yolo_runs` | `.../experiments/F02_YOLO26S_CLS_AUG200` |
| `save_dir` | `.../F01_YOLO26M_CLS_AUG200/yolo_runs/train` | `.../F02_YOLO26S_CLS_AUG200/train` |

### 6.2 Effective training, validation, and reproducibility fields

| Field | Saved value | Interpretation |
|---|---:|---|
| `task` | `classify` | Classification task |
| `mode` | `train` | Training run |
| `data` | `.../balanced_train/aug200_imagefolder` | Aug-200 ImageFolder root |
| `epochs` | `200` | Maximum epoch budget |
| `time` | `null` | No time-budget override |
| `patience` | `30` | Early-stopping patience |
| `batch` | `64` | Batch size |
| `imgsz` | `224` | Input size |
| `save` | `true` | Checkpoint saving enabled |
| `save_period` | `-1` | No periodic checkpoint schedule |
| `cache` | `false` | Dataset caching disabled |
| `device` | `'0'` | First CUDA device requested |
| `workers` | `8` | Data-loader workers |
| `name` | `train` | Run subdirectory name |
| `exist_ok` | `true` | Existing path permitted |
| `pretrained` | `true` | Pretrained starting weights |
| `optimizer` | `auto` | Optimiser delegated to Ultralytics |
| `verbose` | `true` | Verbose logging |
| `seed` | `42` | Campaign seed |
| `deterministic` | `true` | Ultralytics deterministic request |
| `single_cls` | `false` | Full 20-class task retained |
| `rect` | `false` | Rectangular batching disabled |
| `cos_lr` | `false` | Cosine LR flag disabled |
| `resume` | `false` | Fresh run, not resumed |
| `amp` | `true` | Automatic mixed precision enabled |
| `fraction` | `1.0` | Full training dataset used |
| `profile` | `false` | Profiling disabled |
| `freeze` | `null` | No layer-freeze list saved |
| `multi_scale` | `0.0` | Multi-scale training disabled |
| `compile` | `false` | Model compilation disabled |
| `dropout` | `0.0` | No additional dropout setting |
| `val` | `true` | Validation enabled |
| `split` | `val` | Validation split selected |
| `plots` | `true` | Training plots enabled |
| `lr0` | `0.01` | Initial LR argument |
| `lrf` | `0.01` | Final LR fraction argument |
| `momentum` | `0.937` | Saved optimiser momentum argument |
| `weight_decay` | `0.0005` | Weight decay argument |
| `warmup_epochs` | `3.0` | Warm-up duration |
| `warmup_momentum` | `0.8` | Warm-up momentum argument |
| `warmup_bias_lr` | `0.1` | Warm-up bias LR argument |
| `nbs` | `64` | Nominal batch-size scaling reference |

### 6.3 Saved augmentation-related fields

| Field | Value |
|---|---:|
| `hsv_h` | `0.015` |
| `hsv_s` | `0.7` |
| `hsv_v` | `0.4` |
| `degrees` | `0.0` |
| `translate` | `0.1` |
| `scale` | `0.5` |
| `shear` | `0.0` |
| `perspective` | `0.0` |
| `flipud` | `0.0` |
| `fliplr` | `0.5` |
| `bgr` | `0.0` |
| `mosaic` | `1.0` |
| `mixup` | `0.0` |
| `cutmix` | `0.0` |
| `copy_paste` | `0.0` |
| `copy_paste_mode` | `flip` |
| `auto_augment` | `randaugment` |
| `erasing` | `0.4` |

These are saved generic Ultralytics arguments. Their presence does not mean every detection-oriented field was applied by the classification data pipeline. The effective classification transforms were established separately from the preserved Ultralytics classification source path and are frozen in Section 7.

### 6.4 Generic detection/segmentation/pose fields retained in the YAML

The following stored values are configuration-schema fields but are not evidence that the corresponding detection, segmentation, pose, or tracking mechanism affected these classification runs:

```text
close_mosaic=10
bar/box-related: box=7.5, cls=0.5, cls_pw=0.0, dfl=1.5
pose/segmentation-related: pose=12.0, kobj=1.0, rle=1.0, angle=1.0
overlap_mask=true, mask_ratio=4
distill_model=null, dis=6.0
```

In particular, `cls=0.5` must not be used to reconstruct or rename the native classification-loss implementation.

### 6.5 Generic inference, visualisation, export, and tracking fields

The raw YAML also preserves:

```text
save_json=false
conf=null
iou=0.7
max_det=300
quantize=null
dnn=false
end2end=null
source=null
vid_stride=1
stream_buffer=false
visualize=false
augment=false
agnostic_nms=false
classes=null
retina_masks=false
embed=null
show=false
save_frames=false
save_txt=false
save_conf=false
save_crop=false
show_labels=true
show_conf=true
show_boxes=true
line_width=null
format=torchscript
keras=false
optimize=false
dynamic=false
simplify=true
opset=null
workspace=null
nms=false
cfg=null
tracker=tracktrack.yaml
```

These fields mostly belong to the wider Ultralytics schema. They do not define the training objective or prove that detection-style post-processing was used for classification.

The saved `augment: false` is an inference-augmentation option. It does **not** disable the classification training transformations described next.

---

## 7. Effective online transforms

F01 and F02 used the same effective online classification pipeline.

### 7.1 Training pipeline

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
   - operation space includes identity, shear, translation, rotation, brightness, colour, contrast, sharpness, posterize, solarize, autocontrast, and equalize.
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

### 7.2 Validation and official-test pipeline

1. resize the shorter edge to 224;
2. `CenterCrop(224)`;
3. tensor conversion to `[0,1]`;
4. identity normalisation.

No random evaluation augmentation was used.

### 7.3 Offline versus online augmentation

Aug-200 is an offline training-file expansion. The pipeline above is dynamic online augmentation. They are separate interventions and must not be conflated.

---

## 8. Optimiser and native-loss claim boundary

### 8.1 Optimiser

The only locked request is:

```text
optimizer=auto
```

The saved LR, momentum, warm-up, and weight-decay arguments are documented, but the final evidence does not freeze the resolved optimiser class, runtime parameter groups, or any automatic override logic. Therefore the resolved optimiser remains:

> **To be verified**

It must not be retrospectively labelled MuSGD, SGD, AdamW, or another optimiser from memory or an older preliminary register.

### 8.2 Objective

The experiments used the native Ultralytics classification objective with:

- no custom class weights;
- no custom focal loss;
- no campaign-specific imbalance-aware loss;
- label smoothing recorded as zero in the FINAL register.

The exact version-specific native loss implementation and internal function path remain **To be verified**. Generic YAML fields must not be used to invent a more specific loss name.

---

## 9. Checkpoint selection rule

For F01 and F02, the validation fitness was reconstructed as:

\[
\mathrm{fitness}
=
\frac{\mathrm{Top1}_{val}+\mathrm{Top5}_{val}}{2}.
\]

This was native Ultralytics classification fitness, not validation Macro F1.

The official test set did not participate in this calculation.

---

## 10. F01 chronology — `F01_YOLO26M_CLS_AUG200`

### 10.1 Configuration

| Item | Locked value |
|---|---|
| Model | `yolo26m-cls.pt` |
| Train/val/test | 7,284 / 1,312 / 1,612 |
| Input | 224×224 |
| Batch | 64 |
| Workers | 8 |
| Max epochs | 200 |
| Patience | 30 |
| Optimiser request | `auto` |
| LR arguments | `lr0=0.01`, `lrf=0.01` |
| Warm-up | 3 epochs |
| Momentum argument | 0.937 |
| Weight decay | 0.0005 |
| AMP | enabled |
| Seed | 42 |
| Deterministic request | true |

### 10.2 Training history and early stopping

- completed epochs: **32**;
- maximum budget: 200;
- best-fitness epoch: **2**;
- no later epoch exceeded the epoch-2 validation fitness;
- the 32-epoch termination is consistent with patience-30 early stopping after the early best.

The exact internal bad-epoch counter implementation is not separately frozen, so the chronology should be described as consistent with the saved patience rather than as a reconstructed line-by-line trainer state.

### 10.3 Locked validation result

| Metric | Value |
|---|---:|
| Validation Top-1 | 87.8811% |
| Validation Top-5 | 99.390244% |
| Reconstructed native fitness | 93.635672% |

### 10.4 Safety freeze before test

Immediately after training, the F01 safety package preserved:

- `weights/best.pt`;
- `results.csv`;
- `args.yaml`;
- manifest/README evidence.

The safety package explicitly excluded:

- the dataset;
- `last.pt`;
- official-test metrics.

This provides direct evidence that the locked training checkpoint existed before official-test evaluation.

### 10.5 Checkpoint and history integrity

| Item | Value |
|---|---|
| Checkpoint | `weights/best.pt` |
| Best checkpoint SHA-256 | `aa489def1443a6cdab42a0a20a530b9e341d574dcaaf234bfe6c7983113e8e5a` |
| Checkpoint bytes | 20,926,870 |
| `results.csv` SHA-256 | `e4ca9bb5cc021c4f793b797c7f496e2515243ea973bea951f8396e32de63bcb3` |

### 10.6 Official unchanged test result

| Metric | F01 |
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

### 10.7 Test evidence verification

The final audit independently verified:

- 1,612 saved predictions;
- the 20×20 confusion matrix;
- Top-1/2/3/5 values;
- classification metrics;
- native/manual agreement.

### 10.8 Class-balance interpretation

F01's Weighted F1 exceeded Macro F1 by **8.4341 percentage points**. This gap indicates that strong performance on high-support classes coexisted with weaker equal-class performance. Aug-200 improved minority representation but did not remove the residual imbalance or make the YOLO26m branch uniformly strong across all classes.

The full F01 classification report is preserved in the package. This phase does not invent individual F01 class values that were not surfaced in the controlling summary evidence.

---

## 11. F02 chronology — `F02_YOLO26S_CLS_AUG200`

### 11.1 Configuration

| Item | Locked value |
|---|---|
| Model | `yolo26s-cls.pt` |
| Train/val/test | 7,284 / 1,312 / 1,612 |
| Input | 224×224 |
| Batch | 64 |
| Workers | 8 |
| Max epochs | 200 |
| Patience | 30 |
| Optimiser request | `auto` |
| LR arguments | `lr0=0.01`, `lrf=0.01` |
| Warm-up | 3 epochs |
| Momentum argument | 0.937 |
| Weight decay | 0.0005 |
| AMP | enabled |
| Seed | 42 |
| Deterministic request | true |

### 11.2 Training history, tie, and early stopping

- completed epochs: **68**;
- maximum budget: 200;
- first maximum-fitness epoch: **38**;
- equal maximum-fitness epoch: **49**;
- preserved `best.pt`: epoch **49**;
- equal-fitness tie: verified.

The run ended before the 200-epoch budget. The saved chronology and patience value support an early-stopping interpretation. The exact version-specific handling of equal-fitness ties and bad-epoch-counter resets remains **To be verified** and should not be reconstructed beyond the verified first-best/tied-best/locked-checkpoint sequence.

### 11.3 Locked validation result

| Metric | Value |
|---|---:|
| Validation Top-1 | 86.0518% |
| Validation Top-5 | 98.7043% |
| Reconstructed native fitness | 92.378050% |

### 11.4 Checkpoint integrity

| Item | Value |
|---|---|
| Checkpoint | `weights/best.pt` |
| Locked checkpoint epoch | 49 |
| Best checkpoint SHA-256 | `38a4948f97f8ed3194d4c3ce7d56132180f8a1b3b84a4c575fba45d1380ab302` |
| Checkpoint bytes | 11,078,338 |

### 11.5 Official unchanged test result

| Metric | F02 |
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

### 11.6 Test evidence verification

The final audit verified:

- 1,612 predictions;
- stored confusion matrices;
- Top-1/2/3/5 values;
- Macro and Weighted metrics;
- native/manual metric agreement.

### 11.7 Weak class findings

The official F02 README records particularly weak recalls for:

| Class | Recall |
|---|---:|
| `flatworm` | 23.08% |
| `marine_dolphin` | 30.00% |
| `seaCucumber` | 30.00% |
| `octopus` | 40.00% |
| `eel` | 48.78% |

F02's Weighted F1 exceeded Macro F1 by **12.4468 percentage points**, a wider gap than F01. This supports the interpretation that reducing model scale disproportionately harmed balanced minority-class recognition, even though headline Top-1 declined by only 1.24 points.

---

## 12. F01 versus F02 controlled comparison

### 12.1 Validation comparison

| Metric | F01 | F02 | F01 − F02 |
|---|---:|---:|---:|
| Val Top-1 | 87.8811% | 86.0518% | +1.8293 pp |
| Val Top-5 | 99.390244% | 98.7043% | +0.685944 pp |
| Native fitness | 93.635672% | 92.378050% | +1.257622 pp |

### 12.2 Official-test comparison

| Metric | F01 | F02 | F01 − F02 |
|---|---:|---:|---:|
| Top-1 | 84.677419% | 83.436725% | +1.240694 pp |
| Top-2 | 94.230769% | 92.307692% | +1.923077 pp |
| Top-3 | 97.332506% | 95.905707% | +1.426799 pp |
| Top-5 | 98.945409% | 97.952854% | +0.992555 pp |
| Macro Precision | 83.305358% | 80.910310% | +2.395048 pp |
| Macro Recall | 73.989533% | 65.256457% | +8.733076 pp |
| Macro F1 | 75.990769% | 70.390895% | +5.599874 pp |
| Weighted F1 | 84.424872% | 82.837646% | +1.587226 pp |

### 12.3 Scientific interpretation

The medium-scale F01 model was stronger on every frozen validation/test metric. The most important difference was not the 1.24-point Top-1 gap, but the much larger:

- **8.73-point Macro Recall advantage**;
- **5.60-point Macro F1 advantage**.

Therefore, within this controlled branch, reducing YOLO model scale harmed equal-class recognition more strongly than aggregate accuracy. This is consistent with the preserved F02 weak-class recalls.

The result does **not** prove that model size alone causes this pattern universally. It is one completed run per configuration, without training-seed replication, controlled compute benchmarking, or an architecture-independent capacity study.

### 12.4 Aug-200 conclusion boundary

Both experiments still showed substantial Macro-versus-Weighted gaps. Therefore:

- Aug-200 alone did not eliminate minority-class difficulty;
- the experiment supports a negative/configuration-level finding;
- it does not establish that all data-level balancing is ineffective;
- it does not establish a causal comparison against an unbalanced YOLO control, because no original-distribution YOLO counterpart is included in this F01–F02 pair.

---

## 13. Package and integrity freeze

| Item | F01 | F02 |
|---|---|---|
| Package | `aqua20_f01_yolo26m_cls_aug200_outputs.zip` | `aqua20_f02_yolo26s_cls_aug200_outputs.zip` |
| ZIP SHA-256 | `c0ac776d43a1bd7ab9c673baa855c04d009afd71f0796a4d5438ba7ea154e2b1` | `121bc5cf88ce4cfbb9eedeade74524c2c1f48c5684c41cbbffe5be4e748a2fc7` |
| ZIP bytes | 19,971,213 | 10,909,233 |
| Extracted files | 27 | 16 |
| Extracted-tree bytes | 22,154,877 | 12,318,987 |
| Tree SHA-256 | `397344dcea8eaea56e0ce936f2f37109ba36cd0d9fb6b0e82a1adbdb45a45a2a` | `727a7e1f93aec1eecf51fc7ab08e65cac6d6aaa6b610320fa8f233b8acfbed16` |
| Checkpoint | `weights/best.pt` | `weights/best.pt` |
| Checkpoint SHA-256 | `aa489def1443a6cdab42a0a20a530b9e341d574dcaaf234bfe6c7983113e8e5a` | `38a4948f97f8ed3194d4c3ce7d56132180f8a1b3b84a4c575fba45d1380ab302` |
| Checkpoint bytes | 20,926,870 | 11,078,338 |
| Prediction support | 1,612 | 1,612 |
| Confusion matrix | verified | verified |
| Metrics | independently recomputed | independently recomputed |

### Package-content boundary

Verified content categories include:

- locked checkpoint;
- training arguments/history;
- best-epoch/checkpoint summaries;
- official-test predictions;
- Top-k and classification metrics;
- classification report;
- confusion matrices and figures;
- comparison-ready summary/README evidence.

The dataset and `last.pt` were excluded from the preserved release packages.

---

## 14. Environment statement inherited from Phase 3

The directly observed campaign runtime was a Kaggle-hosted Linux notebook with Python 3.12.13, PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118, CUDA 11.8, and one Tesla P100-PCIE-16GB GPU. Ultralytics 8.4.89 was explicitly restored, printed, and asserted as an F01-compatible environment record.

A complete original run-local environment manifest was not preserved separately for every F01/F02 training stage. Consequently, exact bitwise reproduction is not claimed.

Approved disclosure:

> Complete run-specific hardware and software-version records were not preserved for every archived experiment and are therefore not reconstructed retrospectively.

---

## 15. Locked claims and prohibited claims

### Approved

- F01 is `F01_YOLO26M_CLS_AUG200`.
- F02 is `F02_YOLO26S_CLS_AUG200`.
- Both used the same train-only Aug-200 YOLO protocol.
- F01 outperformed F02 across all frozen metrics.
- The largest model-scale differences appeared in Macro Recall and Macro F1.
- F02 showed severe weakness on several minority classes.
- Test evaluation occurred after checkpoint locking.

### Do not write

- F02 was a ConvNeXt-Small cross-entropy baseline.
- F04 metrics belong to F02.
- the exact resolved optimiser was MuSGD/SGD/AdamW without new direct proof;
- `cls=0.5` proves a specific native loss formula;
- all saved generic detection augmentation fields were active in classification;
- the runs were fully bitwise deterministic;
- Aug-200 completely balanced AQUA20;
- F01–F02 proves a general causal law about model capacity;
- official-test results selected either checkpoint.

---

## 16. Remaining items marked To be verified

1. original pretrained `yolo26m-cls.pt` and `yolo26s-cls.pt` source hashes and immutable upstream provenance;
2. original training-run-local Ultralytics version proof for both runs, separate from the later 8.4.89 restoration;
3. resolved runtime optimiser and parameter-group configuration under `optimizer=auto`;
4. exact version-specific native classification-loss implementation;
5. exact internal early-stopping counter/tie-reset semantics;
6. complete run-local `pip freeze`, immutable Kaggle image ID, NVIDIA driver, direct cuDNN runtime value, CPU/RAM record;
7. controlled parameter count, FLOPs, peak GPU memory, latency, throughput, and energy evidence under a common protocol;
8. training-seed variance from repeated runs;
9. full duplicate/near-duplicate and immutable dataset-revision boundaries carried from Phase 2.

---

## 17. Reviewer and viva question bank

### Q1. What is F01?

`F01_YOLO26M_CLS_AUG200`, a pretrained YOLO26m classification model trained on the 7,284-image Aug-200 training set at 224×224.

### Q2. What is F02?

`F02_YOLO26S_CLS_AUG200`, the smaller YOLO26s model under the same Aug-200 protocol.

### Q3. Is F02 the ConvNeXt cross-entropy baseline?

No. F04 is the ConvNeXt-Small + Aug-200 + standard cross-entropy configuration.

### Q4. What factor is primarily compared by F01 versus F02?

YOLO model scale, with the dataset, image size, batch, major training controls, and selection principle held the same.

### Q5. What data did they use?

7,284 Aug-200 training images, 1,312 unchanged validation images, and 1,612 unchanged official-test images.

### Q6. Was Aug-200 used on validation or test?

No. It was restricted to training.

### Q7. Did Aug-200 produce perfect balance?

No. It raised minority classes to a floor of 200 but retained larger classes unchanged.

### Q8. What pretrained weights were used?

`yolo26m-cls.pt` for F01 and `yolo26s-cls.pt` for F02. Original upstream file hashes were not archived.

### Q9. What optimiser was used?

The saved request is `optimizer=auto`. The resolved runtime optimiser remains To be verified.

### Q10. Did the models use class-balanced focal loss?

No. They used the native Ultralytics classification objective without custom class weights or custom focal loss.

### Q11. What was the checkpoint-selection metric?

Native Ultralytics validation classification fitness based on the mean of validation Top-1 and Top-5.

### Q12. Was Macro F1 used to choose F01/F02 checkpoints?

No.

### Q13. What was F01's best epoch?

Epoch 2.

### Q14. How many epochs did F01 complete?

32 of the maximum 200.

### Q15. What happened in F02's training history?

The first maximum fitness occurred at epoch 38, an equal maximum occurred at epoch 49, and the preserved best checkpoint corresponds to epoch 49. The run completed 68 epochs.

### Q16. Why can the tie not be explained more precisely?

The exact version-specific tie handling and early-stopping counter semantics were not fully archived.

### Q17. What were the Top-1 results?

F01 achieved 84.6774%; F02 achieved 83.4367% on the unchanged 1,612-image test set.

### Q18. Which metric showed the largest F01 advantage?

Macro Recall, with F01 ahead by 8.7331 percentage points.

### Q19. What was the Macro F1 difference?

F01 was ahead by 5.5999 percentage points.

### Q20. Why are Macro metrics important here?

They weight each class equally and therefore expose minority-class weakness that can be hidden by aggregate accuracy and Weighted F1.

### Q21. Which F02 classes were especially weak?

Flatworm, marine dolphin, sea cucumber, octopus, and eel had recalls of 23.08%, 30%, 30%, 40%, and 48.78%, respectively.

### Q22. Does F02 prove that small models are unsuitable for marine classification?

No. It is one configuration and one completed run. It shows a negative result within this controlled campaign only.

### Q23. Was the official test used for tuning?

No. It was evaluated only after checkpoint selection and locking.

### Q24. How was package integrity protected?

Both ZIPs, extracted trees, and `best.pt` checkpoints have frozen SHA-256 values; predictions and metrics were independently recomputed.

### Q25. Were all YAML fields operationally relevant?

No. Ultralytics saves a broad cross-task schema. Detection, segmentation, export, tracking, and display fields were separated from the effective classification pipeline.

### Q26. Does `augment: false` mean no training augmentation?

No. It is an inference-augmentation setting. Training used RandomResizedCrop, horizontal flip, RandAugment, and random erasing.

### Q27. Why is identity normalisation reported?

The effective Ultralytics classification pipeline scaled tensors to `[0,1]` and used mean `[0,0,0]`, standard deviation `[1,1,1]`, rather than ImageNet mean/std.

### Q28. Can the runs be reproduced bit-for-bit?

That cannot be guaranteed because complete run-local environment, worker-state, container, and backend records were not uniformly preserved.

### Q29. What is F01's later campaign role?

It is the frozen YOLO source used in the F08 heterogeneous probability ensemble.

### Q30. What is the main scientific conclusion of Phase 4?

Under the same Aug-200 YOLO protocol, YOLO26m was consistently stronger than YOLO26s, and the scale reduction caused a much larger loss in balanced-class recall/F1 than in headline Top-1 accuracy.

---

## 18. Phase 4 completion decision

Phase 4 is complete because the following are now frozen:

- official F01/F02 identities and roles;
- complete saved-argument categorisation;
- pretrained-identifier claim boundary;
- effective online transforms;
- training and stopping chronology;
- validation-fitness selection rule;
- best/tied-best epochs;
- checkpoint, package, and tree hashes;
- official-test Top-k and classification metrics;
- verified prediction/CM support;
- F02 weak-class findings;
- controlled F01-versus-F02 interpretation;
- unresolved values and prohibited claims;
- viva/reviewer question bank.
