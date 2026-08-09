# F06 — ConvNeXt-Small HR384 Gentle Class-Balanced Focal Continuation

## 1. Experiment identity and status

**Official experiment ID**

`F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL`

**Repository file**

`documentation/experiments/F06_convnext_hr384.md`

**Campaign**

`04-07-2026 AQUA20 FINAL BEAT BASE PLAN`

**Scientific role**

F06 is the high-resolution continuation of the frozen F03 imbalance-aware ConvNeXt branch. It is the campaign's **primary proposed single model** and **strongest balanced-class model**.

Approved interpretation:

> Continuing the selected F03 checkpoint at 384×384 with additional low-learning-rate training produced the campaign's strongest balanced single-model result.

Prohibited interpretation:

> Increasing resolution alone caused the improvement.

F06 is a continuation-stage complete configuration. Relative to F03, it changes resolution, adds continuation training, lowers the learning rate, reduces batch size, and uses a new training budget.

---

## 2. Evidence hierarchy used for this record

Evidence was reconciled in the following order:

1. latest final manuscript;
2. FINAL evidence freeze and frozen terminology/index records;
3. direct F06 campaign notebook and `README_F06.txt`;
4. F06 locked protocol/configuration, training history, termination record, validation summary, checkpoint metadata, predictions, report, and confusion-matrix evidence;
5. F03 source-checkpoint/package evidence and the frozen F03 documentation;
6. package/tree integrity records;
7. earlier audits and handovers for historical context only.

Direct run-level F06 evidence takes priority where it resolves older incomplete registry fields.

The official F02 identity remains:

`F02_YOLO26S_CLS_AUG200`

F02 must never be relabelled as a ConvNeXt-Small cross-entropy experiment.

---

## 3. Why F06 continued F03 rather than F05

F03 is the original-distribution ConvNeXt-Small branch trained with Gentle Class-Balanced Focal Loss and is frozen as the campaign's **best isolated imbalance intervention**.

F05 combines Aug-200 with the same loss family and is frozen as a completed combined-intervention negative/over-correction ablation. Its test Macro F1 was lower than the F03 result, and its scientific role was not to become the continuation source.

F06 therefore continued the frozen validation-selected F03 checkpoint rather than F05.

This choice preserves the campaign hierarchy:

- F03: best isolated imbalance intervention;
- F05: combined Aug-200 + loss-level negative ablation;
- F06: continuation of the selected F03 branch and primary proposed single model.

This does not prove that the F03 ingredients are independently causal. It records which branch was selected and continued in the completed campaign.

---

## 4. Frozen F03 source checkpoint

| Field | Locked value |
|---|---|
| Source experiment | `F03_CONVNEXT_SMALL_GENTLE_CB_FOCAL` |
| Source role | Validation-selected F03 checkpoint |
| Source best epoch | 13 |
| Source resolution | `224×224` |
| Source checkpoint | `weights/best_model.pth` |
| Source SHA-256 | `928c8af60aef70537263e2c8cc1cf13782747bc5974f86ff054fbe352065c26f` |
| Source checkpoint bytes | 198,009,769 |

The F03 checkpoint was a dictionary containing the model state, validation metrics, class names, and training configuration. Direct inspection found 344 state tensors and a 20-class classifier with:

```text
classifier.0.weight : (768,)
classifier.0.bias   : (768,)
classifier.2.weight : (20, 768)
classifier.2.bias   : (20,)
```

The F06 reconstruction therefore retained the ConvNeXt-Small 20-class architecture and loaded the frozen F03 model state as the continuation source.

No claim is made that F06 independently restarted from a new ImageNet-pretrained checkpoint. Its scientific initialization is the frozen F03 checkpoint.

---

## 5. Dataset, split, and class-order protocol

| Item | Locked F06 value |
|---|---:|
| Dataset | AQUA20 |
| Classes | 20 |
| Training distribution | Original, naturally imbalanced |
| Training images | 5,247 |
| Validation images | 1,312 |
| Official test images | 1,612 |
| Aug-200 | Disabled |
| Fine-tuning resolution | `384×384` |
| Campaign/split seed | 42 |

No Aug-200 synthetic expansion was used in F06.

The validation and official test partitions remained unchanged. The official test split was excluded from training, manual stopping decisions, and checkpoint selection.

### Frozen class order

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

---

## 6. HR384 preprocessing

### 6.1 Training transform

The effective F06 training pipeline is frozen as:

```text
Resize 440×440
→ RandomResizedCrop 384×384
   scale = 0.80–1.00
   ratio = 0.75–1.3333
→ RandomHorizontalFlip p=0.5
→ RandomRotation −10° to +10°
→ ColorJitter
   brightness = 0.88–1.12
   contrast   = 0.88–1.12
   saturation = 0.88–1.12
   hue        = −0.03 to +0.03
→ ToTensor
→ ImageNet normalisation
```

For the preserved ConvNeXt transform family:

- resize/crop used bilinear interpolation with antialiasing;
- rotation used nearest-neighbour interpolation;
- rotation expansion was disabled;
- rotation fill was 0.

### 6.2 Validation and official-test transform

```text
Direct Resize 384×384
→ ToTensor
→ ImageNet normalisation
```

There was:

- no CenterCrop;
- no random validation augmentation;
- no random official-test augmentation in F06.

### 6.3 ImageNet normalisation

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

F07 later inherits this deterministic 384×384 evaluation pipeline, but F07 is a separate inference-only experiment.

---

## 7. Gentle Class-Balanced Focal objective

F06 retained the Gentle Class-Balanced Focal objective from the original-distribution F03 branch.

Constants:

```text
beta  = 0.999
gamma = 1.0
```

For class \(y\) containing \(n_y\) training samples, the raw effective-number weight is

\[
\widetilde{w}_y =
\frac{1-\beta}{1-\beta^{n_y}}.
\]

The weights are normalised to mean one:

\[
w_y =
\frac{\widetilde{w}_y}
{\operatorname{mean}(\widetilde{w})}.
\]

The per-sample objective is

\[
\mathcal{L}
=
-w_y(1-p_y)^\gamma \log(p_y).
\]

Preserved implementation path:

```text
unreduced cross-entropy
→ p_t = exp(−CE)
→ focal factor = (1−p_t)^gamma
→ target-indexed class weight
→ per-sample multiplication
→ mean reduction
```

“Gentle” refers only to the moderate focal exponent `gamma=1.0`. The loss components are not claimed as a newly invented loss family.

No label smoothing, inverse-frequency weighting, or weighted sampler is introduced into the frozen F06 description.

---

## 8. Original-distribution class counts and F06 class weights

The source F03 class-weight table is the F06 weight basis because F06 uses the same original 5,247-image training distribution.

| ID | Class | Train count | Normalised CB weight |
|---:|---|---:|---:|
| 00 | coral | 1,249 | 0.062978 |
| 01 | crab | 34 | 1.343341 |
| 02 | diver | 41 | 1.117873 |
| 03 | eel | 128 | 0.373767 |
| 04 | fish | 1,760 | 0.054254 |
| 05 | fishInGroups | 218 | 0.229268 |
| 06 | flatworm | 40 | 1.145251 |
| 07 | jellyfish | 78 | 0.598467 |
| 08 | marine_dolphin | 16 | 2.829116 |
| 09 | octopus | 16 | 2.829116 |
| 10 | rayfish | 305 | 0.170836 |
| 11 | seaAnemone | 712 | 0.088178 |
| 12 | seaCucumber | 28 | 1.626336 |
| 13 | seaSlug | 63 | 0.735485 |
| 14 | seaUrchin | 92 | 0.510912 |
| 15 | shark | 57 | 0.810493 |
| 16 | shrimp | 18 | 2.517280 |
| 17 | squid | 19 | 2.385981 |
| 18 | starfish | 133 | 0.360597 |
| 19 | turtle | 240 | 0.210468 |

The mean class weight is 1.0.

The largest frozen weights belong to `marine_dolphin` and `octopus`, each with only 16 training images. The smallest belongs to `fish`, which has 1,760 training images.

---

## 9. Training configuration and runtime controls

### 9.1 Locked F06 configuration

| Field | Locked value |
|---|---|
| Source | Frozen F03 epoch-13 checkpoint |
| Resolution | `384×384` |
| Batch size | 8 |
| Maximum epochs | 15 |
| Retained fully completed epochs | 9 |
| Optimiser | AdamW |
| Learning rate | `5e-6` |
| Weight decay | `1e-4` |
| Scheduler | CosineAnnealingLR |
| Scheduler `T_max` | **To be verified** |
| Scheduler `eta_min` | **To be verified** |
| Loss | Gentle Class-Balanced Focal |
| `beta` | 0.999 |
| `gamma` | 1.0 |
| Gradient clipping | Maximum norm 1.0 |
| AMP | Enabled |
| Seed | 42 |
| Checkpoint metric | Validation Macro F1 |
| Workers | **To be verified** |
| Full run-specific backend trace | **To be verified** |

A preserved F06 training output recorded:

```text
GPU: Tesla P100-PCIE-16GB
resolution: 384×384
batch size: 8
train batches: 656
validation batches: 164
maximum epochs: 15
learning rate: 5e-6
```

The campaign notebook also preserves a repaired PyTorch CUDA 11.8 stack, but a complete immutable F06 run-local software/container/driver manifest is not frozen. Therefore exact reproducibility claims must remain bounded.

### 9.2 Optimizer and schedule boundaries

The evidence freezes the optimizer family, learning rate, weight decay, and scheduler family.

The following must not be reconstructed from generic defaults unless direct run-local evidence is added:

- AdamW `betas`;
- AdamW `eps`;
- AdamW `amsgrad`;
- exact parameter groups;
- exact F06 `T_max`;
- exact F06 `eta_min`;
- warm-up status;
- gradient-accumulation status.

---

## 10. Epoch chronology, interruption, and manual termination

### 10.1 Planned budget

The maximum planned F06 budget was 15 epochs.

### 10.2 Retained completed training

The frozen history contains:

- fully completed epochs: **1–9**;
- retained completed epochs: **9**;
- interrupted epoch: **10**;
- epoch 10 included in the official history: **No**.

Only fully completed epochs were retained as official training evidence.

### 10.3 Best checkpoint

Checkpoint selection used validation Macro F1.

The best F06 epoch was:

```text
epoch = 2
validation Macro F1 = 88.5122%
```

Seven fully completed epochs followed the best epoch without replacing it.

The frozen checkpoint was independently verified as epoch 2.

### 10.4 Manual termination versus early stopping

F06 must **not** be described as having triggered a normal archived patience-based early-stopping rule.

The run was manually terminated after the validation Macro-F1 trajectory had plateaued. Epoch 10 was interrupted and excluded.

Therefore the correct terminology is:

> manual validation-plateau termination after nine fully completed epochs, with interrupted epoch 10 excluded.

The official test split was not used to decide when to stop.

### 10.5 Test isolation

The official test split was evaluated only after the F06 epoch-2 checkpoint was locked.

This excludes test-based:

- training;
- stopping;
- checkpoint replacement;
- configuration tuning.

---

## 11. Frozen F06 validation result

Best epoch: **2**

| Validation metric | Value |
|---|---:|
| Top-1 | 92.9878% |
| Top-2 | 98.6280% |
| Top-3 | 99.5427% |
| Top-5 | 99.7713% |
| Macro Precision | 88.4581% |
| Macro Recall | 90.6790% |
| Macro F1 | 88.5122% |
| Weighted F1 | 93.0475% |

Important interpretation boundary:

F06's best validation Macro F1 is not evidence that the F06 continuation improved the F03 validation Macro F1. The F03 source checkpoint had a frozen validation Macro F1 of 89.3301%. F06 is retained as the primary balanced single model based on the completed campaign's official-test reporting hierarchy, not because its validation Macro F1 exceeded F03.

---

## 12. Frozen F06 official-test result

Official unchanged test support: **1,612**

| Test metric | Value |
|---|---:|
| Top-1 | 92.1216% |
| Top-2 | 97.9529% |
| Top-3 | 99.2556% |
| Top-5 | 99.7519% |
| Macro Precision | 90.4312% |
| Macro Recall | 87.4362% |
| Macro F1 | 88.1699% |
| Weighted F1 | 92.0810% |

Weighted F1 exceeds Macro F1 by approximately **3.9111 percentage points**, showing that class-wise performance was still not uniform despite F06 being the campaign's strongest balanced single model.

The official package preserves:

- 1,612 test predictions;
- classification report;
- raw confusion matrix;
- row-normalised confusion matrix;
- aggregate Top-k and classification metrics.

These assets passed the frozen campaign audit.

---

## 13. Class-wise evidence and error structure

The final manuscript and R06 per-class analysis records support the following F06 class-wise interpretation.

### Comparatively strong classes

The per-class F1 analysis identifies the following as comparatively strong:

- starfish;
- turtle;
- diver;
- fish;
- rayfish.

### Weaker classes

The weakest F1 was reported for:

1. marine_dolphin;
2. octopus;
3. squid;
4. shark.

These classes have relatively small official-test supports, so a small number of changed predictions can produce large changes in precision, recall, and F1.

### Verified confusion directions

The row-normalised F06 confusion evidence shows examples including:

- marine_dolphin → shark;
- marine_dolphin → fishInGroups;
- octopus → eel;
- octopus → crab;
- octopus → fish.

These are descriptive error patterns from the frozen F06 predictions. They are not proof of biological similarity or a causal mechanism.

The exact 20-class report and confusion matrices remain the controlling numerical evidence for exhaustive per-class values.

---

## 14. F06 versus F03 — continuation-stage complete-configuration comparison

F03 official test:

- Top-1: 91.6253%;
- Top-5: 99.6898%;
- Macro Precision: 90.7850%;
- Macro Recall: 86.9302%;
- Macro F1: 87.9645%;
- Weighted F1: 91.5656%.

F06 minus F03:

| Metric | Difference |
|---|---:|
| Top-1 | +0.4963 pp |
| Top-5 | +0.0621 pp |
| Macro Precision | −0.3538 pp |
| Macro Recall | +0.5060 pp |
| Macro F1 | +0.2054 pp |
| Weighted F1 | +0.5154 pp |

Approved conclusion:

> The completed F06 continuation configuration improved Top-1, Macro Recall, Macro F1, and Weighted F1 relative to F03 on the official test split, while Macro Precision was slightly lower.

Required limitation:

F03→F06 is **not** a resolution-only ablation because F06 simultaneously changed:

- input resolution: 224→384;
- continuation stage;
- learning rate: `1e-5`→`5e-6`;
- batch size: 16→8;
- maximum training budget: 40→15 for the new continuation stage;
- training chronology/termination.

Do not write:

> 384×384 resolution caused the +0.2054 pp Macro-F1 improvement.

---

## 15. Why F06 remains primary although F08 has slightly higher Top-1

The final reporting hierarchy is frozen as:

```text
Primary proposed method / primary proposed single model:
F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL

Strongest balanced-class model:
F06

Complementary final ensemble:
F08_CONVNEXT_SMALL_HR384_YOLO26M_PROB_ENSEMBLE

Strongest ranking model:
F08
```

F08 achieved a slightly higher official-test Top-1:

```text
F08 Top-1 = 92.2457%
F06 Top-1 = 92.1216%
difference = +0.1241 pp for F08
```

However, F06 retained stronger balanced-class values:

```text
F06 Macro Recall = 87.4362%
F08 Macro Recall = 87.0170%
F06 advantage     = +0.4192 pp

F06 Macro F1 = 88.1699%
F08 Macro F1 = 87.9406%
F06 advantage = +0.2293 pp
```

F08 is also an inference-level heterogeneous ensemble of two frozen models, whereas F06 is one trained ConvNeXt-Small model and represents the core imbalance-aware continuation method.

Therefore:

- F06 remains the primary single-model scientific contribution;
- F08 is reported as a complementary ranking-oriented ensemble;
- the 0.1241 pp Top-1 difference must not be used to relabel F08 as the primary method.

---

## 16. Checkpoint and package integrity

### 16.1 F06 best checkpoint

| Field | Value |
|---|---|
| File | `weights/best_model.pth` |
| Best epoch | 2 |
| SHA-256 | `d0c225c28b25f09822bc99518a37f9b1fe642b2d01213da7f68e2d76a255832a` |
| Exact checkpoint byte size | **To be verified** |

### 16.2 F06 last checkpoint

| Field | Value |
|---|---|
| File | `weights/last_model.pth` |
| SHA-256 | `23d80cf55a3538cce9e39b2b214e469eb1214f77b455f27bf2dc831f20d1d96a` |

### 16.3 Official package

```text
aqua20_f06_convnext_small_hr384_gentle_focal_outputs.zip
```

Notebook-reported ZIP size:

```text
351.17 MB
```

This is a displayed approximate size, not a frozen exact byte count.

Frozen extracted-package integrity:

| Field | Value |
|---|---|
| Curated files | 19 |
| Extracted bytes | 397,397,341 |
| Extracted-tree SHA-256 | `c4b35588260023f1fcf9e08987a51fd249263bd9d61497e4525a68b8cf63c35a` |
| Original ZIP SHA-256 | **Not collected / To be verified** |
| Exact original ZIP byte count | **To be verified** |

### 16.4 Curated package contents

```text
F06_locked_protocol.json
README_F06.txt
figures/F06_confusion_matrix.png
figures/F06_confusion_matrix_normalized.png
figures/F06_validation_history.png
tables/F06_F03_F06_scientific_comparison.csv
tables/F06_best_epoch_summary.csv
tables/F06_campaign_target_comparison.csv
tables/F06_classification_report.csv
tables/F06_comparison_ready_summary.csv
tables/F06_confusion_matrix.csv
tables/F06_confusion_matrix_normalized.csv
tables/F06_test_metrics_summary.csv
tables/F06_test_predictions.csv
tables/F06_training_history.csv
tables/F06_training_termination.json
weights/F06_best_model_sha256.txt
weights/best_model.pth
weights/last_model.pth
```

---

## 17. Reproducibility boundaries and items still To be verified

The following are not reconstructed beyond direct evidence:

- F06 DataLoader worker count;
- exact F06 scheduler `T_max`;
- exact F06 scheduler `eta_min`;
- AdamW `betas`, `eps`, `amsgrad`, and exact parameter groups;
- gradient-accumulation status;
- warm-up status;
- complete worker-seeding/generator-state trace;
- complete F06 backend/cudNN deterministic/benchmark trace;
- immutable Kaggle/container image identifier;
- complete original F06 `pip freeze`;
- NVIDIA driver and complete `nvidia-smi`;
- direct native-cuDNN runtime version;
- exact run-local scikit-learn, NumPy, pandas, Pillow, and auxiliary-package versions;
- CPU, host RAM, and storage configuration;
- exact F06 best-checkpoint byte size;
- original F06 ZIP SHA-256;
- exact original F06 ZIP byte count;
- immutable dataset revision/fingerprint/content hash;
- split-manifest checksum;
- complete duplicate/near-duplicate and decode-audit boundaries;
- per-image random augmentation draws;
- model parameter count under one controlled reporting convention;
- FLOPs under one controlled convention;
- peak memory;
- inference latency;
- throughput;
- energy/compute cost;
- repeated-seed optimisation variance.

A complete bitwise-reproduction claim is therefore not permitted.

---

## 18. Allowed and prohibited claims

### Allowed

- F06 is the high-resolution continuation of the frozen F03 branch.
- F06 used the original 5,247-image training distribution and no Aug-200.
- F06 continued the F03 epoch-13 checkpoint whose SHA-256 is frozen.
- F06 used 384×384 inputs, batch 8, AdamW, `5e-6` learning rate, `1e-4` weight decay, CosineAnnealingLR, Gentle CB Focal, gradient clipping 1.0, AMP, and seed 42 where directly frozen.
- F06 had nine retained completed epochs; interrupted epoch 10 was excluded.
- F06 was manually terminated after validation-Macro-F1 plateau behaviour; this is not described as ordinary patience-based early stopping.
- F06 checkpoint selection used validation Macro F1 only.
- F06 official-test Top-1 was 92.1216% and Macro F1 was 88.1699%.
- F06 is the campaign's primary proposed single model and strongest balanced-class model.
- F08 has a slightly higher Top-1 point estimate but remains complementary.

### Prohibited

- “Resolution alone caused F06 to improve.”
- “F06 proved 384×384 is superior to 224×224.”
- “F06 validation Macro F1 improved over F03.”
- “F06 used F05 as its source checkpoint.”
- “F06 used Aug-200.”
- “Epoch 10 was a completed official epoch.”
- “F06 early stopping automatically triggered at epoch 10.”
- “The official test was used to decide when to stop.”
- “F08 is the primary method because its Top-1 is higher.”
- “F06 is globally state of the art.”
- “The archived run is fully bitwise reproducible.”

---

## 19. Reviewer / viva question bank

### Q1. What exactly is F06?

F06 is `F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL`, a ConvNeXt-Small continuation experiment initialized from the frozen F03 checkpoint and fine-tuned at 384×384 with Gentle Class-Balanced Focal Loss.

### Q2. Why was F03 selected as the source?

F03 is the campaign's frozen best isolated imbalance intervention on the original training distribution. F05 was a combined Aug-200 + focal configuration and was retained as a negative/over-correction ablation rather than the continuation source.

### Q3. What checkpoint initialized F06?

The F03 epoch-13 validation-selected checkpoint with SHA-256 `928c8af60aef70537263e2c8cc1cf13782747bc5974f86ff054fbe352065c26f`.

### Q4. Did F06 train from scratch?

No. It continued the frozen F03 model state.

### Q5. Did F06 use Aug-200?

No. F06 used the original 5,247-image naturally imbalanced training split.

### Q6. What were the validation and test sizes?

1,312 validation images and 1,612 unchanged official-test images.

### Q7. What changed from F03 to F06?

Resolution, continuation training, learning rate, batch size, and training budget changed.

### Q8. Why is F03→F06 not a resolution-only ablation?

Because several training factors changed simultaneously. The comparison is between complete configurations.

### Q9. What was the F06 input resolution?

384×384.

### Q10. What was the training preprocessing?

Resize to 440, RandomResizedCrop to 384 with scale 0.80–1.00, horizontal flip, ±10° rotation, ColorJitter, tensor conversion, and ImageNet normalisation.

### Q11. What was the evaluation preprocessing?

Direct resize to 384×384, tensor conversion, and ImageNet normalisation, with no random evaluation augmentation.

### Q12. What loss did F06 use?

Gentle Class-Balanced Focal Loss with `beta=0.999` and `gamma=1.0`, using effective-number weights normalised to mean one.

### Q13. Why is it called “gentle”?

Only because the focal exponent is the moderate value `gamma=1.0`. The term does not imply a new loss family.

### Q14. Which classes received the largest class weights?

`marine_dolphin` and `octopus`, each with 16 training images and weight 2.829116.

### Q15. What was the F06 learning rate?

`5e-6`.

### Q16. What was the batch size?

8.

### Q17. What was the maximum planned epoch budget?

15 epochs.

### Q18. How many epochs were officially completed?

Nine fully completed epochs.

### Q19. What happened to epoch 10?

It was manually interrupted and excluded from the official history.

### Q20. Was F06 stopped by early stopping?

The archived evidence supports manual validation-plateau termination, not a normal patience-based early-stopping trigger.

### Q21. What metric selected the checkpoint?

Validation Macro F1.

### Q22. What was the best epoch?

Epoch 2.

### Q23. What was the best validation Macro F1?

88.5122%.

### Q24. Was the official test used for stopping or model selection?

No.

### Q25. What is the official-test Top-1?

92.1216%.

### Q26. What is the official-test Macro F1?

88.1699%.

### Q27. How did F06 compare with F03 on Macro F1?

F06 was +0.2054 percentage points higher on the official test, but this is a complete-configuration difference, not a resolution-only effect.

### Q28. Which F06 classes were comparatively strongest?

Starfish, turtle, diver, fish, and rayfish were reported as comparatively strong in the per-class F1 analysis.

### Q29. Which F06 classes were weakest?

Marine dolphin was lowest, followed by octopus, squid, and shark.

### Q30. What notable confusion directions were observed?

Examples include marine dolphin to shark/fishInGroups and octopus to eel/crab/fish.

### Q31. Why can minority-class metrics change sharply?

Several weak classes have small test supports, so one or two prediction changes can materially alter recall or F1.

### Q32. Why is F06 still primary when F08 has higher Top-1?

F06 is the single-model scientific contribution and has stronger Macro Recall and Macro F1. F08 is a complementary heterogeneous ensemble with a small Top-1 advantage.

### Q33. How much higher is F08 Top-1 than F06?

Approximately 0.1241 percentage points.

### Q34. Which has higher Macro F1, F06 or F08?

F06: 88.1699% versus F08: 87.9406%.

### Q35. What is the F06 best-checkpoint SHA-256?

`d0c225c28b25f09822bc99518a37f9b1fe642b2d01213da7f68e2d76a255832a`.

### Q36. Is the original F06 ZIP SHA-256 available?

No. It was not collected; the extracted-tree SHA-256 is frozen instead.

### Q37. What is the extracted-tree SHA-256?

`c4b35588260023f1fcf9e08987a51fd249263bd9d61497e4525a68b8cf63c35a`.

### Q38. Can exact bitwise reproduction be claimed?

No. Several run-local environment, backend, scheduler, worker, and immutable-source details remain unresolved.

---

## 20. Phase 8 completion decision

Phase 8 is complete because this record now freezes:

- exact F06 identity and scientific role;
- F03 source identity, epoch, checkpoint structure, and SHA-256;
- rationale for continuing F03 rather than F05;
- original 5,247-image training distribution and explicit absence of Aug-200;
- unchanged 1,312 validation and 1,612 official test splits;
- exact HR384 training/evaluation preprocessing;
- Gentle Class-Balanced Focal formulation and original-distribution class weights;
- core F06 training configuration;
- 15-epoch plan, nine retained completed epochs, interrupted epoch-10 exclusion, and manual termination;
- validation-Macro-F1 checkpoint selection and best epoch 2;
- complete frozen validation and official-test metrics;
- class-wise evidence and bounded confusion interpretation;
- F03→F06 complete-configuration comparison;
- F06/F08 reporting hierarchy;
- checkpoint/package/tree integrity;
- unresolved reproducibility fields;
- reviewer/viva question bank.

The next documentation phase is F07, the deterministic two-view inference-only TTA extension of the frozen F06 checkpoint.
