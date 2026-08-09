# D05 — AQUA20 F03 ConvNeXt-Small Gentle Class-Balanced Focal Experiment Documentation

## Document status

**Phase:** 5 — F03 ConvNeXt-Small Gentle Class-Balanced Focal Experiment Documentation  
**Status:** Completed and locked  
**Canonical manuscript:** *Improving Underwater Marine Species Classification through Imbalance-Aware Learning and Cross-Dataset Evaluation*  
**Official experiment identity:** `F03_CONVNEXT_SMALL_GENTLE_CB_FOCAL`

---

## 1. Scope and source-control decision

This document records only the F03 experiment. It follows the locked source hierarchy:

1. latest final manuscript;
2. FINAL evidence-freeze and frozen-terminology/index records;
3. official F03 package, README, configuration, history, loss/weight records, checkpoint metadata, predictions, reports, confusion matrices, and package manifests;
4. preserved campaign notebook;
5. older drafts and audits for historical context only.

Where an older record conflicts with the final hierarchy, the final manuscript and FINAL evidence-freeze take precedence.

### Mandatory identity boundary

F03 is:

```text
F03_CONVNEXT_SMALL_GENTLE_CB_FOCAL
```

It is the ConvNeXt-Small branch trained on the **original 5,247-image training distribution** with Gentle Class-Balanced Focal Loss at `224×224`.

F02 remains:

```text
F02_YOLO26S_CLS_AUG200
```

F02 must never be relabelled as a ConvNeXt-Small cross-entropy baseline. The preserved ConvNeXt-Small + Aug-200 + standard cross-entropy experiment is F04.

---

## 2. Scientific role and permitted interpretation

F03 is the campaign's **loss-level imbalance-aware ConvNeXt branch** and is retained in the final reporting hierarchy as the **strongest isolated imbalance intervention within the tested campaign**.

This wording is intentionally bounded. The collected campaign does **not** contain an architecture-matched ConvNeXt-Small + standard cross-entropy experiment trained on the same original 5,247-image distribution. Therefore:

- F03 may be described as the strongest isolated imbalance intervention among the tested configurations;
- F03 may not be claimed to quantify the causal benefit of Gentle Class-Balanced Focal Loss over an original-distribution ConvNeXt-Small cross-entropy control;
- F03 versus F04 changes both the loss and the training distribution;
- F03 versus F05 changes the training distribution and maximum training budget;
- F03 versus F06 changes resolution, continuation training, learning rate, batch size, and training budget.

F03 is not the final campaign-winning model. It is the validated source checkpoint later continued by F06.

---

## 3. Dataset and split protocol

| Item | Locked value |
|---|---:|
| Dataset | AQUA20 |
| Classes | 20 |
| Training distribution | Original, naturally imbalanced |
| Training images | 5,247 |
| Validation images | 1,312 |
| Official test images | 1,612 |
| Input resolution | `224×224` |
| Split seed | 42 |
| Aug-200 used in F03 | No |

The validation and official test partitions remained unchanged. F03 training, early stopping, and checkpoint selection did not use the official test split. The test split was evaluated only after the epoch-13 validation-selected checkpoint had been locked and preserved.

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

## 4. Model construction and pretrained weights

F03 used torchvision ConvNeXt-Small:

```python
weights = ConvNeXt_Small_Weights.DEFAULT
model = convnext_small(weights=weights)
original_in_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(original_in_features, 20)
```

Under the preserved torchvision environment, `ConvNeXt_Small_Weights.DEFAULT` resolved to:

```text
ConvNeXt_Small_Weights.IMAGENET1K_V1
```

Locked construction facts:

| Field | Value |
|---|---|
| Constructor | `torchvision.models.convnext_small` |
| Pretrained weights | `ConvNeXt_Small_Weights.IMAGENET1K_V1` |
| Classifier input features | 768 |
| Classifier output features | 20 |
| Training mode | End-to-end fine-tuning; no frozen-backbone claim is made |

The immutable upstream pretrained-weight file hash and complete upstream provenance record were not archived and remain **To be verified**.

---

## 5. Effective online transforms

### 5.1 Training transform

```text
Resize(256×256)
→ RandomResizedCrop(
      224,
      scale=0.80–1.00,
      ratio=0.75–1.3333
  )
→ RandomHorizontalFlip(p=0.5)
→ RandomRotation(−10° to +10°)
→ ColorJitter
→ ToTensor
→ ImageNet normalisation
```

ColorJitter ranges:

- brightness factor: 0.88–1.12;
- contrast factor: 0.88–1.12;
- saturation factor: 0.88–1.12;
- hue: −0.03 to +0.03.

Interpolation details:

- resize and crop: bilinear with antialiasing;
- rotation: nearest-neighbour;
- rotation expansion: disabled;
- rotation fill: 0.

### 5.2 Validation and official-test transform

```text
Resize(224×224)
→ ToTensor
→ ImageNet normalisation
```

This is a direct tuple resize. It is not a shorter-edge resize followed by CenterCrop. No random validation or test augmentation was used.

### 5.3 Normalisation

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

## 6. Gentle Class-Balanced Focal Loss

### 6.1 Terminology boundary

The word **Gentle** refers only to the moderate focal exponent:

```text
gamma = 1.0
```

The effective-number weighting, focal modulation, and cross-entropy components are established methods. The campaign does not claim a new loss-family invention.

### 6.2 Effective-number class weight

For class `y` with `n_y` training examples:

\[
\widetilde{w}_y
=
\frac{1-\beta}{1-\beta^{n_y}}.
\]

F03 used:

\[
\beta=0.999.
\]

The raw effective-number weights were normalised to a mean of one:

\[
w_y
=
\frac{\widetilde{w}_y}
{\frac{1}{C}\sum_{c=1}^{C}\widetilde{w}_c},
\qquad C=20.
\]

This additional normalisation keeps the average class-weight scale at one. It is a scaling choice, not a novel contribution.

### 6.3 Per-sample loss

For logits `z`, target `y`, and correct-class probability `p_y`:

\[
\mathcal{L}_{\mathrm{F03}}
=
-w_y(1-p_y)^{\gamma}\log(p_y),
\qquad
\gamma=1.0.
\]

The preserved implementation is equivalent to:

```python
ce_loss = F.cross_entropy(logits, targets, reduction="none")
pt = torch.exp(-ce_loss)
focal_factor = torch.pow(1.0 - pt, gamma)
sample_weights = class_weights[targets]
loss = sample_weights * focal_factor * ce_loss
return loss.mean()
```

Implementation implications:

- standard un-smoothed per-sample cross-entropy base;
- correct-class probability reconstructed as `exp(-CE)`;
- class weight selected by the target label;
- focal factor applied per sample;
- final batch reduction by arithmetic mean;
- no raw inverse-frequency formula substituted for effective-number weighting.

### 6.4 Frozen original-distribution class weights

The following values are reconstructed from the frozen F03 training counts, `beta=0.999`, and mean-one normalisation. Values are rounded to six decimals.

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

The weights have mean 1.0. The highest weights belong to the two 16-image classes, `marine_dolphin` and `octopus`; the lowest belongs to the 1,760-image `fish` class. This reflects the intended effective-number rebalancing of the original class distribution.

---

## 7. Training configuration

| Field | Locked F03 value |
|---|---|
| Maximum epochs | 40 |
| Completed epochs | 21 |
| Batch size | 16 |
| Workers | 4 |
| Prefetch factor | 2 |
| Optimiser | AdamW |
| Initial learning rate | `1e-5` |
| Weight decay | `1e-4` |
| Scheduler | CosineAnnealingLR |
| Scheduler `T_max` | 40 |
| Scheduler `eta_min` | **To be verified** |
| Early-stopping patience | 8 |
| Checkpoint metric | Validation Macro F1 |
| Checkpoint comparison | Strict `>` improvement |
| Gradient clipping | Maximum norm 1.0 |
| AMP | Enabled |
| Recovery checkpoint | Every 5 epochs |
| Seed | 42 |
| cuDNN deterministic | True |
| cuDNN benchmark | False |
| Recorded GPU | Tesla P100-PCIE-16GB |

The reviewed evidence does not explicitly freeze AdamW `betas`, `eps`, `amsgrad`, or any custom parameter-group separation. They must not be reconstructed from generic defaults without the exact run-local source/version context.

No warm-up stage or gradient accumulation is claimed because these fields were not explicitly frozen for F03.

---

## 8. Epoch chronology, model selection, and early stopping

### 8.1 Selection rule

At each epoch, validation Top-1/2/3 and Macro Precision/Recall/F1 were computed. The best checkpoint condition was:

```python
is_best = val_macro_f1 > best_val_macro_f1
```

Therefore:

- only a **strictly greater** validation Macro F1 replaced the best checkpoint;
- an equal value did not replace it;
- on improvement, `epochs_without_improvement` reset to zero;
- otherwise, the counter increased by one.

The official test set was not involved.

### 8.2 Locked chronology

| Item | Value |
|---|---:|
| Planned maximum | 40 epochs |
| Best validation epoch | 13 |
| Completed epochs | 21 |
| Patience | 8 consecutive non-improving epochs |
| Stop condition | `epochs_without_improvement >= 8` |

Epoch 13 uniquely maximised validation Macro F1. Epochs 14 through 21 did not exceed it. At the end of epoch 21, the non-improvement counter reached eight and the training loop stopped.

The scheduler was stepped once per completed epoch after checkpoint evaluation. A recovery checkpoint containing model, optimiser, scheduler, scaler, best-epoch, best-score, counter, and training configuration was saved every five epochs.

### 8.3 Checkpoint preservation sequence

1. train and validate using the original training distribution;
2. select the strictly best validation Macro-F1 checkpoint;
3. stop after eight consecutive non-improving epochs;
4. preserve epoch-13 `best_model.pth` and its hash;
5. audit checkpoint metadata against history/configuration;
6. only then evaluate the unchanged official test split.

This sequence preserves validation-only model selection and prevents test leakage.

---

## 9. Locked validation result

The epoch-13 checkpoint produced:

| Validation metric | Value |
|---|---:|
| Top-1 | 92.5305% |
| Top-2 | 97.6372% |
| Top-3 | 99.4665% |
| Macro Precision | 87.9929% |
| Macro Recall | 92.6189% |
| Macro F1 | 89.3301% |

Selection was based only on **Validation Macro F1 = 89.3301%**. Top-1, Top-2, and Top-3 were reporting metrics, not checkpoint-selection substitutes.

---

## 10. Official unchanged test result

| Test metric | Value |
|---|---:|
| Top-1 | 91.6253% |
| Top-2 | 97.7047% |
| Top-3 | 99.1315% |
| Top-5 | 99.6898% |
| Macro Precision | 90.7850% |
| Macro Recall | 86.9302% |
| Macro F1 | 87.9645% |
| Weighted Precision | 91.8777% |
| Weighted Recall | 91.6253% |
| Weighted F1 | 91.5656% |
| Support | 1,612 |

The saved 1,612 prediction rows independently reproduced the Top-k and classification metrics. The classification report and confusion matrices passed the package audit.

### Metric interpretation

Weighted F1 exceeded Macro F1 by:

```text
91.5656% − 87.9645% = 3.6011 percentage points
```

This gap confirms that performance remained uneven across classes despite the imbalance-aware objective. F03 improved balanced recognition relative to the earlier YOLO Aug-200 branch, but it did not eliminate class-wise variation.

---

## 11. Class-wise behaviour and error evidence

The official package preserves:

- the complete 20-class classification report;
- raw and row-normalised confusion matrices;
- per-image predictions and probability evidence;
- comparison-ready tables.

### Locked minority-class observation

Flatworm test recall:

| Configuration | Flatworm recall |
|---|---:|
| F01 — YOLO26m-cls + Aug-200 | 15.38% |
| F02 — YOLO26s-cls + Aug-200 | 23.08% |
| F03 — ConvNeXt-Small + Gentle CB Focal | 76.92% |

The F03 flatworm recall was:

- `+61.54` percentage points above F01;
- `+53.84` percentage points above F02.

This is a strong configuration-level minority-class result. It should not be written as proof that the loss alone caused the improvement because architecture and training protocol also differ from F01/F02.

### Support caution

Flatworm has only 13 official-test samples. A one-image change shifts recall by approximately 7.69 percentage points. Per-class results for low-support categories must therefore be interpreted cautiously and should be accompanied by support counts when discussed.

No unsupported complete ranking of the 20 classes is introduced here. The archived `F03_classification_report.csv` and confusion matrices remain the authoritative source for any later exhaustive class-wise analysis.

---

## 12. Valid comparison boundaries

### 12.1 F03 versus F04

Frozen selected differences, calculated as F03 minus F04:

| Metric | Difference |
|---|---:|
| Top-1 | +0.2481 pp |
| Top-5 | +0.0620 pp |
| Macro Precision | +1.4609 pp |
| Macro Recall | +2.6066 pp |
| Macro F1 | +1.9610 pp |
| Weighted F1 | +0.3015 pp |

**Permitted conclusion:** F03 produced the stronger balanced result among these evaluated complete configurations.

**Not permitted:** “Gentle Class-Balanced Focal Loss caused a 1.961-point Macro-F1 improvement over cross-entropy.”

Reason: F03 used the original 5,247-image distribution and richer RandomResizedCrop/ColorJitter transforms, whereas F04 used the 7,284-image Aug-200 distribution, cross-entropy, and a different training transform.

### 12.2 F03 versus F05

Frozen selected differences, calculated as F03 minus F05:

| Metric | Difference |
|---|---:|
| Top-1 | +0.4963 pp |
| Top-5 | −0.0621 pp |
| Macro Precision | +1.0354 pp |
| Macro Recall | +4.2755 pp |
| Macro F1 | +3.6158 pp |
| Weighted F1 | +0.5897 pp |

F03 and F05 used the same loss family but different training distributions and different maximum budgets, 40 versus 25 epochs. The result supports F03 as the stronger tested isolated loss-level branch; it does not establish a universal rule that Aug-200 and class-balanced focal loss are incompatible.

### 12.3 F03 versus F06

F06 continued the frozen F03 checkpoint at `384×384`. F06 minus F03:

| Metric | Difference |
|---|---:|
| Top-1 | +0.4963 pp |
| Top-5 | +0.0621 pp |
| Macro Precision | −0.3538 pp |
| Macro Recall | +0.5060 pp |
| Macro F1 | +0.2054 pp |
| Weighted F1 | +0.5154 pp |

**Permitted conclusion:** continuing the selected F03 checkpoint at high resolution with additional low-learning-rate training produced the campaign's strongest balanced single-model configuration.

**Not permitted:** “Increasing resolution alone improved F03.”

F06 also changed continuation training, learning rate, batch size, and training budget.

---

## 13. Checkpoint and package integrity

### 13.1 Locked checkpoint

| Field | Value |
|---|---|
| File | `weights/best_model.pth` |
| Best epoch | 13 |
| SHA-256 | `928c8af60aef70537263e2c8cc1cf13782747bc5974f86ff054fbe352065c26f` |
| Bytes | 198,009,769 |

The checkpoint metadata retains the epoch, model state, locked validation metrics, class names, and training configuration. The safety/audit workflow recomputed the hash and matched it to the preserved fingerprint.

### 13.2 Official package

| Field | Value |
|---|---|
| ZIP | `aqua20_f03_convnext_small_gentle_cb_focal_outputs.zip` |
| ZIP SHA-256 | `973a0bc8d15398992a157b33d5171b7f04d9b4caa058b9e162f6c8395278a3b1` |
| ZIP bytes | 184,461,150 |
| Files | 23 |
| CRC | PASS |
| Prediction rows | 1,612 |
| Prediction audit | PASS |
| Confusion matrices | PASS |
| Metric recomputation | PASS |
| Decision | Official F03 verified |

### 13.3 Package contents

```text
README_F03.txt

figures/
    F03_training_metric_curves.png
    F03_training_loss_curves.png
    F03_confusion_matrix.png
    F03_confusion_matrix_normalized.png

tables/
    F03_best_epoch_summary.csv
    F03_class_counts_and_cb_weights.csv
    F03_classification_report.csv
    F03_comparison_ready_summary.csv
    F03_confusion_matrix.csv
    F03_confusion_matrix_normalized.csv
    F03_gentle_cb_focal_config.json
    F03_locked_validation_metrics.csv
    F03_step3c_audit_summary.csv
    F03_test_metrics_summary.csv
    F03_test_predictions.csv
    F03_test_vs_base_targets.csv
    F03_top_validation_epochs.csv
    F03_training_config.json
    F03_training_history.csv
    F03_validation_vs_base_targets.csv

weights/
    F03_best_model_sha256.txt
    best_model.pth
```

The extracted-tree SHA-256 was not located in the reviewed final manifests and remains **To be verified**.

---

## 14. Reproducibility status

### Verified for F03

- fixed split and class order;
- original 5,247-image training distribution;
- exact model constructor and 20-class classifier replacement;
- ImageNet-1K V1 pretrained-weight identifier;
- effective training and evaluation transforms;
- beta, gamma, class-count formula, mean-one weights, and loss computation;
- batch, workers, prefetch, optimiser family, LR, weight decay, scheduler family and `T_max`;
- gradient clipping, AMP, seed, cuDNN deterministic/benchmark flags;
- strict validation Macro-F1 checkpoint comparison;
- patience counter and stop threshold;
- 40/21 epoch chronology and best epoch 13;
- checkpoint hash and package hash;
- validation/test metrics, predictions, reports, and confusion matrices.

### Reproducibility boundary

F03 is well documented at the run-configuration and output-evidence level. Complete bitwise reproduction is not guaranteed because the project does not retain every worker random state, every sampled online augmentation, a full immutable container record, or all library/build details from the exact original training instant.

---

## 15. Items still **To be verified**

1. Immutable upstream file hash and download provenance for `ConvNeXt_Small_Weights.IMAGENET1K_V1`.
2. Exact run-local AdamW `betas`, `eps`, `amsgrad`, and parameter-group construction if non-default or explicitly supplied.
3. CosineAnnealingLR `eta_min` for F03.
4. Whether gradient accumulation was absent; no accumulation is claimed without direct evidence.
5. Complete original run-local `pip freeze`, Kaggle image/container identifier, NVIDIA driver, native cuDNN runtime value, CPU model, and host RAM.
6. Exact scikit-learn, pandas, NumPy, and Pillow versions at the original F03 training instant.
7. Extracted F03 package tree SHA-256.
8. Controlled parameter count, FLOPs, peak memory, latency, and throughput under one frozen benchmark protocol.
9. Repeated-seed optimisation variance.
10. Immutable dataset revision/content checksum, complete duplicate/near-duplicate audit, and complete all-file decode boundary inherited from Phase 2.
11. Any exhaustive class-wise interpretation beyond the preserved report/confusion evidence.

---

## 16. Locked manuscript-ready description

> F03 (`F03_CONVNEXT_SMALL_GENTLE_CB_FOCAL`) fine-tuned an ImageNet-1K-pretrained ConvNeXt-Small on the original 5,247-image AQUA20 training distribution at 224×224. The classifier head was replaced with a 20-class linear layer. Training used Gentle Class-Balanced Focal Loss with effective-number weights (`β=0.999`) normalised to mean one and a moderate focal exponent (`γ=1.0`). AdamW was used with learning rate `1×10⁻⁵`, weight decay `1×10⁻⁴`, cosine annealing, batch size 16, mixed precision, maximum gradient norm 1.0, and seed 42. The checkpoint was selected strictly by validation Macro F1; epoch 13 was the unique best epoch, and training stopped after eight subsequent non-improving epochs, completing 21 of 40 planned epochs. On the unchanged 1,612-image test split, F03 achieved 91.63% Top-1 accuracy, 90.79% Macro Precision, 86.93% Macro Recall, and 87.96% Macro F1. F03 was the strongest 224×224 ConvNeXt configuration in Macro Precision and Macro F1 and is retained as the strongest isolated imbalance intervention within the tested campaign. Because no original-distribution ConvNeXt-Small cross-entropy control was found, this result is not interpreted as a loss-only causal ablation.

---

## 17. Claims ledger

### Permitted

- F03 used the original 5,247-image training distribution.
- F03 used ConvNeXt-Small with ImageNet-1K V1 pretrained weights and a 20-class classifier.
- F03 used Gentle Class-Balanced Focal Loss with `beta=0.999`, `gamma=1.0`, and mean-one effective-number weights.
- Epoch 13 was selected by validation Macro F1 only.
- F03 completed 21 of 40 planned epochs and stopped after eight consecutive non-improving epochs.
- F03 achieved 87.9645% Macro F1 on the unchanged official test split.
- F03 was the strongest 224×224 ConvNeXt configuration in Macro Precision and Macro F1.
- F03 is the strongest isolated imbalance intervention within the tested campaign.
- Flatworm recall was 76.92% for F03.

### Not permitted

- F03 proves the causal superiority of focal loss over cross-entropy.
- F03 was compared against an original-distribution ConvNeXt-Small cross-entropy baseline.
- F03 used Aug-200.
- All F03 classes were solved equally well.
- F03 is the campaign's final primary model.
- F06's difference from F03 was caused only by higher resolution.
- The result is robust across seeds or hardware platforms without repeated-run evidence.
- The archived configuration guarantees bitwise reproduction.

---

## 18. F03 viva and reviewer question bank

### Q1. What is the exact identity of F03?

`F03_CONVNEXT_SMALL_GENTLE_CB_FOCAL`: ConvNeXt-Small, original AQUA20 training distribution, Gentle Class-Balanced Focal Loss, 224×224.

### Q2. Why is F03 called an isolated imbalance intervention?

It applies the loss-level imbalance mechanism without Aug-200. Within the tested campaign it is the cleanest preserved loss-level branch, but it is not matched to an original-distribution cross-entropy control.

### Q3. Is “Gentle Class-Balanced Focal Loss” a new loss?

No. “Gentle” only denotes `gamma=1.0`. The focal and effective-number components are established methods; mean-one normalisation is a scaling choice.

### Q4. How were class weights calculated?

Using `(1-beta)/(1-beta^n_c)` with `beta=0.999`, followed by division by the mean weight across 20 classes.

### Q5. Why normalise the class weights to mean one?

To retain relative reweighting while keeping the average loss scale comparable to unweighted cross-entropy.

### Q6. How does the implementation obtain `p_t`?

It calculates unreduced cross-entropy and uses `p_t = exp(-CE)` for each sample.

### Q7. What exactly was the checkpoint-selection metric?

Validation Macro F1 only. A checkpoint replaced the current best only when the new value was strictly greater.

### Q8. Was Top-1 used to choose epoch 13?

No. Top-1 was reported, but selection was based solely on validation Macro F1.

### Q9. Why did training finish at epoch 21 when the maximum was 40?

Epoch 13 was the unique best. Epochs 14–21 did not exceed it, so the non-improvement counter reached the patience value of eight.

### Q10. Was the official test split used for stopping or model selection?

No. It was evaluated after the epoch-13 checkpoint had been locked and preserved.

### Q11. What is the strongest F03 result?

Its official-test Macro F1 was 87.9645%, the strongest among the tested 224×224 ConvNeXt configurations. It also reached 91.6253% Top-1.

### Q12. Why is F03 not a strict loss ablation against F04?

F03 and F04 differ in both training distribution and objective, and their effective training transforms also differ.

### Q13. What does the flatworm result show?

F03 reached 76.92% recall on 13 flatworm test images, much higher than the earlier YOLO Aug-200 configurations. Because support is small and other factors differ, it is a configuration-level observation, not proof of loss causality.

### Q14. Why was F03 selected as the source for F06 instead of F05?

F03 preserved the isolated loss-level branch. Continuing F05 would also carry the Aug-200 data-level intervention into the high-resolution stage.

### Q15. What is the principal reproducibility limitation?

The exact run configuration and outputs are well preserved, but a complete immutable original container/software manifest and repeated-seed runs are unavailable.

### Q16. Can F03 be called the final proposed method?

No. F06 is the primary balanced single model; F08 is the complementary ensemble. F03 is the strongest isolated imbalance intervention and the source checkpoint for F06.

---

## 19. Evidence ledger

Primary evidence used for this phase:

- latest final manuscript: *Improving Underwater Marine Species Classification through Imbalance-Aware Learning and Cross-Dataset Evaluation*;
- `P00_AQUA20_Master_Evidence_Freeze_FINAL.xlsx`;
- `P00_AQUA20_Evidence_Freeze_Report_FINAL.md`;
- `aqua20_f03_convnext_small_gentle_cb_focal_outputs.zip`;
- `README_F03.txt`;
- `F03_training_config.json`;
- `F03_training_history.csv`;
- `F03_best_epoch_summary.csv`;
- `F03_locked_validation_metrics.csv`;
- `F03_class_counts_and_cb_weights.csv`;
- `F03_gentle_cb_focal_config.json`;
- `F03_test_metrics_summary.csv`;
- `F03_test_predictions.csv`;
- `F03_classification_report.csv`;
- F03 raw and normalised confusion matrices;
- F03 checkpoint/hash records;
- `P00_Batch1_F01_F03_Package_Audit.md`;
- `P00_source_file_manifest.csv`;
- `04-07-2026-aqua20-final-beat-base-plan.ipynb`;
- `D03_AQUA20_Environment_Reproducibility_and_Common_Protocol.md`;
- `H04_Phase4_to_Phase5_Handover.md`.

---

## 20. Phase 5 completion decision

Phase 5 is complete.

The F03 identity, scientific role, data distribution, model construction, pretrained-weight identifier, effective transforms, Gentle Class-Balanced Focal implementation, class weights, training controls, strict validation selection, early-stopping chronology, checkpoint/package integrity, validation/test results, selected class-wise behaviour, comparison boundaries, unsupported fields, and viva/reviewer answers are now documented and locked.

The next phase is:

**Phase 6 — F04 ConvNeXt-Small Aug-200 Cross-Entropy Experiment Documentation.**
