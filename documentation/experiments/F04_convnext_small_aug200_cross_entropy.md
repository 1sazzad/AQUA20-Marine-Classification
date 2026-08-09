# AQUA20 Phase 6 Documentation

## F04 ConvNeXt-Small Aug-200 Cross-Entropy Experiment

**Official experiment identity:** `F04_CONVNEXT_SMALL_AUG200`  
**Documentation phase:** Phase 6  
**Campaign:** `04-07-2026 AQUA20 FINAL BEAT BASE PLAN`

---

## 1. Purpose and scientific role

F04 is the campaign's **ConvNeXt-Small data-level balancing control**. It combines:

- ImageNet-pretrained ConvNeXt-Small;
- the training-only Aug-200 distribution;
- standard cross-entropy optimisation;
- 224×224 input resolution;
- validation Macro F1 checkpoint selection.

Its scientific purpose is to evaluate how the Aug-200 data-level intervention behaves within the ConvNeXt-Small branch when no class weighting and no focal term are used.

F04 is not:

- F02;
- an original-distribution ConvNeXt-Small cross-entropy control;
- a class-balanced focal-loss experiment;
- the final campaign-winning model.

The locked identities are:

```text
F02 = F02_YOLO26S_CLS_AUG200
F04 = F04_CONVNEXT_SMALL_AUG200
```

No architecture-matched ConvNeXt-Small + standard cross-entropy experiment on the original 5,247-image training distribution was found. Therefore F04 cannot be used as a strict loss-only control for F03.

---

## 2. Source hierarchy and conflict rule

This documentation follows the locked source hierarchy:

1. latest final manuscript, *Improving Underwater Marine Species Classification through Imbalance-Aware Learning and Cross-Dataset Evaluation*;
2. FINAL evidence-freeze and frozen terminology/index records;
3. direct F04 records:
   - `F04_training_config.json`;
   - `F04_official_final_summary.json`;
   - frozen F04 tables and figures;
   - checkpoint/hash records;
   - source-integrity manifest;
4. preserved campaign notebook, `04-07-2026-aqua20-final-beat-base-plan.ipynb`;
5. older audits and preliminary records for historical context only.

F04 has no standalone README. Its identity, configuration, chronology, metrics, and package contents are therefore controlled by the direct JSON/table evidence, final registry, and preserved notebook.

Later FINAL records and direct run-level files supersede older records that incorrectly assigned the ConvNeXt cross-entropy role to F02.

---

## 3. Locked experiment identity

| Field | Locked value |
|---|---|
| Experiment ID | `F04` |
| Official name | `F04_CONVNEXT_SMALL_AUG200` |
| Scientific group | `G2_data_level_balancing_control` |
| Architecture | torchvision ConvNeXt-Small |
| Training distribution | Aug-200, training only |
| Objective | standard `CrossEntropyLoss` |
| Class weights | none |
| Focal term | none |
| Resolution | 224×224 |
| Selection metric | validation Macro F1 |
| Best epoch | 20 |
| Status | completed |

Approved description:

> F04 is the ConvNeXt-Small Aug-200 data-level balancing branch trained with standard cross-entropy at 224×224.

Prohibited description:

> F02 is the ConvNeXt-Small cross-entropy baseline.

---

## 4. Data protocol

### 4.1 Frozen partitions

| Partition | Images | F04 use |
|---|---:|---|
| Aug-200 training | 7,284 | model optimisation |
| Original validation | 1,312 | checkpoint selection and reporting |
| Original official test | 1,612 | final evaluation only |
| Classes | 20 | unchanged class space |

The campaign split seed was 42.

The official test split was not used for:

- training;
- early stopping;
- hyperparameter tuning;
- checkpoint replacement;
- model selection.

### 4.2 Meaning of Aug-200

Aug-200 imposed a minimum training representation of 200 images for classes below that threshold. It did not downsample larger classes.

Locked dataset-level facts:

- original campaign training images: 5,247;
- classes augmented: 14;
- synthetic training images added: 2,037;
- Aug-200 total: 7,284;
- validation and test images added: zero;
- the post-Aug-200 distribution remained imbalanced because larger classes were retained unchanged.

### 4.3 Offline Aug-200 generation versus online transforms

Aug-200 is an **offline file-generation intervention**. The preserved generator applied:

```text
RGB conversion
→ isotropic random crop, scale 0.82–1.00
→ bicubic resize to the source dimensions
→ horizontal flip, p=0.5
→ rotation −18° to +18°, bicubic, no expansion
→ brightness factor 0.82–1.18
→ contrast factor 0.85–1.20
→ saturation factor 0.80–1.22
→ sharpness factor 0.85–1.25
→ JPEG quality-95 saving
```

These operations created additional training files. They are distinct from the online transformations applied dynamically when F04 loaded each training image.

### 4.4 Frozen class order

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

## 5. Model construction and pretrained initialisation

The preserved F04 code constructed the model as follows:

```python
model = convnext_small(
    weights=ConvNeXt_Small_Weights.DEFAULT
)

in_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_features, 20)
```

Locked construction facts:

| Item | Value |
|---|---|
| Constructor | `torchvision.models.convnext_small` |
| Saved weight identifier | `ConvNeXt_Small_Weights.DEFAULT` |
| Downloaded upstream filename | `convnext_small-0c510722.pth` |
| Classifier input features | 768 |
| Classifier replacement | `nn.Linear(768, 20)` |
| Output classes | 20 |

The immutable SHA-256 of the upstream pretrained-weight file and a separately archived upstream provenance manifest are **To be verified**.

---

## 6. Effective F04 transforms

### 6.1 Online training transform

```text
Direct Resize 224×224
→ RandomHorizontalFlip, p=0.5
→ RandomRotation, −10° to +10°
→ ToTensor
→ ImageNet normalisation
```

The preserved torchvision calls were:

```python
transforms.Resize((224, 224))
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomRotation(10)
transforms.ToTensor()
transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
```

No `RandomResizedCrop`, `CenterCrop`, `ColorJitter`, RandAugment, random erasing, MixUp, or CutMix operation appears in the preserved F04 transform sequence.

### 6.2 Validation and official-test transform

```text
Direct Resize 224×224
→ ToTensor
→ ImageNet normalisation
```

No random validation/test augmentation and no CenterCrop were used.

### 6.3 ImageNet normalisation

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### 6.4 Transform claim boundary

The direct F04 source verifies the operation sequence. Low-level torchvision defaults not explicitly set in the code, including exact interpolation/antialias/fill details under the archived runtime, should not be retrospectively restated beyond what the source preserves.

---

## 7. Objective and loss boundary

F04 used:

```python
criterion = nn.CrossEntropyLoss()
```

For a sample with true class \(y\) and predicted correct-class probability \(p_y\), the standard cross-entropy term is:

\[
\mathcal{L}_{\mathrm{CE}}=-\log(p_y).
\]

Locked facts:

- no class-weight tensor was supplied;
- `class_weights=false` in the frozen configuration;
- `focal_loss=false` in the frozen configuration;
- no focal modulation term \((1-p_y)^\gamma\) was used;
- no effective-number class weighting was used;
- no custom loss-family claim is permitted.

The constructor was invoked without explicit optional arguments. Exact version-dependent defaults beyond the archived call should not be expanded from memory.

---

## 8. Training configuration and runtime controls

### 8.1 Locked configuration

| Field | Value |
|---|---:|
| Image size | 224 |
| Batch size | 16 |
| Workers | 4 |
| Maximum epochs | 40 |
| Early-stopping patience | 8 |
| Optimiser | AdamW |
| Learning rate | `1e-5` |
| Weight decay | `1e-4` |
| Scheduler | CosineAnnealingLR |
| `T_max` | 40 |
| `eta_min` | `5e-7` |
| Loss | CrossEntropyLoss |
| Class weights | false |
| Focal loss | false |
| AMP | enabled |
| Seed | 42 |
| Checkpoint metric | validation Macro F1 |

### 8.2 DataLoader controls found in source

Training loader:

- `shuffle=True`;
- `batch_size=16`;
- `num_workers=4`;
- `pin_memory=True`.

Validation loader:

- `shuffle=False`;
- `batch_size=16`;
- `num_workers=4`;
- `pin_memory=True`.

Persistent workers, an explicit prefetch factor, worker-specific seeding, and gradient accumulation were not frozen in the direct F04 configuration.

### 8.3 Reproducibility calls

The F04 training cell explicitly seeded:

```text
Python random = 42
NumPy = 42
PyTorch CPU = 42
PyTorch CUDA/all CUDA devices = 42
```

The run-specific F04 code also set:

```python
torch.backends.cudnn.benchmark = True
```

A run-specific `torch.backends.cudnn.deterministic` assignment was not found in the direct F04 training block. It must therefore remain **To be verified**, rather than being copied from F03.

### 8.4 Preserved environment evidence

The F04 verification output recorded:

| Item | Recorded value |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.7.1+cu118 |
| torchvision | 0.22.1+cu118 |
| CUDA available | true |
| GPU | Tesla P100-PCIE-16GB |

A complete immutable original package/container, driver, native cuDNN, CPU, RAM, and all-library manifest was not preserved.

### 8.5 Unsupported runtime details

The following remain **To be verified**:

- AdamW betas, epsilon, amsgrad, and exact parameter groups;
- gradient clipping status;
- gradient accumulation status;
- exact AMP scaler/autocast implementation details;
- persistent-worker and prefetch settings;
- warm-up status;
- exact run-local versions of scikit-learn, pandas, NumPy, and Pillow;
- complete driver/native-cuDNN/CPU/RAM/container manifest.

---

## 9. Checkpoint-selection rule and chronology

### 9.1 Strict selection condition

The preserved condition was:

```python
if val_macro_f1 > best_val_macro_f1:
```

Therefore:

- only a strictly higher validation Macro F1 replaced the best checkpoint;
- a tie did not replace it;
- an improvement reset `epochs_without_improvement` to zero;
- otherwise, the counter increased by one;
- the official test set was not involved.

### 9.2 Early stopping

The stop condition was:

```python
if epochs_without_improvement >= PATIENCE:
    break
```

with:

```text
PATIENCE = 8
```

### 9.3 Locked chronology

| Item | Value |
|---|---:|
| Planned maximum | 40 epochs |
| Completed | 28 epochs |
| Best epoch | 20 |
| Patience | 8 non-improving epochs |
| Stop point | after epoch 28 |

Epoch 18 produced an earlier best validation Macro F1 of approximately 89.17%. Epoch 20 improved it to the final locked 89.8088%. Epochs 21–28 did not exceed epoch 20, so the non-improvement counter reached eight and early stopping triggered.

The scheduler was stepped once after each completed epoch.

### 9.4 Recovery/checkpoint records

The training loop wrote:

- the current validation-selected best checkpoint;
- a lightweight `last_checkpoint.pth` after each epoch containing model, optimiser, scheduler, current best information, counter, and history.

The official package retains only the curated locked best checkpoint, not the transient last checkpoint.

---

## 10. Locked validation result

The independently re-evaluated epoch-20 checkpoint produced:

| Validation metric | Value |
|---|---:|
| Top-1 | 92.8354% |
| Top-2 | 97.7896% |
| Top-3 | 99.3140% |
| Top-5 | 99.8476% |
| Macro Precision | 90.7527% |
| Macro Recall | 90.5123% |
| Macro F1 | 89.8088% |

Checkpoint selection used only:

```text
Validation Macro F1 = 89.8088%
```

The fresh validation pass asserted agreement with the values stored in the locked checkpoint.

---

## 11. Official unchanged test result

| Official-test metric | Value |
|---|---:|
| Top-1 | 91.3772% |
| Top-2 | 97.2705% |
| Top-3 | 99.0074% |
| Top-5 | 99.6278% |
| Macro Precision | 89.3241% |
| Macro Recall | 84.3236% |
| Macro F1 | 86.0035% |
| Weighted Precision | 91.4067% |
| Weighted Recall | 91.3772% |
| Weighted F1 | 91.2641% |
| Support | 1,612 |

The saved evidence includes:

- 1,612 prediction rows;
- classification report;
- raw confusion matrix;
- row-normalised confusion matrix;
- test metric summary;
- comparison-ready summary;
- raw and normalised confusion-matrix figures.

The final audit verified that the 1,612 predictions, confusion matrix, and classification metrics matched. Per-sample saved Top-k rankings were not retained; the aggregate Top-k values are frozen in the official summary.

### 11.1 Macro-versus-weighted gap

```text
Weighted F1 − Macro F1
= 91.2641% − 86.0035%
= 5.2606 percentage points
```

This gap shows that high aggregate accuracy coexisted with uneven class-wise performance. Aug-200 improved minority representation but did not fully remove the effect of class imbalance.

---

## 12. Class-wise finding: flatworm

F04 official-test flatworm performance was:

| Metric | Value |
|---|---:|
| Precision | 81.8182% |
| Recall | 69.2308% |
| F1 | 75.0000% |
| Support | 13 |

Related frozen recalls:

| Configuration | Flatworm recall |
|---|---:|
| F01 | 15.38% |
| F02 | 23.08% |
| F03 | 76.92% |
| F04 | 69.23% |

With support 13, one correct or incorrect sample changes recall by approximately 7.69 percentage points. The result should therefore be reported as a configuration-level observation, not as proof that Aug-200 alone caused the change.

The complete classification report and confusion matrices should remain the basis for any broader per-class interpretation.

---

## 13. F04 versus F03

F03 exceeded F04 by:

| Metric | F03 minus F04 |
|---|---:|
| Top-1 | +0.2481 pp |
| Top-5 | +0.0620 pp |
| Macro Precision | +1.4609 pp |
| Macro Recall | +2.6066 pp |
| Macro F1 | +1.9610 pp |
| Weighted F1 | +0.3015 pp |

Interpretation boundary:

- both use ConvNeXt-Small and 224×224 evaluation;
- F03 uses the original 5,247-image training distribution;
- F04 uses the 7,284-image Aug-200 training distribution;
- F03 uses Gentle Class-Balanced Focal Loss;
- F04 uses standard cross-entropy;
- their effective online training transforms also differ.

Therefore this is a **complete-configuration comparison**, not a loss-only or data-only causal ablation.

Approved wording:

> Under the tested complete configurations, F03 achieved higher official-test Macro Recall and Macro F1 than F04.

Prohibited wording:

> Focal loss alone improved Macro F1 by 1.9610 percentage points.

---

## 14. F04 versus F05

F04 and F05 are the closest available loss-focused comparison because they share:

- ConvNeXt-Small;
- the Aug-200 training distribution;
- 224×224 input;
- batch size 16;
- AdamW;
- learning rate `1e-5`;
- weight decay `1e-4`;
- validation-based checkpoint selection.

Their official-test differences were:

| Metric | F04 minus F05 |
|---|---:|
| Top-1 | +0.2482 pp |
| Top-5 | −0.1241 pp |
| Macro Precision | −0.4255 pp |
| Macro Recall | +1.6689 pp |
| Macro F1 | +1.6548 pp |
| Weighted F1 | +0.2882 pp |

However, the comparison is not a strict one-factor ablation:

- F04 maximum budget: 40 epochs;
- F05 maximum budget: 25 epochs;
- F04 patience: 8;
- F05 patience and some scheduler/runtime details are not fully archived.

Approved wording:

> F04 and F05 provide a loss-focused comparison within the same Aug-200 ConvNeXt branch. F04 obtained higher Macro Recall and Macro F1 in the completed runs, but the difference cannot be attributed exclusively to the loss because the maximum epoch budgets and some stopping/scheduler details differed.

Prohibited wording:

> Adding focal loss caused performance to decline.

---

## 15. Scientific interpretation

The evidence supports the following conclusions:

1. F04 demonstrates that Aug-200 was substantially more effective with ConvNeXt-Small than in the earlier YOLO Aug-200 configurations.
2. F04 reached strong headline performance, including 91.3772% Top-1 and 99.6278% Top-5.
3. Its Macro F1 of 86.0035% remained below its Weighted F1 by 5.2606 percentage points, indicating residual class-wise imbalance.
4. F03 ranked above F04 on Macro Recall and Macro F1 under the tested complete configurations.
5. F04 ranked above F05 on Macro Recall and Macro F1 in their completed runs, but this does not establish a pure causal effect of the loss function.
6. F04 was not selected as the source checkpoint for F06; F06 continued the isolated F03 branch.
7. F04 is an important data-level control, not the final proposed model.

---

## 16. Checkpoint and package integrity

### 16.1 Locked checkpoint

| Item | Value |
|---|---|
| File | `weights/F04_best_model_locked.pth` |
| Epoch | 20 |
| SHA-256 | `4e2c4abddb528e52d9721c6e1f85ea389788495f315d2891989e4ff2e403a1a7` |
| Checkpoint byte size | **To be verified** |

The hash was recalculated during the notebook audit and matched the saved hash record and checkpoint metadata.

### 16.2 Official package

| Item | Value |
|---|---|
| ZIP | `aqua20_f04_convnext_small_aug200_outputs.zip` |
| Notebook-reported ZIP size | 524.86 MiB |
| Curated files | 14 |
| Package audit | PASS |
| Extracted bytes | 594,978,599 |
| Extracted-tree SHA-256 | `725d25f97bd5cbeacc2dd63070eb62b6e15972d0106e0a629fe0e77a34310eb5` |
| Original ZIP SHA-256 | **Not collected / To be verified** |

### 16.3 Curated package contents

```text
weights/F04_best_model_locked.pth
F04_training_config.json
tables/F04_training_history.csv
tables/F04_locked_validation_summary.csv
tables/F04_test_metrics_summary.csv
tables/F04_classification_report.csv
tables/F04_test_predictions.csv
tables/F04_confusion_matrix.csv
tables/F04_confusion_matrix_normalized.csv
tables/F04_comparison_ready_summary.csv
figures/F04_confusion_matrix.png
figures/F04_confusion_matrix_normalized.png
F04_best_model_sha256.txt
F04_official_final_summary.json
```

A dedicated F04 training-curve PNG is not present in the frozen package.

---

## 17. Reproducibility and claim limitations

The following remain **To be verified** or unavailable:

- immutable upstream pretrained-weight file hash and complete provenance manifest;
- original official ZIP SHA-256 and exact original ZIP byte count;
- checkpoint byte size;
- AdamW betas, epsilon, amsgrad, and exact parameter groups;
- gradient clipping status;
- gradient accumulation status;
- exact AMP scaler/autocast path;
- run-specific cuDNN deterministic flag;
- persistent workers, prefetch factor, and worker-seeding details;
- warm-up status;
- complete original container/package/driver/native-cuDNN/CPU/RAM manifest;
- exact run-local scikit-learn, pandas, NumPy, and Pillow versions;
- controlled parameter count, FLOPs, GPU memory, latency, throughput, and energy use;
- repeated-seed optimisation variance;
- immutable dataset revision and full dataset content hash;
- exhaustive duplicate/near-duplicate and all-file decode audits inherited from Phase 2;
- exhaustive causal class-wise interpretation beyond the preserved report and confusion matrices.

Approved general disclosure:

> The direct F04 configuration, source code, checkpoint chronology, checkpoint hash, validation/test summaries, predictions, reports, and confusion matrices are preserved. Some low-level runtime, immutable environment, efficiency, and repeated-seed evidence was not archived and is therefore not reconstructed retrospectively.

---

## 18. Viva and reviewer question bank

### Q1. What is the exact F04 identity?

`F04_CONVNEXT_SMALL_AUG200`.

### Q2. Is F04 the experiment previously called F02?

No. F02 is `F02_YOLO26S_CLS_AUG200`. F04 is the ConvNeXt-Small Aug-200 standard-cross-entropy experiment.

### Q3. What was F04 designed to test?

It tested the Aug-200 data-level balancing strategy within a ConvNeXt-Small branch trained with standard cross-entropy.

### Q4. What data did F04 use?

It trained on 7,284 Aug-200 images, selected its checkpoint on the unchanged 1,312-image validation split, and was finally evaluated on the unchanged 1,612-image official test split.

### Q5. Was Aug-200 a perfectly balanced dataset?

No. It raised minority classes to a floor of 200 without downsampling larger classes, so the resulting distribution remained imbalanced.

### Q6. Is offline Aug-200 the same as online training augmentation?

No. Aug-200 created additional training files offline. F04 then applied a separate online sequence consisting of direct resize, horizontal flip, rotation, tensor conversion, and ImageNet normalisation.

### Q7. How was ConvNeXt-Small constructed?

The preserved code used `convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)` and replaced the 768-input classifier layer with `nn.Linear(768, 20)`.

### Q8. What loss was used?

Standard `nn.CrossEntropyLoss()` with no class weights and no focal term.

### Q9. What optimiser and scheduler were used?

AdamW with learning rate `1e-5` and weight decay `1e-4`, followed by `CosineAnnealingLR` with `T_max=40` and `eta_min=5e-7`.

### Q10. How was the best checkpoint selected?

Only a strictly higher validation Macro F1 replaced the current best checkpoint. Ties did not replace it.

### Q11. Why did training finish after 28 rather than 40 epochs?

Epoch 20 was the best. Epochs 21–28 did not exceed it, so eight consecutive non-improving epochs reached the patience threshold and triggered early stopping.

### Q12. Was the official test set used for selection?

No. It was evaluated only after the epoch-20 checkpoint had been locked and audited.

### Q13. What were the principal F04 test results?

Top-1 was 91.3772%, Top-5 was 99.6278%, Macro F1 was 86.0035%, and Weighted F1 was 91.2641%.

### Q14. What does the Macro-versus-Weighted F1 gap show?

Weighted F1 exceeded Macro F1 by 5.2606 percentage points, showing that performance remained uneven across classes.

### Q15. Can F03 versus F04 be called a loss-only ablation?

No. Their training distributions, objectives, and online training transforms differ.

### Q16. Is F04 versus F05 a strict loss-only comparison?

No. It is the closest loss-focused comparison, but their maximum epoch budgets were 40 and 25, and some F05 stopping/scheduler details are incomplete.

### Q17. What did F04 show for flatworm?

It reached 69.23% recall on 13 test images. Because one sample changes recall by approximately 7.69 percentage points, the finding must be interpreted cautiously.

### Q18. Why was F04 not used as the F06 source checkpoint?

F06 continued F03 to preserve the isolated original-distribution imbalance-aware branch. Continuing F04 would carry the Aug-200 data-level intervention into the high-resolution stage.

### Q19. Is F04 the final proposed method?

No. F06 is the primary balanced single model, while F08 is the complementary probability ensemble. F04 is a data-level control.

### Q20. What is the main reproducibility limitation?

The central run configuration, source, checkpoint, metrics, predictions, and matrices are preserved, but the original ZIP hash, complete immutable environment, some low-level optimiser/runtime details, efficiency measurements, and repeated-seed results are unavailable.

---

## 19. Evidence ledger

Primary evidence used for this phase:

- latest final manuscript, *Improving Underwater Marine Species Classification through Imbalance-Aware Learning and Cross-Dataset Evaluation*;
- `P00_AQUA20_Master_Evidence_Freeze_FINAL.xlsx`;
- `P00_AQUA20_Evidence_Freeze_Report_FINAL.md` and final numerical/methodology audits;
- `experiment_readme_index.md`;
- `F04_training_config.json`;
- `F04_official_final_summary.json`;
- `F04_training_history.csv`;
- `F04_locked_validation_summary.csv`;
- `F04_test_metrics_summary.csv`;
- `F04_test_predictions.csv`;
- `F04_classification_report.csv`;
- F04 raw and normalised confusion matrices and figures;
- `F04_comparison_ready_summary.csv`;
- `F04_best_model_sha256.txt`;
- `P00_source_integrity_manifest.csv`;
- `04-07-2026-aqua20-final-beat-base-plan.ipynb`;
- `D02_AQUA20_Dataset_and_Aug200_Documentation.md`;
- `D03_AQUA20_Environment_Reproducibility_and_Common_Protocol.md`;
- `D05_AQUA20_F03_ConvNeXt_Small_Gentle_CB_Focal_Experiment_Documentation.md`;
- `H05_Phase5_to_Phase6_Handover.md`.

---

## 20. Phase 6 completion decision

Phase 6 is complete.

The following are now documented and locked:

- exact F04 identity and scientific role;
- separation of F04 from F02;
- Aug-200 data protocol and offline/online augmentation distinction;
- ConvNeXt-Small construction and preserved pretrained-weight identifier;
- effective train/validation/test transforms;
- standard cross-entropy objective with no class weights or focal term;
- optimiser, scheduler, AMP, seed, workers, and verified backend fields;
- strict validation Macro-F1 checkpoint selection;
- 40/28 epoch chronology, patience 8, and epoch-20 best checkpoint;
- independently confirmed validation result;
- official 1,612-image test result;
- predictions, report, confusion-matrix, and flatworm evidence;
- valid F03/F04 and F04/F05 comparison boundaries;
- checkpoint and package integrity;
- unsupported fields marked **To be verified**;
- F04 viva/reviewer answers.

The next phase is:

**Phase 7 — F05 ConvNeXt-Small Aug-200 Gentle Class-Balanced Focal Experiment Documentation.**
