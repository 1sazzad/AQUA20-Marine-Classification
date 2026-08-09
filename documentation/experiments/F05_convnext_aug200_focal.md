# F05 — ConvNeXt-Small + Aug-200 + Gentle Class-Balanced Focal

**Official experiment ID:** `F05_CONVNEXT_SMALL_AUG200_GENTLE_CB_FOCAL`  
**Campaign:** `04-07-2026 AQUA20 FINAL BEAT BASE PLAN`  
**Scientific role:** combined data-level and loss-level imbalance intervention  
**Input resolution:** `224×224`  
**Status:** completed and frozen for reporting

---

## 1. Scope and evidence hierarchy

This document records the final evidence-supported configuration, chronology, results, integrity information, interpretation boundaries, and reviewer-facing answers for F05.

Evidence priority for this record is:

1. the latest final manuscript, *Improving Underwater Marine Species Classification through Imbalance-Aware Learning and Cross-Dataset Evaluation*;
2. the FINAL evidence-freeze and frozen experiment-index records;
3. direct F05 evidence from the preserved campaign notebook and the official F05 package, including `README_F05.txt`, training history, checkpoint metadata, loss implementation, transforms, predictions, classification report, and confusion matrices;
4. earlier audits and handovers only when they do not conflict with newer direct evidence.

Where an exact value is not supported by the surviving F05 evidence, it is marked **To be verified** rather than reconstructed from another experiment.

---

## 2. Experiment identity and scientific role

The exact experiment identity is:

```text
F05_CONVNEXT_SMALL_AUG200_GENTLE_CB_FOCAL
```

F05 combines:

```text
ConvNeXt-Small
+
Aug-200 train-only balancing
+
Gentle Class-Balanced Focal Loss
```

Its purpose is to test whether the campaign's data-level balancing intervention and loss-level imbalance intervention remain complementary when applied together in the same `224×224` ConvNeXt-Small branch.

F05 is **not**:

- the F02 experiment;
- a YOLO experiment;
- the original-distribution ConvNeXt focal experiment;
- the final proposed single model;
- evidence that class-balanced focal loss is universally harmful.

The related identities remain:

```text
F02 = F02_YOLO26S_CLS_AUG200
F03 = F03_CONVNEXT_SMALL_GENTLE_CB_FOCAL
F04 = F04_CONVNEXT_SMALL_AUG200
F05 = F05_CONVNEXT_SMALL_AUG200_GENTLE_CB_FOCAL
F06 = F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL
```

---

## 3. Data protocol

### 3.1 Frozen dataset partitions

AQUA20 contains 8,171 images across 20 classes. The campaign split was fixed before the model experiments:

| Partition | Images | F05 role |
|---|---:|---|
| Training | 7,284 | Aug-200 training distribution |
| Validation | 1,312 | unchanged original validation split |
| Official test | 1,612 | unchanged original test split |
| Classes | 20 | fixed class order |

The original campaign training partition contained 5,247 images. Aug-200 added 2,037 synthetic training images and increased every class below 200 training examples to a floor of 200 while retaining larger classes unchanged.

Therefore, Aug-200 is a **minimum-floor balancing strategy**, not a fully uniform class distribution.

No Aug-200 files were added to validation or test.

### 3.2 Frozen class order

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

## 4. Offline Aug-200 generation versus F05 online transforms

Two different augmentation concepts must remain separate.

### 4.1 Offline Aug-200 file generation

Aug-200 permanently created extra training files. The preserved generator applied:

```text
RGB conversion
→ isotropic random crop, scale 0.82–1.00
→ bicubic resize back to the source dimensions
→ horizontal flip, p=0.5
→ rotation −18° to +18°, bicubic, no expansion
→ brightness factor 0.82–1.18
→ contrast factor 0.85–1.20
→ saturation factor 0.80–1.22
→ sharpness factor 0.85–1.25
→ JPEG quality 95
```

The preserved Aug-200 generator does not support claims that it used vertical flip, shear, perspective transformation, blur, additive noise, MixUp, CutMix, or copy-paste augmentation.

### 4.2 F05 online training transform

F05 then applied dynamic torchvision-v2 transforms when the Aug-200 files were loaded:

```text
Direct Resize 224×224
→ RandomHorizontalFlip(p=0.5)
→ RandomRotation(−10° to +10°)
→ ToImage
→ ToDtype(float32, scale=True)
→ ImageNet normalisation
```

The preserved code used:

```python
weights = ConvNeXt_Small_Weights.DEFAULT
weights_transforms = weights.transforms()

train_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=10),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=weights_transforms.mean,
        std=weights_transforms.std,
    ),
])
```

### 4.3 Validation and official-test transform

```text
Direct Resize 224×224
→ ToImage
→ ToDtype(float32, scale=True)
→ ImageNet normalisation
```

No random augmentation was applied during validation or official-test evaluation.

### 4.4 Normalisation

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

The online transform sequence is distinct from the offline Aug-200 image-generation process and must not be described as though they were a single augmentation pipeline.

---

## 5. Model construction and pretrained initialisation

The preserved F05 notebook constructed a torchvision ConvNeXt-Small classifier.

The run used:

```text
ConvNeXt_Small_Weights.DEFAULT
```

and the preserved runtime resolved that identifier to:

```text
ConvNeXt_Small_Weights.IMAGENET1K_V1
```

The model's final classifier was replaced for the 20 AQUA20 classes:

```python
model = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)

in_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_features, 20)
```

Locked construction facts:

| Field | Value |
|---|---|
| Constructor | `torchvision.models.convnext_small` |
| Requested pretrained enum | `ConvNeXt_Small_Weights.DEFAULT` |
| Runtime-resolved enum | `ConvNeXt_Small_Weights.IMAGENET1K_V1` |
| Classifier input features | 768 |
| Classifier outputs | 20 |
| Training | end-to-end fine-tuning; no frozen-backbone claim |

The immutable SHA-256 of the upstream pretrained-weight file and a separately archived upstream provenance manifest remain **To be verified**.

---

## 6. Gentle Class-Balanced Focal Loss

### 6.1 Terminology boundary

The campaign term **Gentle** refers only to the moderate focal exponent:

```text
gamma = 1.0
```

The effective-number weighting, focal modulation, and cross-entropy components are established methods. F05 does not introduce a new loss family.

### 6.2 Effective-number weighting

For class \(y\), containing \(n_y\) training examples, the raw effective-number weight is:

\[
\widetilde{w}_y =
\frac{1-\beta}
     {1-\beta^{n_y}},
\]

with:

\[
\beta = 0.999.
\]

The weights are then normalised to have mean one:

\[
w_y =
\frac{\widetilde{w}_y}
{\frac{1}{C}\sum_{c=1}^{C}\widetilde{w}_c},
\qquad C=20.
\]

### 6.3 Focal objective

For the correct-class probability \(p_y\):

\[
\mathcal{L}_{\mathrm{CBF}}
=
-w_y(1-p_y)^\gamma\log(p_y),
\]

with:

```text
beta  = 0.999
gamma = 1.0
```

The preserved implementation follows:

```text
unreduced per-sample cross-entropy
→ p_t = exp(−CE)
→ focal factor = (1−p_t)^gamma
→ target-indexed normalised class weight
→ per-sample multiplication
→ mean reduction
```

The F05 loss configuration explicitly excludes:

- label smoothing;
- simple inverse-frequency weighting;
- weighted sampling.

---

## 7. Aug-200 class counts and F05 class weights

The loss weights were calculated from the actual 7,284-image Aug-200 training distribution, not from the original 5,247-image distribution.

| ID | Class | Aug-200 train count | Normalised CB weight |
|---:|---|---:|---:|
| 00 | coral | 1,249 | 0.293991 |
| 01 | crab | 200 | 1.156491 |
| 02 | diver | 200 | 1.156491 |
| 03 | eel | 200 | 1.156491 |
| 04 | fish | 1,760 | 0.253264 |
| 05 | fishInGroups | 218 | 1.070262 |
| 06 | flatworm | 200 | 1.156491 |
| 07 | jellyfish | 200 | 1.156491 |
| 08 | marine_dolphin | 200 | 1.156491 |
| 09 | octopus | 200 | 1.156491 |
| 10 | rayfish | 305 | 0.797488 |
| 11 | seaAnemone | 712 | 0.411629 |
| 12 | seaCucumber | 200 | 1.156491 |
| 13 | seaSlug | 200 | 1.156491 |
| 14 | seaUrchin | 200 | 1.156491 |
| 15 | shark | 200 | 1.156491 |
| 16 | shrimp | 200 | 1.156491 |
| 17 | squid | 200 | 1.156491 |
| 18 | starfish | 200 | 1.156491 |
| 19 | turtle | 240 | 0.982500 |

The weight vector was explicitly checked to be finite and to have mean approximately 1.0.

### Interpretation

Aug-200 already raises low-count classes to 200 images. Consequently, many previously rare classes receive identical or near-identical F05 class-balanced weights. F05 therefore applies a second imbalance-aware mechanism to a training distribution that has already been partially equalised by offline augmentation.

This observation motivates the scientific question tested by F05, but it does not by itself establish why F05 performed worse than F03 or F04.

---

## 8. Training configuration and reproducibility controls

### 8.1 Locked F05 configuration

| Field | F05 value |
|---|---|
| Resolution | 224×224 |
| Batch size | 16 |
| Workers | 4 |
| Maximum epochs | 25 |
| Completed epochs | 25 |
| Optimiser | AdamW |
| Learning rate | `1e-5` |
| Weight decay | `1e-4` |
| Scheduler | CosineAnnealingLR |
| Scheduler `T_max` | 25 |
| Explicit custom `eta_min` | not found in preserved constructor |
| Loss | Gentle Class-Balanced Focal |
| Beta | 0.999 |
| Gamma | 1.0 |
| Seed | 42 |
| Checkpoint metric | validation Macro F1 |
| AMP | enabled |
| Training shuffle | true |
| Validation shuffle | false |
| `pin_memory` | true |
| `persistent_workers` | true |
| Train loader generator | seeded with 42 |
| `torch.use_deterministic_algorithms` | false |
| cuDNN deterministic | false |
| cuDNN benchmark | true |

The direct F05 notebook therefore resolves several fields that older campaign summaries had left open, especially the pretrained-weight enum, worker count, AMP path, `T_max`, persistent-worker usage, and backend settings.

### 8.2 AMP

The preserved F05 path used CUDA automatic mixed precision with a CUDA gradient scaler. The exact lower-level kernel/precision behaviour should not be extended beyond the archived code.

### 8.3 Early-stopping mechanism

The training loop contains an `epochs_without_improvement` counter and an early-stop condition of the form:

```python
if epochs_without_improvement >= PATIENCE:
    break
```

However, the exact numeric assignment of `PATIENCE` has not been recovered from the currently frozen direct F05 evidence. It therefore remains:

**To be verified.**

Observed run behaviour is unambiguous:

- maximum budget: 25 epochs;
- completed: 25 epochs;
- best epoch: 20;
- epochs 21–25 did not exceed epoch 20;
- training therefore reached its planned maximum rather than being truncated before epoch 25.

The F03/F04 patience value of 8 must not be copied into F05 without direct evidence.

### 8.4 Recovery checkpointing

The preserved training block includes a recovery-checkpoint mechanism labelled as saving every five epochs and stores model, optimiser, scheduler, scaler, best score/epoch, non-improvement counter, and training configuration.

### 8.5 Environment evidence

The preserved campaign runtime records:

| Item | Recorded value |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.7.1+cu118 |
| torchvision | 0.22.1+cu118 |
| CUDA build | 11.8 |
| GPU | Tesla P100-PCIE-16GB |
| GPU memory | approximately 15.89 GiB |
| Compute capability | 6.0 |

This does not constitute a complete immutable run-local environment manifest.

---

## 9. Checkpoint selection and training chronology

### 9.1 Selection rule

F05 selected its checkpoint using a strict validation Macro-F1 improvement:

```python
is_best = val_macro_f1 > best_val_macro_f1
```

Therefore:

- only a strictly higher validation Macro F1 replaced the saved best model;
- ties did not replace the checkpoint;
- an improvement reset the non-improvement counter;
- a non-improvement incremented the counter;
- the official test set was not used for checkpoint selection.

### 9.2 Locked chronology

| Item | Value |
|---|---:|
| Planned maximum | 25 epochs |
| Completed | 25 epochs |
| Best epoch | 20 |
| Best validation Macro F1 | 89.9326% |
| Total recorded training time | 57.96 minutes |
| Official test used during training | No |

At epoch 20, the recorded validation values were:

| Validation metric | Value |
|---|---:|
| Top-1 | 93.2927% |
| Macro Precision | 90.0635% |
| Macro Recall | 91.0146% |
| Macro F1 | 89.9326% |

The FINAL evidence register does not freeze F05 validation Top-2, Top-3, Top-5, or validation Weighted F1 as official values. They should remain unreported unless retrieved directly from the preserved history/checkpoint evidence.

### 9.3 Post-best epochs

Recorded validation Macro F1 after the best checkpoint:

| Epoch | Validation Macro F1 |
|---:|---:|
| 20 | **89.9326%** |
| 21 | 88.9476% |
| 22 | 88.9135% |
| 23 | 88.9476% |
| 24 | 89.1033% |
| 25 | 89.1033% |

No epoch after 20 exceeded the locked epoch-20 validation Macro F1.

---

## 10. Official unchanged test result

The epoch-20 checkpoint was evaluated only after model selection had been fixed.

| Test metric | F05 |
|---|---:|
| Top-1 | 91.1290% |
| Top-2 | 97.4566% |
| Top-3 | 98.9454% |
| Top-5 | 99.7519% |
| Macro Precision | 89.7496% |
| Macro Recall | 82.6547% |
| Macro F1 | 84.3487% |
| Weighted F1 | 90.9759% |
| Support | 1,612 |

The checkpoint hash was verified before official-test evaluation, and the official package preserves the predictions, classification report, raw confusion matrix, and row-normalised confusion matrix.

### Macro-versus-weighted gap

```text
Weighted F1 − Macro F1
= 90.9759 − 84.3487
= 6.6272 percentage points
```

The sizeable gap shows that aggregate performance remained uneven across classes despite the combined imbalance interventions.

---

## 11. Class-wise evidence and interpretation boundary

The official package preserves:

```text
tables/F05_classification_report.csv
tables/F05_test_predictions.csv
tables/F05_confusion_matrix.csv
tables/F05_confusion_matrix_normalized.csv
figures/F05_confusion_matrix.png
figures/F05_confusion_matrix_normalized.png
```

These files are the authoritative source for exhaustive per-class discussion.

This phase does **not** invent a complete class ranking from partial evidence. Specific class-wise values should be quoted only after the corresponding preserved report/CM row is directly verified.

Safe current conclusions are:

- F05's Macro F1 is materially below its Weighted F1;
- balanced recognition remained less even than the headline Top-1 score suggests;
- the result is compatible with class-specific weaknesses, but the exact causal source of those weaknesses is not established by a single run.

---

## 12. F05 versus F04 — closest loss-focused comparison

F04 and F05 share:

- ConvNeXt-Small;
- Aug-200 7,284-image training distribution;
- unchanged validation/test sets;
- 224×224 input;
- batch size 16;
- AdamW;
- learning rate `1e-5`;
- weight decay `1e-4`;
- validation Macro-F1 checkpoint selection.

They differ principally in the objective, but their maximum epoch budgets are not identical and the exact F05 patience value is unresolved.

### Official-test comparison

| Metric | F04 | F05 | F05 − F04 |
|---|---:|---:|---:|
| Top-1 | 91.3772% | 91.1290% | −0.2482 pp |
| Top-5 | 99.6278% | 99.7519% | +0.1241 pp |
| Macro Precision | 89.3241% | 89.7496% | +0.4255 pp |
| Macro Recall | 84.3236% | 82.6547% | −1.6689 pp |
| Macro F1 | 86.0035% | 84.3487% | −1.6548 pp |
| Weighted F1 | 91.2641% | 90.9759% | −0.2882 pp |

### Approved interpretation

> Within the completed Aug-200 ConvNeXt configurations, F05 produced slightly higher Top-5 and Macro Precision but lower Top-1, Macro Recall, Macro F1, and Weighted F1 than F04. Because the maximum epoch budgets differed (25 versus 40) and the exact F05 stopping configuration is incomplete, the difference cannot be attributed exclusively to the loss function.

### Prohibited causal wording

Do not write:

> Adding focal loss caused performance to decline.

Do not write:

> Class-balanced focal loss is worse than cross-entropy.

F04–F05 is the campaign's **closest loss-focused comparison**, not a perfectly controlled one-factor causal ablation.

---

## 13. F05 versus F03 — complete-configuration comparison

F03 and F05 both use Gentle Class-Balanced Focal Loss, but they do not isolate the effect of Aug-200 because the complete training configurations differ.

Key differences include:

| Factor | F03 | F05 |
|---|---|---|
| Training distribution | original 5,247 | Aug-200 7,284 |
| Maximum epochs | 40 | 25 |
| Online training transform | richer crop/jitter pipeline | direct resize + flip + rotation |
| Best epoch | 13 | 20 |

Official-test differences, reported as F03 minus F05:

| Metric | F03 − F05 |
|---|---:|
| Top-1 | +0.4963 pp |
| Top-5 | −0.0621 pp |
| Macro Precision | +1.0354 pp |
| Macro Recall | +4.2755 pp |
| Macro F1 | +3.6158 pp |
| Weighted F1 | +0.5897 pp |

### Approved interpretation

> F03 was the stronger balanced 224×224 Gentle-CB-Focal configuration in the tested campaign. F05 did not demonstrate complementarity between Aug-200 and the Gentle Class-Balanced Focal objective under its completed configuration.

### Important boundary

The campaign result does **not** prove that augmentation and class-balanced focal loss are fundamentally incompatible. It reports only what happened under the tested F03/F05 configurations.

---

## 14. Scientific finding

The evidence supports the following configuration-level conclusion:

> The data-level and loss-level imbalance interventions were not complementary under the tested F05 configuration. The original-distribution F03 branch retained stronger official-test balanced metrics, and the Aug-200 cross-entropy F04 branch also achieved higher Macro Recall and Macro F1 than F05.

The higher validation Macro F1 of F05 did not translate into correspondingly stronger balanced performance on the unchanged official test set.

F05 therefore remains a valuable **negative/over-correction ablation** in the campaign, but it does not replace:

- F03 as the strongest isolated imbalance-intervention branch at 224×224; or
- F06 as the primary balanced single-model result.

---

## 15. Checkpoint and package integrity

### 15.1 Locked checkpoint

```text
File: weights/best_model.pth
Best epoch: 20
SHA-256:
548e7c7d925f0272d06a5e57b682364b2affe302ff5c7c5071461ebde0b2e3f3
```

The checkpoint SHA-256 was recomputed and matched before the official-test evaluation.

Checkpoint byte size: **To be verified**.

### 15.2 Official package

```text
aqua20_f05_convnext_small_aug200_gentle_focal_outputs.zip
```

Notebook-reported package size:

```text
524.63 MB
```

The package contains 13 curated files:

```text
README_F05.txt

figures/
    F05_confusion_matrix.png
    F05_confusion_matrix_normalized.png

tables/
    F05_best_epoch_summary.csv
    F05_classification_report.csv
    F05_confusion_matrix.csv
    F05_confusion_matrix_normalized.csv
    F05_scientific_comparison.csv
    F05_target_comparison.csv
    F05_test_metrics_summary.csv
    F05_test_predictions.csv
    F05_training_history.csv

weights/
    best_model.pth
```

The original ZIP SHA-256, exact ZIP byte count, and extracted-tree SHA-256 remain **To be verified**.

---

## 16. Remaining items to verify

The following should not be guessed:

1. exact numeric F05 `PATIENCE` assignment;
2. immutable upstream pretrained-weight file SHA-256 and full provenance manifest;
3. original F05 ZIP SHA-256;
4. exact original ZIP byte count;
5. F05 checkpoint byte size;
6. extracted-tree SHA-256;
7. AdamW betas, epsilon, amsgrad, and exact parameter-group details beyond the preserved constructor;
8. gradient-clipping status for F05;
9. gradient-accumulation status;
10. warm-up status;
11. complete immutable container/image identifier;
12. NVIDIA driver and direct native-cuDNN runtime record;
13. CPU, host RAM, storage, and full machine manifest;
14. exact run-local versions of scikit-learn, pandas, NumPy, Pillow, and all auxiliary packages;
15. immutable dataset revision/content hash and complete duplicate/decode boundaries;
16. per-image Aug-200 random-draw log;
17. controlled parameter-count, FLOPs, GPU memory, latency, throughput, and energy measurements;
18. repeated-seed optimisation variance;
19. exhaustive class-wise interpretation beyond directly verified archived rows.

---

## 17. Approved and prohibited claims

### Approved

- F05 is `F05_CONVNEXT_SMALL_AUG200_GENTLE_CB_FOCAL`.
- F05 combines Aug-200 and Gentle Class-Balanced Focal Loss.
- F05 completed 25 epochs and selected epoch 20 by validation Macro F1.
- F05 achieved 91.1290% Top-1 and 84.3487% Macro F1 on the unchanged 1,612-image test set.
- Under the tested configuration, F05 did not improve Macro Recall or Macro F1 over either F03 or F04.
- F04–F05 is the closest loss-focused comparison in the Aug-200 ConvNeXt branch, with an unequal maximum epoch budget caveat.
- F03–F05 is a complete-configuration comparison.
- The official test set was not used to select the F05 checkpoint.

### Prohibited

- “F05 proves focal loss is worse than cross-entropy.”
- “Aug-200 and class-balanced focal loss are always incompatible.”
- “F04–F05 is a perfectly controlled loss-only ablation.”
- “F03–F05 isolates only the effect of augmentation.”
- “F05 used patience 8” unless direct F05 evidence for that value is located.
- “F05 was fully bitwise deterministic.”
- “F05 is the final proposed method.”
- “F02 is the ConvNeXt cross-entropy baseline.”

---

## 18. Reviewer and viva question bank

### Q1. What exactly is F05?

F05 is a ConvNeXt-Small classifier trained at `224×224` on the 7,284-image Aug-200 training distribution using Gentle Class-Balanced Focal Loss.

### Q2. Why was F05 included?

To test whether the data-level balancing intervention and loss-level imbalance intervention were complementary when combined.

### Q3. What is meant by “Gentle” focal loss?

Only that the focal exponent is moderate, `gamma=1.0`. The underlying focal and effective-number components are established methods.

### Q4. What beta value was used?

`beta=0.999`.

### Q5. How were class weights calculated?

From the actual Aug-200 training counts using effective-number weighting, followed by mean-one normalisation.

### Q6. Were the original class counts used for F05 weights?

No. The weights were calculated from the 7,284-image Aug-200 distribution.

### Q7. Was Aug-200 applied to validation or test?

No. It was training only.

### Q8. Is offline Aug-200 the same as online training augmentation?

No. Aug-200 permanently created training files. F05 also applied dynamic resize/flip/rotation transforms when loading those files.

### Q9. What model initialization was used?

Torchvision ConvNeXt-Small with `ConvNeXt_Small_Weights.DEFAULT`, which resolved in the preserved runtime to `IMAGENET1K_V1`, followed by replacement of the classifier with a 20-class linear layer.

### Q10. What optimizer and learning rate were used?

AdamW with learning rate `1e-5` and weight decay `1e-4`.

### Q11. What scheduler was used?

CosineAnnealingLR with `T_max=25`. No custom `eta_min` argument was found in the preserved constructor.

### Q12. How was the best checkpoint chosen?

By a strict increase in validation Macro F1. Ties did not replace the current best.

### Q13. What was the best epoch?

Epoch 20.

### Q14. Did training stop early?

The run completed all 25 planned epochs. An early-stopping mechanism existed, but the exact numeric F05 patience value remains To be verified.

### Q15. Was the official test set used for model selection?

No. Official test evaluation occurred after the checkpoint had been fixed.

### Q16. What were the main official-test results?

Top-1 was 91.1290%, Top-5 was 99.7519%, Macro Recall was 82.6547%, Macro F1 was 84.3487%, and Weighted F1 was 90.9759%.

### Q17. What did F05 show relative to F04?

F05 had slightly higher Top-5 and Macro Precision but lower Macro Recall and Macro F1. This is a configuration-level result, not proof that focal loss caused the decline.

### Q18. What did F05 show relative to F03?

F03 retained higher Top-1, Macro Precision, Macro Recall, Macro F1, and Weighted F1. Because the training distributions, budgets, and online transforms differ, this is a complete-configuration comparison.

### Q19. Why is the Macro-F1 result important?

Macro F1 weights each class equally, so it better reflects balanced recognition under the severe class imbalance than headline Top-1 alone.

### Q20. What is the principal reproducibility limitation for F05?

The core model, data protocol, transforms, loss, training chronology, checkpoint, and metrics are substantially recoverable, but some low-level optimiser/stopping details, immutable package hashes, complete environment metadata, efficiency measurements, and repeated-seed variance remain unavailable.

---

## 19. Evidence ledger

Primary evidence used to freeze this phase:

- latest final manuscript, *Improving Underwater Marine Species Classification through Imbalance-Aware Learning and Cross-Dataset Evaluation*;
- `P00_AQUA20_Master_Evidence_Freeze_FINAL.xlsx`;
- `P00_AQUA20_Evidence_Freeze_Report_FINAL.md`;
- `experiment_readme_index.md`;
- `methodology_control_audit.md`;
- `README_F05.txt`;
- `04-07-2026-aqua20-final-beat-base-plan.ipynb`;
- `F05_best_epoch_summary.csv`;
- `F05_training_history.csv`;
- `F05_test_metrics_summary.csv`;
- `F05_target_comparison.csv`;
- `F05_scientific_comparison.csv`;
- `F05_classification_report.csv`;
- `F05_test_predictions.csv`;
- `F05_confusion_matrix.csv`;
- `F05_confusion_matrix_normalized.csv`;
- `F05_confusion_matrix.png`;
- `F05_confusion_matrix_normalized.png`;
- `weights/best_model.pth`;
- prior F03 documentation;
- prior F04 documentation;
- Phase 6-to-Phase 7 handover.

---

## 20. Phase 7 completion decision

Phase 7 is complete.

The following are now documented and locked:

- exact F05 identity and combined-intervention role;
- separation of F05 from F02/F04;
- 7,284-image Aug-200 training distribution and unchanged validation/test splits;
- offline Aug-200 versus online F05 transform distinction;
- ConvNeXt-Small model construction and resolved ImageNet pretrained enum;
- F05 torchvision-v2 training/evaluation transforms;
- Gentle Class-Balanced Focal implementation, `beta=0.999`, `gamma=1.0`, Aug-200 class counts, and mean-one class weights;
- AdamW, learning rate, weight decay, `T_max=25`, workers, AMP, loader, seed, and backend evidence;
- 25/25 epoch chronology and epoch-20 validation-Macro-F1 checkpoint;
- unchanged official-test metrics;
- package contents and checkpoint SHA-256;
- F04/F05 and F03/F05 comparison boundaries;
- unsupported fields retained as **To be verified**;
- F05 reviewer/viva answers.

The next phase is:

**Phase 8 — F06 ConvNeXt-Small HR384 Gentle Class-Balanced Focal Continuation Experiment Documentation.**
