# F03 — ConvNeXt-Small Gentle Class-Balanced Focal

## 1. Official identity

```text
F03_CONVNEXT_SMALL_GENTLE_CB_FOCAL
```

**Repository file:** `documentation/experiments/F03_convnext_focal.md`

**Scientific role:** loss-level imbalance-aware ConvNeXt branch, strongest isolated imbalance intervention within the tested campaign, and frozen source checkpoint later continued by F06.

F03 is not the final proposed single model. F06 is the primary balanced single model.

### Claim boundary

No architecture-matched ConvNeXt-Small + standard cross-entropy run on the same original 5,247-image training distribution was found. Therefore F03 must not be described as a quantified loss-only causal improvement over such a control.

F02 remains `F02_YOLO26S_CLS_AUG200`; F04 is the ConvNeXt-Small + Aug-200 + standard cross-entropy experiment.

## 2. Data protocol

| Item | Locked value |
|---|---:|
| Dataset | AQUA20 |
| Classes | 20 |
| Training distribution | Original, naturally imbalanced |
| Training images | 5,247 |
| Validation images | 1,312 |
| Official test images | 1,612 |
| Resolution | 224×224 |
| Split/campaign seed | 42 |
| Aug-200 | disabled |

The official test was excluded from training, early stopping, and checkpoint selection.

## 3. Model construction

F03 used torchvision ConvNeXt-Small:

```python
weights = ConvNeXt_Small_Weights.DEFAULT
model = convnext_small(weights=weights)
model.classifier[2] = nn.Linear(768, 20)
```

The preserved runtime mapping resolves `DEFAULT` to:

```text
ConvNeXt_Small_Weights.IMAGENET1K_V1
```

Locked fields:

| Field | Value |
|---|---|
| Constructor | `torchvision.models.convnext_small` |
| Pretrained weights | ImageNet-1K V1 |
| Classifier input | 768 |
| Classifier output | 20 |
| Training | end-to-end fine-tuning |

The immutable upstream pretrained-weight file hash/provenance remains **To be verified**.

## 4. Effective preprocessing

### Training

```text
Resize 256×256
→ RandomResizedCrop 224
   scale 0.80–1.00
   ratio 0.75–1.3333
→ RandomHorizontalFlip p=0.5
→ RandomRotation −10° to +10°
→ ColorJitter
   brightness 0.88–1.12
   contrast   0.88–1.12
   saturation 0.88–1.12
   hue       −0.03 to +0.03
→ ToTensor
→ ImageNet normalisation
```

Resize/crop used bilinear interpolation with antialiasing. Rotation used nearest-neighbour interpolation, no expansion, fill 0.

### Validation / official test

```text
Direct Resize 224×224
→ ToTensor
→ ImageNet normalisation
```

No CenterCrop and no random evaluation augmentation.

### Normalisation

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

## 5. Gentle Class-Balanced Focal objective

```text
beta  = 0.999
gamma = 1.0
```

For a class `y` with `n_y` training examples:

\[
\widetilde{w}_y =
\frac{1-\beta}{1-\beta^{n_y}}
\]

The class weights were normalised to mean one:

\[
w_y =
\frac{\widetilde{w}_y}
{\operatorname{mean}(\widetilde{w})}
\]

Per-sample loss:

\[
\mathcal{L}
=
-w_y(1-p_y)^\gamma\log(p_y)
\]

Preserved implementation path:

```text
unreduced cross-entropy
→ pt = exp(−CE)
→ focal factor
→ target-indexed class weight
→ per-sample product
→ mean
```

“Gentle” refers only to `gamma=1.0`; this is not claimed as a newly invented loss family.

### Frozen class counts and normalised weights

| Class | Train | Weight |
|---|---:|---:|
| coral | 1249 | 0.062978 |
| crab | 34 | 1.343341 |
| diver | 41 | 1.117873 |
| eel | 128 | 0.373767 |
| fish | 1760 | 0.054254 |
| fishInGroups | 218 | 0.229268 |
| flatworm | 40 | 1.145251 |
| jellyfish | 78 | 0.598467 |
| marine_dolphin | 16 | 2.829116 |
| octopus | 16 | 2.829116 |
| rayfish | 305 | 0.170836 |
| seaAnemone | 712 | 0.088178 |
| seaCucumber | 28 | 1.626336 |
| seaSlug | 63 | 0.735485 |
| seaUrchin | 92 | 0.510912 |
| shark | 57 | 0.810493 |
| shrimp | 18 | 2.517280 |
| squid | 19 | 2.385981 |
| starfish | 133 | 0.360597 |
| turtle | 240 | 0.210468 |

## 6. Training configuration

| Field | Locked value |
|---|---|
| Maximum epochs | 40 |
| Completed epochs | 21 |
| Batch | 16 |
| Workers | 4 |
| Prefetch factor | 2 |
| Optimiser | AdamW |
| Learning rate | `1e-5` |
| Weight decay | `1e-4` |
| Scheduler | CosineAnnealingLR |
| `T_max` | 40 |
| `eta_min` | **To be verified** |
| Patience | 8 |
| Selection | validation Macro F1 |
| Gradient clip | max norm 1.0 |
| AMP | enabled |
| Seed | 42 |
| cuDNN deterministic | true |
| cuDNN benchmark | false |
| Recorded GPU | Tesla P100-PCIE-16GB |

AdamW non-explicit defaults/parameter groups and gradient-accumulation status must not be inferred.

## 7. Epoch chronology and selection

Checkpoint replacement rule:

```python
is_best = val_macro_f1 > best_val_macro_f1
```

Therefore ties did not replace the current best.

| Item | Value |
|---|---:|
| Planned maximum | 40 |
| Best epoch | 13 |
| Completed epochs | 21 |
| Patience | 8 |
| Stop condition | eight consecutive non-improving epochs |

Epoch 13 was the unique best validation Macro-F1 epoch. Epochs 14–21 did not exceed it, so training stopped after the patience counter reached eight.

The official test was evaluated only after the epoch-13 checkpoint had been locked.

## 8. Locked validation result

| Metric | Value |
|---|---:|
| Top-1 | 92.5305% |
| Top-2 | 97.6372% |
| Top-3 | 99.4665% |
| Macro Precision | 87.9929% |
| Macro Recall | 92.6189% |
| Macro F1 | 89.3301% |

Checkpoint selection used **Macro F1 = 89.3301%** only.

## 9. Official unchanged test result

| Metric | Value |
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

The 1,612 preserved prediction rows independently reproduce the frozen metrics. Classification report and confusion matrices passed package audit.

Weighted F1 exceeded Macro F1 by approximately **3.6011 percentage points**.

## 10. Class-wise evidence

One locked minority-class observation is flatworm recall:

| Experiment | Flatworm recall |
|---|---:|
| F01 | 15.38% |
| F02 | 23.08% |
| F03 | 76.92% |

F03 exceeded F01 by 61.54 pp and F02 by 53.84 pp on flatworm recall.

Flatworm support is only 13 test images, so one image changes recall by roughly 7.69 pp. This result is therefore descriptive configuration-level evidence, not proof that the loss alone caused the improvement.

## 11. Valid comparison boundaries

### F03 versus F04

F03 minus F04:

| Metric | Difference |
|---|---:|
| Top-1 | +0.2481 pp |
| Top-5 | +0.0620 pp |
| Macro Precision | +1.4609 pp |
| Macro Recall | +2.6066 pp |
| Macro F1 | +1.9610 pp |
| Weighted F1 | +0.3015 pp |

This is a **complete-configuration** comparison because training distribution, objective, and effective transforms differ.

Do not write that the focal loss alone caused the +1.9610 pp Macro-F1 difference.

### F03 versus F05

F03 minus F05:

| Metric | Difference |
|---|---:|
| Top-1 | +0.4963 pp |
| Top-5 | −0.0621 pp |
| Macro Precision | +1.0354 pp |
| Macro Recall | +4.2755 pp |
| Macro F1 | +3.6158 pp |
| Weighted F1 | +0.5897 pp |

Both use the same loss family, but training distributions and maximum budgets differ.

### F03 versus F06

F06 minus F03:

| Metric | Difference |
|---|---:|
| Top-1 | +0.4963 pp |
| Top-5 | +0.0621 pp |
| Macro Precision | −0.3538 pp |
| Macro Recall | +0.5060 pp |
| Macro F1 | +0.2054 pp |
| Weighted F1 | +0.5154 pp |

F06 is a continuation-stage complete configuration. The difference must not be attributed to resolution alone.

## 12. Checkpoint and package integrity

### Best checkpoint

```text
weights/best_model.pth
epoch = 13
SHA-256 =
928c8af60aef70537263e2c8cc1cf13782747bc5974f86ff054fbe352065c26f

bytes = 198,009,769
```

### Official package

```text
aqua20_f03_convnext_small_gentle_cb_focal_outputs.zip
```

| Integrity field | Value |
|---|---|
| ZIP SHA-256 | `973a0bc8d15398992a157b33d5171b7f04d9b4caa058b9e162f6c8395278a3b1` |
| ZIP bytes | 184,461,150 |
| Files | 23 |
| CRC | PASS |
| Predictions | 1,612 |
| Confusion matrices | PASS |
| Metrics | PASS — recomputed |
| Extracted-tree SHA-256 | **To be verified** |

## 13. Allowed and prohibited claims

### Allowed

- F03 used the original 5,247-image training distribution.
- F03 used ConvNeXt-Small with ImageNet-1K V1 pretrained weights.
- F03 used Gentle Class-Balanced Focal Loss with `beta=0.999`, `gamma=1.0`, and mean-one effective-number weights.
- Epoch 13 was selected strictly by validation Macro F1.
- F03 completed 21 of 40 planned epochs.
- F03 achieved 87.9645% Macro F1 on the official test.
- F03 is the strongest isolated imbalance intervention within the tested campaign.
- F03 is the source checkpoint for F06.

### Do not write

- F03 proves focal loss is causally superior to an original-distribution ConvNeXt cross-entropy baseline.
- such a matched original-distribution CE control existed in the frozen campaign.
- F03 used Aug-200.
- F03 is the final primary model.
- F06's gain over F03 was caused only by higher resolution.
- all classes were equally solved.
- the result is robust across training seeds without repeated-run evidence.
- exact bitwise reproduction is guaranteed.

## 14. Still To be verified

- immutable upstream pretrained-weight SHA/provenance;
- AdamW betas, epsilon, amsgrad, exact parameter groups;
- scheduler `eta_min`;
- gradient-accumulation status;
- complete immutable original run-local package/container/driver/native-cuDNN/CPU/RAM manifest;
- exact auxiliary-library versions;
- extracted package tree SHA-256;
- controlled parameters/FLOPs/memory/latency/throughput;
- repeated-seed optimisation variance;
- immutable dataset revision/content hash and complete duplicate/decode boundaries;
- exhaustive class-wise interpretation beyond archived report/confusion evidence.

## 15. Reviewer / viva quick questions

**What is F03?**  
ConvNeXt-Small trained on the original 5,247-image distribution with Gentle Class-Balanced Focal Loss at 224×224.

**Why is F03 called the strongest isolated imbalance intervention?**  
It applies the loss-level imbalance mechanism without Aug-200 and produced the strongest tested isolated imbalance-aware result, but there is no matched original-distribution CE control.

**What is the loss configuration?**  
Effective-number class weighting with `beta=0.999`, mean-one normalisation, and focal `gamma=1.0`.

**How was the checkpoint selected?**  
Strictly by validation Macro F1.

**What was the best epoch?**  
Epoch 13.

**Why did training stop at 21 epochs?**  
Eight consecutive later epochs failed to exceed the epoch-13 validation Macro F1.

**What was the official-test Macro F1?**  
87.9645%.

**Why can F03 versus F04 not be called a loss-only ablation?**  
Their training distributions, objectives, and effective transforms differ.

**Why did F06 continue F03 rather than F05?**  
F03 preserved the isolated original-distribution loss-level branch; continuing F05 would also carry Aug-200 into the high-resolution stage.
