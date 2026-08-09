# F07 — ConvNeXt-Small HR384 Deterministic Two-View TTA

## 1. Experiment identity and status

**Official experiment ID**

`F07_CONVNEXT_SMALL_HR384_TTA`

**Repository file**

`documentation/experiments/F07_convnext_hr384_tta.md`

**Campaign**

`04-07-2026 AQUA20 FINAL BEAT BASE PLAN`

**Status**

Completed, packaged, frozen, and retained as an inference-only ablation.

**Scientific role**

F07 is a deterministic two-view test-time-augmentation inference experiment applied to the already-frozen F06 high-resolution ConvNeXt-Small checkpoint.

F07 performs:

- no training;
- no fine-tuning;
- no optimiser or scheduler step;
- no early stopping;
- no new checkpoint creation;
- no test-driven protocol selection.

Approved interpretation:

> F07 directly compares frozen F06 single-view inference with one pre-specified deterministic two-view rule: original plus horizontal-flip probability averaging.

Prohibited interpretation:

> F07 is a newly trained model.

Prohibited generalisation:

> Test-time augmentation is harmful in general.

The evidence supports only a conclusion about this specific F07 two-view protocol on this fixed AQUA20 evaluation.

---

## 2. Evidence hierarchy used for this record

Evidence was reconciled in the following order:

1. latest final manuscript, *Improving Underwater Marine Species Classification through Imbalance-Aware Learning and Cross-Dataset Evaluation*;
2. FINAL master evidence-freeze and frozen terminology/index records;
3. official F07 package/README and locked F07 protocol;
4. frozen F06 source-checkpoint verification;
5. inherited F06 384×384 deterministic evaluation preprocessing;
6. F07 validation single-view-versus-TTA comparison and predictions;
7. official-test predictions, metrics, classification report, raw and normalised confusion matrices;
8. FINAL source-integrity/package records;
9. preserved `04-07-2026-aqua20-final-beat-base-plan.ipynb` campaign notebook;
10. older handovers/audits only as corroborating historical context.

Where older summaries are less complete than direct F07 run evidence or the FINAL evidence freeze, the later direct/frozen evidence controls this document.

---

## 3. Exact scientific relationship to F06

F07 does not alter the learned model parameters. It reuses the frozen F06 best checkpoint exactly.

| Field | Locked value |
|---|---|
| Source experiment | `F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL` |
| Source model | ConvNeXt-Small |
| Source best epoch | 2 |
| Source resolution | `384×384` |
| Source checkpoint | F06 `best_model.pth` |
| Source SHA-256 | `d0c225c28b25f09822bc99518a37f9b1fe642b2d01213da7f68e2d76a255832a` |
| F07 training | None |
| F07 new checkpoint | None |

The preserved F07 workflow recomputed the SHA-256 of the F06 checkpoint and asserted equality with the expected frozen hash before packaging F07.

Therefore F06-versus-F07 is a **direct inference-rule comparison**:

```text
same frozen F06 weights
same 20-class output space
same validation/test image sets
same 384×384 base evaluation preprocessing

changed factor:
single original view
vs
original + horizontal-flip probability averaging
```

No training-stage causal claim is attached to F07.

---

## 4. Dataset, split, and test-isolation protocol

F07 uses the same fixed AQUA20 partitions and class order as the completed campaign.

| Item | Locked value |
|---|---:|
| Classes | 20 |
| Validation images | 1,312 |
| Official test images | 1,612 |
| Validation/test class order | Frozen AQUA20 20-class order |
| Resolution per view | `384×384` |
| Training data used by F07 | None |
| Aug-200 used by F07 | No |

The official 1,612-image test split was not used to:

- train or fine-tune a model;
- choose a checkpoint;
- invent or select the horizontal-flip view;
- choose a TTA weight;
- alter the aggregation rule;
- tune F07 after validation inspection.

The preserved notebook states that the TTA configuration was already locked before validation comparison and was not modified afterward. The official test was then used only for the final evaluation of that fixed rule.

This distinction is important: F07 has **no checkpoint selection** and **no validation-driven TTA search**. Validation was used to record the behaviour of an already-fixed protocol, not to create a new rule.

---

## 5. Inherited F06 evaluation preprocessing

F07 inherits F06's deterministic HR384 evaluation transform.

```text
Direct Resize 384×384
→ ToTensor
→ ImageNet normalisation
```

ImageNet normalisation:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

For the F07 implementation, the horizontally reflected second view is generated from the transformed input tensor using a width-axis horizontal flip.

There is no F07 evaluation-time use of:

- random crop;
- rotation;
- colour jitter;
- vertical flip;
- multi-scale inference;
- test-time training;
- checkpoint ensembling;
- learned or validation-tuned TTA weights.

---

## 6. Locked deterministic two-view inference rule

Let the frozen F06 model produce logits for the deterministic HR384 input tensor `x` and for its horizontal reflection `flip(x)`.

### 6.1 View 1 — original

\[
\mathbf{p}_{\mathrm{orig}}
=
\operatorname{softmax}(f_{\mathrm{F06}}(x)).
\]

### 6.2 View 2 — horizontal reflection

\[
\mathbf{p}_{\mathrm{flip}}
=
\operatorname{softmax}(f_{\mathrm{F06}}(\operatorname{flip}(x))).
\]

### 6.3 Arithmetic probability mean

\[
\mathbf{p}_{\mathrm{F07}}
=
\frac{
\mathbf{p}_{\mathrm{orig}}
+
\mathbf{p}_{\mathrm{flip}}
}{2}.
\]

### 6.4 Final predicted class

\[
\hat{y}_{\mathrm{F07}}
=
\arg\max_c
\mathbf{p}_{\mathrm{F07},c}.
\]

Top-k rankings are also computed from the averaged probability vector rather than from either individual view alone.

The preserved dry run verified:

- two `8×20` logit outputs from an `8×3×384×384` input batch;
- one `8×20` averaged TTA probability tensor;
- probability sum of `1.000000` for the checked sample;
- class-order consistency;
- successful TTA pipeline execution.

---

## 7. Protocol timing and anti-leakage decision

The F07 rule was fixed independently of the official-test outcome.

Frozen sequence:

```text
1. Restore and SHA-verify frozen F06 checkpoint.
2. Lock deterministic F07 rule.
3. Verify HR384 model/class-order/TTA pipeline.
4. Record validation single-view versus fixed-TTA behaviour.
5. Do not alter the protocol from validation results.
6. Evaluate the unchanged 1,612-image official test split once under the locked rule.
7. Preserve predictions, metrics, report, confusion matrices, and package outputs.
```

This means the official-test result is an evaluation result, not a selection signal.

---

## 8. Validation comparison — frozen F06 single view versus F07

The validation comparison uses 1,312 unchanged validation images.

| Metric | F06-style single view | F07 TTA | F07 − F06 |
|---|---:|---:|---:|
| Top-1 | 92.9878% | 92.6829% | −0.3049 pp |
| Top-2 | 98.6280% | 98.6280% | 0.0000 pp |
| Top-3 | 99.5427% | 99.4665% | −0.0762 pp |
| Top-5 | 99.7713% | 99.6951% | −0.0762 pp |
| Macro Precision | 88.4581% | 88.5306% | +0.0726 pp |
| Macro Recall | 90.6790% | 91.2770% | +0.5981 pp |
| Macro F1 | 88.5122% | 88.9836% | **+0.4714 pp** |
| Weighted F1 | 93.0475% | 92.7625% | −0.2850 pp |

Validation correct predictions recorded in the preserved notebook:

```text
single view = 1,220 / 1,312
F07 TTA    = 1,216 / 1,312
```

### Validation interpretation

The fixed F07 rule improved validation Macro Recall and Macro F1 even though validation Top-1 and Weighted F1 decreased.

The primary balanced-class observation is:

```text
Macro F1:
88.5122% → 88.9836%
Δ = +0.4714 pp
```

This positive validation effect did not authorize changing the rule; the rule was already frozen.

---

## 9. Official unchanged test result

The official test contains 1,612 images across the frozen 20-class AQUA20 output space.

### 9.1 F07 official-test metrics

| Metric | F07 |
|---|---:|
| Top-1 | **92.121588%** |
| Top-2 | **97.890819%** |
| Top-3 | **99.317618%** |
| Top-5 | **99.689826%** |
| Macro Precision | **89.876543%** |
| Macro Recall | **86.532374%** |
| Macro F1 | **87.402551%** |
| Weighted F1 | **92.077110%** |
| Support | **1,612** |

Rounded manuscript-style values:

```text
Top-1           92.1216%
Top-2           97.8908%
Top-3           99.3176%
Top-5           99.6898%
Macro Precision 89.8765%
Macro Recall    86.5324%
Macro F1        87.4026%
Weighted F1     92.0771%
```

### 9.2 Auxiliary NLL

The preserved F07 evaluation reports standard softmax negative log-likelihood:

```text
NLL = 0.248791
```

This value must **not** be compared directly with the previously reported F06 Gentle Class-Balanced Focal test loss because they are different objectives/scales.

---

## 10. Direct official-test comparison with frozen F06

F06-versus-F07 is the cleanest interpretation of F07 because the model weights remain fixed and only the inference rule changes.

| Metric | F06 single view | F07 TTA | F07 − F06 |
|---|---:|---:|---:|
| Top-1 | 92.1216% | 92.1216% | **0.0000 pp** |
| Top-2 | 97.9529% | 97.8908% | −0.0620 pp |
| Top-3 | 99.2556% | 99.3176% | +0.0620 pp |
| Top-5 | 99.7519% | 99.6898% | −0.0620 pp |
| Macro Precision | 90.4312% | 89.8765% | −0.5547 pp |
| Macro Recall | 87.4362% | 86.5324% | **−0.9038 pp** |
| Macro F1 | 88.1699% | 87.4026% | **−0.7674 pp** |
| Weighted F1 | 92.0810% | 92.0771% | −0.0039 pp |

The F07 evaluation code also reproduced the frozen F06 single-view official-test metrics before comparing the TTA result, including the same Top-1 and Macro-F1 values. This is an important internal consistency check.

---

## 11. Central F07 scientific finding

The required negative-result record is:

> Deterministic original-plus-horizontal-flip probability averaging improved validation Macro F1 but did not improve F06 on the official test; balanced test metrics decreased under the tested F07 protocol.

More specifically:

```text
Validation Macro F1:
+0.4714 pp

Official-test Top-1:
0.0000 pp

Official-test Macro Recall:
−0.9038 pp

Official-test Macro F1:
−0.7674 pp
```

Therefore F07 does not replace F06 as the stronger completed branch.

F06 remains the primary proposed single model and the stronger balanced-class model in the frozen campaign hierarchy.

### Claim boundary

Permitted:

> The tested deterministic two-view horizontal-flip TTA rule was not complementary to F06 on the official AQUA20 test split.

Not permitted:

> TTA does not work for marine classification.

Not permitted:

> Horizontal flipping always harms ConvNeXt.

Not permitted:

> F07 proves TTA is generally ineffective.

The campaign evaluated one deterministic two-view construction, not the space of possible TTA policies.

---

## 12. Predictions, classification report, and confusion evidence

The official F07 package preserves:

- validation single-view-versus-TTA comparison;
- validation predictions;
- official-test single-view-versus-TTA comparison;
- official F07 test metrics;
- 1,612 official-test prediction rows;
- classification report;
- raw confusion matrix;
- row-normalised confusion matrix;
- raw and normalised confusion-matrix figures;
- F06-versus-F07 scientific comparison.

The FINAL evidence freeze records that:

- source hash and locked protocol were verified;
- all 1,612 official-test prediction rows were present;
- classification metrics were independently recomputed;
- the confusion matrix matched the predictions;
- official Top-k summary evidence was present.

### Class-wise interpretation boundary

The current retrieved Phase 9 evidence confirms the existence and integrity of the complete F07 classification report and confusion matrices, but it does not surface a reliable full set of per-class rows for independent textual interpretation here.

Therefore no new F07-specific strongest/weakest-class story is invented in this document.

**Exhaustive F07 per-class interpretation: To be verified from the direct `F07_classification_report.csv` and confusion-matrix rows if later required.**

This restriction does not affect the frozen aggregate F07 result.

---

## 13. Inference runtime fields supported by direct evidence

The preserved F07 validation/test notebook directly supports the following runtime fields for the shown inference path:

| Field | Directly supported value |
|---|---|
| Device abstraction | CUDA (`DEVICE` printed as `cuda`) |
| Model mode | `model.eval()` |
| Autograd mode | `torch.inference_mode()` |
| Batch size | 8 |
| Test shuffle | False |
| DataLoader workers | 2 |
| `pin_memory` | True |
| Device transfer | non-blocking enabled in evaluation loop |
| Views per sample | 2 |
| Per-view softmax | Yes |
| Probability aggregation | Arithmetic mean |

The preserved F07 inference loop does not show an explicit autocast wrapper. This does not justify reconstructing an exact low-level precision statement beyond the archived code path.

Still **To be verified**:

- exact run-local GPU model for the F07 execution itself;
- exact floating-point/autocast runtime state beyond the shown code;
- complete software/container/driver environment for F07;
- exact scikit-learn version used for the run;
- wall-clock inference latency;
- images/second throughput;
- GPU-memory consumption;
- energy/compute cost;
- controlled single-view-versus-two-view efficiency comparison.

---

## 14. Official package and integrity record

**Official package**

`aqua20_f07_convnext_small_hr384_tta_outputs.zip`

The notebook displayed an approximate ZIP size of:

```text
0.10 MB
```

This display value is not treated as an exact byte count.

### 14.1 Curated package contents

The official F07 package contains 14 files:

```text
F07_locked_protocol.json
F07_source_checkpoint_sha256.txt
README_F07.txt
figures/F07_confusion_matrix.png
figures/F07_confusion_matrix_normalized.png
tables/F07_classification_report.csv
tables/F07_confusion_matrix.csv
tables/F07_confusion_matrix_normalized.csv
tables/F07_official_test_metrics.csv
tables/F07_scientific_comparison.csv
tables/F07_test_predictions.csv
tables/F07_test_single_vs_tta.csv
tables/F07_validation_predictions.csv
tables/F07_validation_single_vs_tta.csv
```

No F07 weight file is included because F07 produces no new checkpoint.

### 14.2 Frozen extracted-package integrity

| Field | Locked value |
|---|---|
| Curated files | 14 |
| Extracted bytes | 297,339 |
| Extracted-tree SHA-256 | `e2d6066da6fa53632f2cd12853b3c6938d2341eae8065ae74810f00d0f15d681` |
| Checkpoints inside F07 package | 0 |
| Original ZIP SHA-256 | **Not collected / To be verified** |
| Exact original ZIP byte count | **To be verified** |

The deterministic extracted-tree hash is the frozen package-integrity record currently available for F07.

---

## 15. Reproducibility boundaries

The following statements are supported:

- exact F07 identity is frozen;
- exact F06 source identity, epoch, and SHA-256 are frozen;
- no F07 training occurred;
- no new F07 checkpoint exists;
- deterministic two-view rule is frozen;
- inherited 384×384 evaluation transform is frozen;
- validation and official-test sizes are frozen;
- the official test was evaluated only after the rule was locked;
- full aggregate validation and official-test metrics are frozen;
- test prediction support, report, confusion matrix, and Top-k evidence passed audit;
- F07 package extracted tree is integrity-hashed.

The following remain **To be verified** and must not be reconstructed from generic defaults:

1. original F07 ZIP SHA-256;
2. exact original ZIP byte count;
3. exact F07 run-local hardware model;
4. complete F07 software/container/driver manifest;
5. exact F07 precision/autocast runtime state beyond the preserved inference loop;
6. controlled latency/throughput/GPU-memory/energy measurements;
7. exhaustive class-wise narrative unless direct F07 report rows are separately surfaced;
8. immutable source-dataset revision/content hash and other dataset-level boundaries inherited from earlier phases.

---

## 16. F07 reviewer / viva question bank

### Q1. What is the exact F07 experiment identity?

`F07_CONVNEXT_SMALL_HR384_TTA`.

### Q2. Was F07 trained?

No. F07 is inference-only.

### Q3. Did F07 create a new checkpoint?

No. It reuses the frozen F06 checkpoint and produces no new model weights.

### Q4. Which checkpoint does F07 use?

The validation-selected F06 best checkpoint from epoch 2.

### Q5. What is the exact F06 source SHA-256?

`d0c225c28b25f09822bc99518a37f9b1fe642b2d01213da7f68e2d76a255832a`.

### Q6. How was the source checkpoint verified?

The preserved F07 packaging workflow recomputed its SHA-256 and asserted equality with the frozen expected hash.

### Q7. What are the two F07 views?

The deterministic original HR384 input and its horizontal reflection.

### Q8. What preprocessing is inherited before inference?

Direct resize to `384×384`, tensor conversion, and ImageNet normalisation.

### Q9. Does F07 use random evaluation augmentation?

No. Its only additional deterministic view is the specified horizontal flip.

### Q10. Is the flip applied before or after the deterministic tensor pipeline in the preserved code?

The preserved implementation flips the transformed input tensor along the width dimension using `torch.flip(..., dims=[3])`.

### Q11. How are the two views combined?

Softmax is applied separately to each view's logits, then the two class-probability vectors are averaged arithmetically.

### Q12. Are logits averaged before softmax?

No. The frozen rule averages **per-view softmax probabilities**.

### Q13. How is the final class selected?

By `argmax` of the averaged probability vector.

### Q14. How is Top-k computed?

By ranking classes from the averaged F07 probability vector.

### Q15. Was the F07 rule selected on the official test?

No. The rule was fixed before official-test evaluation.

### Q16. Was the rule changed after validation results were seen?

No. The preserved notebook explicitly states that the already-locked configuration would not be changed from the validation results.

### Q17. What was validation Macro F1 for frozen F06 single-view inference?

88.5122%.

### Q18. What was F07 validation Macro F1?

88.9836%.

### Q19. How much did validation Macro F1 change?

It increased by approximately 0.4714 percentage points.

### Q20. Did all validation metrics improve?

No. Macro Recall and Macro F1 improved, while Top-1, Top-3, Top-5, and Weighted F1 decreased; Top-2 was unchanged.

### Q21. How many official-test images were used?

1,612 unchanged AQUA20 test images.

### Q22. What was F07 official-test Top-1?

92.1216%.

### Q23. Did F07 improve official-test Top-1 over F06?

No. Top-1 was unchanged.

### Q24. What was F07 official-test Macro Recall?

86.5324%.

### Q25. What was F07 official-test Macro F1?

87.4026%.

### Q26. How much did official-test Macro Recall change versus F06?

Approximately −0.9038 percentage points.

### Q27. How much did official-test Macro F1 change versus F06?

Approximately −0.7674 percentage points.

### Q28. Why is F06-versus-F07 a direct inference ablation?

Because the source checkpoint, class space, evaluation images, and base HR384 preprocessing are fixed; the intended changed factor is the single-view versus fixed two-view probability-averaging inference rule.

### Q29. Why does F07 remain scientifically useful if it is a negative result?

It documents that a plausible deterministic inference refinement improved validation balanced metrics but failed to reproduce that benefit on the untouched official test, preventing selective reporting and post-hoc method switching.

### Q30. Can we conclude that TTA is generally ineffective?

No. Only one deterministic original-plus-horizontal-flip protocol was tested.

### Q31. What is the auxiliary F07 NLL?

0.248791 using standard softmax negative log-likelihood.

### Q32. Can that NLL be compared directly with F06's Gentle CB Focal test loss?

No. They are different loss definitions/scales.

### Q33. What runtime fields are directly supported for F07 inference?

CUDA device abstraction, batch size 8, two workers, `shuffle=False`, `pin_memory=True`, `model.eval()`, and `torch.inference_mode()` are directly preserved in the F07 notebook path.

### Q34. Is exact inference latency frozen?

No. Controlled latency and throughput remain To be verified.

### Q35. What is the F07 extracted-tree SHA-256?

`e2d6066da6fa53632f2cd12853b3c6938d2341eae8065ae74810f00d0f15d681`.

### Q36. Is the original F07 ZIP SHA-256 available?

No. It was not collected; this remains To be verified.

### Q37. How many curated files are in the F07 extracted package?

14 files.

### Q38. Why is there no weights file in the F07 package?

Because F07 is inference-only and does not create a new checkpoint.

### Q39. Does F07 replace F06 as the final model?

No. F06 remains the primary proposed single model and stronger balanced-class branch.

### Q40. What is the safest one-sentence F07 conclusion?

> The pre-specified original-plus-horizontal-flip probability average improved validation Macro F1 but did not improve frozen F06 on the official test, where balanced-class metrics declined under this tested two-view rule.

---

## 17. Evidence ledger

Primary evidence used to freeze Phase 9:

- latest final manuscript, *Improving Underwater Marine Species Classification through Imbalance-Aware Learning and Cross-Dataset Evaluation*;
- `P00_AQUA20_Master_Evidence_Freeze_FINAL.xlsx`;
- `P00_AQUA20_Evidence_Freeze_Report_FINAL.md`;
- frozen terminology/index and methodology-control audit records;
- `experiment_readme_index.md`;
- `README_F07.txt`;
- `F07_locked_protocol.json` as reproduced/preserved by the campaign notebook;
- `F07_source_checkpoint_sha256.txt` as reproduced/preserved by the campaign notebook;
- `F07_validation_single_vs_tta.csv`;
- `F07_validation_predictions.csv`;
- `F07_test_single_vs_tta.csv`;
- `F07_official_test_metrics.csv`;
- `F07_test_predictions.csv`;
- `F07_classification_report.csv`;
- `F07_confusion_matrix.csv`;
- `F07_confusion_matrix_normalized.csv`;
- F07 raw and normalised confusion-matrix figures;
- FINAL source-integrity manifest/tree-hash record;
- `04-07-2026-aqua20-final-beat-base-plan.ipynb`;
- `F06_convnext_hr384.md`;
- `H08_to_09.md`.

---

## 18. Phase 9 completion decision

Phase 9 is complete.

The following are now documented and locked:

- exact F07 identity;
- inference-only scientific role;
- no-training/no-new-checkpoint boundary;
- frozen F06 source experiment, best epoch, and SHA-256;
- inherited deterministic HR384 preprocessing;
- exact original-plus-horizontal-flip views;
- softmax per view;
- arithmetic probability averaging;
- final `argmax` prediction rule;
- Top-k ranking from averaged probabilities;
- protocol fixed before validation/test analysis and unchanged after validation;
- official-test isolation;
- complete frozen validation metrics and F06 comparison;
- complete frozen official-test metrics and direct F06 comparison;
- central validation-positive/test-negative result;
- package contents, extracted bytes, and extracted-tree SHA-256;
- directly supported F07 inference-loader/runtime fields;
- classification-report/confusion/prediction integrity;
- unsupported fields retained as **To be verified**;
- reviewer/viva question bank;
- strict claim boundary that the negative result applies only to the tested deterministic two-view protocol.

The next documentation phase is **Phase 10 — F08 ConvNeXt-Small HR384 + YOLO26m Validation-Selected Probability Ensemble Documentation**.
