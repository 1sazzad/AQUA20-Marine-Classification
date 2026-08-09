# F08 ConvNeXt-Small HR384 + YOLO26m Validation-Selected Probability Ensemble

## 1. Experiment identity and documentation status

**Official experiment identity**

```text
F08_CONVNEXT_SMALL_HR384_YOLO26M_PROB_ENSEMBLE
```

**Repository document**

```text
documentation/experiments/F08_convnext_yolo_ensemble.md
```

**Campaign**

```text
04-07-2026 AQUA20 FINAL BEAT BASE PLAN
```

**Scientific mode**

```text
Inference-only heterogeneous probability ensemble
Training: NONE
Fine-tuning: NONE
New checkpoint: NONE
```

F08 is a validation-governed probability-level ensemble of two already completed and frozen model families:

- F06 ConvNeXt-Small HR384;
- F01 YOLO26m-cls.

F08 does not create a new trained model. It combines the class probabilities of the two frozen source models after preserving each source model's own evaluation pipeline and verifying identical AQUA20 class order.

---

## 2. Evidence basis and source hierarchy

Phase 10 follows the locked project source hierarchy:

1. latest final manuscript;
2. FINAL evidence freeze and frozen terminology/index;
3. official `README_F08.txt` and F08 package evidence;
4. `F08_experiment_lock.json` and preserved F08 notebook execution;
5. frozen F06 and F01 source-checkpoint verification;
6. F08 class-order and probability-compatibility verification;
7. source-specific preprocessing records;
8. validation probability files and predefined weight-grid comparison;
9. selected validation-rule record;
10. official-test probability, prediction, metric, report, and confusion evidence;
11. package/source-integrity records;
12. preserved campaign notebook;
13. prior completed F06/F07 documentation for lineage and comparison boundaries.

Direct F08 execution evidence and FINAL frozen records supersede older incomplete registry fields where they resolve them.

---

## 3. Scientific role and no-training boundary

F08 asks a narrowly defined inference question:

> Can a pre-governed probability ensemble of the frozen F06 ConvNeXt-Small HR384 model and the frozen F01 YOLO26m-cls model improve ranking behaviour without retraining either source model?

The experiment therefore contains no:

- optimiser;
- scheduler;
- training epoch;
- early stopping;
- fine-tuning;
- parameter update;
- checkpoint creation;
- test-time weight search.

The only selection decision introduced by F08 is the probability weight, and that decision was restricted to a **predefined validation-only grid**. The final weight was locked before official-test evaluation.

---

## 4. Frozen source checkpoints

| Source | Exact experiment identity | Checkpoint | Best epoch | SHA-256 | Role in F08 |
|---|---|---|---:|---|---|
| F06 | `F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL` | `best_model.pth` | 2 | `d0c225c28b25f09822bc99518a37f9b1fe642b2d01213da7f68e2d76a255832a` | ConvNeXt probability source |
| F01 | `F01_YOLO26M_CLS_AUG200` | `best.pt` | 2 | `aa489def1443a6cdab42a0a20a530b9e341d574dcaaf234bfe6c7983113e8e5a` | YOLO probability source |

The preserved F08 Step 1 workflow recomputed and verified both hashes before any F08 validation selection.

F08 itself contains no new checkpoint. Its reproducibility therefore depends on preserving these exact two source checkpoints and the F08 inference protocol.

---

## 5. Dataset partitions, class order, and test isolation

The campaign used the unchanged AQUA20 partitions:

```text
Training split     = 5,247 original images
Validation split   = 1,312 images
Official test      = 1,612 images
Classes            = 20
Split seed         = 42
```

F08 performs no new training, so the training split is relevant only through the already-frozen source models.

The frozen 20-class order is:

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

Before validation ensembling, F08 explicitly verified that F06 and F01 returned 20-class probability vectors under an identical class-name mapping.

The official test split was not used to:

- choose either source checkpoint;
- define the candidate weight grid;
- select the final weight;
- modify the tie-break order;
- alter preprocessing;
- add a new candidate rule.

---

## 6. Source-specific evaluation preprocessing

F08 does **not** force both models through one shared preprocessing pipeline. Each source retains its own frozen evaluation path.

### 6.1 F06 ConvNeXt-Small HR384

```text
RGB image
→ Direct Resize 384×384
→ ToTensor
→ ImageNet normalisation
```

ImageNet normalisation:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

No random evaluation augmentation is used.

### 6.2 F01 YOLO26m-cls

The preserved effective Ultralytics classification evaluation path is:

```text
RGB image
→ resize shorter edge to 224 while preserving aspect ratio
→ CenterCrop 224×224
→ tensor scaling to [0,1]
→ identity normalisation
```

Identity normalisation:

```text
mean = [0, 0, 0]
std  = [1, 1, 1]
```

No random evaluation augmentation is used.

### 6.3 Important boundary

F08 combines **probabilities after source-specific preprocessing and inference**. It does not:

- feed 384×384 F06 tensors into YOLO;
- feed YOLO-centre-cropped 224 inputs into F06;
- average model logits from incompatible pipelines;
- replace either model's native frozen evaluation representation.

---

## 7. Compatibility and sample alignment

F08 Step 2 verified:

```text
Validation images       = 1,312
Classes                 = 20
Class order             = VERIFIED IDENTICAL
F06 probability shape  = (20,)
F01 probability shape  = (20,)
Probability sums        ≈ 1.0 for each source
```

The full validation and test workflows then aligned the two sources by sample identity before fusion.

The preserved workflow:

1. uses the F06 ImageFolder ordering as the canonical sample order;
2. obtains F01 predictions/probabilities for the same image set;
3. maps F01 outputs by unique filename;
4. verifies exact filename-set agreement;
5. reorders F01 probability rows into the F06 canonical order;
6. verifies expected shapes:
   - validation: `(1312, 20)`;
   - test: `(1612, 20)`;
7. verifies probability-vector sums;
8. only then performs class-aligned probability fusion.

This prevents a valid probability vector from being combined with the wrong image or wrong class index.

---

## 8. Frozen probability-ensemble rule

For an image \(x\), let:

- \(\mathbf{p}_{F06}(x)\) be the frozen 20-class F06 softmax probability vector;
- \(\mathbf{p}_{F01}(x)\) be the frozen 20-class F01 class-probability vector.

For an F06 weight \(w\), F08 computes:

\[
\mathbf{p}_{F08}(x;w)
=
w\,\mathbf{p}_{F06}(x)
+
(1-w)\,\mathbf{p}_{F01}(x).
\]

The selected rule is:

\[
\mathbf{p}_{F08}(x)
=
0.70\,\mathbf{p}_{F06}(x)
+
0.30\,\mathbf{p}_{F01}(x).
\]

Final prediction:

```text
predicted class = argmax(F08 averaged probability vector)
```

Top-k rankings are produced from the same fused probability vector.

F08 is therefore:

- probability-level fusion;
- weighted arithmetic averaging;
- not majority voting;
- not logit averaging;
- not checkpoint averaging;
- not stacking with a learned meta-model;
- not joint training.

---

## 9. Predefined validation-only weight grid

The candidate grid was locked before full validation evaluation:

| Candidate | F06 weight | F01 weight |
|---|---:|---:|
| A | 0.50 | 0.50 |
| B | 0.60 | 0.40 |
| C | 0.70 | 0.30 |
| D | 0.80 | 0.20 |

No additional weights were permitted after validation inspection.

### Locked selection criterion

```text
Primary: validation Macro F1
```

### Locked tie-break order

```text
1. Higher validation Macro F1
2. Higher validation Macro Recall
3. Higher validation Top-1
4. Prefer equal weighting if still tied
```

The official test was still untouched when this grid and tie-break protocol were locked.

---

## 10. Validation weight-grid result

The preserved validation comparison exposed the following three governing metrics for all four candidates:

| F06 / F01 weight | Validation Top-1 | Validation Macro Recall | Validation Macro F1 |
|---|---:|---:|---:|
| 0.50 / 0.50 | 93.1402% | 90.8628% | 89.8842% |
| 0.60 / 0.40 | 93.3689% | 92.1709% | 90.2689% |
| **0.70 / 0.30** | **93.4451%** | **93.0674%** | **90.7359%** |
| 0.80 / 0.20 | 93.0640% | 91.2224% | 88.7436% |

The selected rule was therefore:

```text
0.70 F06 + 0.30 F01
```

because it had the highest validation Macro F1 in the pre-locked grid. The tie-break sequence was not required to overturn that choice.

For context, the frozen F06 single-model validation values were:

```text
Top-1       = 92.9878%
Macro Recall= 90.6790%
Macro F1    = 88.5122%
```

The selected F08 rule therefore showed a strong validation balanced-metric gain before official-test evaluation.

---

## 11. Selected-validation metric completeness boundary

The FINAL master evidence register resolves the selected rule's available frozen aggregate validation metrics as:

| Metric | Selected 0.70/0.30 validation value | Status |
|---|---:|---|
| Top-1 | 93.4451% | **Verified** |
| Top-2 | — | **Not frozen / N/A in the selected-rule record** |
| Top-3 | — | **Not frozen / N/A in the selected-rule record** |
| Top-5 | — | **Not frozen / N/A in the selected-rule record** |
| Macro Precision | 89.8120% | **Verified from FINAL master evidence freeze** |
| Macro Recall | 93.0674% | **Verified** |
| Macro F1 | 90.7359% | **Verified** |
| Weighted F1 | 93.4968% | **Verified from FINAL master evidence freeze** |

The corresponding full-precision frozen register values are:

```text
Top-1           = 93.445122%
Macro Precision = 89.812017%
Macro Recall    = 93.067447%
Macro F1        = 90.735878%
Weighted F1     = 93.496818%
```

Selected-validation Top-2/Top-3/Top-5 are blank in the FINAL frozen register. They are therefore reported as unavailable rather than reconstructed from test results or other weights.

---

## 12. Final rule lock and official-test governance

The experiment chronology is critical:

```text
Step 1
→ verify F06/F01 checkpoints
→ lock four-weight grid
→ lock validation Macro-F1 selection
→ lock tie-break order
→ official test NOT evaluated

Step 2
→ verify model/class/probability compatibility
→ official test NOT evaluated

Step 3
→ generate validation probabilities
→ evaluate only the four pre-locked candidates
→ select 0.70/0.30
→ lock final F08 rule
→ close weight search
→ official test NOT evaluated

Step 4
→ evaluate the unchanged 1,612-image official test
→ apply only the already-locked 0.70/0.30 rule
```

No new ensemble weights were tested after the official-test result was observed.

This distinguishes F08 from F07:

- F07 had a deterministic inference rule fixed before validation and had no validation-selected hyperparameter;
- F08 had a pre-locked finite candidate grid and used validation to choose one rule;
- neither experiment used the official test to alter its inference rule.

---

## 13. Frozen official-test result

Official test support:

```text
images  = 1,612
classes = 20
```

F08 official-test metrics:

| Metric | F08 |
|---|---:|
| Top-1 | **92.2457%** |
| Top-2 | **98.0769%** |
| Top-3 | **99.3797%** |
| Top-5 | **99.8759%** |
| Macro Precision | **90.6725%** |
| Macro Recall | **87.0170%** |
| Macro F1 | **87.9406%** |
| Weighted F1 | **92.1750%** |

These are the frozen official campaign values for the selected validation-governed ensemble rule.

---

## 14. Direct F08 versus F06 comparison

F06 is the closest scientific reference because F08 is explicitly constructed by augmenting the frozen F06 probability output with F01 probabilities.

| Metric | F06 | F08 | F08 interpretation |
|---|---:|---:|---|
| Top-1 | 92.1216% | **92.2457%** | +0.1241 pp |
| Top-2 | 97.9529% | **98.0769%** | improved |
| Top-3 | 99.2556% | **99.3797%** | improved |
| Top-5 | 99.7519% | **99.8759%** | +0.1241 pp in the locked F08 comparison |
| Macro Precision | 90.4312% | **90.6725%** | improved |
| Macro Recall | **87.4362%** | 87.0170% | −0.4192 pp |
| Macro F1 | **88.1699%** | 87.9406% | −0.2293 pp |
| Weighted F1 | 92.0810% | **92.1750%** | improved |

The Top-1 change corresponds to only **two additional correct predictions among 1,612 test images**. The final manuscript therefore treats the result as a marginal ranking gain rather than definitive overall superiority.

### Central generalisation observation

The selected rule produced large validation gains in Macro Recall and Macro F1 relative to F06, but these balanced-metric gains did not reproduce on the unchanged official test. On test:

- Top-1 and Top-k ranking improved slightly;
- Macro Precision improved slightly;
- Weighted F1 improved slightly;
- Macro Recall decreased;
- Macro F1 decreased.

Therefore the validation-selected ensemble is useful as a **complementary ranking result**, not as a replacement for the stronger balanced-class F06 model.

---

## 15. F01 context and comparison boundary

The frozen F01 source model achieved:

| Metric | F01 |
|---|---:|
| Top-1 | 84.6774% |
| Top-2 | 94.2308% |
| Top-3 | 97.3325% |
| Top-5 | 98.9454% |
| Macro Precision | 83.3054% |
| Macro Recall | 73.9895% |
| Macro F1 | 75.9908% |
| Weighted F1 | 84.4249% |

F08 is substantially stronger than F01 on the same unchanged test, but this should not be interpreted as a clean one-factor F01→F08 ablation because F08 directly incorporates both F01 and F06.

The purpose of F01 in F08 is heterogeneous model-family complementarity, not to establish a causal law about YOLO versus ConvNeXt.

---

## 16. Frozen scientific conclusion and reporting hierarchy

Approved experiment-level conclusion:

> The validation-selected 0.70 F06 + 0.30 F01 probability ensemble improved official-test Top-1, Top-2, Top-3, Top-5, Macro Precision, and Weighted F1 relative to F06, but Macro Recall and Macro F1 were slightly lower. F08 is therefore retained as a complementary ranking ensemble, while F06 remains the primary balanced single model.

The validation balanced-metric improvement should also be reported honestly:

> F08 produced strong validation Macro Recall and Macro F1 gains, but those gains did not fully generalise to the unchanged official test.

Frozen campaign hierarchy:

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

F08 achieved the highest campaign Top-1 point estimate, but this does not change the primary-method selection because F06 retained the stronger official-test Macro Recall and Macro F1 and is the single-model scientific contribution.

The internal 92.68% Top-1 beat-base target was **not** beaten:

```text
Reference target = 92.6800%
F08 Top-1       = 92.2457%
Gap             = -0.4343 percentage points
Status          = NOT BEATEN
```

---

## 17. Official F08 package and integrity record

**Official package**

```text
aqua20_f08_convnext_hr384_yolo26m_prob_ensemble_outputs.zip
```

The notebook displayed an approximate package size of:

```text
0.96 MB
```

This is only a notebook display value and is not treated as an exact original ZIP byte count.

### 17.1 Curated package contents

The official curated F08 package contains 17 files:

```text
README_F08.txt
figures/F08_confusion_matrix.png
figures/F08_confusion_matrix_normalized.png
probabilities/F08_test_probabilities.npz
probabilities/F08_validation_probabilities.npz
protocol/F08_experiment_lock.json
tables/F08_classification_report.csv
tables/F08_confusion_matrix.csv
tables/F08_confusion_matrix_normalized.csv
tables/F08_scientific_model_comparison.csv
tables/F08_source_checkpoint_hashes.csv
tables/F08_test_metrics_comparison.csv
tables/F08_test_metrics_summary.csv
tables/F08_test_predictions.csv
tables/F08_validation_selected_rule.csv
tables/F08_validation_weight_comparison.csv
tables/F08_vs_F06_test_delta.csv
```

No source checkpoint is embedded in this package because F08 produces no new checkpoint. Source identity is preserved through hash records and the separately frozen F06/F01 packages.

### 17.2 Frozen extracted-package integrity

| Integrity field | Locked value |
|---|---|
| Curated files | 17 |
| Extracted bytes | 1,428,871 |
| Checkpoints inside F08 package | 0 |
| Extracted-tree SHA-256 | `d86fe8e910fce75aac8e3b55319daace6df41df8cf33003bce3192b57b7f4137` |
| Original ZIP SHA-256 | **Not collected / To be verified** |
| Exact original ZIP byte count | **To be verified** |

The extracted-tree hash is the currently frozen package-integrity record.

---

## 18. Output and consistency audit

The FINAL evidence freeze records successful verification of the F08 core evidence:

- F06 source checkpoint identity and SHA-256;
- F01 source checkpoint identity and SHA-256;
- identical 20-class mapping;
- validation-selected 0.70/0.30 rule;
- 1,612 official-test prediction rows;
- independently recomputed aggregate classification metrics;
- 20×20 confusion matrix consistency;
- official Top-k summary;
- preserved classification report;
- raw confusion matrix;
- row-normalised confusion matrix;
- raw validation probability archive;
- raw test probability archive;
- validation weight-comparison table;
- selected-rule record;
- F08-vs-F06 scientific comparison tables.

No new class-wise F08 narrative is invented from incomplete snippets. The class-wise report is an archived evidence asset and should be used directly if a later phase requires detailed class-specific interpretation.

---

## 19. F08 inference runtime evidence and reproducibility boundary

### 19.1 Directly supported campaign-session environment

The preserved campaign notebook session that executed the F08 workflow records:

```text
Python       = 3.12.13
PyTorch      = 2.7.1+cu118
torchvision  = 0.22.1+cu118
CUDA toolkit = 11.8
GPU          = Tesla P100-PCIE-16GB
GPU memory   ≈ 15.89 GB
Compute cap. = 6.0
Device       = cuda:0
```

This is a preserved execution-session record. It should not be overstated as an immutable complete container manifest for every historical source run.

### 19.2 F06 branch inside F08

The preserved F08 validation/test inference code directly supports:

```text
batch size           = 16
workers              = 2
shuffle              = False
pin_memory           = True
torch.inference_mode = used
non_blocking transfer= used
CUDA autocast path   = enabled when CUDA is active
softmax probability  = used
```

### 19.3 F01 branch inside F08

The preserved Ultralytics prediction call supports:

```text
imgsz    = 224
batch    = 64
device   = 0 when CUDA is active
stream   = True
verbose  = False
```

F01 class probabilities are then converted to aligned NumPy probability rows for fusion.

### 19.4 Still To be verified

The following must not be reconstructed retrospectively:

1. exact original F08 ZIP SHA-256;
2. exact original ZIP byte count;
3. complete immutable F08 container/image identifier;
4. complete driver/native-cuDNN/CPU/RAM manifest;
5. exact F08 run-local scikit-learn version;
6. complete F08 run-local package freeze;
7. controlled end-to-end ensemble latency;
8. controlled F06-only and F01-only latency under the same benchmark protocol;
9. throughput;
10. peak GPU memory;
11. energy/compute cost;
12. combined parameters and FLOPs under one standardised counting method;
13. exhaustive class-wise textual interpretation unless the archived report rows are directly surfaced;
14. immutable dataset revision/content checksum and remaining dataset-level integrity boundaries inherited from earlier phases.

A controlled efficiency comparison must include the cost of **both source models**, not only the ConvNeXt branch.

---

## 20. Approved and prohibited claims

### Approved

- F08 is `F08_CONVNEXT_SMALL_HR384_YOLO26M_PROB_ENSEMBLE`.
- F08 is inference-only.
- F08 trained no model and created no checkpoint.
- F08 uses the frozen F06 and F01 checkpoints with verified SHA-256 values.
- The two sources use different frozen preprocessing pipelines.
- Identical 20-class mapping was verified before fusion.
- Only four pre-locked validation weights were evaluated.
- Validation Macro F1 selected the final rule, with frozen tie-breaks.
- The selected rule is `0.70 F06 + 0.30 F01`.
- The selected rule was locked before official-test evaluation.
- F08 improved official-test Top-1/Top-k, Macro Precision, and Weighted F1 relative to F06.
- F08 had lower official-test Macro Recall and Macro F1 than F06.
- F08 is the complementary final ensemble / strongest ranking model.
- F06 remains the primary proposed single model / strongest balanced-class model.
- The 92.68% Top-1 target was not beaten.

### Do not write

- F08 replaced F06 as the primary proposed method.
- F08 is universally better than F06.
- Ensembling improved balanced-class performance on the official test.
- The official test was used to select the 0.70/0.30 rule.
- Additional weights were explored after official-test inspection.
- F08 produced a new trained checkpoint.
- Both source models used one common input transform.
- F01 and F06 logits were averaged directly.
- The two-extra-correct Top-1 difference establishes definitive overall superiority.
- The 92.68% target was beaten.
- Missing selected-validation metrics may be inferred from the official-test record.
- F08 class-wise behaviour may be generalised without direct report evidence.

---

## 21. Reviewer / viva question bank

### Q1. What is the exact F08 identity?

`F08_CONVNEXT_SMALL_HR384_YOLO26M_PROB_ENSEMBLE`.

### Q2. Was F08 trained?

No. F08 is inference-only.

### Q3. Did F08 create a new checkpoint?

No. It combines probabilities from two frozen source checkpoints.

### Q4. Which two experiments are the sources?

F06 ConvNeXt-Small HR384 Gentle CB Focal and F01 YOLO26m-cls Aug-200.

### Q5. What is the frozen F06 hash?

`d0c225c28b25f09822bc99518a37f9b1fe642b2d01213da7f68e2d76a255832a`.

### Q6. What is the frozen F01 hash?

`aa489def1443a6cdab42a0a20a530b9e341d574dcaaf234bfe6c7983113e8e5a`.

### Q7. Were the class mappings assumed to match?

No. Identical 20-class class-name mapping was explicitly verified before validation ensembling.

### Q8. Did F08 use one preprocessing pipeline for both models?

No. F06 retained its direct 384×384 ImageNet-normalised path, while F01 retained its Ultralytics 224 resize/centre-crop path with identity normalisation.

### Q9. How are the two model outputs combined?

By a weighted arithmetic mean of aligned 20-class probability vectors.

### Q10. What weights were eligible?

`0.50/0.50`, `0.60/0.40`, `0.70/0.30`, and `0.80/0.20` for F06/F01.

### Q11. Why were only those four weights tested?

The finite grid was pre-locked to prevent open-ended post-hoc search.

### Q12. What selected the final weight?

Validation Macro F1.

### Q13. What was the tie-break order?

Macro F1, then Macro Recall, then Top-1, then equal-weight preference.

### Q14. Which rule won?

`0.70 F06 + 0.30 F01`.

### Q15. What were the verified selected-validation values?

Top-1 93.4451%, Macro Precision 89.8120%, Macro Recall 93.0674%, Macro F1 90.7359%, and Weighted F1 93.4968%. Selected-validation Top-2/Top-3/Top-5 were not frozen in the final selected-rule register.

### Q16. Was the test used to choose 0.70/0.30?

No. The final rule was locked before official-test evaluation.

### Q17. What was F08 official-test Top-1?

92.2457%.

### Q18. What was F08 official-test Macro F1?

87.9406%.

### Q19. Did the ensemble improve every metric over F06?

No. Top-1/Top-k, Macro Precision, and Weighted F1 improved, but Macro Recall and Macro F1 were lower.

### Q20. Why does F08 not replace F06?

F06 remains stronger on balanced-class Macro Recall and Macro F1, is a single trained model, and represents the core imbalance-aware method. F08 provides a complementary ranking gain.

### Q21. How large was the Top-1 improvement over F06 in sample terms?

Two additional correct predictions among 1,612 test images.

### Q22. What is the central validation-to-test lesson?

A strong validation Macro Recall/Macro-F1 gain did not fully generalise to the official test, while modest ranking gains did.

### Q23. Is F08 probability averaging the same as checkpoint ensembling?

No. The source checkpoints remain independent; only their aligned class probabilities are fused.

### Q24. What is the package tree hash?

`d86fe8e910fce75aac8e3b55319daace6df41df8cf33003bce3192b57b7f4137`.

### Q25. Which selected-validation metrics remain unavailable?

Selected-validation Top-2, Top-3, and Top-5 are not populated in the FINAL frozen selected-rule register. They are therefore left unavailable rather than reconstructed. Macro Precision and Weighted F1 are frozen as 89.8120% and 93.4968%, respectively.

---

## 22. Evidence ledger

Phase 10 was grounded in the following evidence families:

- latest final AQUA20 manuscript;
- FINAL P00 evidence-freeze report and master experiment register;
- `README_F08.txt`;
- F08 protocol lock and Step 1 checkpoint-hash verification;
- F08 Step 2 model/class/probability compatibility verification;
- F08 Step 3 validation probability generation and pre-locked weight selection;
- F08 official-test evaluation and package creation in the preserved campaign notebook;
- F01 frozen checkpoint/test evidence;
- F06 frozen checkpoint/test evidence;
- common environment and preprocessing documentation;
- F09 final reporting hierarchy and campaign freeze;
- Phase 9→10 handover.

Where older P00 rows contained incomplete configuration fields, later direct preserved notebook evidence was used when available. Unsupported values remain explicitly unresolved.

---

## 23. Phase 10 completion decision

**Phase 10 — F08 ConvNeXt-Small HR384 + YOLO26m Validation-Selected Probability Ensemble Documentation is complete.**

The following are now documented and locked:

- exact F08 identity;
- inference-only / no-training / no-new-checkpoint boundary;
- exact F06 and F01 source identities;
- both source SHA-256 values;
- identical 20-class mapping;
- source-specific F06 and F01 preprocessing;
- sample/probability alignment process;
- weighted arithmetic probability-fusion formula;
- exact four-candidate validation grid;
- validation Macro-F1 selection and tie-break order;
- selected `0.70 F06 + 0.30 F01` rule;
- proof that the selected rule preceded official-test evaluation;
- all available frozen selected-validation aggregate metrics: Top-1, Macro Precision, Macro Recall, Macro F1, and Weighted F1;
- honest unavailable boundary for selected-validation Top-2/Top-3/Top-5;
- complete official-test Top-1/2/3/5, Macro Precision, Macro Recall, Macro F1, and Weighted F1;
- direct F08-versus-F06 comparison;
- the two-additional-correct-predictions interpretation;
- F06/F08 final reporting hierarchy;
- negative boundary that validation balanced gains did not fully generalise;
- package contents and extracted-tree integrity;
- output audit;
- directly supported F08 inference/runtime fields;
- unsupported fields retained as **To be verified**;
- reviewer/viva question bank;
- strict approved/prohibited claim boundaries.

The next documentation phase is:

**Phase 11 — F09 Final Method Freeze and Campaign Consolidation Documentation.**
