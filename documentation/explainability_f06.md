# AQUA20 — Explainability and Frozen F06 Qualitative Diagnostics

**Documentation phase:** Phase 15  
**Repository target:** `documentation/analysis/explainability_f06.md`  
**Status:** Completed with explicit X03 and integrity limitations  
**Frozen source model:** `F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL`  
**Included XAI stages:** `X01_F06_FINAL_GRADCAM`, `X02_F06_FINAL_LIME`  
**X03 status:** Direct comparison package/protocol/metrics not surfaced; **To be verified / not claimed complete**

---

## 1. Phase scope and roadmap control

The Phase 14 handover required the latest Documentation Master Plan / phase index to control the exact Phase 15 title. The File Library search did not surface a controlling document with an explicit numbered Phase 15 row.

The latest final manuscript, FINAL evidence freeze, X01/X02 official READMEs, XAI inventory, and preserved campaign records nevertheless converge on explainability as the remaining major documentation stream after the model campaign, internal error/statistical analysis, and external evaluation.

Accordingly, this phase proceeds under the working title:

> **Phase 15 — Explainability and Frozen F06 Qualitative Diagnostic Documentation**

This is a documentation/evidence-consolidation phase. It performs no training, model selection, checkpoint selection, or new XAI sample selection.

If a newer controlling numbered master plan is later surfaced, that plan may replace the phase title/numbering, but it does not override the frozen X01/X02 evidence documented here.

---

## 2. Scientific role

The explainability stream is intended to inspect prediction behaviour of the already-selected F06 model on a frozen subset of AQUA20 official-test images.

The manuscript-level scientific boundary is deliberately conservative:

- Grad-CAM and LIME are **post-hoc local explanation tools**;
- they may indicate image regions associated with a particular F06 prediction;
- they are used for qualitative diagnosis of correct, difficult, incorrect, and weak/minority-class cases;
- they are **not evidence of causal, biological, or human-like model reasoning**;
- visually plausible explanation maps do not prove that the classifier relies on biologically meaningful features.

This phase therefore documents explanation protocols and qualitative diagnostic use, not a new predictive result.

---

## 3. Frozen source model

All XAI work uses the same unchanged F06 checkpoint.

| Field | Frozen value |
|---|---|
| Model identity | `F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL` |
| Architecture | Torchvision ConvNeXt-Small |
| Source branch | F03 continuation at high resolution |
| Input resolution | `384×384` |
| Checkpoint | `best_model.pth` |
| Best epoch | 2 |
| Checkpoint SHA-256 | `d0c225c28b25f09822bc99518a37f9b1fe642b2d01213da7f68e2d76a255832a` |
| Training during XAI | **NONE** |
| Fine-tuning during XAI | **NONE** |
| New checkpoint | **NONE** |
| Model selection during XAI | **NONE** |

F06 remains the primary proposed single model and strongest balanced-class model from the frozen campaign hierarchy. Explainability does not reopen F09 model selection.

---

## 4. Frozen XAI sample registry

### 4.1 Registry design

X01 froze the XAI sample set **before** Grad-CAM generation.

```text
Frozen samples:       20
Groups:               4
Samples per group:     5
Publication subset:   12
```

The four frozen groups are:

| Group | Frozen role |
|---|---|
| A | Correct high-confidence predictions |
| B | Correct difficult / lower-confidence predictions |
| C | High-confidence misclassified predictions |
| D | Minority / weak-performing class examples |

No image was selected because its Grad-CAM heatmap looked visually attractive, and no frozen image was replaced after inspecting Grad-CAM outputs.

### 4.2 Registry reuse in X02

X02 reused the same X01 registry unchanged:

- same 20 image IDs;
- same sample order;
- same A/B/C/D groups;
- same true labels and official F06 predictions;
- same 12-image publication subset;
- same explanation target rule.

The copied X01 registry in X02 was recorded as content-identical to the original closed X01 registry.

### 4.3 Frozen 20-image ID verification list

The preserved campaign verification record lists:

```text
Group A: 614, 1573, 1567, 642, 1596
Group B: 1605, 1372, 500, 430, 482
Group C: 577, 3, 217, 944, 1199
Group D: 1061, 1071, 1486, 1397, 1456
```

These IDs are verification aids only. The authoritative registry remains:

```text
X01_frozen_xai_sample_registry.csv
```

### 4.4 Frozen 12-image publication subset

The publication subset was defined as the **first three frozen samples per group**, before explanation appearance was used for any decision:

```text
A: 614, 1573, 1567
B: 1605, 1372, 500
C: 577, 3, 217
D: 1061, 1071, 1486
```

No publication sample should be replaced simply because one method produces a cleaner or more visually appealing map.

---

## 5. Exact frozen image representation

X01 established the image representation used for the XAI registry:

```text
AQUA20 official test image
→ RGB conversion
→ JPEG export at quality 95
→ JPEG reload
→ Resize to 384×384
→ tensor conversion
→ ImageNet normalisation
```

ImageNet normalisation:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

This representation reproduced all 20 official F06 predictions before Grad-CAM generation.

Frozen reproduction evidence:

```text
Samples verified:          20 / 20
Prediction agreement:      20 / 20
Maximum confidence delta:  0.00033830
```

X02 reused the same exact JPEG-quality-95 representation and reproduced the same 20 F06 predictions before LIME generation.

This representation is part of the XAI freeze and should not be casually replaced by a different encoding/preprocessing path when reproducing the frozen figures.

---

## 6. Common explanation-target rule

Both X01 and X02 explain the **F06 predicted class**.

```text
Target model:       Frozen F06
Target class rule:  F06 predicted class
```

For a correct prediction:

```text
true class = predicted class = explanation target
```

For a misclassified prediction:

```text
true class ≠ predicted class
explanation target = predicted class
```

This is essential because the purpose is to explain evidence associated with the decision the frozen model actually made, including its errors.

Do not switch to the true class for misclassified examples merely because that explanation is easier to interpret.

---

# 7. X01 — F06 Final Grad-CAM

## 7.1 Identity and role

```text
Official stage:  X01_F06_FINAL_GRADCAM
Role:            XAI / interpretability / visual explanation
Training:        NONE
Fine-tuning:     NONE
New checkpoint:  NONE
```

X01 was completed, packaged, and closed.

## 7.2 Grad-CAM target layer

The frozen target layer is:

```text
features.7.2
```

This corresponds to the final locked ConvNeXt feature block used for Grad-CAM generation.

## 7.3 Frozen Grad-CAM flow

```text
F06 forward pass
→ predicted-class logit
→ gradients at features.7.2
→ channel-wise gradient averaging
→ weighted activation aggregation
→ ReLU
→ heatmap normalisation
→ resize to image dimensions
→ overlay generation
```

The weights of F06 are never modified during this process.

## 7.4 X01 official output inventory

Frozen X01 evidence records:

```text
Frozen registry rows:      20
Grad-CAM metadata rows:    20
Original images:           20
Raw heatmaps:              20
Grad-CAM overlays:         20
Figures:                    2
```

Figures:

```text
X01_gradcam_publication_panel.png
X01_gradcam_group_comparison.png
```

The preserved source-integrity manifest records:

```text
Package folder:        aqua20_x01_f06_final_gradcam_outputs
Files:                 65
Extracted bytes:       16,296,485
Extracted-tree SHA-256:
6c0d3c77820dcea3a9ee75ba00581430f9d34566896867d2ead9844a6f4ea138
```

### ZIP-hash boundary

An earlier campaign handover recorded an X01 ZIP SHA-256 value, but the FINAL source-integrity manifest leaves the original ZIP-hash field empty and states that the original ZIP hash was not collected. Under the locked source hierarchy, the final integrity manifest controls.

Therefore:

```text
Authoritative original X01 ZIP SHA-256 = To be verified
Authoritative extracted-tree SHA-256   = 6c0d3c...ea138
```

Do not present the older ZIP-hash record as the final immutable package hash unless the original ZIP bytes are recovered and recomputed.

---

# 8. X02 — F06 Final LIME

## 8.1 Identity and role

```text
Official stage:  X02_F06_FINAL_LIME
Role:            XAI / interpretability / model-agnostic local explanation
Training:        NONE
Fine-tuning:     NONE
New checkpoint:  NONE
```

X02 was completed, packaged, and closed.

## 8.2 Registry and prediction reproduction

X02 reused the frozen X01 registry and publication subset. No new XAI sample-selection process was performed.

Before final LIME generation:

```text
Frozen samples verified:  20 / 20
Prediction agreement:     20 / 20
Maximum confidence delta: 0.00033830
```

## 8.3 Frozen LIME protocol

The protocol was locked before final explanation generation.

| Parameter | Frozen value |
|---|---|
| Random seed | `42` |
| Perturbation samples | `1000` |
| Segmentation | `SLIC` |
| Requested SLIC segments | `100` |
| Compactness | `10` |
| Sigma | `1` |
| Start label | `0` |
| Requested positive features | `10` |
| Positive-only | `True` |
| Hide-rest explanation | `True` |
| Distance metric | `cosine` |
| Kernel width | `0.25` |
| Feature selection | `auto` |
| Target rule | F06 predicted class |
| Image representation | `RGB_JPEG_quality95_reload` |
| Model input resolution | `384×384` |
| Normalisation | ImageNet mean/std |

No LIME parameter was changed because an explanation looked diffuse, visually weak, or inconsistent with Grad-CAM.

## 8.4 X02 official output inventory

Frozen X02 records contain:

```text
Copied frozen registry:        20 rows
LIME protocol lock:             1
LIME metadata:                 20 rows
Original images:               20
Positive explanations:         20
Binary positive masks:         20
Boundary overlays:             20
Figures:                        2
```

Figures:

```text
X02_lime_publication_panel.png
X02_lime_group_comparison.png
```

The preserved source-integrity manifest records:

```text
Package folder:        aqua20_x02_f06_final_lime_outputs
Files:                 86
Extracted bytes:       14,819,361
Extracted-tree SHA-256:
11ca45b02e0f0ab98278c94b39c72fe0d8f655f4879e5f8747ef871b35f36db8
```

The same FINAL integrity record states that the original ZIP hash was not collected.

Therefore:

```text
Authoritative original X02 ZIP SHA-256 = To be verified
Authoritative extracted-tree SHA-256   = 11ca45...36db8
```

---

## 9. X01–X02 direct comparability

X01 and X02 were intentionally designed to support case-level comparison under matched conditions.

The following are frozen identically across both stages:

| Dimension | X01 | X02 |
|---|---|---|
| Model | F06 | F06 |
| Checkpoint epoch | 2 | 2 |
| Checkpoint SHA-256 | same | same |
| Images | same frozen 20 | same frozen 20 |
| Sample order | frozen X01 order | reused X01 order |
| A/B/C/D groups | same | same |
| Publication subset | same frozen 12 | same frozen 12 |
| Image representation | JPEG95 reload | JPEG95 reload |
| Resolution | 384×384 | 384×384 |
| Explanation target | F06 predicted class | F06 predicted class |
| Correct/error rule | unchanged | unchanged |

This design supports scientifically cleaner side-by-side inspection because a difference between maps is not confounded by changing the sample or target class.

However, matched inputs alone do not create a quantitative agreement result. Such a claim requires a separately frozen X03 comparison protocol and direct X03 outputs.

---

## 10. X03 status resolution

### 10.1 Conflicting historical records

Earlier planning/supervisor materials sometimes referred to Grad-CAM/LIME “comparison” as completed or planned for quantitative polish.

The FINAL evidence inventory is more restrictive:

```text
X03 — Grad-CAM versus LIME comparison
Protocol:      not found
Package:       not found
Output:        not found
Status:        Missing
Instruction:   do not claim completion without physical evidence
```

The FINAL inventory therefore controls.

### 10.2 What exists without X03

There is preserved code for rebuilding a manuscript/review panel from frozen X01/X02 outputs using the first frozen sample in each A/B/C/D group. That code performs no new inference and can display:

```text
Original | Grad-CAM | LIME
```

A side-by-side figure is useful for qualitative inspection, but it does **not** establish:

- a frozen X03 protocol;
- Grad-CAM thresholding rule;
- mask resizing rule;
- quantitative IoU/Dice calculation;
- group-level similarity statistics;
- statistical significance of explanation agreement;
- an official X03 package.

Therefore the repository documentation must not convert a qualitative manuscript panel into a claimed completed X03 experiment.

### 10.3 X03 fields retained as To be verified

```text
X03 official README
X03 protocol lock
Grad-CAM threshold rule
LIME mask rule for comparison
mask-resizing rule
IoU definition
Dice definition
per-image similarity table
group-level similarity summary
correct-vs-error similarity analysis
weak/minority-group similarity analysis
X03 figures
X03 package manifest
X03 tree hash
X03 ZIP hash
```

Until direct evidence is recovered, any numerical Grad-CAM-versus-LIME agreement claim is prohibited.

---

## 11. Qualitative findings permitted by the final manuscript

The final manuscript uses the XAI outputs cautiously.

The supported qualitative conclusion is:

- several correct cases showed highlighted regions overlapping the visible organism;
- other examples showed explanation emphasis on water, substrate, neighbouring objects, or background structures;
- correct predictions with diffuse localisation were observed;
- incorrect predictions could still produce visually plausible explanation maps.

These observations justify using Grad-CAM and LIME as **diagnostic inspection tools**.

They do not justify claims such as:

```text
"F06 understands marine biology"
"the model reasons like a human"
"the heatmap proves the cause of the prediction"
"attention on the organism proves the model will generalise"
"Grad-CAM and LIME agree quantitatively"
```

No new sample-specific causal narrative is introduced in this documentation phase beyond what the manuscript directly supports.

---

## 12. Relationship to internal error analysis

The frozen XAI groups deliberately include:

- high-confidence correct cases;
- difficult correct cases;
- high-confidence errors;
- minority/weak-performing classes.

This complements the internal class-wise error analysis by providing qualitative evidence around individual predictions.

The two evidence types must remain distinct:

```text
classification report / confusion matrix
= outcome-level class performance

Grad-CAM / LIME
= post-hoc local explanation of a selected prediction
```

A heatmap should not be used to retroactively create an unsupported causal explanation for a confusion-matrix error.

---

## 13. Relationship to external evaluation

X01/X02 explain selected **AQUA20 official-test** predictions using the frozen F06 model.

They do not directly explain the E01 external-dataset predictions unless separately generated external XAI evidence exists.

Therefore do not state that the internal Grad-CAM/LIME panels explain the cause of the external E01A/E01B performance drop.

The following remain separate conclusions:

- internal XAI: qualitative diagnostic of selected AQUA20 F06 predictions;
- external evaluation: source-only generalisation behaviour on the Sea Animals Image Dataset;
- E01C: limited known–unknown score separation;
- no causal link between a particular XAI visual pattern and the measured external-domain performance has been established.

---

## 14. Manuscript-ready methods wording

Approved concise methods wording:

> Grad-CAM and LIME were applied to the frozen F06 model using the same preselected 20-image AQUA20 test registry. Both methods targeted the F06 predicted class for correct and incorrect cases. Grad-CAM used ConvNeXt layer `features.7.2`. LIME used 1,000 perturbations with seed 42 and SLIC segmentation with 100 requested segments, compactness 10, and sigma 1. The 12 publication examples were fixed from the registry before explanation appearance was inspected.

Approved limitation wording:

> The explanation maps are interpreted as qualitative post-hoc diagnostics. They indicate image regions associated with the frozen model's prediction but are not treated as proof of causal, biological, or human-like reasoning.

Approved X03 boundary wording:

> Although Grad-CAM and LIME were generated on the same frozen cases, no verified direct X03 quantitative comparison package was recovered; numerical claims of cross-method spatial agreement are therefore not made.

---

## 15. Manuscript-ready evidence summary table

| Item | X01 Grad-CAM | X02 LIME |
|---|---|---|
| Model | Frozen F06 | Frozen F06 |
| Checkpoint | Epoch 2 | Epoch 2 |
| Samples | 20 | Same 20 |
| Publication subset | 12 | Same 12 |
| Target | F06 predicted class | F06 predicted class |
| Main method lock | `features.7.2` | seed 42; 1000 perturbations; SLIC 100/10/1 |
| Prediction reproduction | 20/20 | 20/20 |
| Max confidence delta | 0.00033830 | 0.00033830 |
| Core per-image outputs | original, heatmap, overlay | original, positive explanation, mask, overlay |
| Package file count | 65 | 86 |
| Extracted-tree SHA-256 | `6c0d3c...ea138` | `11ca45...36db8` |
| Original ZIP SHA-256 | **To be verified** | **To be verified** |
| Status | Completed / packaged / closed | Completed / packaged / closed |

---

## 16. Claim boundaries

Do **not**:

1. claim that Grad-CAM or LIME proves causal reasoning;
2. claim that highlighted areas are biologically meaningful without independent annotation/validation;
3. claim XAI establishes external-domain robustness;
4. replace frozen samples because explanations look poor;
5. remove errors or weak/minority examples from the frozen set;
6. switch misclassified cases from predicted-class targets to true-class targets for the main protocol;
7. change LIME parameters after inspecting explanation appearance;
8. treat the 12 publication samples as a post-hoc “best heatmap” subset;
9. describe X02 as training or fine-tuning;
10. describe X01/X02 as creating a new model/checkpoint;
11. claim numerical Grad-CAM/LIME agreement without direct X03 evidence;
12. invent IoU, Dice, overlap, localisation concentration, or stability metrics;
13. treat a side-by-side manuscript figure as proof that X03 was completed;
14. state an original X01/X02 ZIP checksum as authoritative when the FINAL source-integrity manifest records that the original ZIP hash was not collected;
15. use internal XAI to causally explain the E01 external performance drop.

---

## 17. Reproducibility and integrity status

### Directly supported

```text
F06 identity and checkpoint SHA-256
X01/X02 stage identities
20-image frozen registry
4 groups × 5 samples
12-image fixed publication subset
JPEG quality-95 reload representation
384×384 input
ImageNet normalisation
20/20 reproduced predictions
maximum confidence delta 0.00033830
predicted-class target for both methods
Grad-CAM layer features.7.2
full LIME protocol lock
X01 output count
X02 output count
X01 extracted-tree hash
X02 extracted-tree hash
```

### To be verified / not fully closed

```text
original X01 ZIP SHA-256 from original bytes
original X02 ZIP SHA-256 from original bytes
complete immutable run/container/driver/CPU/RAM manifests
controlled XAI runtime benchmark
controlled latency / throughput
peak GPU memory
energy / compute cost
full file-by-file release checksum manifest beyond recorded tree hash
X03 package/protocol/metrics
quantitative spatial-agreement statistics
foreground-object annotations for localisation validation
explanation stability under controlled perturbations/repeated runs
formal human/expert agreement study
```

---

## 18. Reviewer / viva question bank

### Q1. Why was F06 used for explainability rather than F08?
F06 is the frozen primary proposed single model and strongest balanced-class model. F08 is a complementary two-model ensemble, which would make explanation attribution more complex and would not replace the selected primary method.

### Q2. Did X01 or X02 retrain F06?
No. Both are post-hoc inference/analysis stages with no training, fine-tuning, or new checkpoint.

### Q3. Why freeze the sample registry before Grad-CAM?
To prevent cherry-picking visually attractive heatmaps and to keep the qualitative analysis protocol independent of explanation appearance.

### Q4. Why include misclassified examples?
Because explanation analysis should inspect failure behaviour as well as successful predictions; excluding errors would bias the qualitative picture.

### Q5. Why include minority/weak-class examples?
The study emphasises balanced recognition under class imbalance, so selected qualitative diagnostics should not be restricted to dominant easy classes.

### Q6. Why use the predicted class as the target?
The goal is to explain the decision F06 actually made. For an error, explaining the true class would answer a different question.

### Q7. What Grad-CAM layer was used?
`features.7.2`, the frozen final ConvNeXt feature block used in X01.

### Q8. What are the key LIME settings?
Seed 42, 1,000 perturbations, SLIC with 100 requested segments, compactness 10, sigma 1, start label 0, 10 requested positive features, positive-only explanations, cosine distance, kernel width 0.25, and automatic feature selection.

### Q9. Were LIME parameters tuned after seeing the maps?
No. The protocol was frozen before final explanation generation.

### Q10. Were X01 and X02 run on the same images?
Yes. X02 reused the exact frozen 20-image X01 registry and the same 12 publication examples.

### Q11. Were the model predictions reproducible after the JPEG95 image reconstruction?
Yes. All 20 official F06 predictions were reproduced; the maximum confidence delta was 0.00033830.

### Q12. What does agreement between Grad-CAM and LIME prove?
Without a verified X03 protocol and metrics, no numerical agreement claim is made. Even visual agreement would still not prove causal reasoning.

### Q13. Was X03 completed?
A verified X03 package/protocol/metric output was not surfaced in the FINAL evidence inventory. Therefore X03 is not claimed complete.

### Q14. But does a combined Grad-CAM/LIME manuscript panel exist?
Preserved review code can build a deterministic side-by-side panel from frozen X01/X02 outputs. That is a qualitative presentation artifact, not a verified X03 quantitative experiment.

### Q15. What did the qualitative manuscript inspection find?
Some correct cases overlapped the visible organism, while other cases highlighted background or neighbouring structures. Diffuse correct explanations and visually plausible incorrect explanations were also observed.

### Q16. Can the XAI maps explain why external E01 performance dropped?
No. No controlled causal link between internal explanation maps and the external source-shift performance reduction was established.

### Q17. Why is the 12-image publication subset defensible?
It was deterministically defined as the first three frozen samples from each of the four groups, before explanation appearance was used for selection.

### Q18. What is the principal remaining XAI reproducibility gap?
Direct X03 evidence and authoritative original ZIP hashes are missing, along with complete run-specific environment/efficiency records and quantitative localisation-validation evidence.

---

## 19. Evidence ledger

This phase was controlled by the following evidence hierarchy:

1. latest final manuscript, *Improving Underwater Marine Species Classification through Imbalance-Aware Learning and Cross-Dataset Evaluation*;
2. `P00_AQUA20_Master_Evidence_Freeze_FINAL.xlsx`;
3. `P00_AQUA20_Evidence_Freeze_Report_FINAL.md`;
4. FINAL source-integrity manifest;
5. `README_X01.txt`;
6. `X01_frozen_xai_sample_registry.csv` and `X01_gradcam_metadata.csv` as preserved in the extracted-package evidence;
7. `README_X02.txt`;
8. `X02_lime_protocol_lock.csv` and `X02_lime_metadata.csv` as preserved in the extracted-package evidence;
9. preserved campaign notebook and detailed X01/X02 handover records;
10. current manuscript XAI results section and deterministic R08 XAI figure-rebuild script;
11. Phase 14 external-evaluation handover for cross-phase claim boundaries.

Where older supervisor/planning material conflicts with the FINAL XAI inventory—especially regarding X03 completion—the FINAL evidence inventory controls.

---

## 20. Phase 15 completion decision

Phase 15 is complete as a documentation phase.

The following are now documented and locked:

- roadmap-controlled working Phase 15 scope;
- X01 and X02 exact stage identities;
- unchanged F06 checkpoint, epoch, and SHA-256;
- no-training/no-fine-tuning/no-new-checkpoint boundary;
- frozen 20-image A/B/C/D registry;
- deterministic 12-image publication subset;
- JPEG quality-95 reload representation and 20/20 prediction reproduction;
- common predicted-class explanation target for correct and incorrect cases;
- X01 Grad-CAM target layer and output inventory;
- X02 frozen LIME protocol and output inventory;
- X01/X02 direct comparability conditions;
- FINAL extracted-tree integrity values and authoritative ZIP-hash limitations;
- manuscript-supported qualitative findings;
- strict non-causal XAI interpretation boundary;
- explicit separation of XAI from external E01 causal interpretation;
- X03 status resolved as **not supported by direct package/protocol evidence**;
- reviewer/viva question bank;
- unresolved quantitative-XAI and integrity items retained as **To be verified**.

No new scientific result was fabricated to close X03.

---

## 21. Handover

No explicit numbered Phase 16 master-plan row was surfaced during this phase.

Given the older campaign roadmap, and because internal error analysis, statistical robustness documentation, external evaluation, and X01/X02 explainability documentation are now already covered, the next logical documentation stream is final publication-table and manuscript/repository consolidation.

That next phase must still search for a controlling Documentation Master Plan / phase index before forcing an exact Phase 16 title.
