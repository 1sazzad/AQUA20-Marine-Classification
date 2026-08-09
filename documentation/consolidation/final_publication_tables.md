# AQUA20 — Final Publication Table Consolidation

**Documentation phase:** Phase 16  
**Campaign stage:** `P01 — FINAL PAPER TABLE CONSOLIDATION`  
**Status:** Completed with explicitly retained evidence gaps  
**Recommended repository target:** `documentation/consolidation/final_publication_tables.md`

---

## 1. Scope-control decision

The preserved official campaign roadmap explicitly separates:

1. `P01 — FINAL PAPER TABLE CONSOLIDATION`
2. `P02 — MANUSCRIPT WRITING`

Therefore Phase 16 is restricted to **P01 only**. It consolidates and reconciles publication tables, claim boundaries, documentation indexing, and unresolved evidence. It does **not** begin P02 manuscript rewriting.

This phase performs:

```text
Training: NONE
Fine-tuning: NONE
New inference: NONE
New checkpoint: NONE
New ensemble/TTA search: NONE
Final-method reselection: NONE
XAI sample reselection: NONE
```

The F01–F09 model campaign remains closed.

---

## 2. Controlling source hierarchy

The P01 consolidation follows this hierarchy:

1. latest final manuscript, *Improving Underwater Marine Species Classification through Imbalance-Aware Learning and Cross-Dataset Evaluation*;
2. FINAL evidence-freeze and frozen-terminology/index records;
3. `F09_final_method_freeze.md` and direct frozen F01–F08 campaign comparison evidence;
4. completed experiment documentation;
5. `internal_classwise_error_analysis.md`;
6. `statistical_uncertainty.md`;
7. `external_evaluation.md`;
8. `explainability_f06.md`;
9. current manuscript LaTeX tables/figures;
10. older audits/handovers only where they do not conflict with later frozen evidence.

Unsupported values are not reconstructed from memory. They remain:

> **To be verified**

---

## 3. Frozen final reporting hierarchy

```text
PRIMARY PROPOSED METHOD:
F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL

PRIMARY PROPOSED SINGLE MODEL:
F06

STRONGEST BALANCED-CLASS MODEL:
F06

COMPLEMENTARY FINAL ENSEMBLE:
F08_CONVNEXT_SMALL_HR384_YOLO26M_PROB_ENSEMBLE

STRONGEST RANKING MODEL:
F08

BEST ISOLATED IMBALANCE INTERVENTION WITHIN THE TESTED CAMPAIGN:
F03 — Gentle Class-Balanced Focal Loss
```

The hierarchy is role-aware rather than Top-1-only.

The internal campaign reference target was:

```text
Reference Top-1: 92.6800%
Best campaign Top-1: 92.2457% (F08)
Gap: -0.4343 percentage points
Status: NOT BEATEN
```

This reference target must not be described as an official benchmark unless its independent provenance/comparability is separately established.

---

## 4. Final experiment/stage identity table

| ID | Frozen identity | Main role | Training status |
|---|---|---|---|
| F01 | `F01_YOLO26M_CLS_AUG200` | YOLO26m Aug-200 reference; F08 source | Trained |
| F02 | `F02_YOLO26S_CLS_AUG200` | Smaller YOLO scale ablation under Aug-200 | Trained |
| F03 | `F03_CONVNEXT_SMALL_GENTLE_CB_FOCAL` | Original-distribution imbalance-aware ConvNeXt; best isolated imbalance intervention within tested campaign | Trained |
| F04 | `F04_CONVNEXT_SMALL_AUG200` | ConvNeXt Aug-200 + standard cross-entropy data-level control | Trained |
| F05 | `F05_CONVNEXT_SMALL_AUG200_GENTLE_CB_FOCAL` | Combined Aug-200 + Gentle CB Focal configuration | Trained |
| F06 | `F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL` | Primary balanced single model; F03 continuation at 384×384 | Fine-tuned continuation |
| F07 | `F07_CONVNEXT_SMALL_HR384_TTA` | Deterministic two-view TTA inference comparison | Inference only |
| F08 | `F08_CONVNEXT_SMALL_HR384_YOLO26M_PROB_ENSEMBLE` | Validation-selected heterogeneous probability ensemble | Inference only |
| F09 | `F09_FINAL_METHOD_FREEZE_AND_CAMPAIGN_COMPARISON` | Final comparison/reporting freeze | Analysis only |

### Mandatory identity rule

`F02` is **YOLO26s-cls + Aug-200**.  
The ConvNeXt-Small + Aug-200 + standard cross-entropy experiment is **F04**.

---

## 5. Table P01-A — frozen F01–F08 official-test results

All rows use the unchanged **1,612-image AQUA20 official test split**.

| Experiment | Top-1 | Top-2 | Top-3 | Top-5 | Macro P | Macro R | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F01 | 84.6774% | N/A | N/A | 98.9454% | 83.3054% | 73.9895% | 75.9908% | 84.4249% |
| F02 | 83.4367% | N/A | N/A | 97.9529% | 80.9103% | 65.2565% | 70.3909% | 82.8376% |
| F03 | 91.6253% | 97.7047% | 99.1315% | 99.6898% | **90.7850%** | 86.9302% | 87.9645% | 91.5656% |
| F04 | 91.3772% | 97.2705% | 99.0074% | 99.6278% | 89.3241% | 84.3236% | 86.0035% | 91.2641% |
| F05 | 91.1290% | 97.4566% | 98.9454% | 99.7519% | 89.7496% | 82.6547% | 84.3487% | 90.9759% |
| **F06** | 92.1216% | 97.9529% | 99.2556% | 99.7519% | 90.4312% | **87.4362%** | **88.1699%** | 92.0810% |
| F07 | 92.1216% | 97.8908% | 99.3176% | 99.6898% | 89.8765% | 86.5324% | 87.4026% | 92.0771% |
| **F08** | **92.2457%** | **98.0769%** | **99.3797%** | **99.8759%** | 90.6725% | 87.0170% | 87.9406% | **92.1750%** |

### Missing-metric rule

The final F09 registry intentionally preserves F01/F02 Top-2 and Top-3 as `N/A`. P01 does not back-fill those cells from older isolated records because doing so would silently alter the final frozen campaign table.

For manuscript presentation, the current six-column Top-1/Top-5/Macro P/Macro R/Macro F1/Weighted F1 table remains fully populated and consistent with this registry.

---

## 6. Table P01-B — final F06 versus F08 role-aware comparison

| Metric | F06 | F08 | F08 − F06 |
|---|---:|---:|---:|
| Top-1 | 92.1216% | 92.2457% | +0.1241 pp |
| Top-2 | 97.9529% | 98.0769% | +0.1240 pp |
| Top-3 | 99.2556% | 99.3797% | +0.1241 pp |
| Top-5 | 99.7519% | 99.8759% | +0.1240 pp |
| Macro Precision | 90.4312% | 90.6725% | +0.2413 pp |
| Macro Recall | **87.4362%** | 87.0170% | −0.4192 pp |
| Macro F1 | **88.1699%** | 87.9406% | −0.2293 pp |
| Weighted F1 | 92.0810% | **92.1750%** | +0.0940 pp |

The Top-1 difference corresponds to only two additional correct predictions out of 1,612 images.

### Frozen interpretation

> F08 has the strongest ranking-oriented point estimates, including slightly higher Top-1, while F06 retains higher Macro Recall and Macro F1 and remains the primary balanced single-model method.

Do not write that F08 universally outperformed or replaced F06.

---

## 7. Table P01-C — intervention/comparison summary

| Comparison | Selected official-test deltas | Valid interpretation |
|---|---|---|
| F02 − F01 | Top-1 −1.2407 pp; Macro R −8.7330 pp; Macro F1 −5.5999 pp | Under the same Aug-200 YOLO protocol, the smaller F02 model performed worse on these reported metrics. This is a model-scale comparison, not an augmentation ablation. |
| F03 − F04 | Top-1 +0.2481 pp; Macro R +2.6066 pp; Macro F1 +1.9610 pp | F03 is stronger on these balanced metrics, but this is a **complete-configuration** comparison because both loss and training distribution differ. |
| F05 − F04 | Top-1 −0.2482 pp; Top-5 +0.1241 pp; Macro R −1.6689 pp; Macro F1 −1.6548 pp | Closest loss-focused comparison within the Aug-200 ConvNeXt branch; maximum epoch budgets differ, so interpretation remains configuration-level. |
| F06 − F03 | Top-1 +0.4963 pp; Macro R +0.5060 pp; Macro F1 +0.2054 pp; Weighted F1 +0.5154 pp | F06 improves the continued F03 branch, but F06 includes additional continuation training and other stage changes. Do not attribute the gain to resolution alone. |
| F07 − F06 | Top-1 0.0000 pp; Macro R −0.9038 pp; Macro F1 −0.7673 pp | The frozen two-view TTA rule improved validation Macro F1 but did not improve the official-test balanced metrics. Do not generalise this to all TTA. |
| F08 − F06 | Top-1 +0.1241 pp; Macro P +0.2413 pp; Macro R −0.4192 pp; Macro F1 −0.2293 pp; Weighted F1 +0.0940 pp | The ensemble improves ranking/Top-k point estimates while slightly reducing balanced recall/F1. F08 remains complementary. |

### Critical missing orthogonal control

No verified **ConvNeXt-Small + standard cross-entropy on the original 5,247-image training distribution** exists in the collected final campaign evidence.

Therefore:

- F03 may be called the strongest isolated imbalance intervention **within the tested campaign**.
- The paper must not claim a quantified loss-only causal gain of Gentle CB Focal over an architecture-matched original-distribution cross-entropy baseline.

---

## 8. Table P01-D — statistical uncertainty summary

### Frozen protocol

```text
Evaluation universe: 1,612 fixed AQUA20 test images
Bootstrap replicates: 10,000
Seed: 42
Confidence level: 95%
Top-1 CI: ordinary bootstrap
Macro F1 CI: true-class-stratified bootstrap
Pairwise differences: matched/paired resampling
Top-1 correctness comparison: two-sided exact McNemar
```

These analyses quantify uncertainty from the finite fixed test sample **conditional on the preserved trained models**. They do not estimate training-seed variability.

| Pair | Top-1 point difference | 95% paired bootstrap CI | Exact McNemar p | Phase-16 status |
|---|---:|---:|---:|---|
| F03 → F06 | +0.496 pp | [−0.496, +1.489] pp | 0.396 | Manuscript-preserved value; direct standalone R09 result table not surfaced in Phase 13 retrieval |
| F06 → F08 | +0.124 pp | [−0.434, +0.682] pp | 0.780 | Manuscript-preserved value; direct standalone R09 result table not surfaced in Phase 13 retrieval |

Both intervals include zero. On the preserved fixed test sample, these rows do not provide evidence of a statistically supported Top-1 difference under the stated paired procedure.

Do **not** convert “not statistically supported” into “equivalent”.

Still **To be verified** from direct standalone R09 outputs:

- model-level bootstrap interval table;
- pairwise Macro-F1 intervals;
- discordant-count tables;
- F04/F05 paired statistics;
- F06/F08 correctness-switch summary;
- prediction-source manifest.

---

## 9. Table P01-E — internal minority/error-analysis summary

### Official test support registry

Several classes have very small support, making recall highly sensitive to one or two predictions.

| Class | Support | One changed correct prediction changes recall by |
|---|---:|---:|
| `marine_dolphin` | 10 | 10.0000 pp |
| `octopus` | 10 | 10.0000 pp |
| `seaCucumber` | 10 | 10.0000 pp |
| `squid` | 10 | 10.0000 pp |
| `crab` | 11 | 9.0909 pp |
| `shrimp` | 11 | 9.0909 pp |
| `diver` | 13 | 7.6923 pp |
| `flatworm` | 13 | 7.6923 pp |
| `shark` | 19 | 5.2632 pp |

The full test split contains 1,612 images over 20 classes; large supports include `fish` (538) and `coral` (348).

### Frozen narrative boundary

Earlier frozen records describe F06 as comparatively strong on starfish, turtle, diver, fish, and rayfish, and weak on marine dolphin, octopus, squid, and shark.

However, Phase 12 did not surface the complete direct final F06 classification-report rows and raw confusion-matrix rows during its retrieval. Therefore P01 does **not** invent exact class-wise P/R/F1 values or raw confusion counts from figures.

Exact strongest/weakest rankings and major raw confusion counts remain:

> **To be verified from direct rows**

This limitation does not change the frozen aggregate F06 result.

---

## 10. Table P01-F — external generalisation

### E01A and E01B aggregate results

Both rows use the same 9,738 images from the 15 strictly shared external classes and the unchanged frozen F06 checkpoint.

| Evaluation | Output rule | Top-1 | Top-2 | Top-3 | Top-5 | Macro P | Macro R | Macro F1 | Weighted F1 | Role |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **E01A** | Raw F06 20-way | **76.5044%** | 87.1534% | 91.8464% | 95.7486% | **81.3575%** | 74.1616% | 75.2774% | 78.3601% | **Primary external result** |
| **E01B** | Restricted + renormalised 15-way | 78.6815% | 89.1456% | 93.5716% | 96.7652%* | 80.4852% | 76.7121% | 76.0247% | 79.0111% | Secondary taxonomy diagnostic |

\*The stored full-precision value rounds to 96.7652% at four decimals; both older 96.7653% text and the full-precision value round to 96.77% at manuscript precision. Use the full-precision recomputation as the source of truth.

E01A raw 20-way Top-1 prediction rate into the five AQUA20-only outputs:

```text
4.7546%
```

### External registry

```text
External dataset: Sea Animals Image Dataset
Source identifier: vencerlanz09/sea-animals-image-dataste
Folders/classes: 23
Images: 13,711
Strict shared classes: 15
Strict shared images: 9,738
External-only classes: 8
External-only images: 3,973
Decode errors: 0
Preserved probability matrix: 13,711 × 20
External training/fine-tuning/adaptation/model selection: NONE
```

### E01C known–unknown diagnostic

| Diagnostic | Known/shared | External-only |
|---|---:|---:|
| Images | 9,738 | 3,973 |
| Mean Top-1 probability | 0.8467 | 0.6418 |
| Median Top-1 probability | 0.9554 | 0.6866 |
| Mean predictive entropy | 0.5514 | 1.2062 |
| Mean normalised entropy | 0.1841 | 0.4026 |
| Mean maximum shared probability | 0.8173 | 0.6073 |
| Mean outside-shared probability mass | 0.0608 | 0.1046 |

Frozen unknown-separation scores:

| Unknown score | AUROC | AUPR (unknown positive) | FPR @ 95% unknown TPR |
|---|---:|---:|---:|
| `1 - max probability` | 0.7151 | 0.5570 | 0.8276 |
| **Normalised predictive entropy** | **0.7197** | **0.5687** | **0.8300** |
| `1 - max shared probability` | 0.7066 | 0.4715 | 0.8257 |
| Outside-shared probability mass | 0.6955 | 0.4307 | 0.8258 |

Interpretation:

> E01C shows limited known–unknown score separation and closed-set behaviour. It is not ordinary 23-class accuracy and it does not establish reliable open-set rejection.

Do not use XAI as a causal explanation for the external performance reduction.

---

## 11. Table P01-G — qualitative explainability summary

All XAI uses the same frozen source model:

```text
F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL
checkpoint: best_model.pth
best epoch: 2
SHA-256:
d0c225c28b25f09822bc99518a37f9b1fe642b2d01213da7f68e2d76a255832a
```

Common registry:

```text
Frozen images: 20
Groups: 4
Samples/group: 5
Publication subset: 12
Image representation: official test image -> RGB -> JPEG quality 95 -> reload -> Resize 384×384 -> tensor -> ImageNet normalisation
Explanation target: F06 predicted class
```

| Stage | Frozen protocol | Output role | Integrity status |
|---|---|---|---|
| X01 Grad-CAM | Target layer `features.7.2`; same 20 images; predicted-class target | Qualitative spatial diagnostic | 65 files; extracted-tree SHA-256 `6c0d3c77820dcea3a9ee75ba00581430f9d34566896867d2ead9844a6f4ea138`; original ZIP SHA-256 **To be verified** |
| X02 LIME | Seed 42; 1000 perturbations; SLIC requested 100 segments; compactness 10; sigma 1; 10 positive features; positive-only; hide-rest; cosine; kernel width 0.25 | Qualitative local superpixel diagnostic | 86 files; extracted-tree SHA-256 `11ca45b02e0f0ab98278c94b39c72fe0d8f655f4879e5f8747ef871b35f36db8`; original ZIP SHA-256 **To be verified** |
| X03 | Direct protocol/package/metrics not surfaced | **Not claimed complete** | IoU/Dice/mask comparison metrics **To be verified / unsupported** |

### Frozen qualitative interpretation

Supported:

- some correct cases overlap the visible organism;
- some explanations emphasise background, water, substrate, or neighbouring structures;
- some correct explanations are diffuse;
- visually plausible explanations can also occur for incorrect predictions.

Not supported:

- causal reasoning;
- biological understanding;
- human-like reasoning;
- quantitative Grad-CAM/LIME agreement without X03 evidence.

A side-by-side manuscript figure built from frozen X01/X02 outputs may be used as a qualitative panel. It must not be labelled as a completed quantitative X03 experiment.

---

## 12. Manuscript reconciliation status

### 12.1 Internal result table

The current manuscript/LaTeX six-metric F01–F08 table agrees with the frozen F09 values after two-decimal rounding.

**Status: RECONCILED**

### 12.2 Final model roles

Current manuscript language retains:

- F06 as primary balanced single model;
- F08 as complementary ensemble/ranking model;
- F03 as strongest isolated imbalance intervention within tested campaign.

**Status: RECONCILED**

### 12.3 Causal/ablation boundary

Current methodology correctly states that:

- F01–F02 mainly compare model scale under the same Aug-200 protocol;
- F04–F05 are the closest loss-focused comparison, with unequal maximum budgets;
- F03/F04, F03/F06 and F06/F08 require broader complete-configuration interpretation.

**Status: RECONCILED**

### 12.4 Statistical wording

The manuscript's statement that the F06/F08 Top-1 difference is statistically unsupported is consistent with the manuscript-preserved R09 interval and McNemar p-value. Direct standalone R09 result files were not surfaced in Phase 13.

**Status: SCIENTIFICALLY CONSISTENT, DIRECT OUTPUT ARTIFACT STILL TO BE VERIFIED**

### 12.5 External table

Current manuscript E01A/E01B values match frozen evidence at manuscript precision.

**Status: RECONCILED**

### 12.6 Explainability

Current qualitative Grad-CAM/LIME boundary is compatible with Phase 15. A combined X01/X02 panel is acceptable only as qualitative presentation.

**Status: RECONCILED WITH X03 NON-CLAIM**

### 12.7 Class-wise detailed claims

Exact per-class rankings and raw confusion counts should remain cautious until direct final rows are surfaced.

**Status: PARTIALLY RECONCILED / ROW-LEVEL EVIDENCE TO BE VERIFIED**

---

## 13. P01 publication-table readiness decision

The following table streams are ready for P02 manuscript use under the frozen boundaries:

```text
READY:
- final experiment-configuration table;
- F01–F08 aggregate AQUA20 result table;
- F06/F08 primary/complementary comparison;
- intervention/comparison summary with causal-boundary notes;
- E01A/E01B external-result table;
- E01C known–unknown diagnostic summary;
- X01/X02 qualitative XAI summary;
- fixed-test statistical-summary table with direct-output caveat;
- support-aware internal error-analysis summary.
```

The following must not be fabricated merely to make a table look complete:

```text
DO NOT INVENT:
- X03 IoU/Dice or quantitative Grad-CAM/LIME agreement;
- missing direct R09 row-level/statistical outputs;
- missing exact F06 per-class rows/raw confusion counts;
- missing package hashes/environment/efficiency measurements;
- repeated-seed optimisation variance;
- immutable dataset revision/content hashes.
```

Phase 16/P01 is complete. P02 manuscript writing is intentionally deferred to the next phase.
