# AQUA20 — Internal Class-Wise Error Analysis and Final-Model Diagnostics

## Document status

**Documentation phase:** Phase 12 — Internal Class-Wise Error Analysis and Final-Model Diagnostics  
**Repository target:** `documentation/analysis/internal_classwise_error_analysis.md`  
**Status:** Completed with explicit row-level verification boundaries  
**Experiment identity created:** None  
**Training:** None  
**Fine-tuning:** None  
**New checkpoint:** None  
**Model selection reopened:** No

This is a cross-experiment analysis/documentation record. It interprets already frozen AQUA20 official-test evidence and does not create an `F10` experiment.

---

## 1. Locked reporting hierarchy

Phase 12 inherits the final F09 hierarchy without modification:

```text
Primary proposed method:
F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL

Primary proposed single model:
F06

Strongest balanced-class model:
F06

Complementary final ensemble:
F08_CONVNEXT_SMALL_HR384_YOLO26M_PROB_ENSEMBLE

Strongest ranking model:
F08

Best isolated imbalance intervention within the tested campaign:
F03 — Gentle Class-Balanced Focal Loss
```

Class-wise test evidence is used only for diagnostic interpretation. It is not used to reselect the final method, revise the ensemble rule, or create a new experiment.

---

## 2. Evidence hierarchy and retrieval rule

The Phase 12 evidence hierarchy is:

1. latest final manuscript;
2. FINAL evidence freeze and frozen terminology/index;
3. direct frozen class-wise package assets for F01–F08;
4. numerical and methodology audits;
5. completed F06–F09 documentation;
6. preserved campaign notebook and R06 class-wise analysis code;
7. earlier handovers and supervisor-facing summaries for lineage only.

The controlling rule for this phase is strict:

> A strongest/weakest-class ranking may be frozen only from direct classification-report rows, and a major confusion direction may be frozen only from direct raw confusion-matrix rows.

Narrative text, plotted bars, heatmaps, or a row-normalised figure must not be reverse-engineered into missing precision, recall, F1, or raw error counts.

Where direct rows were not surfaced by the File Library retrieval used in this phase, the value is marked **To be verified**.

---

## 3. Test split, class order, and support registry

All internal class-wise analysis refers to the unchanged AQUA20 official test split:

```text
Test images = 1,612
Classes     = 20
```

The frozen class order is:

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

### 3.1 Exact official-test supports

| ID | Class | Test support | One changed correct prediction: recall step | Phase-12 caution |
|---:|---|---:|---:|---|
| 00 | `coral` | 348 | 0.2874 pp | No Phase-12 small-support flag |
| 01 | `crab` | 11 | 9.0909 pp | Very small support |
| 02 | `diver` | 13 | 7.6923 pp | Very small support |
| 03 | `eel` | 41 | 2.4390 pp | No Phase-12 small-support flag |
| 04 | `fish` | 538 | 0.1859 pp | No Phase-12 small-support flag |
| 05 | `fishInGroups` | 72 | 1.3889 pp | No Phase-12 small-support flag |
| 06 | `flatworm` | 13 | 7.6923 pp | Very small support |
| 07 | `jellyfish` | 25 | 4.0000 pp | Small support |
| 08 | `marine_dolphin` | 10 | 10.0000 pp | Very small support |
| 09 | `octopus` | 10 | 10.0000 pp | Very small support |
| 10 | `rayfish` | 95 | 1.0526 pp | No Phase-12 small-support flag |
| 11 | `seaAnemone` | 221 | 0.4525 pp | No Phase-12 small-support flag |
| 12 | `seaCucumber` | 10 | 10.0000 pp | Very small support |
| 13 | `seaSlug` | 20 | 5.0000 pp | Small support |
| 14 | `seaUrchin` | 29 | 3.4483 pp | Small support |
| 15 | `shark` | 19 | 5.2632 pp | Small support |
| 16 | `shrimp` | 11 | 9.0909 pp | Very small support |
| 17 | `squid` | 10 | 10.0000 pp | Very small support |
| 18 | `starfish` | 40 | 2.5000 pp | No Phase-12 small-support flag |
| 19 | `turtle` | 76 | 1.3158 pp | No Phase-12 small-support flag |
| **Total** | **20 classes** | **1,612** | — | — |

The support total is exactly 1,612.

### 3.2 Meaning of the recall-step column

For a true class containing support \(n_c\), changing the number of correct predictions by one changes that class's recall by:

\[
\Delta \mathrm{Recall}_c = \frac{1}{n_c}.
\]

The percentage-point values above are therefore exact support-based sensitivity illustrations for recall.

They are **not** fixed step sizes for precision or F1. Precision depends on how many samples from other classes are predicted into the class, and F1 depends jointly on precision and recall.

The most fragile supports include:

```text
marine_dolphin = 10  → 10.0000 pp recall per one correct prediction
octopus        = 10  → 10.0000 pp
squid          = 10  → 10.0000 pp
seaCucumber    = 10  → 10.0000 pp
crab           = 11  →  9.0909 pp
shrimp         = 11  →  9.0909 pp
diver          = 13  →  7.6923 pp
flatworm       = 13  →  7.6923 pp
shark          = 19  →  5.2632 pp
```

Accordingly, point estimates for small classes must not be presented as statistically stable without separate uncertainty evidence.

---

## 4. Campaign-wide class-wise evidence inventory

The campaign packages and preserved notebook/documentation establish that class-wise outputs were created and/or archived for F01–F08. Phase 12 distinguishes **asset-level evidence** from **direct row-level evidence surfaced in this retrieval**.

| Experiment | Classification report | Raw confusion matrix | Row-normalised confusion matrix | Test predictions | Direct complete 20-row report surfaced in Phase 12? | Phase-12 use |
|---|---|---|---|---|---|---|
| F01 | Preserved/created | Preserved/created | Preserved/created | 1,612 rows | No | Asset inventory only; full class narrative **To be verified** |
| F02 | Preserved/created | Preserved/created | Preserved/created | 1,612 rows | No | Asset inventory only; retain exact identity `F02_YOLO26S_CLS_AUG200` |
| F03 | Package audit preserved | Package audit preserved | Package audit preserved | 1,612 rows | No | Selected historical class evidence exists, but exhaustive 20-row analysis not rebuilt |
| F04 | Curated package preserved | Curated package preserved | Curated package preserved | 1,612 rows | No | Selected flatworm row previously documented; exhaustive 20-row analysis not rebuilt |
| F05 | Official evaluation produced | Official evaluation produced | Official evaluation produced | 1,612 rows | No | Asset inventory only; exhaustive class narrative **To be verified** |
| F06 | Package audit preserved | Package audit preserved | Package audit preserved | 1,612 rows | **No** | Primary diagnostic target; exact row/ranking re-verification remains **To be verified** |
| F07 | Package audit preserved | Package audit preserved | Package audit preserved | 1,612 rows | **No** | No new F07 strongest/weakest-class narrative |
| F08 | Curated package preserved | Curated package preserved | Curated package preserved | 1,612 rows | **No** | No F08 strongest/weakest-class narrative invented |

This inventory does not imply that the archived CSVs do not exist. It records that their complete row contents were not surfaced as directly inspectable File Library results in the present Phase 12 retrieval.

---

## 5. F06 as the primary class-wise diagnostic model

### 5.1 Frozen aggregate context

F06 official-test result:

| Metric | F06 |
|---|---:|
| Top-1 | 92.1216% |
| Top-2 | 97.9529% |
| Top-3 | 99.2556% |
| Top-5 | 99.7519% |
| Macro Precision | 90.4312% |
| Macro Recall | 87.4362% |
| Macro F1 | 88.1699% |
| Weighted F1 | 92.0810% |
| Support | 1,612 |

The Weighted-F1 minus Macro-F1 gap is:

```text
92.0810% − 88.1699% = 3.9111 pp
```

This is compatible with non-uniform class-wise performance. It does not identify which classes are weak; that requires direct per-class rows.

### 5.2 Previously documented F06 strong/weak narrative

The final manuscript, F06 documentation, and earlier F06 handover consistently report:

**Comparatively strong F1 classes**

```text
starfish
turtle
diver
fish
rayfish
```

**Weakest reported F1 sequence**

```text
marine_dolphin
octopus
squid
shark
```

However, the Phase 11 handover requires these statements to be re-verified against direct `F06_classification_report.csv` rows before Phase 12 freezes them as a final numerical class-wise table.

The complete direct CSV rows were not surfaced in this Phase 12 retrieval. Therefore the safe status is:

| Class | Test support | Previously documented status | Phase-12 re-verification status |
|---|---:|---|---|
| `starfish` | 40 | Previously reported as comparatively strong | **To be verified** — Direct F06 metric row not surfaced in Phase 12 retrieval |
| `turtle` | 76 | Previously reported as comparatively strong | **To be verified** — Direct F06 metric row not surfaced in Phase 12 retrieval |
| `diver` | 13 | Previously reported as comparatively strong | **To be verified** — Direct F06 metric row not surfaced in Phase 12 retrieval |
| `fish` | 538 | Previously reported as comparatively strong | **To be verified** — Direct F06 metric row not surfaced in Phase 12 retrieval |
| `rayfish` | 95 | Previously reported as comparatively strong | **To be verified** — Direct F06 metric row not surfaced in Phase 12 retrieval |
| `marine_dolphin` | 10 | Previously reported as lowest F1 | **To be verified** — Direct F06 metric row not surfaced in Phase 12 retrieval |
| `octopus` | 10 | Previously reported as next-lowest F1 | **To be verified** — Direct F06 metric row not surfaced in Phase 12 retrieval |
| `squid` | 10 | Previously reported as next-lowest F1 | **To be verified** — Direct F06 metric row not surfaced in Phase 12 retrieval |
| `shark` | 19 | Previously reported as next-lowest F1 | **To be verified** — Direct F06 metric row not surfaced in Phase 12 retrieval |

**Phase-12 decision:** the prior narrative is preserved as historical/frozen manuscript context, but it is **not newly re-certified from direct row values in this document**. No precision, recall, or F1 number is reconstructed from the per-class figure.

### 5.3 Why the low-support warning matters for the weak sequence

The reported weak sequence includes three 10-image classes and one 19-image class:

| Class | Support | One-correct-prediction recall step |
|---|---:|---:|
| `marine_dolphin` | 10 | 10.0000 pp |
| `octopus` | 10 | 10.0000 pp |
| `squid` | 10 | 10.0000 pp |
| `shark` | 19 | 5.2632 pp |

Therefore a very small number of prediction changes can materially alter their recall and can also materially alter F1. The correct interpretation is descriptive and sample-sensitive, not a claim of stable population-level ordering.

---

## 6. Confusion-matrix interpretation

### 6.1 Raw versus row-normalised confusion matrices

For raw confusion matrix \(C\):

\[
C_{ij} =
\text{number of true-class } i \text{ samples predicted as class } j.
\]

For the row-normalised matrix \(R\):

\[
R_{ij} =
\frac{C_{ij}}{\sum_j C_{ij}}
=
\frac{C_{ij}}{n_i}.
\]

They answer different questions:

- **Raw matrix:** how many actual errors occurred in a direction;
- **Row-normalised matrix:** what fraction of one true class was assigned to each predicted class.

A high row-normalised error rate in a 10-image class can correspond to fewer images than a low percentage in a 538-image class. Major confusion directions by error count therefore must be identified from the raw matrix, not from heatmap intensity alone.

### 6.2 Previously documented candidate F06 directions

Earlier F06 records and the manuscript describe the following directions:

```text
marine_dolphin → shark
marine_dolphin → fishInGroups
octopus        → eel
octopus        → crab
octopus        → fish
```

The Phase 12 rule is stricter than merely repeating those narrative examples: a final major-confusion table requires direct raw `F06_confusion_matrix.csv` rows and exact counts.

Because the raw matrix rows were not surfaced directly in the current retrieval:

- exact raw counts are **To be verified**;
- a rank ordering of these confusion directions is **not frozen here**;
- no count is reconstructed from a normalised heatmap;
- no causal explanation is assigned to any direction.

### 6.3 Causal-claim boundary

A confusion matrix can establish **what the model predicted**, not **why it predicted it**.

Not permitted from the matrix alone:

- “visual similarity caused this error”;
- “background complexity caused this confusion”;
- “class imbalance caused this individual error”;
- “the model learned the wrong biological feature.”

Those hypotheses require a separate controlled error-taxonomy or explanation study. Phase 12 keeps them outside the frozen class-wise evidence.

---

## 7. F06 versus F07 class-wise comparison boundary

F07 is an inference-only deterministic two-view TTA experiment. Its aggregate official-test change relative to F06 is frozen:

```text
Top-1        =  0.0000 pp
Macro Recall = −0.9038 pp
Macro F1     = −0.7674 pp
```

The F07 package preserves a classification report, raw confusion matrix, row-normalised confusion matrix, and 1,612 prediction rows. The completed F07 documentation already states that its full per-class rows were not surfaced for independent textual interpretation.

Therefore Phase 12 does **not** state which classes improved or deteriorated under F07.

**F06-versus-F07 aligned per-class delta table:** **To be verified** from direct `F06_classification_report.csv` and `F07_classification_report.csv` rows, with class order and support checked before subtraction.

The negative F07 aggregate result remains locked and does not depend on a new class-wise narrative.

---

## 8. F06 versus F08 class-wise comparison boundary

F08 is the validation-selected 0.70 F06 + 0.30 F01 heterogeneous probability ensemble.

Aggregate official-test differences, F08 minus F06:

```text
Top-1           = +0.1241 pp
Top-2           = +0.1240 pp
Top-3           = +0.1241 pp
Top-5           = +0.1240 pp
Macro Precision = +0.2413 pp
Macro Recall    = −0.4192 pp
Macro F1        = −0.2293 pp
Weighted F1     = +0.0940 pp
```

The Top-1 difference corresponds to only two additional correct predictions among 1,612 test images.

The F08 package contains:

```text
tables/F08_classification_report.csv
tables/F08_confusion_matrix.csv
tables/F08_confusion_matrix_normalized.csv
tables/F08_test_predictions.csv
```

But the complete F08 class rows were not surfaced in the current Phase 12 retrieval. Therefore:

- no F08 strongest/weakest-class ranking is created;
- no F06-versus-F08 per-class F1 delta is created;
- no F08 confusion direction is inferred from aggregate metrics;
- no class-wise explanation is used to reopen the F09 selection hierarchy.

The final interpretation remains:

> F08 provides the strongest ranking-oriented result, while F06 retains the stronger official-test Macro Recall and Macro F1 and remains the primary balanced single-model method.

---

## 9. Manuscript-ready evidence-safe tables

### 9.1 Support and sample-sensitivity table

A concise manuscript table may safely report support and recall sensitivity because those values are directly determined by the frozen test registry:

| Class | Support | Recall change from one correct prediction |
|---|---:|---:|
| `marine_dolphin` | 10 | 10.00 pp |
| `octopus` | 10 | 10.00 pp |
| `squid` | 10 | 10.00 pp |
| `shark` | 19 | 5.26 pp |
| `starfish` | 40 | 2.50 pp |
| `turtle` | 76 | 1.32 pp |
| `diver` | 13 | 7.69 pp |
| `fish` | 538 | 0.19 pp |
| `rayfish` | 95 | 1.05 pp |

Caption-safe wording:

> Selected official-test supports and the percentage-point change in class recall associated with one additional correct prediction. The sensitivity values apply to recall only; precision and F1 depend on the full prediction distribution.

### 9.2 F06 class-wise metric table — controlled placeholder

The final numerical table must be populated only from the direct frozen report:

| Class | Precision | Recall | F1 | Support | Status |
|---|---:|---:|---:|---:|---|
| `starfish` | **To be verified** | **To be verified** | **To be verified** | 40 | Await direct F06 row |
| `turtle` | **To be verified** | **To be verified** | **To be verified** | 76 | Await direct F06 row |
| `diver` | **To be verified** | **To be verified** | **To be verified** | 13 | Await direct F06 row |
| `fish` | **To be verified** | **To be verified** | **To be verified** | 538 | Await direct F06 row |
| `rayfish` | **To be verified** | **To be verified** | **To be verified** | 95 | Await direct F06 row |
| `marine_dolphin` | **To be verified** | **To be verified** | **To be verified** | 10 | Await direct F06 row |
| `octopus` | **To be verified** | **To be verified** | **To be verified** | 10 | Await direct F06 row |
| `squid` | **To be verified** | **To be verified** | **To be verified** | 10 | Await direct F06 row |
| `shark` | **To be verified** | **To be verified** | **To be verified** | 19 | Await direct F06 row |

This placeholder is intentionally preferable to reverse-engineering plot values.

### 9.3 Confusion-summary table — controlled placeholder

| True class | Predicted class | Raw error count | Row-normalised rate | Phase-12 status |
|---|---|---:|---:|---|
| `marine_dolphin` | `shark` | **To be verified** | **To be verified from direct matrix** | Previously documented direction |
| `marine_dolphin` | `fishInGroups` | **To be verified** | **To be verified from direct matrix** | Previously documented direction |
| `octopus` | `eel` | **To be verified** | **To be verified from direct matrix** | Previously documented direction |
| `octopus` | `crab` | **To be verified** | **To be verified from direct matrix** | Previously documented direction |
| `octopus` | `fish` | **To be verified** | **To be verified from direct matrix** | Previously documented direction |

No “major” label should be attached until the direct raw counts are available and ranked.

---

## 10. Evidence-safe manuscript prose

The following wording is safe for a manuscript or technical report without inventing unavailable rows:

> Internal class-wise analysis used the unchanged 1,612-image AQUA20 official test split and the frozen 20-class order. The class distribution is highly uneven at evaluation time: several minority categories contain only 10–19 test images, so one changed prediction can alter class recall by approximately 5–10 percentage points. The preserved F06 report and confusion-matrix assets therefore need to be interpreted with explicit support awareness. Earlier frozen manuscript records describe starfish, turtle, diver, fish, and rayfish as comparatively strong F1 classes and marine dolphin, octopus, squid, and shark as the weakest sequence; however, Phase 12 does not reconstruct their numerical precision/recall/F1 values from figures. Likewise, previously described confusion directions are retained as descriptive leads rather than re-ranked without direct raw confusion-matrix counts. These class-wise observations are diagnostic and do not reopen the validation-governed final method selection.

If direct F06 rows are later surfaced, this paragraph may be strengthened by replacing the verification caveat with the exact frozen numbers and raw error counts.

---

## 11. Limitations

1. **Small class support.** Several test classes contain only 10–29 images. Their recall and F1 point estimates can move sharply after one or two changed predictions.
2. **Single frozen model realisation.** F06 class-wise values describe one validation-selected trained checkpoint. They do not estimate training-seed or optimisation-run variability.
3. **No causal inference from confusion matrices.** Error destinations do not establish a mechanism.
4. **No reverse-engineering from figures.** Heatmap percentages and plotted F1 bars are not substitutes for direct CSV rows when the controlling CSV is unavailable.
5. **Incomplete Phase-12 row surfacing.** Package-level evidence confirms the existence of F06/F07/F08 class reports and confusion matrices, but the complete direct rows were not surfaced by the current File Library retrieval.
6. **No post-hoc reselection.** Official-test class-wise behaviour is not used to choose a different model or ensemble rule.
7. **No claim of statistical stability.** Class-wise uncertainty intervals are not frozen in this phase.
8. **No universal class-difficulty claim.** A class that is weak for one frozen checkpoint on this split is not declared intrinsically difficult across datasets or training runs.

---

## 12. Reviewer / viva question bank

### Q1. What is Phase 12?

A cross-experiment internal error-analysis documentation phase. It performs no training, fine-tuning, new inference, or checkpoint creation.

### Q2. Did Phase 12 create F10?

No. No new experiment identity is created.

### Q3. What model is primary for detailed class-wise diagnosis?

F06, because F09 already froze it as the primary proposed single model and strongest balanced-class model.

### Q4. Does F08's higher Top-1 change that hierarchy?

No. F08 is the complementary final ensemble and strongest ranking model; F06 retains stronger official-test Macro Recall and Macro F1.

### Q5. How many internal official-test images are analysed?

1,612.

### Q6. How many classes are there?

20, in the frozen campaign class order.

### Q7. Why is support essential in class-wise interpretation?

Because a class with support 10 changes recall by 10 percentage points when the number of correct predictions changes by one.

### Q8. Is that 10-point step also the exact F1 step?

No. F1 also depends on precision, which depends on predictions arriving from other true classes.

### Q9. What is the difference between raw and row-normalised confusion matrices?

Raw matrices record counts. Row-normalised matrices divide each true-class row by its support and report within-class proportions.

### Q10. Which matrix should identify major confusion directions by number of errors?

The raw confusion matrix.

### Q11. Can a normalised heatmap be used to reconstruct a missing raw count?

Not in Phase 12. The direct raw matrix is the controlling evidence.

### Q12. Which F06 classes were previously reported as comparatively strong?

Starfish, turtle, diver, fish, and rayfish.

### Q13. Which were previously reported as weakest?

Marine dolphin, octopus, squid, and shark.

### Q14. Are those rankings newly re-verified numerically in Phase 12?

No. The complete direct `F06_classification_report.csv` rows were not surfaced in the Phase 12 retrieval, so the prior narrative is preserved but the numerical re-verification remains **To be verified**.

### Q15. What confusion directions were previously documented for F06?

Marine dolphin to shark/fishInGroups and octopus to eel/crab/fish.

### Q16. Are raw counts for those directions frozen here?

No. Direct raw rows were not surfaced, so exact counts and rankings remain **To be verified**.

### Q17. Why is no F07-specific strongest/weakest narrative added?

The F07 package contains the report and matrices, but the complete per-class rows were not surfaced for direct analysis.

### Q18. Why is no F08-specific strongest/weakest narrative added?

For the same evidence-control reason. The F08 package contains the assets, but Phase 12 does not invent class-level conclusions from incomplete snippets.

### Q19. Can a confusion direction prove visual similarity caused an error?

No. It describes an output pattern, not a causal mechanism.

### Q20. Can Phase 12 change the selected method after seeing test class metrics?

No. Model and inference-rule selection were already validation-governed and frozen in F09.

### Q21. What is the key unresolved evidence needed to close the class-wise numerical table?

The direct final `F06_classification_report.csv` and `F06_confusion_matrix.csv` rows; aligned F07/F08 rows are needed for direct per-class comparisons.

### Q22. What is the main scientific value of Phase 12 despite the unresolved rows?

It freezes the evidence standard, exact support structure, sample-sensitivity limits, class-wise interpretation boundaries, and a non-post-hoc diagnostic framework without fabricating unavailable numbers.

---

## 13. Verification checklist retained for future evidence completion

### Required to fully re-freeze F06 class-wise numbers

```text
F06_classification_report.csv
F06_confusion_matrix.csv
F06_confusion_matrix_normalized.csv
F06_test_predictions.csv
```

Required checks:

```text
20 class rows
support sum = 1,612
class order exactly matches frozen registry
classification-report recall = diagonal of row-normalised CM
raw CM row sums = exact class supports
raw CM total = 1,612
metrics reproduce from predictions
```

### Required for F06-versus-F07 aligned per-class comparison

```text
F07_classification_report.csv
F07_confusion_matrix.csv
F07_confusion_matrix_normalized.csv
F07_test_predictions.csv
```

### Required for F06-versus-F08 aligned per-class comparison

```text
F08_classification_report.csv
F08_confusion_matrix.csv
F08_confusion_matrix_normalized.csv
F08_test_predictions.csv
```

No values should be copied from plots when these direct files are the intended controlling evidence.

---

## 14. Phase 12 completion decision

Phase 12 is complete as an evidence-controlled documentation phase.

The following are now locked:

- no new experiment or checkpoint;
- F09 reporting hierarchy preserved;
- unchanged 1,612-image test split preserved;
- exact frozen 20-class order preserved;
- exact 20-class official-test support registry;
- support-based recall sensitivity documented;
- campaign-wide class-wise asset inventory;
- raw-versus-normalised confusion-matrix interpretation;
- F06 as the primary diagnostic model;
- prior F06 strong/weak narrative preserved without inventing numeric rows;
- prior F06 confusion directions preserved without inventing raw counts;
- F07/F08 class-wise narratives withheld where direct rows were not surfaced;
- no causal explanation from confusion evidence alone;
- no test-based reopening of final method selection;
- reviewer/viva question bank;
- explicit direct-file checklist for closing remaining row-level verification.

The unresolved exact class rows and raw confusion counts remain **To be verified**, exactly as required by the Phase 11 handover.

The next documentation phase is:

**Phase 13 — Statistical Uncertainty and Paired Model-Comparison Documentation.**
