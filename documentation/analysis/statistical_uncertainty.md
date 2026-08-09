# AQUA20 Statistical Uncertainty and Paired Model-Comparison Documentation

## 1. Document status

**Documentation phase:** Phase 13 — Statistical Uncertainty and Paired Model-Comparison Documentation  
**Status:** Completed with explicit direct-output verification caveats  
**Repository target:** `documentation/analysis/statistical_uncertainty.md`

No new `F10` or `F11` experiment identity is created for this analysis.

---

## 2. Phase 13 role

Phase 13 is:

```text
CROSS-EXPERIMENT STATISTICAL ANALYSIS
PAIRED MODEL COMPARISON
FINITE-TEST-SAMPLE UNCERTAINTY
DOCUMENTATION
```

It performs:

```text
Training: NONE
Fine-tuning: NONE
New model inference: NONE
New checkpoint: NONE
Final-method reselection: NONE
```

The statistical analysis uses preserved per-image predictions from already frozen campaign models. It does not update model parameters and does not reopen the F09 final reporting hierarchy.

---

## 3. Source-control hierarchy used

Phase 13 was documented using the established AQUA20 evidence hierarchy:

1. latest final manuscript and manuscript source;
2. FINAL evidence-freeze and frozen terminology records;
3. direct `R09_statistical_uncertainty.py`;
4. F09 final method freeze and campaign-consolidation documentation;
5. Phase 12 internal class-wise error-analysis documentation and handover;
6. preserved experiment records for F03/F04/F05/F06/F08;
7. older supervisor/audit material for chronology and context only.

### Important R09 retrieval boundary

The direct R09 analysis script was surfaced and inspected. It defines the statistical protocol, alignment safeguards, model set, pairwise comparisons, output filenames, and interpretation boundaries.

However, the following standalone generated R09 result files were **not surfaced directly** in the Phase 13 File Library retrieval:

```text
README_R09.md
R09_bootstrap_protocol.json
R09_model_bootstrap_confidence_intervals.csv
R09_pairwise_statistical_comparisons.csv
R09_prediction_source_manifest.csv
R09_*_prediction_disagreements.csv
R09_F06_F08_correctness_switch_summary.csv
```

The direct standalone F03/F04/F05/F06/F08 prediction CSVs were also not surfaced as row-level files in this retrieval.

Therefore, values that require those direct outputs are retained as:

> **To be verified**

They are not reconstructed from rounded manuscript values.

---

## 4. Frozen evaluation universe

The statistical analysis is defined on the unchanged AQUA20 official test split:

```text
Official test images = 1,612
Classes              = 20
Class order           = frozen campaign order
```

Frozen class order:

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

R09 uses F06 as the alignment anchor and requires the anchor to contain exactly 1,612 prediction rows. It also explicitly checks that the aligned true labels contain exactly 20 classes.

No alternate test reconstruction is introduced in Phase 13.

---

## 5. Frozen R09 statistical protocol

The direct R09 script fixes the following protocol:

| Component | Frozen rule |
|---|---|
| Bootstrap replicates | 10,000 |
| Random seed | 42 |
| Confidence level | 95% |
| Alpha | 0.05 |
| Top-1 model CI | Ordinary percentile bootstrap |
| Macro-F1 model CI | True-class-stratified percentile bootstrap |
| Pairwise Top-1 difference | Paired resampling using matched bootstrap observations |
| Pairwise Macro-F1 difference | Paired true-class-stratified resampling |
| Top-1 paired test | Two-sided exact McNemar test |
| Evaluation data | Unchanged 1,612-image official test split |
| Model state | Preserved frozen predictions; no retraining |
| Training-seed variance | Not estimated |

### 5.1 Top-1 bootstrap

For Top-1 accuracy, R09 samples test-image indices with replacement from the full 1,612-image test set and computes the model's correctness rate for each replicate.

This is an ordinary non-parametric percentile bootstrap over the fixed test observations.

### 5.2 Macro-F1 bootstrap

For Macro F1, R09 resamples **within each true class**, with replacement, while preserving the original number of observations from each class in every replicate.

This protects the frozen class-support structure while estimating finite-test-sample uncertainty for the macro-averaged metric.

### 5.3 Paired model differences

Pairwise comparisons use the **same resampled observations for both models**. Therefore the resampling preserves the paired structure of predictions on the same test images.

### 5.4 McNemar test

Top-1 correctness changes are additionally evaluated using a two-sided exact McNemar test based on the discordant correctness cases:

```text
reference correct / candidate wrong
candidate correct / reference wrong
```

The direct R09 script uses an exact binomial test for those discordant counts.

---

## 6. Direct prediction-alignment safeguards

A paired test is valid only when the compared predictions refer to the same underlying test observations in the same true-label space.

R09 contains explicit safeguards.

### 6.1 Column and label handling

The script:

- searches for recognised true-label columns;
- searches for recognised predicted-label columns;
- searches for optional sample/image key columns;
- prefers semantic class names when both semantic names and numeric IDs exist;
- extracts and validates ID-to-name mappings when both are preserved;
- canonicalises class labels before cross-file comparison.

### 6.2 Key-based alignment

When both files provide unique sample keys and the key sets match:

1. the candidate file is reordered to the F06 anchor key order;
2. true labels are canonicalised;
3. the reordered true-label sequence must match the F06 anchor;
4. otherwise the script raises an error.

### 6.3 Verified row-order fallback

If a common unique key is unavailable, R09 permits row-order alignment only if:

```text
row counts are equal
AND
canonical true-label sequences are exactly identical
```

If those conditions fail, the analysis stops rather than silently pairing mismatched observations.

### 6.4 Executed alignment outcome

The direct R09 source manifest that records the alignment method used for the preserved files was not surfaced during Phase 13.

Therefore:

```text
Executed F03↔F06 alignment record = To be verified
Executed F06↔F08 alignment record = To be verified
Executed F04↔F05 alignment record = To be verified
Prediction source SHA-256 records = To be verified
```

The **alignment procedure** is directly verified from the script; the **executed source-manifest rows** remain to be verified from the generated R09 outputs.

---

## 7. Models and intended comparisons

R09 defines the stable model order from the available preserved predictions:

```text
F03
F04
F05
F06
F08
```

The predefined pairwise comparisons are:

```text
F03 vs F06
F06 vs F08
F04 vs F05
```

Phase 13 prioritises:

1. F03 versus F06;
2. F06 versus F08;
3. F04 versus F05 only when both direct prediction files and the direct R09 pairwise output are surfaced and aligned.

---

## 8. Model-level uncertainty table

The frozen F09 point estimates are included below only to identify the models. The requested confidence intervals must come from direct R09 output and are not reconstructed.

| Model | Frozen Top-1 point estimate | Top-1 95% bootstrap CI | Frozen Macro F1 point estimate | Macro-F1 95% stratified-bootstrap CI | Direct R09 CI status |
|---|---:|---:|---:|---:|---|
| F03 | 91.6253% | **To be verified** | 87.9645% | **To be verified** | R09 CI CSV not surfaced |
| F04 | 91.3772% | **To be verified** | 86.0035% | **To be verified** | R09 CI CSV not surfaced |
| F05 | 91.1290% | **To be verified** | 84.3487% | **To be verified** | R09 CI CSV not surfaced |
| F06 | 92.1216% | **To be verified** | 88.1699% | **To be verified** | R09 CI CSV not surfaced |
| F08 | 92.2457% | **To be verified** | 87.9406% | **To be verified** | R09 CI CSV not surfaced |

### Interpretation rule

Standalone model confidence intervals are descriptive uncertainty intervals for each frozen model's performance on a finite test sample.

They must **not** be used as a substitute for the paired difference interval when the scientific question is whether two models differ on the same 1,612 images.

---

## 9. Pairwise statistical evidence

### 9.1 Manuscript-preserved Top-1 results

The final manuscript preserves the following Top-1 paired-comparison results:

| Comparison | Candidate − reference Top-1 difference | 95% paired-bootstrap CI | Exact McNemar p | Direct R09-output re-verification |
|---|---:|---:|---:|---|
| F03 → F06 | +0.496 pp | [-0.496, +1.489] pp | 0.396 | **To be verified** |
| F06 → F08 | +0.124 pp | [-0.434, +0.682] pp | 0.780 | **To be verified** |

These values are retained as **manuscript-preserved values**, not newly re-certified from the missing standalone R09 pairwise CSV.

### 9.2 F03 → F06

Manuscript-preserved result:

```text
Top-1 difference, F06 − F03 = +0.496 pp
95% paired-bootstrap CI      = [-0.496, +1.489] pp
Exact McNemar p              = 0.396
```

Direct R09 fields not surfaced:

```text
Prediction disagreement count                    = To be verified
F03 correct / F06 wrong                          = To be verified
F06 correct / F03 wrong                          = To be verified
Macro-F1 difference, F06 − F03                   = To be verified
Macro-F1 95% stratified paired-bootstrap CI      = To be verified
Bootstrap Monte-Carlo p-value, if reported        = To be verified
Executed alignment/source-manifest record         = To be verified
```

Scientific boundary:

F03→F06 compares **complete configurations**. F06 changes more than image resolution: it also introduces continuation training, a lower learning rate, a smaller batch size, and a new training budget.

Therefore the paired statistical result must not be described as a resolution-only causal effect.

### 9.3 F06 → F08

Manuscript-preserved result:

```text
Top-1 difference, F08 − F06 = +0.124 pp
95% paired-bootstrap CI      = [-0.434, +0.682] pp
Exact McNemar p              = 0.780
```

The frozen F09 campaign record also shows that F08's Top-1 point estimate corresponds to a net gain of only two correct predictions over F06 among 1,612 images.

That net gain is **not** sufficient to reconstruct the two discordant McNemar cells. The actual disagreement counts remain:

```text
Prediction disagreement count                    = To be verified
F06 correct / F08 wrong                          = To be verified
F08 correct / F06 wrong                          = To be verified
Macro-F1 difference, F08 − F06                   = To be verified
Macro-F1 95% stratified paired-bootstrap CI      = To be verified
Executed alignment/source-manifest record         = To be verified
```

Scientific boundary:

F08 is a probability ensemble, not another single checkpoint. Its slightly higher Top-1 point estimate does not replace F06 as the primary balanced single model.

### 9.4 F04 → F05

R09 defines F04 versus F05 as an optional third pair.

However, the direct Phase 13 retrieval did not surface:

```text
F04_test_predictions.csv
F05_test_predictions.csv
R09_pairwise_statistical_comparisons.csv row for F04 vs F05
```

Therefore all direct paired-statistical values for F04→F05 remain:

> **To be verified**

The aggregate frozen campaign metrics may still be discussed separately, but they must not be presented as if a paired-bootstrap interval or McNemar result had been directly verified.

Scientific boundary:

F04 and F05 are the closest loss-focused Aug-200 ConvNeXt configurations, but their maximum training budgets differ. The comparison is not a perfectly controlled loss-only causal ablation.

---

## 10. Statistical interpretation locked for manuscript use

### 10.1 What the manuscript-facing Top-1 intervals support

The manuscript-preserved F03→F06 and F06→F08 95% paired-bootstrap intervals both include zero.

The manuscript therefore reports that neither comparison provides statistically supported evidence of a Top-1 correctness difference on the fixed 1,612-image test split.

Phase 13 retains that wording as manuscript context while marking direct standalone R09-output re-verification as outstanding.

### 10.2 What non-significance does not mean

A non-significant McNemar result or a paired interval containing zero does **not** establish:

```text
model equivalence
practical equivalence
non-inferiority
identical performance
proof of no effect
```

No pre-specified equivalence margin or non-inferiority margin is part of the frozen R09 protocol.

Therefore those claims are not permitted.

### 10.3 Why paired inference is preferred for these comparisons

The models are evaluated on the same test images. Pairing uses within-image changes and therefore answers the comparison question more directly than comparing two unrelated standalone confidence intervals.

Do not infer significance from whether standalone model confidence intervals overlap or do not overlap.

### 10.4 Scope of uncertainty

These intervals quantify:

> uncertainty from finite resampling of the fixed test sample, conditional on the preserved trained models.

They do **not** quantify:

```text
random training-seed variability
optimizer-run variability
alternative checkpoint variability
alternative dataset reconstruction variability
hyperparameter-search variability
future-domain shift
```

Repeated independent training runs would be required to characterise training-run variability.

---

## 11. F09 hierarchy remains frozen

Statistical uncertainty analysis does not reopen final method selection.

The reporting hierarchy remains:

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

Statistical analysis is used to qualify the strength of comparative claims, not to perform post-hoc model switching after observing the official-test outcomes.

---

## 12. Manuscript-ready statistical methodology

Statistical uncertainty was assessed from preserved per-image predictions on the unchanged 1,612-image AQUA20 official test split, without retraining or new model inference. Top-1 accuracy uncertainty was estimated with 10,000 ordinary percentile bootstrap resamples. Macro F1 was assessed with 10,000 true-class-stratified percentile bootstrap resamples that preserved the original support of every true class in each replicate. Pairwise model differences used matched resamples so that both models were evaluated on the same resampled observations. Changes in Top-1 correctness were additionally assessed with a two-sided exact McNemar test. The bootstrap random seed was fixed at 42. These analyses quantify finite-test-sample uncertainty conditional on the preserved trained models; they do not estimate variability across independent training seeds or optimisation runs.

---

## 13. Manuscript-ready paired Top-1 table

**Use condition:** The table below reproduces the final manuscript values. Before treating it as directly R09-reverified documentation, replace the verification-status column only after the standalone R09 pairwise output is surfaced.

| Comparison | Top-1 difference | 95% paired-bootstrap CI | Exact McNemar p | Status |
|---|---:|---:|---:|---|
| F03 → F06 | +0.496 pp | [-0.496, +1.489] pp | 0.396 | Manuscript-preserved; direct R09 output **To be verified** |
| F06 → F08 | +0.124 pp | [-0.434, +0.682] pp | 0.780 | Manuscript-preserved; direct R09 output **To be verified** |

Recommended interpretation:

> The preserved paired Top-1 intervals for F03→F06 and F06→F08 both include zero, and the corresponding exact McNemar tests do not support a definitive Top-1 superiority claim on the fixed AQUA20 test split. The F03→F06 comparison concerns complete configurations rather than a resolution-only effect, while the F06→F08 result supports describing the ensemble's Top-1 advantage as marginal rather than definitive. These intervals reflect finite-test-sample uncertainty conditional on the frozen trained models and do not estimate training-run variability.

---

## 14. Direct files required for full numerical closure

To convert the remaining Phase 13 caveats into direct re-verification, surface the following preserved R09 outputs:

```text
README_R09.md
protocol/R09_bootstrap_protocol.json
tables/R09_model_bootstrap_confidence_intervals.csv
tables/R09_pairwise_statistical_comparisons.csv
tables/R09_prediction_source_manifest.csv
tables/R09_F03_vs_F06_prediction_disagreements.csv
tables/R09_F06_vs_F08_prediction_disagreements.csv
tables/R09_F04_vs_F05_prediction_disagreements.csv   # only if generated
tables/R09_F06_F08_correctness_switch_summary.csv
figures/R09_top1_bootstrap_ci.*
figures/R09_macro_f1_stratified_bootstrap_ci.*
figures/R09_top1_paired_difference_ci.*
figures/R09_macro_f1_paired_difference_ci.*
```

Also surface the exact prediction files used by R09:

```text
F03_test_predictions.csv
F04_test_predictions.csv
F05_test_predictions.csv
F06_test_predictions.csv
F08_test_predictions.csv
```

Required direct checks:

```text
F06 anchor rows = 1,612
all included models align to the same 1,612 observations
exact frozen 20-class taxonomy
true-label sequence equality after alignment
source path and SHA-256 records match the source manifest
bootstrap replicates = 10,000
seed = 42
alpha = 0.05
model-level CI rows reproduce README values
pairwise rows reproduce manuscript values
discordant correctness counts reproduce exact McNemar p-values
Macro-F1 paired intervals match the direct pairwise CSV
```

No rounded manuscript value should be used to reconstruct a missing direct output.

---

## 15. Reviewer / viva question bank

### Q1. What is Phase 13?

A statistical-analysis and documentation phase over preserved test predictions. It performs no model training, fine-tuning, new inference, or checkpoint creation.

### Q2. Did Phase 13 create F10 or F11?

No. Statistical analysis does not create a new experiment identity merely because it produces tables or confidence intervals.

### Q3. How many test images are used?

The unchanged 1,612-image AQUA20 official test split.

### Q4. How many classes are used?

Twenty, in the frozen campaign class order.

### Q5. How many bootstrap replicates are used?

10,000.

### Q6. What is the bootstrap seed?

42.

### Q7. How is Top-1 uncertainty estimated?

With an ordinary percentile bootstrap over test-image observations.

### Q8. Why is Macro F1 bootstrapped differently?

Macro F1 gives equal weight to classes. R09 therefore uses true-class-stratified bootstrap resampling so every replicate preserves the original class-support structure.

### Q9. Are pairwise differences bootstrapped independently for the two models?

No. The same resampled observations are used for both models, preserving the paired comparison.

### Q10. What paired test is used for Top-1 correctness changes?

A two-sided exact McNemar test.

### Q11. What observations enter McNemar's test?

Only discordant cases: reference-correct/candidate-wrong and candidate-correct/reference-wrong observations.

### Q12. Why is prediction alignment important?

Because a paired test is meaningful only when both predictions refer to the same underlying test observations and true labels.

### Q13. What happens if prediction files cannot be safely aligned?

The R09 script raises an error rather than silently performing a mismatched paired comparison.

### Q14. What is the manuscript-preserved F03→F06 Top-1 result?

F06 minus F03 is +0.496 percentage points, with a 95% paired-bootstrap interval of [-0.496, +1.489] points and exact McNemar p = 0.396.

### Q15. Does that prove F06 and F03 are equivalent?

No. A non-significant paired comparison is not an equivalence test.

### Q16. Is F03→F06 a resolution-only statistical comparison?

No. F06 also changes continuation training, learning rate, batch size, and training budget.

### Q17. What is the manuscript-preserved F06→F08 Top-1 result?

F08 minus F06 is +0.124 percentage points, with a 95% paired-bootstrap interval of [-0.434, +0.682] points and exact McNemar p = 0.780.

### Q18. Does the higher F08 Top-1 make F08 the primary model?

No. F09 already froze F06 as the primary balanced single model and F08 as the complementary ranking ensemble.

### Q19. What does a confidence interval containing zero mean here?

It means the paired resampling distribution includes no difference among plausible finite-test-sample differences at the stated 95% interval level. It does not prove the true performances are identical.

### Q20. Can overlapping standalone model confidence intervals be used as a significance test?

No. The paired difference distribution and paired test are the appropriate comparison tools for these matched predictions.

### Q21. Do these intervals estimate random-seed variability?

No. They are conditional on the preserved trained models and quantify finite-test-sample uncertainty only.

### Q22. Were multiple training seeds evaluated?

No repeated-seed campaign is frozen for these comparisons. Training-run variability therefore remains outside R09.

### Q23. Why are the disagreement counts not listed numerically in Phase 13?

Because the direct R09 disagreement outputs were not surfaced. They are kept **To be verified** instead of being reconstructed from net accuracy changes.

### Q24. Why are the model-level Top-1 and Macro-F1 CIs not filled in?

Because the direct `R09_model_bootstrap_confidence_intervals.csv` was not surfaced. The documentation does not invent or recompute the missing frozen R09 rows.

### Q25. Why is F04 versus F05 not reported with a p-value here?

Although R09 defines the pair, the direct F04/F05 prediction alignment and direct R09 pairwise result were not surfaced during this phase.

### Q26. Can non-significance be called non-inferiority?

No. Non-inferiority requires a pre-specified margin and design. R09 is not a non-inferiority analysis.

### Q27. Can statistical analysis be used to switch the final method after seeing test outcomes?

No. F09 final method selection remains frozen. Phase 13 only qualifies uncertainty around already preserved comparisons.

### Q28. What is the main statistical limitation?

The analysis does not characterise variability across independent training runs, seeds, optimisation paths, or alternative dataset reconstructions.

---

## 16. Phase 13 completion decision

Phase 13 documentation is complete under the evidence available in the File Library.

The following are directly documented and locked from the R09 source and frozen campaign evidence:

- analysis-only role;
- no retraining and no new inference;
- unchanged 1,612-image test split;
- frozen 20-class taxonomy;
- 10,000 bootstrap replicates;
- seed 42;
- 95% confidence level;
- ordinary percentile bootstrap for Top-1;
- true-class-stratified percentile bootstrap for Macro F1;
- paired resampling for model differences;
- two-sided exact McNemar testing;
- explicit prediction-alignment safeguards;
- F03↔F06, F06↔F08, and optional F04↔F05 comparison design;
- finite-test-sample uncertainty boundary;
- prohibition on equivalence/non-inferiority interpretation;
- prohibition on post-hoc final-method reselection;
- F09 reporting hierarchy unchanged.

The final manuscript's two paired Top-1 rows are preserved as manuscript-facing evidence, but their standalone direct R09-output re-verification remains **To be verified** because the generated pairwise CSV was not surfaced.

The following also remain **To be verified**:

```text
model-level Top-1 confidence intervals
model-level Macro-F1 confidence intervals
pairwise disagreement counts
discordant correctness cells
pairwise Macro-F1 differences
pairwise Macro-F1 confidence intervals
F04-vs-F05 paired numerical results
executed R09 prediction-source manifest
executed alignment method per source file
R09 output/package integrity hashes
```

No unsupported value was reconstructed.
