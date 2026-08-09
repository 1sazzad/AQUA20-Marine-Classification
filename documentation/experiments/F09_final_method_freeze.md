# F09 Final Method Freeze and Campaign Consolidation

## 1. Document status

**Documentation phase:** Phase 11 — F09 Final Method Freeze and Campaign Consolidation  
**Status:** Completed and locked  
**Repository target:** `documentation/experiments/F09_final_method_freeze.md`  
**Official campaign:** `04-07-2026 AQUA20 FINAL BEAT BASE PLAN`  
**Official stage identity:** `F09_FINAL_METHOD_FREEZE_AND_CAMPAIGN_COMPARISON`

F09 is the campaign-closing analysis stage. It does not introduce a new trainable model and it does not create a new checkpoint.

```text
MODE:
ANALYSIS
COMPARISON
FINAL METHOD SELECTION
CAMPAIGN FREEZE

TRAINING:
NONE

FINE-TUNING:
NONE

NEW CHECKPOINT:
NONE
```

---

## 2. Scope and source-control rule

This document consolidates the completed F01–F08 campaign, freezes the final scientific reporting hierarchy, records the source checkpoints used by the final methods, preserves the internal beat-base target status, and defines the claim boundaries that must be retained in later documentation and manuscript work.

Evidence was interpreted using the locked hierarchy:

1. latest final manuscript, *Improving Underwater Marine Species Classification through Imbalance-Aware Learning and Cross-Dataset Evaluation*;
2. FINAL evidence-freeze and frozen-terminology/index records;
3. official `README_F09.txt` and F09 final campaign package evidence;
4. preserved F09 execution in `04-07-2026-aqua20-final-beat-base-plan.ipynb`;
5. frozen F01–F08 experiment evidence and completed experiment documentation;
6. `F09_F01_F08_campaign_comparison.csv`;
7. `F09_primary_metric_comparison.csv`;
8. `F09_intervention_effect_comparison.csv`;
9. `F09_validation_vs_test_generalization.csv`;
10. `F09_final_method_selection.csv`;
11. `F09_beat_base_target_comparison.csv`;
12. `F09_source_checkpoint_freeze_record.csv`;
13. `F09_scientific_conclusions.csv`;
14. F09 comparison figures and package-integrity evidence;
15. older reports and handovers only for historical lineage when they do not conflict with later frozen evidence.

Direct F09 execution evidence and FINAL frozen records supersede older incomplete or obsolete descriptions.

Unsupported values are not reconstructed. They are marked:

> **To be verified**

---

## 3. Critical experiment identities

The final campaign identities are:

```text
F01 = F01_YOLO26M_CLS_AUG200
F02 = F02_YOLO26S_CLS_AUG200
F03 = F03_CONVNEXT_SMALL_GENTLE_CB_FOCAL
F04 = F04_CONVNEXT_SMALL_AUG200
F05 = F05_CONVNEXT_SMALL_AUG200_GENTLE_CB_FOCAL
F06 = F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL
F07 = F07_CONVNEXT_SMALL_HR384_TTA
F08 = F08_CONVNEXT_SMALL_HR384_YOLO26M_PROB_ENSEMBLE
F09 = F09_FINAL_METHOD_FREEZE_AND_CAMPAIGN_COMPARISON
```

### Mandatory identity correction

`F02` is the YOLO26s-cls + Aug-200 experiment.

It must never be relabelled as a ConvNeXt-Small cross-entropy baseline. The ConvNeXt-Small + Aug-200 + standard cross-entropy experiment is `F04_CONVNEXT_SMALL_AUG200`.

---

## 4. What F09 does — and does not do

F09 performs evidence consolidation and final reporting selection over already completed experiments.

F09 does:

- consolidate the frozen F01–F08 official-test results;
- compare the evaluated campaign interventions;
- preserve validation-versus-test generalisation evidence for F06, F07, and F08;
- freeze the primary and complementary final reporting roles;
- freeze the final F06 and F01 source-checkpoint records;
- close the internal 92.68% beat-base target comparison;
- create final comparison tables and publication-ready figures;
- close the model campaign without post-hoc method switching.

F09 does **not**:

- train a model;
- fine-tune a model;
- update model parameters;
- choose a new optimiser or scheduler;
- perform early stopping;
- search for a new checkpoint;
- search for a new ensemble weight;
- search for a new TTA policy;
- use the official test set to create another model configuration;
- produce an `F09` model checkpoint.

The final campaign hierarchy is therefore an analysis decision over frozen evidence, not another optimisation stage.

---

## 5. Frozen F01–F08 official-test comparison

All results below refer to the unchanged **1,612-image AQUA20 official test split**.

| Experiment | Top-1 | Top-2 | Top-3 | Top-5 | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F01 | 84.6774% | N/A | N/A | 98.9454% | 83.3054% | 73.9895% | 75.9908% | 84.4249% |
| F02 | 83.4367% | N/A | N/A | 97.9529% | 80.9103% | 65.2565% | 70.3909% | 82.8376% |
| F03 | 91.6253% | 97.7047% | 99.1315% | 99.6898% | 90.7850% | 86.9302% | 87.9645% | 91.5656% |
| F04 | 91.3772% | 97.2705% | 99.0074% | 99.6278% | 89.3241% | 84.3236% | 86.0035% | 91.2641% |
| F05 | 91.1290% | 97.4566% | 98.9454% | 99.7519% | 89.7496% | 82.6547% | 84.3487% | 90.9759% |
| **F06** | **92.1216%** | **97.9529%** | **99.2556%** | **99.7519%** | **90.4312%** | **87.4362%** | **88.1699%** | **92.0810%** |
| F07 | 92.1216% | 97.8908% | 99.3176% | 99.6898% | 89.8765% | 86.5324% | 87.4026% | 92.0771% |
| **F08** | **92.2457%** | **98.0769%** | **99.3797%** | **99.8759%** | **90.6725%** | **87.0170%** | **87.9406%** | **92.1750%** |

### Missing-metric rule

The final F09 comparison explicitly records the following as unavailable:

```text
F01 Top-2: N/A
F01 Top-3: N/A
F02 Top-2: N/A
F02 Top-3: N/A
```

These values must remain unavailable unless direct preserved evidence is later surfaced. They must not be interpolated, inferred, or reconstructed from other Top-k metrics.

For F03–F08, the complete aggregate Top-1/Top-2/Top-3/Top-5 and Macro/Weighted metrics shown above are preserved in the final campaign evidence.

---

## 6. Final reporting hierarchy

F09 freezes the final scientific reporting hierarchy as:

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

This hierarchy is metric- and role-aware. It does not simply rank all systems by Top-1 accuracy.

---

## 7. Why F06 remains the primary method

### 7.1 F06 frozen official-test result

```text
Top-1           = 92.1216%
Top-2           = 97.9529%
Top-3           = 99.2556%
Top-5           = 99.7519%
Macro Precision = 90.4312%
Macro Recall    = 87.4362%
Macro F1        = 88.1699%
Weighted F1     = 92.0810%
support         = 1,612
```

F06 is the strongest completed **balanced single-model** configuration in the campaign and retains the highest official-test Macro Recall and Macro F1 among the final F06/F08 candidates.

### 7.2 F08 frozen official-test result

```text
Top-1           = 92.2457%
Top-2           = 98.0769%
Top-3           = 99.3797%
Top-5           = 99.8759%
Macro Precision = 90.6725%
Macro Recall    = 87.0170%
Macro F1        = 87.9406%
Weighted F1     = 92.1750%
support         = 1,612
```

F08 is the strongest campaign **ranking-oriented** result.

### 7.3 F08 versus F06

| Metric | F06 | F08 | F08 − F06 |
|---|---:|---:|---:|
| Top-1 | 92.1216% | 92.2457% | +0.1241 pp |
| Top-2 | 97.9529% | 98.0769% | +0.1240 pp |
| Top-3 | 99.2556% | 99.3797% | +0.1241 pp |
| Top-5 | 99.7519% | 99.8759% | +0.1240 pp |
| Macro Precision | 90.4312% | 90.6725% | +0.2413 pp |
| Macro Recall | 87.4362% | 87.0170% | −0.4192 pp |
| Macro F1 | 88.1699% | 87.9406% | −0.2293 pp |
| Weighted F1 | 92.0810% | 92.1750% | +0.0940 pp |

The Top-1 difference corresponds to only two additional correct predictions among 1,612 test images.

### Final selection logic

F08 improves Top-1, Top-2, Top-3, Top-5, Macro Precision, and Weighted F1, but F06 retains stronger Macro Recall and Macro F1. The study is explicitly concerned with balanced recognition under class imbalance, so the primary method is not selected by Top-1 alone.

Approved interpretation:

> The heterogeneous F08 ensemble produced a slightly higher Top-1 point estimate than F06, while F06 retained higher Macro Recall and Macro F1. F06 is therefore frozen as the primary balanced single-model method, while F08 is retained as the complementary ranking ensemble.

Do not write:

> F08 universally outperformed F06.

Do not write:

> F08 replaced F06 as the primary method because its Top-1 was higher.

---

## 8. F08 rule retained by F09

F09 does not reopen F08 selection.

The frozen ensemble remains:

```text
p_F08 = 0.70 * p_F06 + 0.30 * p_F01
```

The weight was selected from the predefined validation-only grid:

```text
0.50 / 0.50
0.60 / 0.40
0.70 / 0.30
0.80 / 0.20
```

Selection order:

```text
validation Macro F1
→ validation Macro Recall
→ validation Top-1
→ equal-weight preference if still tied
```

The final `0.70 F06 + 0.30 F01` rule was locked before official-test evaluation. F09 performs no additional weight search.

---

## 9. Intervention-effect analysis

F09 consolidates the evaluated interventions but does not convert every comparison into a causal ablation.

### 9.1 F03 versus F04 — Gentle CB Focal branch versus Aug-200 cross-entropy branch

Frozen aggregate differences, F03 minus F04:

```text
Top-1           = +0.2481 pp
Top-2           = +0.4342 pp
Top-3           = +0.1241 pp
Top-5           = +0.0620 pp
Macro Precision = +1.4609 pp
Macro Recall    = +2.6066 pp
Macro F1        = +1.9610 pp
Weighted F1     = +0.3015 pp
```

Interpretation:

F03 was stronger than F04 as an evaluated complete configuration and is retained as the strongest isolated imbalance-aware branch within the tested campaign.

Boundary:

F03 and F04 do **not** constitute a strict one-factor loss ablation. They differ in both training distribution and objective, and their online training pipelines also differ. The campaign does not contain an architecture-matched ConvNeXt-Small + standard cross-entropy model trained on the original 5,247-image distribution.

Therefore the study must not claim a quantified causal benefit of the focal objective over a verified original-distribution cross-entropy control.

---

### 9.2 Aug-200 alone

Within the evaluated ConvNeXt configurations, the Aug-200 + cross-entropy F04 branch did not exceed the F03 original-distribution focal branch on the main balanced metrics.

Approved interpretation:

> Train-only Aug-200 was weaker than the F03 focal branch for balanced-class performance in the tested configurations.

Not permitted:

> Oversampling is generally harmful.

---

### 9.3 F03 versus F05 — adding Aug-200 to the focal branch

Frozen aggregate differences, F05 minus F03:

```text
Top-1           = −0.4963 pp
Top-2           = −0.2481 pp
Top-3           = −0.1861 pp
Top-5           = +0.0621 pp
Macro Precision = −1.0354 pp
Macro Recall    = −4.2755 pp
Macro F1        = −3.6158 pp
Weighted F1     = −0.5897 pp
```

Interpretation:

The completed Aug-200 + Gentle CB Focal configuration did not improve the original-distribution Gentle CB Focal branch. The two imbalance interventions were not complementary in this tested configuration.

Boundary:

F03 and F05 also differ in training distribution and maximum training budget. This is a configuration-level result, not a universal proof that data-level balancing and class-balanced focal optimisation cannot work together.

---

### 9.4 F04 versus F05 — closest loss-focused Aug-200 comparison

F04 and F05 share:

- ConvNeXt-Small;
- Aug-200 training distribution;
- 224×224 input;
- AdamW;
- learning rate `1e-5`;
- weight decay `1e-4`;
- batch size 16;
- validation-based checkpoint selection.

However, the maximum training budgets differ:

```text
F04 maximum epochs = 40
F05 maximum epochs = 25
```

Some F05 stopping/scheduler details were historically less complete.

Therefore this is the closest available loss-focused comparison, but it is **not a perfectly controlled loss-only ablation**.

---

### 9.5 F03 versus F06 — HR384 continuation

Frozen aggregate differences, F06 minus F03:

```text
Top-1           = +0.4963 pp
Top-2           = +0.2482 pp
Top-3           = +0.1241 pp
Top-5           = +0.0621 pp
Macro Precision = −0.3538 pp
Macro Recall    = +0.5060 pp
Macro F1        = +0.2054 pp
Weighted F1     = +0.5154 pp
```

The continuation produced the strongest balanced single-model result.

Boundary:

F03→F06 is not a resolution-only experiment. F06 also includes:

- additional continuation training;
- a lower learning rate;
- a smaller batch size;
- a new training budget.

Approved description:

> Continuing the selected F03 checkpoint at 384×384 under the F06 continuation configuration produced the strongest balanced single-model result.

Not permitted:

> Increasing resolution alone caused the F06 improvement.

---

### 9.6 F06 versus F07 — deterministic horizontal-flip TTA

F07 is a direct inference ablation because it uses the same frozen F06 checkpoint and changes only the evaluated inference rule.

Official-test deltas:

```text
Top-1           =  0.0000 pp
Top-2           ≈ −0.0621 pp
Top-3           = +0.0620 pp
Top-5           ≈ −0.0621 pp
Macro Precision = −0.5547 pp
Macro Recall    = −0.9038 pp
Macro F1        = −0.7674 pp
Weighted F1     = −0.0039 pp
```

Validation Macro F1 had increased by approximately `+0.4714 pp`, but that gain did not generalise to the official test.

Scientific conclusion:

> The tested deterministic two-view horizontal-flip TTA rule was a negative official-test intervention for balanced metrics and does not replace F06.

Boundary:

This conclusion applies only to this locked two-view TTA construction. It is not evidence that TTA is universally ineffective.

---

### 9.7 F06 versus F08 — cross-family probability ensemble

F08 improved the ranking-oriented and several aggregate metrics relative to F06:

```text
Top-1           = +0.1241 pp
Top-2           = +0.1240 pp
Top-3           = +0.1241 pp
Top-5           = +0.1240 pp
Macro Precision = +0.2413 pp
Weighted F1     = +0.0940 pp
```

But:

```text
Macro Recall = −0.4192 pp
Macro F1     = −0.2293 pp
```

Scientific conclusion:

> F08 is a positive but complementary ranking result, not a replacement for the stronger balanced single-model F06 result.

Boundary:

F06 versus F08 is not a one-factor model ablation. F08 adds a second architecture and an additional inference path.

---

## 10. Validation-versus-test generalisation audit

F09 preserved direct validation-versus-test comparisons for the final-stage F06–F08 configurations.

| Experiment | Val Top-1 | Test Top-1 | Test−Val | Val Macro Recall | Test Macro Recall | Test−Val | Val Macro F1 | Test Macro F1 | Test−Val |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F06 | 92.9878% | 92.1216% | −0.8662 pp | 90.6790% | 87.4362% | −3.2428 pp | 88.5122% | 88.1699% | −0.3423 pp |
| F07 | 92.6829% | 92.1216% | −0.5613 pp | 91.2771% | 86.5324% | −4.7447 pp | 88.9836% | 87.4026% | −1.5810 pp |
| F08 | 93.4451% | 92.2457% | −1.1994 pp | 93.0674% | 87.0170% | −6.0504 pp | 90.7359% | 87.9406% | −2.7953 pp |

The key scientific lesson is that stronger validation balanced metrics did not guarantee equally strong official-test gains.

- F07 improved validation Macro F1 but reduced official-test Macro F1.
- F08 had the strongest final-stage validation Top-1, Macro Recall, and Macro F1, but the official-test balanced advantage over F06 did not materialise.
- F06 showed the smallest final-stage validation-to-test Macro-F1 drop and remained the strongest balanced test model.

This supports retaining the frozen test as a final reporting set rather than using it to reopen model or inference-rule selection.

---

## 11. Frozen source-checkpoint record

F09 freezes the exact source checkpoints that underlie the final reporting hierarchy.

### F06 primary model

```text
Experiment:
F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL

Checkpoint:
best_model.pth

Best epoch:
2

SHA-256:
d0c225c28b25f09822bc99518a37f9b1fe642b2d01213da7f68e2d76a255832a

Status:
VERIFIED
FROZEN
```

### F01 ensemble source

```text
Experiment:
F01_YOLO26M_CLS_AUG200

Checkpoint:
best.pt

Best epoch:
2

SHA-256:
aa489def1443a6cdab42a0a20a530b9e341d574dcaaf234bfe6c7983113e8e5a

Status:
VERIFIED
FROZEN
```

F09 creates no additional checkpoint.

---

## 12. Internal beat-base target status

The campaign used an internal reference Top-1 target:

```text
Reference Top-1 target = 92.6800%
```

The strongest completed final-campaign Top-1 was:

```text
F08 = 92.2457%
```

Difference:

```text
92.2457 - 92.6800 = -0.4343 percentage points
```

Final status:

```text
TARGET STATUS:
NOT BEATEN
```

### Claim boundary

The `92.68%` value is an **internal campaign target**, not an official AQUA20 benchmark and not a safe state-of-the-art threshold.

Do not write:

> The official AQUA20 benchmark was 92.68%.

Do not write:

> The campaign beat 92.68%.

Do not claim state of the art from this internal comparison.

---

## 13. F09 package audit

Official package:

```text
aqua20_f09_final_method_freeze_campaign_outputs.zip
```

Notebook-displayed approximate ZIP size:

```text
0.34 MB
```

Direct F09 execution reports:

```text
Required final tables = 8 / 8 VERIFIED
CSV tables in package = 8
Publication-ready figures = 3
README = README_F09.txt
Training performed = NONE
New checkpoint produced = NONE
```

### Curated package manifest

```text
README_F09.txt

tables/F09_F01_F08_campaign_comparison.csv
tables/F09_beat_base_target_comparison.csv
tables/F09_final_method_selection.csv
tables/F09_intervention_effect_comparison.csv
tables/F09_primary_metric_comparison.csv
tables/F09_scientific_conclusions.csv
tables/F09_source_checkpoint_freeze_record.csv
tables/F09_validation_vs_test_generalization.csv

figures/F09_campaign_top1_macro_f1.png
figures/F09_campaign_top5_weighted_f1.png
figures/F09_intervention_effect_deltas.png
```

Total listed package assets:

```text
12
```

comprising:

```text
1 README
8 CSV tables
3 figures
```

### Integrity fields not directly frozen

The following values must remain unresolved:

```text
Original F09 ZIP SHA-256:
To be verified

Exact original F09 ZIP byte count:
To be verified

Extracted-tree SHA-256:
To be verified

Exact extracted-tree byte count:
To be verified

Per-file immutable checksum manifest:
To be verified
```

The approximate `0.34 MB` notebook display must not be converted into an invented exact byte count.

---

## 14. Runtime and reproducibility boundary

The preserved campaign notebook contains direct environment evidence elsewhere in the session for:

```text
Python       = 3.12.13
PyTorch      = 2.7.1+cu118
torchvision  = 0.22.1+cu118
CUDA         = 11.8
GPU          = Tesla P100-PCIE-16GB
```

However, F09 itself is an analysis/comparison stage and the archive does not preserve a complete immutable F09-specific runtime/resource manifest.

Therefore the documentation does **not** claim that every F09 analysis operation required or used the GPU.

Still **To be verified** for F09:

- exact F09-specific CPU/GPU execution attribution for every analysis step;
- immutable container image/digest;
- driver version;
- native cuDNN version;
- CPU model;
- RAM allocation;
- complete package freeze;
- exact run-local scikit-learn version;
- exact run-local pandas/NumPy versions unless separately surfaced;
- controlled analysis runtime;
- controlled latency or throughput;
- peak GPU memory;
- energy/compute cost;
- standardised parameters/FLOPs/model-size comparison across architectures;
- complete original ZIP/tree integrity metadata.

F09's scientific conclusions do not depend on inventing these missing environment values because it operates on already frozen experiment outputs.

---

## 15. Final scientific conclusions

F09 freezes the following campaign-level conclusions.

### 15.1 Best isolated imbalance intervention

F03 is retained as:

```text
BEST ISOLATED IMBALANCE INTERVENTION
WITHIN THE TESTED CAMPAIGN
```

This wording is bounded because the campaign lacks an original-distribution ConvNeXt-Small + standard cross-entropy control.

### 15.2 Aug-200

Aug-200 was a valid train-only data-level intervention, but the tested Aug-200 ConvNeXt configurations did not exceed F03 on balanced-class performance.

### 15.3 Combined Aug-200 + focal configuration

F05 did not show complementary gains over F03. This is a negative configuration result, not a universal claim about oversampling plus focal objectives.

### 15.4 High-resolution continuation

Continuing the selected F03 checkpoint under the F06 HR384 configuration produced the strongest balanced single-model result.

The improvement must not be attributed to image resolution alone.

### 15.5 TTA

The fixed two-view horizontal-flip F07 rule is retained as a scientifically useful negative result.

### 15.6 Ensemble

F08 is a positive ranking-oriented result. It achieved the best campaign Top-1, Top-2, Top-3, Top-5, and Weighted F1 point estimates among the final configurations, but it did not improve official-test Macro Recall or Macro F1 over F06.

### 15.7 Final method selection

F06 remains the primary single-model scientific result because the campaign's stated objective includes balanced-class recognition and F06 preserves the stronger Macro Recall and Macro F1 profile while avoiding the second-model inference requirement of F08.

---

## 16. Final claim-control table

| Topic | Approved wording | Do not write |
|---|---|---|
| F02 identity | F02 is YOLO26s-cls + Aug-200 | F02 is a ConvNeXt CE baseline |
| F03 | Strongest isolated imbalance intervention within the tested campaign | CB focal is proven universally superior to CE |
| F03 vs F04 | Broader configuration comparison | Strict loss-only ablation |
| F04 vs F05 | Closest loss-focused Aug-200 comparison, with unequal max epoch budgets | Perfectly controlled loss-only ablation |
| F03 vs F06 | HR384 continuation configuration improved the F03 branch | Resolution alone caused the gain |
| F06 vs F07 | Tested two-view flip TTA did not improve official-test balanced metrics | TTA never works |
| F06 vs F08 | F08 slightly higher Top-1; F06 higher Macro Recall/F1 | F08 universally outperformed F06 |
| 92.68 target | Internal reference target, not beaten | Official benchmark or beaten target |
| F09 | Analysis/comparison/final freeze; no training | F09 trained a new final model |
| XAI/external later work | Separate post-model-freeze evaluation stages | Part of F09 model selection |

---

## 17. Reviewer / viva question bank

### Q1. What exactly is F09?

F09 is the final analysis, comparison, method-selection, and campaign-freeze stage. It consolidates F01–F08; it is not a training experiment.

### Q2. Did F09 train or fine-tune any model?

No. Training and fine-tuning are both `NONE`.

### Q3. Did F09 produce a checkpoint?

No. There is no F09 model checkpoint.

### Q4. What is the primary proposed method after F09?

`F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL`.

### Q5. What is the complementary final ensemble?

`F08_CONVNEXT_SMALL_HR384_YOLO26M_PROB_ENSEMBLE`.

### Q6. Which model has the strongest balanced-class result?

F06, based especially on its official-test Macro Recall of 87.4362% and Macro F1 of 88.1699%.

### Q7. Which system has the strongest ranking result?

F08, with the highest campaign Top-1, Top-2, Top-3, Top-5, and Weighted F1 point estimates among the final configurations.

### Q8. Why is F08 not the primary method when its Top-1 is higher?

Its Top-1 advantage over F06 is only 0.1241 percentage points, while its Macro Recall is 0.4192 points lower and Macro F1 is 0.2293 points lower. The study prioritises balanced recognition as well as accuracy.

### Q9. How many additional test images did F08 classify correctly at Top-1 compared with F06?

Two additional images among 1,612.

### Q10. What is the best isolated imbalance intervention?

F03 — Gentle Class-Balanced Focal Loss — within the tested campaign.

### Q11. Why is that statement qualified with “within the tested campaign”?

Because there is no architecture-matched ConvNeXt-Small + standard cross-entropy control trained on the original 5,247-image distribution.

### Q12. Can F03 versus F04 be called a loss-only ablation?

No. They differ in training distribution as well as loss/objective and other pipeline details.

### Q13. Is F04 versus F05 a strict loss-only ablation?

No. It is the closest loss-focused comparison in the Aug-200 branch, but their maximum epoch budgets differ, 40 versus 25.

### Q14. Did Aug-200 and Gentle CB Focal combine successfully in F05?

Not in the tested configuration. F05 had lower Top-1, Macro Recall, Macro F1, and Weighted F1 than F03.

### Q15. Does F03 versus F06 prove that higher resolution alone improves performance?

No. F06 also adds continuation training with a different learning rate, batch size, and training budget.

### Q16. What did F07 show?

The fixed original-plus-horizontal-flip TTA improved validation Macro F1 but did not improve official-test Top-1 and reduced official-test Macro Recall and Macro F1.

### Q17. Does F07 prove that TTA is generally ineffective?

No. It evaluates one deterministic two-view horizontal-flip rule only.

### Q18. What ensemble rule does F08 use?

`0.70 F06 + 0.30 F01` probability averaging.

### Q19. Was the F08 ensemble weight selected on the official test?

No. It was selected on validation from a predefined grid and locked before official-test evaluation.

### Q20. What is the F06 source-checkpoint hash?

`d0c225c28b25f09822bc99518a37f9b1fe642b2d01213da7f68e2d76a255832a`.

### Q21. What is the F01 source-checkpoint hash?

`aa489def1443a6cdab42a0a20a530b9e341d574dcaaf234bfe6c7983113e8e5a`.

### Q22. What was the internal Top-1 target?

92.6800%.

### Q23. Was that target beaten?

No. The best campaign Top-1 was F08 at 92.2457%, leaving a gap of −0.4343 percentage points.

### Q24. Is 92.68% an official AQUA20 benchmark?

No. It is an internal campaign reference target.

### Q25. Which F01/F02 aggregate metrics are unavailable in the final F09 table?

Top-2 and Top-3 for both F01 and F02.

### Q26. Why are those values not estimated?

The documentation only reports directly supported frozen evidence. Missing values are preserved as N/A.

### Q27. What does the F09 validation-versus-test table show?

The strongest validation balanced-metric gains, especially for F07 and F08, did not fully generalise to the official test. This supports keeping model/rule selection validation-governed and not reopening selection after test inspection.

### Q28. How many final F09 CSV tables were packaged?

Eight.

### Q29. How many F09 publication-ready figures were packaged?

Three.

### Q30. Is the exact original F09 ZIP SHA-256 available?

No. It remains **To be verified**.

### Q31. Can exact bitwise reproduction of F09 be claimed?

No. The final scientific records are strong, but the original ZIP hash, exact ZIP bytes, complete immutable runtime environment, and several resource/efficiency fields were not preserved.

### Q32. What is F09's main scientific purpose?

To prevent post-hoc method switching and freeze an evidence-based reporting hierarchy after the completed experimental campaign.

---

## 18. Evidence ledger

Primary sources controlling this documentation:

- latest final manuscript, *Improving Underwater Marine Species Classification through Imbalance-Aware Learning and Cross-Dataset Evaluation*;
- `P00_AQUA20_Master_Evidence_Freeze_FINAL.xlsx` / final master evidence freeze;
- `P00_AQUA20_Evidence_Freeze_Report_FINAL.md`;
- `experiment_readme_index.md`;
- `methodology_control_audit.md`;
- `README_F09.txt`;
- `04-07-2026-aqua20-final-beat-base-plan.ipynb`;
- `F09_F01_F08_campaign_comparison.csv`;
- `F09_primary_metric_comparison.csv`;
- `F09_intervention_effect_comparison.csv`;
- `F09_validation_vs_test_generalization.csv`;
- `F09_final_method_selection.csv`;
- `F09_beat_base_target_comparison.csv`;
- `F09_source_checkpoint_freeze_record.csv`;
- `F09_scientific_conclusions.csv`;
- `F09_campaign_top1_macro_f1.png`;
- `F09_campaign_top5_weighted_f1.png`;
- `F09_intervention_effect_deltas.png`;
- completed F03, F06, F07, and F08 documentation;
- Phase 10-to-Phase 11 handover.

---

## 19. Remaining items marked To be verified

The following are intentionally not reconstructed:

1. original F09 ZIP SHA-256;
2. exact original F09 ZIP byte count;
3. extracted-tree SHA-256;
4. exact extracted-tree byte count;
5. complete per-file checksum manifest for the original F09 package;
6. immutable F09-specific container image/digest;
7. exact F09-specific driver/native-cuDNN/CPU/RAM manifest;
8. exact run-local scikit-learn version;
9. controlled F09 analysis runtime;
10. controlled latency/throughput;
11. peak GPU memory;
12. energy/compute cost;
13. standardised campaign parameter/FLOP/model-size comparison;
14. any missing F01/F02 Top-2 or Top-3 values;
15. any unsupported per-class narrative not directly recoverable from the preserved class-wise reports and confusion matrices.

---

## 20. Phase 11 completion decision

Phase 11 is complete.

The following are now documented and locked:

- exact F09 identity and analysis-only role;
- no-training/no-new-checkpoint boundary;
- corrected F01–F09 identity registry;
- complete supported F01–F08 official-test comparison;
- explicit preservation of unavailable F01/F02 Top-2 and Top-3 values;
- F06 primary proposed method and strongest balanced-class model;
- F08 complementary final ensemble and strongest ranking model;
- F03 best isolated imbalance intervention within the tested campaign;
- F06-versus-F08 metric-specific selection rationale;
- intervention effects and causal-claim boundaries;
- F07 negative TTA result;
- F08 positive-but-complementary ensemble result;
- F06/F01 source-checkpoint hashes;
- validation-versus-test generalisation evidence;
- internal 92.68% target as NOT BEATEN;
- F09 package manifest and verified table/figure counts;
- unsupported integrity/runtime values retained as **To be verified**;
- final reviewer/viva question bank.

The model-campaign documentation is now closed through F09.

The next documentation phase is a **cross-experiment internal error-analysis phase**, not another training experiment.
