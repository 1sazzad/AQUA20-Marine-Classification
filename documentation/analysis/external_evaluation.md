# AQUA20 — External Evaluation and Cross-Dataset Generalisation

**Documentation phase:** Phase 14  
**Repository target:** `documentation/analysis/external_evaluation.md`  
**Status:** Completed with explicitly retained verification gaps  
**External stage:** `E01_F06_EXTERNAL_SEA_ANIMALS`  
**Frozen source model:** `F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL`

---

## 1. Phase scope and roadmap control

The Phase 13 handover recommended external evaluation as the next documentation stream but required the latest Documentation Master Plan / phase index to control the exact Phase 14 title if such a numbered row could be surfaced.

During Phase 14 retrieval, no controlling master-plan file containing an explicit numbered **Phase 14** row was surfaced. The project foundation, experiment README index, FINAL evidence freeze, final manuscript, preserved E01 protocol, and prior campaign roadmap nevertheless converge on E01 external evaluation as the remaining major post-internal-analysis stream.

Accordingly, this phase proceeds under the working title:

> **Phase 14 — External Evaluation and Cross-Dataset Generalisation Documentation**

This is a documentation and evidence-consolidation phase. It does not create a new experiment identity.

---

## 2. Scientific role

E01 tests how the already-selected F06 model behaves when applied to a separately sourced marine-image collection under a frozen source-only protocol.

The phase is divided into three distinct analyses:

| ID | Role | Scientific interpretation |
|---|---|---|
| **E01A** | Raw 20-way external evaluation | **Primary external result** on the 15 strictly shared semantic classes while retaining F06's native 20-class output space. |
| **E01B** | Restricted 15-way external evaluation | Secondary taxonomy-alignment diagnostic after removing the five AQUA20-only outputs and renormalising the remaining 15 probabilities. |
| **E01C** | Known–unknown confidence / robustness analysis | Diagnostic comparison of shared versus external-only images using confidence, entropy, and prediction destinations. It is not 23-class classification and is not a validated open-set detector. |

The reporting hierarchy is fixed:

```text
E01A = primary external result
E01B = secondary restricted-taxonomy diagnostic
E01C = known–unknown robustness diagnostic
```

E01B must not replace E01A because E01B assumes prior knowledge of which output classes are shared with the external taxonomy.

---

## 3. Frozen-model boundary

The external evaluation used the unchanged primary F06 checkpoint.

| Field | Frozen value |
|---|---|
| Model identity | `F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL` |
| Checkpoint file | `best_model.pth` |
| Best epoch | 2 |
| Checkpoint SHA-256 | `d0c225c28b25f09822bc99518a37f9b1fe642b2d01213da7f68e2d76a255832a` |
| External training | **NONE** |
| External fine-tuning | **NONE** |
| External adaptation | **NONE** |
| External-data tuning | **NONE** |
| External model selection | **NONE** |
| New checkpoint | **NONE** |

The external labels were not used to modify F06. The external stage therefore measures the behaviour of a preserved source-trained model rather than an adapted target-domain model.

---

## 4. External dataset registry

The preserved E01 records identify the external source as:

```text
Dataset: Sea Animals Image Dataset
Source identifier: vencerlanz09/sea-animals-image-dataste
Folders/classes inspected: 23
Images: 13,711
Decode errors during frozen inference: 0
```

The frozen taxonomy partition is:

```text
Strict shared classes: 15
Strict shared images: 9,738
External-only classes: 8
External-only images: 3,973
Total: 23 classes / 13,711 images
```

Count closure:

```text
9,738 + 3,973 = 13,711 images
15 + 8 = 23 folders/classes
```

### 4.1 Licensing / provenance boundary

The preserved dataset audit records the Kaggle data card licence as **“Other (specified in description)”** and notes mixed image provenance. Per-image rights/provenance metadata were not archived in the experiment package.

Therefore the documentation must not describe the entire external collection as uniformly CC0, public-domain, unrestricted, or uniformly permissively licensed.

Approved disclosure:

> The external dataset data card lists the licence as “Other (specified in description)” and describes mixed image provenance. Because per-image rights metadata were not archived in the experimental package, no uniform permissive licence is claimed for the complete collection; reuse should follow the original dataset and source-image terms.

---

## 5. Frozen strict semantic mapping

The mapping was fixed before external metric computation. A folder was treated as shared only when there was a direct class-name match or a defensible semantic/taxonomic synonym. Broad superclass collapse was deliberately avoided.

| External folder | Images | AQUA20 target | Role |
|---|---:|---|---|
| Clams | 497 | — | External-only |
| Corals | 500 | `00_coral` | Shared |
| Crabs | 499 | `01_crab` | Shared |
| Dolphin | 782 | `08_marine_dolphin` | Shared |
| Eel | 497 | `03_eel` | Shared |
| Fish | 494 | `04_fish` | Shared |
| Jelly Fish | 845 | `07_jellyfish` | Shared |
| Lobster | 499 | — | External-only |
| Nudibranchs | 500 | `13_seaSlug` | Shared |
| Octopus | 562 | `09_octopus` | Shared |
| Otter | 500 | — | External-only |
| Penguin | 482 | — | External-only |
| Puffers | 531 | — | External-only |
| Sea Rays | 517 | `10_rayfish` | Shared |
| Sea Urchins | 579 | `14_seaUrchin` | Shared |
| Seahorse | 478 | — | External-only |
| Seal | 414 | — | External-only |
| Sharks | 590 | `15_shark` | Shared |
| Shrimp | 488 | `16_shrimp` | Shared |
| Squid | 483 | `17_squid` | Shared |
| Starfish | 499 | `18_starfish` | Shared |
| Turtle_Tortoise | 1,903 | `19_turtle` | Shared |
| Whale | 572 | — | External-only |

The eight external-only classes are therefore:

```text
Clams
Lobster
Otter
Penguin
Puffers
Seahorse
Seal
Whale
```

Important exclusions retained by the frozen mapping rule:

- Puffers were not collapsed into generic `fish`.
- Seahorse was not collapsed into generic `fish`.
- Lobster was not collapsed into `crab` or `shrimp`.
- Whale was not collapsed into `marine_dolphin`.

### 5.1 AQUA20-only output classes

Five F06 output classes have no shared external ground-truth class:

```text
02_diver
05_fishInGroups
06_flatworm
11_seaAnemone
12_seaCucumber
```

These outputs remain active in E01A and can therefore absorb probability or become Top-1 predictions. They are removed only for E01B before the remaining 15 probabilities are renormalised.

---

## 6. Frozen inference protocol

The E01 GPU evidence records the following inference path:

```text
native external image
→ decode / convert to RGB
→ 384 × 384 model input
→ ImageNet mean/std normalisation
→ frozen F06
→ native 20-class softmax probabilities
```

Verified inference record:

| Field | Value |
|---|---:|
| Images inferred | 13,711 |
| Decode errors | 0 |
| Saved probability matrix | 13,711 × 20 |
| Probability columns | 20 |
| Probability row sums | Verified approximately 1.0 |
| Additional GPU inference required | No |

No new inference was required for E01A, E01B, or E01C after the full 20-way probability matrix had been frozen. Subsequent analyses operate on the preserved probabilities.

Do not add crop mode, interpolation mode, or other preprocessing details that are not directly preserved by the archived transform/protocol evidence.

---

## 7. Preserved prediction evidence and integrity status

The original GPU evidence package was described as containing six files:

```text
README_E01_GPU_EVIDENCE.txt
tables/E01_all23_external_f06_predictions.csv
tables/E01_external_inference_protocol.csv
tables/E01_external_class_taxonomy.csv
tables/E01_strict_shared_class_mapping.csv
tables/E01_external_inference_summary.csv
```

An older preserved session record reports:

```text
GPU evidence ZIP SHA-256:
5c3ad8d1392a252272e1a94924f7e73c4ec2f03368d046aa909d232bde187708

Prediction CSV SHA-256:
1ab3af604f4ddb56e2f6c99940ffba16ddd4dd7b84ffa73950086046a0aa927b
```

These checksum strings are retained as **preserved old-session provenance values**. The current Phase 14 retrieval did not surface the original archive bytes or a fresh direct checksum command, so they are not upgraded here to newly re-verified package hashes.

The FINAL P00 evidence inventory also records that the definitive consolidated E01 package containing E01A/E01B/E01C and final-reporting files had not yet been collected into the master evidence area.

Therefore:

```text
Definitive final E01 package SHA-256 = To be verified
Definitive final E01 package bytes   = To be verified
Definitive extracted-tree hash       = To be verified
Complete final E01 per-file manifest = To be verified
```

---

## 8. E01A — primary raw 20-way external result

### 8.1 Definition

E01A evaluates only the **9,738 images belonging to the 15 strict shared classes**, but the frozen F06 prediction remains its original **20-way** decision.

There is:

```text
Class restriction: NONE
Probability renormalisation: NONE
Model modification: NONE
```

A prediction to any wrong class counts as an error, including predictions to one of the five AQUA20-only output classes.

This is the principal cross-dataset generalisation result because it most faithfully preserves the model exactly as selected on AQUA20.

### 8.2 Independently recomputed aggregate metrics

The FINAL evidence-freeze audit independently recomputed E01A from the preserved probability/prediction evidence and marked the result PASS.

| Metric | E01A |
|---|---:|
| Top-1 | **76.5044%** |
| Top-2 | **87.1534%** |
| Top-3 | **91.8464%** |
| Top-5 | **95.7486%** |
| Macro Precision | **81.3575%** |
| Macro Recall | **74.1616%** |
| Macro F1 | **75.2774%** |
| Weighted F1 | **78.3601%** |
| Top-1 prediction rate into AQUA20-only outputs | **4.7546%** |

The 4.7546% outside-shared Top-1 rate is not an unknown-detection accuracy. It simply records how often shared-class external images were assigned to one of the five AQUA20-only outputs under the raw 20-way classifier.

---

## 9. E01A class-wise evidence

The recomputed external audit preserves direct E01A per-class precision, recall, F1, and support for all 15 shared classes.

| AQUA20 class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `00_coral` | 500 | 57.92% | 70.20% | 63.47% |
| `01_crab` | 499 | 94.03% | 97.80% | 95.87% |
| `03_eel` | 497 | 87.04% | 75.65% | 80.95% |
| `04_fish` | 494 | 35.75% | 78.74% | 49.18% |
| `07_jellyfish` | 845 | 97.34% | 95.27% | 96.29% |
| `08_marine_dolphin` | 782 | 93.86% | 46.93% | 62.57% |
| `09_octopus` | 562 | 66.76% | 42.88% | 52.22% |
| `10_rayfish` | 517 | 88.52% | 77.56% | 82.68% |
| `13_seaSlug` | 500 | 79.52% | 79.20% | 79.36% |
| `14_seaUrchin` | 579 | 97.90% | 96.55% | 97.22% |
| `15_shark` | 590 | 61.18% | 78.81% | 68.89% |
| `16_shrimp` | 488 | 82.41% | 48.98% | 61.44% |
| `17_squid` | 483 | 89.01% | 35.20% | 50.45% |
| `18_starfish` | 499 | 91.30% | 98.80% | 94.90% |
| `19_turtle` | 1,903 | 97.83% | 89.86% | 93.67% |

### 9.1 Directly supported class-wise interpretation

Under raw E01A scoring, the strongest F1 values occur for:

- sea urchin;
- jellyfish;
- crab;
- starfish;
- turtle.

The weakest F1 values occur for:

- fish;
- squid;
- octopus;
- shrimp;
- marine dolphin.

This class-wise pattern is external-source behaviour. It must not be silently substituted for the internal AQUA20 class-wise pattern.

The low fish precision despite substantially higher fish recall is particularly important: many external images from other categories are being absorbed into the broad `04_fish` output. This is a prediction-pattern observation, not proof of the visual cause of those errors.

---

## 10. E01B — restricted 15-way diagnostic

### 10.1 Definition

E01B uses the **same 9,738 shared images** and the **same frozen 13,711 × 20 source probability evidence**.

For each shared image:

1. retain the 15 probabilities corresponding to the frozen shared AQUA20 classes;
2. remove probabilities for the five AQUA20-only outputs;
3. renormalise the remaining 15 probabilities to sum to one;
4. rank within the 15-class restricted space.

No model weights are changed.

Because E01B is given prior knowledge of the target taxonomy, it is a diagnostic of label-space mismatch rather than the primary deployment-style external result.

### 10.2 Independently recomputed metrics

| Metric | E01B |
|---|---:|
| Top-1 | **78.6815%** |
| Top-2 | **89.1456%** |
| Top-3 | **93.5716%** |
| Top-5 | **96.7652%** |
| Macro Precision | **80.4852%** |
| Macro Recall | **76.7121%** |
| Macro F1 | **76.0247%** |
| Weighted F1 | **79.0111%** |

### 10.3 E01B minus E01A

| Metric | Change |
|---|---:|
| Top-1 | **+2.1770 pp** |
| Top-5 | **+1.0166 pp** |
| Macro Precision | **−0.8722 pp** |
| Macro Recall | **+2.5505 pp** |
| Macro F1 | **+0.7473 pp** |
| Weighted F1 | **+0.6510 pp** |

The restricted taxonomy therefore improves several ranking/recall metrics, but **does not improve every metric**. Macro Precision decreases.

Approved interpretation:

> Removing the five non-shared AQUA20 outputs recovers a modest amount of performance, showing that output-space mismatch contributes to the external error burden. The remaining large gap after restriction shows that taxonomy mismatch alone does not explain the cross-dataset degradation.

This is an evidence-based interpretation of the observed point estimates; it is not a causal decomposition of all domain-shift factors.

---

## 11. E01C — known–unknown robustness diagnostic

### 11.1 Evaluation universe

E01C uses **all 13,711 external images**:

```text
Known/shared:      9,738 images / 15 classes
External-only:     3,973 images / 8 classes
Model outputs:     20 AQUA20 classes
```

The eight external-only classes do not have output neurons in F06. Therefore ordinary 23-class classification accuracy is undefined for this frozen classifier and must not be reported.

### 11.2 Locked score definitions

The preserved E01C protocol evaluates unknown-positive scores using:

```text
1 - maximum raw 20-way probability
normalized predictive entropy = entropy / log(20)
1 - maximum probability among the 15 shared outputs
outside-shared probability mass = sum over the five AQUA20-only outputs
```

The primary AUROC/AUPR calculations used no tuned decision threshold. FPR at 95% unknown TPR is a retrospective diagnostic only.

### 11.3 Known versus external-only summary

| Diagnostic | Known/shared | External-only |
|---|---:|---:|
| Images | 9,738 | 3,973 |
| Dataset share | 71.02% | 28.98% |
| Mean Top-1 probability | **0.8467** | **0.6418** |
| Median Top-1 probability | **0.9554** | **0.6866** |
| Mean predictive entropy | **0.5514** | **1.2062** |
| Median predictive entropy | **0.2723** | **1.1137** |
| Mean normalised entropy | **0.1841** | **0.4026** |
| Mean maximum shared probability | **0.8173** | **0.6073** |
| Mean outside-shared probability mass | **0.0608** | **0.1046** |
| Top-1 outside-shared rate | **0.0475** | **0.0639** |

External-only images are, on average, less confident and more entropic than shared images. However, score overlap remains substantial.

### 11.4 Frozen unknown-separation metrics

| Unknown score | AUROC | AUPR (unknown positive) | FPR @ 95% unknown TPR |
|---|---:|---:|---:|
| `1 - max probability` | 0.7151 | 0.5570 | 0.8276 |
| **Normalised predictive entropy** | **0.7197** | **0.5687** | **0.8300** |
| `1 - max shared probability` | 0.7066 | 0.4715 | 0.8257 |
| Outside-shared probability mass | 0.6955 | 0.4307 | 0.8258 |

Normalised predictive entropy has the strongest frozen AUROC/AUPR among the evaluated scores, but the associated FPR at 95% unknown TPR is approximately **83%**.

Approved conclusion:

> The scores provide only limited known–unknown separation. They do not establish reliable unknown rejection or a successful open-set recognition system.

Do not select a threshold retrospectively and describe it as a pre-specified deployment operating point.

---

## 12. External-only prediction destinations

Because F06 is a closed 20-class AQUA20 classifier, external-only organisms are forced into available outputs.

The dominant destination for each frozen external-only folder is:

| External-only class | Images | Dominant F06 destination | Share |
|---|---:|---|---:|
| Clams | 497 | `00_coral` | **42.66%** |
| Lobster | 499 | `16_shrimp` | **74.35%** |
| Otter | 500 | `19_turtle` | **26.20%** |
| Penguin | 482 | `04_fish` | **56.85%** |
| Puffers | 531 | `04_fish` | **94.54%** |
| Seahorse | 478 | `09_octopus` | **41.84%** |
| Seal | 414 | `08_marine_dolphin` | **39.37%** |
| Whale | 572 | `08_marine_dolphin` | **46.33%** |

For Otter, `19_turtle` at 26.20% is only slightly above `08_marine_dolphin` at 26.00%, so it should not be overinterpreted as a strong dominant assignment.

A concise manuscript description may state that external-only categories were commonly absorbed into available labels including fish, coral, shrimp, marine dolphin, octopus, and turtle.

Do not use `seaAnemone` as a representative dominant destination: the frozen class-level summaries do not support that description.

---

## 13. Internal versus external point-estimate comparison

The frozen F06 internal AQUA20 official-test point estimates are:

```text
Top-1    = 92.1216%
Macro F1 = 88.1699%
```

Against E01A:

```text
Top-1 difference    = 76.5044 - 92.1216 = -15.6172 pp
Macro-F1 difference = 75.2774 - 88.1699 = -12.8925 pp
```

Against E01B:

```text
Top-1 difference    = 78.6815 - 92.1216 = -13.4401 pp
Macro-F1 difference = 76.0247 - 88.1699 = -12.1452 pp
```

These are descriptive point-estimate differences, **not paired statistical comparisons**. The internal AQUA20 test and external shared-class set differ in source, class support, class composition, and evaluation context.

Approved interpretation:

> F06 retains substantial recognition ability on the external shared classes but exhibits a marked degradation relative to same-source AQUA20 testing. The restricted E01B diagnostic recovers only a small portion of that gap, indicating that the observed external reduction is not explained solely by the five non-shared output classes.

Do not state that the exact 15.62 pp / 12.89 pp reduction is caused by a single factor such as lighting, camera type, taxonomy, or background unless a controlled analysis directly establishes that factor.

---

## 14. Manuscript-ready external result table

| Evaluation | Images | Output rule | Top-1 | Top-5 | Macro P | Macro R | Macro F1 | Weighted F1 | Role |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| **E01A** | 9,738 | Raw F06 20-way | **76.50%** | **95.75%** | **81.36%** | **74.16%** | **75.28%** | **78.36%** | **Primary external result** |
| **E01B** | 9,738 | Restricted / renormalised 15-way | **78.68%** | **96.77%** | **80.49%** | **76.71%** | **76.02%** | **79.01%** | Secondary taxonomy diagnostic |

Recommended result wording:

> The frozen F06 checkpoint was evaluated on 9,738 images from 15 strictly shared external classes without retraining or adaptation. Under the primary raw 20-way protocol, F06 achieved 76.50% Top-1 accuracy, 95.75% Top-5 accuracy, and 75.28% Macro F1. Restricting and renormalising the outputs to the 15 shared classes increased Top-1 accuracy to 78.68% and Macro F1 to 76.02%, while Macro Precision decreased from 81.36% to 80.49%. The restricted result is therefore reported as a taxonomy diagnostic rather than as a replacement for the raw external evaluation.

Recommended E01C wording:

> External-only images showed lower mean Top-1 confidence (0.642 versus 0.847) and higher normalised predictive entropy (0.403 versus 0.184) than shared-class images. Nevertheless, the strongest frozen unknown-separation score achieved only 0.720 AUROC and approximately 0.830 false-positive rate at 95% unknown recall. The analysis therefore indicates limited score separation and closed-set behaviour rather than reliable unknown rejection.

---

## 15. Claim boundaries

### 15.1 Approved claims

- F06 was frozen before external evaluation.
- No external training, fine-tuning, adaptation, external tuning, or external model selection was performed.
- One frozen inference pass produced 20-class probability vectors for all 13,711 images with no decode errors.
- The inspected external registry contains 23 folders/classes and 13,711 images.
- Fifteen strict shared classes contain 9,738 images.
- Eight external-only classes contain 3,973 images.
- E01A is the primary raw 20-way external result.
- E01B is a restricted-15 taxonomy diagnostic.
- E01C is a known–unknown score-separation diagnostic.
- External performance is materially below the same-source F06 AQUA20 point estimates.
- Restricting the output space recovers only a modest amount of performance and lowers Macro Precision.
- External-only samples tend to have lower confidence and higher entropy on average, but score separation is limited.

### 15.2 Prohibited / unsupported claims

Do not claim that:

- E01 is a 23-class classification experiment;
- F06 can correctly classify the eight external-only classes;
- E01C demonstrates successful or reliable open-set detection;
- E01B is the primary external result;
- restricting to 15 classes improves every metric;
- the external data were used for training, checkpoint selection, tuning, or adaptation;
- the external dataset has the same taxonomy as AQUA20;
- the full external collection has a uniform permissive licence;
- a specific visual/domain factor caused the external performance drop without a controlled study;
- external generalisation has been solved;
- the external source is guaranteed duplicate-free relative to AQUA20 without a preserved cross-dataset duplicate audit;
- the external reduction is a paired statistical effect comparable to the Phase 13 internal paired tests;
- the reported E01C FPR95 threshold was a pre-deployment threshold;
- the historical GPU inference time is a controlled cross-model efficiency benchmark.

---

## 16. Reproducibility and evidence boundaries

### 16.1 Directly frozen / independently verified

- F06 identity, best epoch, and checkpoint SHA-256;
- 23-folder / 13,711-image external registry;
- 15 shared classes / 9,738 images;
- 8 external-only classes / 3,973 images;
- strict semantic mapping;
- RGB, 384×384, ImageNet-normalised evaluation path;
- full 13,711×20 probability evidence and zero decode errors;
- E01A aggregate metrics;
- E01B aggregate metrics;
- E01A/E01B independent recomputation status;
- E01A per-class report rows in the FINAL audit;
- E01C locked score definitions, summary statistics, unknown-score metrics, and prediction destinations.

### 16.2 Preserved but not freshly re-verified in Phase 14

- old-session GPU evidence ZIP checksum;
- old-session prediction-CSV checksum;
- historical notebook-reported external inference duration.

### 16.3 To be verified

1. definitive consolidated `P00_E01_FINAL_EVIDENCE.zip` or equivalent final E01 package;
2. definitive final package SHA-256, exact bytes, extracted-tree hash, and complete per-file checksum manifest;
3. direct inventory/hashes of all final E01A/E01B confusion-matrix files, final-reporting figures, and final README;
4. run-specific E01 software/container/driver/cuDNN/CPU/RAM manifest;
5. controlled E01 latency/throughput and peak-memory benchmark;
6. immutable external-dataset revision/content hash;
7. preserved per-image licence/provenance metadata;
8. direct cross-dataset exact/near-duplicate audit between AQUA20 and the external collection;
9. any preprocessing detail beyond the archived RGB → 384×384 → ImageNet-normalisation evidence, including exact interpolation/crop semantics if not recoverable from source;
10. complete original-source provenance for every external image.

These gaps do not invalidate the frozen point estimates, but they limit claims about exact package reproducibility, dataset independence, licensing uniformity, and deployment efficiency.

---

## 17. Reviewer / viva question bank

### Q1. Why was F06 used for external evaluation instead of F08?
F06 is the frozen primary proposed single model and strongest balanced-class model. F08 is a complementary ensemble. Using F06 preserves the study's primary model hierarchy and avoids changing the external target after test inspection.

### Q2. Was F06 fine-tuned on the Sea Animals Image Dataset?
No. Training, fine-tuning, adaptation, external tuning, and external model selection were all absent.

### Q3. Why are only 15 external classes used for E01A scoring?
Only 15 external folders had strict, pre-frozen semantic counterparts in AQUA20. Broader superclass collapse was deliberately avoided.

### Q4. Why not map Puffers or Seahorse to `fish`?
That would broaden the mapping after the fact and make the external task easier by collapsing distinct external categories into a generic AQUA20 superclass. The strict mapping avoids that manipulation.

### Q5. Why is E01A the primary result if E01B has higher Top-1?
E01A preserves the native frozen 20-way classifier. E01B uses prior knowledge of the shared taxonomy and removes five outputs before renormalisation, so it is a diagnostic rather than the primary deployment-style evaluation.

### Q6. Did E01B improve every metric?
No. It improved Top-1, Top-5, Macro Recall, Macro F1, and Weighted F1, but Macro Precision decreased by 0.8722 percentage points.

### Q7. What does the 4.7546% outside-shared rate mean?
Among E01A shared-class images, 4.7546% received a Top-1 prediction in one of the five AQUA20 outputs that have no shared external ground-truth class. It is not an unknown-detection metric.

### Q8. Why is E01C not 23-class accuracy?
F06 has only 20 AQUA20 output neurons and no outputs for the eight external-only categories, so ordinary 23-class ground-truth classification is not defined.

### Q9. Does AUROC 0.7197 mean F06 is a good open-set detector?
No. The accompanying FPR at 95% unknown TPR is about 0.83, indicating substantial overlap and poor rejection at high unknown recall. The analysis supports only limited score separation.

### Q10. Which unknown classes were strongly absorbed by existing labels?
Examples include Puffers→fish (94.54%), Lobster→shrimp (74.35%), Penguin→fish (56.85%), Whale→marine dolphin (46.33%), and Clams→coral (42.66%).

### Q11. What is the strongest evidence of source shift?
The primary E01A Top-1 and Macro F1 are 76.50% and 75.28%, compared with F06's same-source AQUA20 values of 92.12% and 88.17%. These are descriptive differences across different evaluation sets, not paired causal estimates.

### Q12. Can the external drop be attributed to image quality or lighting?
Not from the preserved evaluation alone. Source, acquisition conditions, class composition, backgrounds, taxonomy, and other factors differ simultaneously. Causal attribution requires controlled evidence.

### Q13. Does E01 prove the external dataset is independent of AQUA20 at the image level?
No. It is separately sourced, but a preserved cross-dataset exact/near-duplicate audit was not surfaced in Phase 14. Stronger “independent with no overlap” language should therefore be avoided unless that audit is later recovered.

### Q14. What is the licensing limitation?
The data card uses a mixed-provenance “Other (specified in description)” licence label, and per-image rights metadata were not archived. The project should cite the source and avoid claiming a single permissive licence for all images.

### Q15. Why are external-only prediction destinations useful?
They show how a closed-set classifier absorbs unseen semantic categories into available labels, helping diagnose failure behaviour without pretending that those assignments are correct classifications.

### Q16. Was the external test used to reselect F06 or change the final method hierarchy?
No. F06 remained the primary model and no external result reopened the already frozen F09 selection decision.

### Q17. Why does strong Top-5 persist when Top-1 drops?
The correct shared class often remains within the model's ranked alternatives even when the first-ranked label is wrong. This indicates useful ranking information but does not erase the Top-1 or Macro-F1 generalisation gap.

### Q18. What is still missing for full reproducibility closure?
The definitive consolidated E01 package and its integrity manifest, run-specific environment/hardware evidence, immutable external-dataset revision record, and direct cross-dataset duplicate audit remain to be verified.

---

## 18. Phase 14 completion decision

Phase 14 is complete as a documentation phase.

The following are now documented and locked:

- roadmap-controlled Phase 14 scope decision;
- exact E01 stage and frozen F06 source identity;
- source-only / no-adaptation boundary;
- F06 checkpoint epoch and SHA-256;
- 23-folder / 13,711-image external registry;
- 15 strict shared classes / 9,738 shared images;
- 8 external-only classes / 3,973 images;
- complete frozen 23-folder semantic mapping;
- five AQUA20-only output classes relevant to E01A/E01B;
- RGB, 384×384, ImageNet-normalised inference protocol;
- one-pass 13,711×20 probability freeze and zero decode errors;
- E01A definition and independently recomputed aggregate metrics;
- E01A direct class-wise report rows;
- E01B definition, independently recomputed metrics, and E01B−E01A deltas;
- E01C locked score definitions and known–unknown summary;
- all four preserved unknown-score AUROC/AUPR/FPR95 rows;
- direct external-only prediction-destination evidence;
- descriptive internal-versus-external point-estimate differences;
- manuscript-ready reporting tables and wording;
- licensing, taxonomy, open-set, source-shift, and causal-interpretation boundaries;
- unresolved package/reproducibility fields retained explicitly as **To be verified**;
- Phase 14 reviewer/viva question bank.

The final E01 consolidated evidence package itself is **not** claimed to be fully integrity-closed because it was not surfaced in the current evidence retrieval.

---

## 19. Handover

The next documentation phase must first search for the latest Documentation Master Plan / phase index and confirm the exact Phase 15 objective.

No Phase 15 title is forced here.

If the controlling roadmap confirms explainability next, the recommended stream is consolidation of the frozen F06 Grad-CAM/LIME evidence, with X03 included only if direct package/output evidence exists. If the roadmap instead assigns final-table consolidation or another objective, the controlling plan must replace that recommendation.
