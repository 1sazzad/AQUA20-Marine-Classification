D01 — AQUA20 Project Foundation
Document status
Phase: 1 — Project Foundation
Status: Completed and locked
Canonical manuscript: Improving Underwater Marine Species Classification through Imbalance-Aware Learning and Cross-Dataset Evaluation
1. Documentation source-control rule
This documentation is a reconstruction of the completed AQUA20 research project. It must not create a new project story from memory or from short user answers.
The following evidence hierarchy controls all statements:
Latest final manuscript — official title, final research framing, final experiment table, reported results, limitations, and conclusion.
FINAL evidence-freeze and frozen-terminology records — exact experiment identities, terminology, selection rules, claim restrictions, and verified numerical records.
Preserved experiment packages — notebooks, CSV files, prediction files, checkpoints, hashes, training histories, scripts, and figures.
Older reports and earlier drafts — historical context only. They must not override later frozen evidence.
A known example of why this hierarchy is necessary is F02. The final manuscript and final evidence freeze identify F02 as YOLO26s-cls trained on Aug-200. Any older document that describes F02 as a ConvNeXt cross-entropy baseline is superseded.
Unsupported details must be written as:
To be verified
They must not be reconstructed from memory.

2. Official project identity
Final title
Improving Underwater Marine Species Classification through Imbalance-Aware Learning and Cross-Dataset Evaluation
Research area
Marine image classification
Underwater species recognition
Class-imbalanced learning
Cross-dataset evaluation
Explainable artificial intelligence
Dataset
The internal benchmark is the 20-class AQUA20 dataset, containing 8,171 underwater images.

3. Project origin and evolution
The project began from the practical difficulty of obtaining reliable multiclass classification on an imbalanced underwater dataset. AQUA20 contains unequal class frequencies, and underwater images also exhibit poor visibility, colour distortion, occlusion, complex backgrounds, and strong visual similarity between categories.
The project was not ultimately framed as a simple attempt to maximise one accuracy value. It evolved from a collection of model trials into a controlled, hypothesis-driven experimental campaign built around:
one fixed train–validation–test protocol;
training-only imbalance interventions;
validation-governed selection;
balanced-class evaluation;
cautious interpretation of model explanations; and
source-only external evaluation.
The final scientific position is therefore broader than “improve classification accuracy.” The study investigates whether strong overall performance can be achieved without hiding weak minority-class behaviour, and whether the selected model remains useful after a change of image source and taxonomy.

4. Background and practical importance
Automated recognition of underwater marine species can support:
biodiversity monitoring;
ecological observation;
underwater survey analysis;
species-oriented image search;
environmental research; and
large-scale marine-image organisation.
A model used for these purposes should not be judged only by its performance on common classes. If under-represented classes are repeatedly missed, a high overall accuracy can give an incomplete or misleading picture of reliability.
This motivates the use of macro-averaged and class-wise metrics alongside Top-k accuracy.

5. Core research problem
Marine species classification in AQUA20 is difficult for several connected reasons:
Class imbalance
Frequent classes contribute more examples during training and can dominate aggregate evaluation.
Underwater image degradation
Images may contain turbidity, colour distortion, poor illumination, suspended material, blur, and occlusion.
Inter-class visual similarity
Different species categories may share shape, texture, colour, habitat, or background characteristics.
Intra-class variation
The same category may appear under different poses, scales, viewpoints, environments, and lighting conditions.
Headline-metric risk
Strong overall accuracy can conceal weak recognition of minority or visually difficult classes.
Source-shift risk
Performance on a held-out split from the same source does not guarantee comparable performance on independently collected images.
Interpretability limits
Grad-CAM and LIME can show influential image regions, but they do not prove causal, biological, or human-like reasoning.
Formal research problem statement
How can an imbalance-aware deep visual classification pipeline achieve strong overall recognition while preserving balanced class-wise performance on AQUA20, under a strict unchanged evaluation protocol, and how does the selected frozen model behave on semantically matched images from an independent external dataset?

6. Final research goal
The final goal of the project is:
To evaluate and improve imbalance-aware underwater marine species classification on AQUA20 using a fixed train–validation–test protocol, balanced evaluation metrics, controlled training and inference interventions, and source-only cross-dataset evaluation.
This goal has four connected parts:
improve internal classification performance;
preserve minority-class and class-balanced recognition;
prevent test-driven model selection and evaluation leakage; and
assess whether internal gains survive an external source shift.

7. Research gap
The project does not claim that class imbalance, focal loss, ConvNeXt, Grad-CAM, or LIME are new.
The defensible gap is:
Although imbalance-aware methods have been studied in fish and marine recognition, their behaviour on AQUA20 had not been sufficiently characterised through one consistent campaign combining data-level balancing, loss-level imbalance handling, high-resolution continuation, inference refinements, balanced-class reporting, qualitative explanation analysis, and frozen source-only external evaluation.
The study therefore focuses on controlled evidence and evaluation discipline, not on claiming the invention of a new backbone or a new loss family.

8. Main research objectives
Objective 1 — Establish a fixed evaluation protocol
Use one stratified AQUA20 split and retain the same split and class order throughout the final campaign.
Objective 2 — Evaluate data-level balancing
Test train-only Aug-200 balancing while leaving validation and test distributions unchanged.
Objective 3 — Evaluate loss-level imbalance handling
Assess Gentle Class-Balanced Focal Loss using effective-number class weighting and a moderate focal exponent.
Objective 4 — Compare evaluated architectures and configurations
Compare YOLO26 classification models and ConvNeXt-Small configurations under the final campaign.
Objective 5 — Evaluate high-resolution continuation
Continue the validation-selected F03 checkpoint at (384\times384) and assess the resulting complete configuration.
Objective 6 — Evaluate inference refinements
Test deterministic two-view horizontal-flip TTA and a validation-selected heterogeneous probability ensemble.
Objective 7 — Report balanced and class-wise behaviour
Use Top-k, Macro Precision, Macro Recall, Macro F1, Weighted F1, per-class scores, and confusion matrices.
Objective 8 — Evaluate external transfer
Run the frozen primary model on the Sea Animals Image Dataset without training, fine-tuning, adaptation, or external model selection.
Objective 9 — Inspect prediction behaviour
Use Grad-CAM and LIME on selected AQUA20 test examples as qualitative diagnostic tools.

9. Research questions
RQ1
How effectively do the evaluated YOLO26 and ConvNeXt-Small configurations classify the 20 AQUA20 categories under the fixed protocol?
RQ2
Does train-only Aug-200 balancing provide sufficient improvement in class-balanced recognition?
RQ3
How does Gentle Class-Balanced Focal Loss behave on the original and Aug-200 training distributions?
RQ4
Does continuing the selected F03 checkpoint at (384\times384) produce a stronger complete configuration?
RQ5
Do deterministic two-view TTA and heterogeneous probability ensembling improve ranking or balanced-class performance?
RQ6
Which classes remain weak, and what error patterns appear in the class-wise results?
RQ7
How much performance is retained when the frozen primary model is evaluated on independently sourced shared-class images?
RQ8
What do Grad-CAM and LIME reveal about correct predictions, difficult predictions, and representative errors?

10. Working hypotheses
The final campaign was guided by the following working hypotheses:
Loss-level imbalance handling may improve Macro F1 and minority-class sensitivity.
Train-only Aug-200 balancing may increase minority-class exposure, but it may not automatically produce the strongest Macro F1.
Combining data-level and loss-level interventions may not yield additive gains.
Higher-resolution continuation may help recognition of morphology and local texture, although F06 cannot isolate resolution from additional continuation training.
TTA and ensembling may improve probability ranking without necessarily improving balanced-class metrics.
External evaluation will reveal a performance gap hidden by same-source testing.
Explanation maps will be useful for diagnosis but will not establish causal model reasoning.
These are experiment-guiding hypotheses, not assumptions that were automatically confirmed.

11. Fixed methodological principles
11.1 Dataset split
A stratified split with seed 42 produced:
Partition
Images
Official training split
5,247
Validation split
1,312
Official test split
1,612
Total
8,171

The same split and class order were retained throughout the study.
11.2 Train-only balancing
Aug-200 raised each class below 200 training samples to 200, producing 7,284 training images. Validation and test data were unchanged.
The exact transformation recipe used to create every Aug-200 image was not fully preserved and must not be retrospectively invented.
11.3 Selection protocol
F01 and F02 used the native Ultralytics classification fitness for checkpoint selection.
F03–F06 used validation Macro F1.
F08 ensemble weights were selected using validation Macro F1, with locked tie-break rules.
F07 created no new trained checkpoint.
The official test split was evaluated only after the relevant checkpoint or inference rule had been fixed.
11.4 Evaluation protocol
The study reports:
Top-1 accuracy;
Top-k accuracy;
Macro Precision;
Macro Recall;
Macro F1;
Weighted F1;
per-class metrics; and
confusion matrices.
Macro F1 is especially important because every class contributes equally to the average.
11.5 External-evaluation protocol
The primary F06 model was frozen before external evaluation. No external training, fine-tuning, adaptation, or external model selection was performed.
11.6 Explainability protocol
Grad-CAM and LIME were used only to inspect influential regions and representative errors. They were not treated as proof of biological understanding or causal reasoning.

12. Final experimental scope
The final campaign contains:
ID
Final configuration
F01
YOLO26m-cls, Aug-200, (224\times224)
F02
YOLO26s-cls, Aug-200, (224\times224)
F03
ConvNeXt-Small, original training distribution, Gentle Class-Balanced Focal Loss, (224\times224)
F04
ConvNeXt-Small, Aug-200, cross-entropy, (224\times224)
F05
ConvNeXt-Small, Aug-200, Gentle Class-Balanced Focal Loss, (224\times224)
F06
F03 checkpoint continuation at (384\times384); primary balanced single model
F07
Frozen F06 with deterministic two-view TTA
F08
Validation-selected probability ensemble: 0.70 F06 + 0.30 F01

Detailed motivation, configuration, training history, and results for each experiment will be documented in later phases.

13. Project outcome snapshot
The detailed result analysis belongs to later phases, but the final project position is anchored by the following verified outcomes:
F06 is the primary proposed single model and strongest balanced single-model configuration in the final campaign.
F06 achieved 92.12% Top-1 accuracy and 88.17% Macro F1.
F08 achieved the highest Top-1 point estimate, 92.25%, but its Macro F1 remained below F06; it is therefore a complementary ranking ensemble rather than the primary method.
F07 did not improve the selected F06 model under the evaluated two-view TTA rule.
The primary raw external evaluation achieved 76.50% Top-1 accuracy and 75.28% Macro F1 on 9,738 shared-class external images.
The external reduction shows that strong internal AQUA20 performance does not solve cross-dataset generalisation.

14. Scientific contribution
The strongest defensible contribution is:
A controlled AQUA20-specific evaluation of data-level and loss-level imbalance interventions, followed by high-resolution continuation, inference-level refinements, balanced-class reporting, qualitative explainability, and frozen source-only external evaluation.
Specific contributions include:
a fixed, validation-controlled AQUA20 campaign;
comparison of original and Aug-200 training distributions;
evaluation of Gentle Class-Balanced Focal Loss;
comparison of YOLO26 and ConvNeXt-Small configurations;
high-resolution continuation of the selected ConvNeXt branch;
deterministic TTA evaluation;
validation-selected heterogeneous probability ensembling;
class-wise and confusion-based error analysis;
external evaluation without adaptation; and
Grad-CAM and LIME diagnosis on selected AQUA20 test images.

15. Claim boundaries
The documentation must not claim:
that this is the first AQUA20 study;
that this is the first class-imbalance study in fish recognition;
that the project invented focal loss, effective-number weighting, or class-balanced focal loss;
that “Gentle” represents a new loss formulation;
that F06 proves a causal benefit from resolution alone;
that F08 is the primary model merely because its Top-1 estimate is slightly higher;
that the work is globally state of the art without a directly comparable benchmark protocol;
that external generalisation has been solved;
that the external dataset has the same 20-class taxonomy as AQUA20;
that Grad-CAM or LIME proves correct reasoning; or
that single-run fixed-split uncertainty represents variation across training seeds.

16. Phase 1 question bank
Q1. What is the central problem of the project?
The central problem is not merely achieving high accuracy. It is achieving strong overall recognition without allowing majority classes to hide weak minority-class performance, while maintaining a strict evaluation protocol and testing behaviour under source shift.
Q2. Why is AQUA20 challenging?
AQUA20 combines class imbalance with underwater degradation, complex backgrounds, intra-class variation, and inter-class visual similarity.
Q3. Why is Top-1 accuracy insufficient?
Top-1 accuracy is influenced more heavily by classes with more images. A model can perform strongly on majority classes and still fail on minority classes.
Q4. Why is Macro F1 important?
Macro F1 calculates F1 independently for every class and averages the class scores equally, making it more informative for imbalanced multiclass evaluation.
Q5. What is the main research goal?
To evaluate and improve imbalance-aware underwater classification on AQUA20 while preserving balanced class-wise performance and testing the frozen selected model on an independent external image source.
Q6. What is the novelty?
The novelty is the controlled combination of imbalance interventions, fixed-protocol evaluation, high-resolution continuation, inference refinements, balanced reporting, external transfer testing, and cautious explainability—not a new backbone or loss family.
Q7. Why was Aug-200 restricted to training?
Balancing validation or test data would alter the evaluation distribution and could make results less representative or introduce evaluation bias.
Q8. Was the official test set used for model selection?
No. The relevant checkpoint or inference rule was locked before official-test evaluation.
Q9. Did every experiment use validation Macro F1 for selection?
No. F01 and F02 used native Ultralytics classification fitness. F03–F06 and the F08 ensemble rule used validation Macro F1.
Q10. Why is F06 the primary model?
F06 provides the strongest balanced single-model result within the final campaign and preserves a cleaner single-model scientific contribution than the ensemble.
Q11. Why is F08 not the primary method?
F08 has a slightly higher Top-1 point estimate but lower Macro F1 than F06 and requires a second model at inference.
Q12. Did TTA improve performance?
The evaluated deterministic two-view horizontal-flip TTA did not improve the selected F06 model.
Q13. Why was external evaluation included?
Internal test performance from the same source cannot establish reliability under changes in image source, background, camera, lighting, and taxonomy.
Q14. Was the external model retrained?
No. F06 remained frozen; no external training, adaptation, fine-tuning, or model selection was performed.
Q15. What do Grad-CAM and LIME prove?
They do not prove model reasoning. They provide qualitative evidence about image regions associated with selected predictions and errors.

17. Phase 1 completion decision
Phase 1 is complete when the following are locked:
official project identity;
canonical source hierarchy;
research background;
core problem;
final goal;
research gap;
objectives;
research questions;
methodological principles;
final campaign scope;
contribution position; and
claim boundaries.
All items above are now locked for subsequent documentation phases.

