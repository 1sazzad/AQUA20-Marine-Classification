D02 — AQUA20 Dataset, Split Construction, and Aug-200 Documentation

Document status

Phase: 2 — Dataset, Class Distribution, Split Construction, and Aug-200 DocumentationStatus: Completed and lockedCanonical manuscript: Improving Underwater Marine Species Classification through Imbalance-Aware Learning and Cross-Dataset Evaluation

1. Source-control decision for this phase

This phase follows the locked evidence hierarchy:

latest final manuscript;

FINAL evidence-freeze and frozen-terminology records;

preserved F00 notebook and run-level dataset evidence;

older reports and drafts for historical context only.

The final evidence-freeze workbook is authoritative for the class registry, split counts, and frozen folder names. The F00 dataset-preparation notebook is authoritative for the actual split-reconstruction code, runtime folder structure, Aug-200 generation algorithm, and integrity assertions.

Evidence-status update

An earlier evidence-freeze record marked the exact Aug-200 transformation operations as missing. The preserved F00 notebook and its extracted transformation record were subsequently found in the File Library. Therefore:

the offline Aug-200 operation sequence and parameter ranges are now verified;

the per-image realised random draws were not separately archived as a table;

the immutable Hugging Face dataset revision or content hash remains To be verified.

The final experiment identity correction remains locked:

F02 is F02_YOLO26S_CLS_AUG200. It is not a ConvNeXt-Small cross-entropy baseline.

2. Dataset identity and scope

2.1 Dataset

Dataset: AQUA20

Public source identifier used by the campaign: taufiktrf/AQUA20

Classes: 20

Images: 8,171

Original public splits observed by F00: 6,559 training-pool images and 1,612 test images

Campaign split: 5,247 official training / 1,312 validation / 1,612 official test

Split seed: 42

Split method: stratified reconstruction of the 6,559-image Hugging Face training pool

Test policy: the original 1,612-image Hugging Face test split was preserved unchanged

2.2 Meaning of “original total” in the registry

In the class registry below, Original total means the full AQUA20 class count across the campaign's official training, validation, and test partitions:

n_c^{\mathrm{train}}+n_c^{\mathrm{val}}+n_c^{\mathrm{test}}.]

It does not refer to the Aug-200 representation.

3. Frozen 20-class registry

ID

Class name

Frozen folder

Original total

Train

Validation

Test

Aug-200 train

Synthetic added

Aug-200 status

0

coral

00_coral

1,910

1,249

313

348

1,249

0

Unchanged

1

crab

01_crab

54

34

9

11

200

166

Raised to 200

2

diver

02_diver

64

41

10

13

200

159

Raised to 200

3

eel

03_eel

201

128

32

41

200

72

Raised to 200

4

fish

04_fish

2,738

1,760

440

538

1,760

0

Unchanged

5

fishInGroups

05_fishInGroups

344

218

54

72

218

0

Unchanged

6

flatworm

06_flatworm

63

40

10

13

200

160

Raised to 200

7

jellyfish

07_jellyfish

122

78

19

25

200

122

Raised to 200

8

marine_dolphin

08_marine_dolphin

30

16

4

10

200

184

Raised to 200

9

octopus

09_octopus

30

16

4

10

200

184

Raised to 200

10

rayfish

10_rayfish

477

305

77

95

305

0

Unchanged

11

seaAnemone

11_seaAnemone

1,111

712

178

221

712

0

Unchanged

12

seaCucumber

12_seaCucumber

45

28

7

10

200

172

Raised to 200

13

seaSlug

13_seaSlug

99

63

16

20

200

137

Raised to 200

14

seaUrchin

14_seaUrchin

144

92

23

29

200

108

Raised to 200

15

shark

15_shark

90

57

14

19

200

143

Raised to 200

16

shrimp

16_shrimp

33

18

4

11

200

182

Raised to 200

17

squid

17_squid

34

19

5

10

200

181

Raised to 200

18

starfish

18_starfish

206

133

33

40

200

67

Raised to 200

19

turtle

19_turtle

376

240

60

76

240

0

Unchanged

Total

20 classes

—

8,171

5,247

1,312

1,612

7,284

2,037

—

3.1 Frozen class order

The class order used throughout the final campaign is:

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

Class order was not inferred from display labels. It was verified against the ImageFolder class-to-index mapping and retained across official train, validation, test, and Aug-200 folders.

4. Class-distribution analysis

4.1 Highest-frequency classes

The three largest classes in the original official training split were:

Rank

Class

Training images

Share of training split

1

fish

1,760

33.54%

2

coral

1,249

23.80%

3

seaAnemone

712

13.57%

Together, these three classes represented 70.92% of the 5,247-image official training split.

4.2 Smallest classes

The smallest official-training classes were:

marine_dolphin: 16;

octopus: 16;

shrimp: 18;

squid: 19;

seaCucumber: 28; and

crab: 34.

The largest-to-smallest official-training ratio was:

[\frac{1,760}{16} = 110:1.]

This confirms that the training distribution was strongly imbalanced before Aug-200.

4.3 Operational minority definition

For the Aug-200 intervention, a class was operationally treated as under-represented when:

[n_c^{\mathrm{train}} < 200.]

Under this rule:

14 classes were augmented;

6 classes were unchanged: fishInGroups, turtle, rayfish, seaAnemone, coral, and fish.

This is an intervention threshold, not a universal biological or statistical definition of a minority class.

4.4 Aug-200 did not fully equalise the dataset

Aug-200 imposed a minimum training representation of 200 images:

[\widetilde{n}_c = \max(n_c, 200).]

It did not downsample the large classes. Consequently, the post-Aug-200 training distribution remained unequal: fish, coral, and seaAnemone still contained substantially more than 200 images.

5. Exact seed-42 split reconstruction

5.1 Source pools

The F00 notebook loaded:

Hugging Face train: 6,559 images;

Hugging Face test: 1,612 images.

Only the 6,559-image Hugging Face training pool was reconstructed into campaign training and validation partitions.

5.2 Reconstruction procedure

The preserved notebook performed the following procedure:

Read the label names from dataset["train"].features["label"].names.

Extract the 6,559 training-pool labels.

Create an index array covering every Hugging Face training example.

Call train_test_split with:

test_size=1312;

random_state=42;

stratify=hf_train_labels;

shuffle=True.

Sort the returned training and validation indices before export.

Record every selected source item in a manifest containing:

Hugging Face dataset ID;

source split;

source index;

campaign partition;

class ID;

class name; and

deterministic source key.

Preserve the original Hugging Face test split as the campaign's official test split.

The resulting relationship was:

[6,559 = 5,247 + 1,312,]

and:

[5,247 + 1,312 + 1,612 = 8,171.]

5.3 Why stratification was used

Stratification retained the class proportions of the 6,559-image source training pool as closely as integer allocation permitted. This prevented the validation split from being formed by an uncontrolled random draw that could omit or severely distort small classes.

5.4 Why the validation and test distributions were unchanged later

The final campaign modified only training exposure. Validation and test data remained unchanged because:

validation governed checkpoint and ensemble decisions;

test was reserved for final locked evaluation;

balancing evaluation data would alter the target distribution;

synthetic evaluation samples could make metrics less representative;

keeping the same files enabled direct comparison across F01–F08.

The F00 notebook compared the relative validation and test file lists between the official and Aug-200 ImageFolder structures and asserted exact equality.

6. Train-only Aug-200 construction

6.1 Balancing rule

The offline generation stage used:

SEED = 42
AUG_TARGET = 200

For every class:

all original training images were retained;

a class below 200 images received synthetic images until it reached exactly 200;

a class already at or above 200 was not augmented for balancing;

no class was downsampled;

validation and test folders were copied unchanged.

The operation added:

[7,284 - 5,247 = 2,037]

synthetic training images.

6.2 Deterministic source selection

The generator used a class-specific random state:

random.Random(42 + class_id * 1000)

Source images inside each class were sorted and selected cyclically:

src_images[aug_i % original_count]

This prevented source selection from depending on arbitrary filesystem traversal order.

6.3 Verified transformation sequence

Each synthetic training image was generated in this order:

Convert the source image to RGB.

Sample an isotropic crop scale uniformly from 0.82 to 1.00.

Crop at a random valid left/top location.

Resize the crop back to the source image's original width and height using PIL bicubic interpolation.

Apply horizontal flipping with probability 0.5.

Apply a rotation sampled uniformly from −18° to +18°, using PIL bicubic interpolation with expand=False.

Adjust brightness using a factor sampled from 0.82 to 1.18.

Adjust contrast using a factor sampled from 0.85 to 1.20.

Adjust colour saturation using a factor sampled from 0.80 to 1.22.

Adjust sharpness using a factor sampled from 0.85 to 1.25.

Save the synthetic output as JPEG with quality 95.

The offline generator did not use:

vertical flipping;

shear;

perspective transformation;

blur;

added noise;

MixUp;

CutMix; or

copy-paste augmentation.

6.4 Offline versus online augmentation

Aug-200 was an offline dataset-expansion stage. It created additional image files before model training.

Online augmentation was model/run specific and transformed images dynamically during optimisation. The two mechanisms must not be described as the same process.

6.5 Per-image random-draw boundary

The algorithm, seed rule, operation order, and parameter ranges are verified. A separate record containing the exact random values applied to every individual synthetic image was not found.

Per-image realised augmentation parameters: To be verified / not separately archived.

7. Runtime folder structure

The preserved F00 notebook used the following project root:

/kaggle/working/AQUA20_FINAL_BEAT_BASE_04072026/

The principal dataset structure was:

AQUA20_FINAL_BEAT_BASE_04072026/
├── official_imagefolder/
│   ├── train/
│   │   ├── 00_coral/
│   │   ├── 01_crab/
│   │   └── ... 19_turtle/
│   ├── val/
│   │   ├── 00_coral/
│   │   └── ... 19_turtle/
│   └── test/
│       ├── 00_coral/
│       └── ... 19_turtle/
├── balanced_train/
│   └── aug200_imagefolder/
│       ├── train/
│       │   ├── 00_coral/
│       │   └── ... 19_turtle/
│       ├── val/
│       │   ├── 00_coral/
│       │   └── ... 19_turtle/
│       └── test/
│           ├── 00_coral/
│           └── ... 19_turtle/
├── data_checks/
├── experiments/
├── tables/
├── figures/
├── xai/
├── ensemble/
├── packages/
└── logs/

The compact F00 handover package intentionally excluded the large official_imagefolder and aug200_imagefolder directories. Their construction code, counts, manifests, and verification outputs were retained.

8. Data-integrity checks and boundaries

8.1 Verified checks

Check

Status

Evidence-backed interpretation

Dataset source identifier

Verified

Campaign used taufiktrf/AQUA20.

Original source split totals

Verified

6,559 train-pool and 1,612 test images.

Campaign split totals

Verified

5,247 train, 1,312 validation, 1,612 test.

Per-class counts

Verified

Frozen for all 20 classes.

Class order

Verified

Stable across official and Aug-200 train/validation/test structures.

Stratified reconstruction

Verified

Seed 42, fixed 1,312 validation count.

Test preservation

Verified

Original Hugging Face test split retained.

Validation/test unchanged under Aug-200

Verified

Relative file lists matched exactly.

Aug-200 totals

Verified

7,284 training images; 2,037 synthetic additions.

ImageFolder loading

Verified

Official and Aug-200 structures loaded successfully.

Tensor sanity check

Verified

One-batch image and label shapes passed at 224×224.

Expected split-count assertions

Verified

Notebook assertions passed.

8.2 Leakage boundary

The reconstructed train and validation sets were generated from disjoint index outputs of one train_test_split call. The original Hugging Face test split was a separate source split. Aug-200 synthetic images were produced only from official-training images and stored only in the training partition.

This verifies the intended split logic. It does not substitute for a content-level duplicate audit.

8.3 Items still not verified

Complete internal duplicate-image audit

No complete pixel-hash, perceptual-hash, or near-duplicate audit across the official training, validation, and test partitions was found.

Complete duplicate-image audit: To be verified.

Corrupt-image policy and exhaustive decode audit

The export/count checks, ImageFolder loading, and batch sanity checks passed. However, no separately archived policy describing corrupt-image rejection, repair, or replacement was found, and no definitive all-file decode-audit report was located.

Corrupt-image handling policy and exhaustive internal decode audit: To be verified.

Immutable source revision

The source identifier is frozen, but no Hugging Face commit SHA, immutable revision, dataset fingerprint, or complete source-content checksum was archived.

Immutable dataset revision/content hash: To be verified.

Split-manifest checksum

The notebook generated a source-index manifest, but a final immutable checksum for the exact manifest was not located in the reviewed evidence.

Final split-manifest checksum: To be verified.

9. Reviewer and viva question bank

Q1. How many images and classes are in AQUA20?

AQUA20 contains 8,171 images across 20 classes.

Q2. What split did the final campaign use?

The campaign used 5,247 official training images, 1,312 validation images, and an unchanged 1,612-image official test split.

Q3. Where did the validation split come from?

It was reconstructed from the 6,559-image Hugging Face training pool using stratified train_test_split, a fixed validation size of 1,312, and random seed 42.

Q4. Was the original Hugging Face test split repartitioned?

No. The 1,612-image Hugging Face test split was preserved unchanged as the official test split.

Q5. Why was stratification necessary?

It retained class proportions as closely as possible and reduced the risk that small classes would be badly represented or omitted in validation.

Q6. What was the most frequent training class?

fish, with 1,760 official-training images.

Q7. What were the smallest training classes?

marine_dolphin and octopus, with 16 images each.

Q8. How severe was the original imbalance?

The largest-to-smallest training-count ratio was 110:1. The three largest classes accounted for 70.92% of the official training split.

Q9. What exactly does Aug-200 mean?

Every class with fewer than 200 official-training images was expanded to 200 through offline augmentation. Larger classes were retained unchanged.

Q10. How many classes were augmented?

Fourteen classes were augmented; six classes already had at least 200 training images and were unchanged.

Q11. How many synthetic images were added?

2,037 synthetic training images were added, increasing training representation from 5,247 to 7,284.

Q12. Did Aug-200 create a perfectly balanced dataset?

No. It imposed a lower floor of 200 but did not downsample large classes, so the post-Aug-200 distribution remained imbalanced.

Q13. Were validation or test images augmented?

No. The intervention was restricted to training. Validation and test relative file lists were verified as unchanged.

Q14. What transformations were used by the offline generator?

RGB conversion, random isotropic crop, bicubic resize, optional horizontal flip, rotation, brightness, contrast, saturation, and sharpness adjustments, followed by JPEG quality-95 saving.

Q15. Is Aug-200 the same as online training augmentation?

No. Aug-200 created additional files offline. Online augmentation was applied dynamically and differed by experiment.

Q16. How was leakage prevented?

Only official-training sources were used for synthetic generation; reconstructed train/validation indices were disjoint; the separate original test split remained untouched.

Q17. Was a complete duplicate audit performed?

No complete content-level duplicate or near-duplicate audit was found. This remains To be verified.

Q18. Were corrupt images documented?

The dataset structures loaded and passed count and batch checks, but a separate exhaustive corrupt-image audit and handling policy were not found.

Q19. Can the exact source dataset version be reproduced indefinitely?

The dataset ID is known, but an immutable Hugging Face revision or full content checksum was not archived. That boundary must be disclosed.

Q20. What is the locked F02 identity?

F02 is F02_YOLO26S_CLS_AUG200; it must not be documented as a ConvNeXt-Small cross-entropy baseline.

10. Evidence ledger

The following sources controlled this phase:

Improving Underwater Marine Species Classification through Imbalance-Aware Learning and Cross-Dataset Evaluation.pdf

P00_AQUA20_Master_Evidence_Freeze_FINAL.xlsx

04-07-2026-aqua20-final-beat-base-plan.ipynb

README_F00.txt

AQUA20_class_distribution.xlsx

Pasted markdown.md containing the verified transform extraction

D01_AQUA20_Project_Foundation.md

H01_Phase1_to_Phase2_Handover.md

Older workbooks that assigned F02 a different identity or marked the transformation algorithm as permanently missing are superseded where later canonical or run-level evidence resolves the issue.

11. Phase 2 completion decision

Phase 2 is complete because the following are now locked:

all 20 class names and exact order;

full per-class original totals;

class-wise train, validation, and test counts;

class-wise Aug-200 counts and synthetic additions;

majority and operational minority interpretation;

exact seed-42 split reconstruction;

unchanged validation/test rationale and file-identity verification;

verified offline Aug-200 transformation sequence;

final runtime ImageFolder structure;

verified integrity checks; and

explicit boundaries for duplicates, corrupt-image handling, immutable revision, and checksums.

Remaining To be verified items carried forward

immutable Hugging Face dataset revision, fingerprint, or full source checksum;

final split-manifest checksum and preserved public location;

complete content-level and near-duplicate audit;

exhaustive internal corrupt-image decode report and handling policy;

per-image realised Aug-200 random-parameter log, if one ever existed.

These unresolved items do not change the frozen counts or split logic, but they limit claims of complete dataset-level reproducibility.