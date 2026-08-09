# AQUA20 Documentation Handover — Phase 1 to Phase 2

## Completed phase

**Phase 1 — Project Foundation**

## Files created

- `D01_AQUA20_Project_Foundation.md`
- `H01_Phase1_to_Phase2_Handover.md`

## Locked source hierarchy

1. Latest final manuscript
2. FINAL evidence-freeze and frozen-terminology records
3. Preserved experiment packages and run-level evidence
4. Older drafts and supervisor reports for historical context only

## Critical correction locked

F02 is:

> **F02_YOLO26S_CLS_AUG200**

It must never be documented as a ConvNeXt-Small cross-entropy baseline.

## Foundation decisions locked

- Final title: *Improving Underwater Marine Species Classification through Imbalance-Aware Learning and Cross-Dataset Evaluation*
- Dataset: AQUA20, 8,171 images, 20 classes
- Official split: 5,247 train / 1,312 validation / 1,612 test
- Split method: stratified, seed 42
- Aug-200: training-only expansion to 7,284 images
- Main research focus: balanced recognition under class imbalance plus external source-shift evaluation
- F06: primary proposed single model / strongest balanced single-model configuration
- F08: complementary ranking ensemble
- XAI: qualitative diagnostic only
- External evaluation: frozen F06, no adaptation
- Unsupported details must be marked `To be verified`

## Next phase

**Phase 2 — Dataset, Class Distribution, Split Construction, and Aug-200 Documentation**

## Phase 2 objectives

1. Lock all 20 AQUA20 class names and exact class order.
2. Record the complete original class distribution.
3. Record class-wise train, validation, and test counts.
4. Identify majority and minority classes.
5. Document how the seed-42 stratified split was reconstructed.
6. Explain why validation and test distributions were unchanged.
7. Document Aug-200 class-by-class changes.
8. Separate verified augmentation facts from missing transformation details.
9. Document data-integrity checks, duplicate checks, corrupt-image handling, and folder structure where evidence exists.
10. Prepare the dataset-related viva/reviewer question bank.

## Evidence to search first in the next session

The assistant must search the project File Library before asking the user for files, including:

- final evidence-freeze workbook;
- dataset split CSV/JSON/TXT records;
- class-order metadata;
- Aug-200 count tables;
- dataset preparation notebooks;
- Fig. 1 source data;
- experiment manifests; and
- any class-distribution scripts.

## Items currently marked To be verified

- Exact transformation sequence used to generate every Aug-200 image
- Whether a complete internal duplicate-image audit was performed
- Corrupt-image handling details
- Exact source folder structure used in the final run
- Any dataset-version identifier beyond the cited official repository
- Complete class-wise counts, to be extracted in Phase 2

## Opening prompt for the next session

> Continue the AQUA20 documentation project from the locked Phase 1 handover. Start Phase 2 — Dataset, Class Distribution, Split Construction, and Aug-200 Documentation. Search the AQUA20 project File Library first and use the source hierarchy: final manuscript, FINAL evidence freeze/frozen terminology, then run-level packages. Do not ask me for already-established information and do not guess. First build a verified 20-class registry with original, train, validation, test, and Aug-200 counts, then document the seed-42 stratified split and all known data-integrity boundaries. Mark unsupported details “To be verified.”
