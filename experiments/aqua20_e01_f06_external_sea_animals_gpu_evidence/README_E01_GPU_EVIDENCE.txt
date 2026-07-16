AQUA20 E01 — F06 EXTERNAL SEA ANIMALS GPU EVIDENCE
============================================================

OFFICIAL CAMPAIGN
------------------------------------------------------------
04-07-2026 AQUA20 FINAL BEAT BASE PLAN

OFFICIAL STAGE
------------------------------------------------------------
E01_F06_EXTERNAL_SEA_ANIMALS

SCIENTIFIC ROLE
------------------------------------------------------------
EXTERNAL GENERALIZATION
KNOWN / UNKNOWN EXTERNAL ROBUSTNESS
FROZEN-MODEL INFERENCE

PRIMARY MODEL
------------------------------------------------------------
F06_CONVNEXT_SMALL_HR384_GENTLE_CB_FOCAL

FROZEN CHECKPOINT
------------------------------------------------------------
Checkpoint : best_model.pth
Epoch      : 2
SHA-256    : d0c225c28b25f09822bc99518a37f9b1fe642b2d01213da7f68e2d76a255832a

The F06 checkpoint was unchanged.

EXTERNAL DATASET
------------------------------------------------------------
Dataset : Sea Animals Image Dataset
Kaggle  : vencerlanz09/sea-animals-image-dataste

External classes : 23
External images  : 13,711

STRICT SHARED-CLASS MAPPING
------------------------------------------------------------
15 strict semantic matches were frozen.

Corals           -> 00_coral
Crabs            -> 01_crab
Dolphin          -> 08_marine_dolphin
Eel              -> 03_eel
Fish             -> 04_fish
Jelly Fish       -> 07_jellyfish
Nudibranchs      -> 13_seaSlug
Octopus          -> 09_octopus
Sea Rays         -> 10_rayfish
Sea Urchins      -> 14_seaUrchin
Sharks           -> 15_shark
Shrimp           -> 16_shrimp
Squid            -> 17_squid
Starfish         -> 18_starfish
Turtle_Tortoise  -> 19_turtle

The strict mapping contains 9,738 external images.

EXTERNAL-ONLY CLASSES
------------------------------------------------------------
Clams
Lobster
Otter
Penguin
Puffers
Seahorse
Seal
Whale

These classes are treated as UNKNOWN EXTERNAL categories for later
mixed known / unknown robustness analysis.

INFERENCE PROTOCOL
------------------------------------------------------------
External image decode : native file -> RGB
Model input resolution: 384 x 384
Normalization          : ImageNet mean / std
Prediction rule        : raw F06 20-way output

Full 20-class probability vectors were saved for every successfully
decoded external image.

INFERENCE VERIFICATION
------------------------------------------------------------
Images inferred     : 13,711
Decode errors       : 0
Probability matrix  : 13,711 x 20
Probability row sums: verified approximately 1.0

TRAINING STATUS
------------------------------------------------------------
Training       : NONE
Fine-tuning    : NONE
External tuning: NONE

The external labels were not used to modify F06.

GPU STATUS
------------------------------------------------------------
Full external F06 inference is complete.

No additional E01 GPU inference is required.

CPU ANALYSIS RESERVED
------------------------------------------------------------
E01A:
Primary strict 15-class shared external generalization using raw
F06 20-way decisions.

E01B:
Optional restricted-15 probability diagnostic.

E01C:
23-class mixed known / unknown external robustness analysis.

Planned CPU analyses include:
- shared-class Top-1 and Top-k performance
- precision, recall, Macro F1, Weighted F1
- per-class analysis
- known-versus-unknown confidence
- predictive entropy
- AUROC
- AUPR
- unknown-class prediction destinations

STATUS
------------------------------------------------------------
E01 GPU INFERENCE COMPLETED
EXTERNAL EVIDENCE FROZEN
READY FOR CPU ANALYSIS