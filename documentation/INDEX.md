# AQUA20 Documentation Index

**Status:** Phase-16 P01 consolidation index  
**Recommended repository target:** `documentation/INDEX.md`

This index records the documentation streams currently represented in the AQUA20 project. It distinguishes older phase-numbered artifacts from later short canonical repository filenames. Existing artifacts are not silently renamed by this index.

---

## 1. Foundation and common protocol

| Phase | File / artifact | Role | Status |
|---|---|---|---|
| 1 | `D01_AQUA20_Project_Foundation.md` | Project scope, research framing, final scientific hierarchy foundations | Completed |
| 2 | `D02_AQUA20_Dataset_and_Aug200_Documentation.md` | AQUA20 dataset, fixed split, class distribution, Aug-200 | Completed |
| 3 | `D03_AQUA20_Environment_Reproducibility_and_Common_Protocol.md` | Environment, reproducibility, shared evaluation/training controls | Completed |

These names are legacy phase-style artifacts. They remain valid historical documentation unless deliberately migrated in a repository-maintenance step.

---

## 2. Experiment documentation

| Scientific ID(s) | Current documentation artifact | Recommended/current repository grouping | Status |
|---|---|---|---|
| F01–F02 | `D04_AQUA20_F01_F02_YOLO26_Aug200_Experiment_Documentation.md` | `documentation/experiments/` | Completed |
| F03 | `F03_convnext_focal.md` (canonical short record; older `D05_...` also exists in history) | `documentation/experiments/F03_convnext_focal.md` | Completed |
| F04 | `D06_AQUA20_F04_ConvNeXt_Small_Aug200_Cross_Entropy_Experiment_Documentation.md` | `documentation/experiments/` if later migrated to a short name | Completed |
| F05 | `F05_convnext_aug200_focal.md` | `documentation/experiments/F05_convnext_aug200_focal.md` | Completed |
| F06 | `F06_convnext_hr384.md` | `documentation/experiments/F06_convnext_hr384.md` | Completed |
| F07 | `F07_convnext_hr384_tta.md` | `documentation/experiments/F07_convnext_hr384_tta.md` | Completed |
| F08 | `F08_convnext_yolo_ensemble.md` | `documentation/experiments/F08_convnext_yolo_ensemble.md` | Completed |
| F09 | `F09_final_method_freeze.md` | `documentation/experiments/F09_final_method_freeze.md` | Completed / campaign closed |

### Identity guard

```text
F02 = F02_YOLO26S_CLS_AUG200
F04 = F04_CONVNEXT_SMALL_AUG200
```

F02 must never be renamed as the ConvNeXt cross-entropy experiment.

---

## 3. Cross-experiment analysis documentation

| Phase | Canonical repository target | Role | Status |
|---|---|---|---|
| 12 | `documentation/analysis/internal_classwise_error_analysis.md` | Internal class-wise diagnostics and error-evidence inventory | Completed with row-level verification boundaries |
| 13 | `documentation/analysis/statistical_uncertainty.md` | Fixed-test bootstrap/McNemar statistical uncertainty documentation | Completed with direct-output verification caveats |
| 14 | `documentation/analysis/external_evaluation.md` | E01A/E01B/E01C cross-dataset evaluation and robustness | Completed with integrity/reproducibility gaps retained |
| 15 | `documentation/analysis/explainability_f06.md` | X01/X02 frozen F06 qualitative explainability | Completed; X03 not claimed complete |

---

## 4. Final consolidation documentation

| Phase | Canonical repository target | Role | Status |
|---|---|---|---|
| 16 / P01 | `documentation/consolidation/final_publication_tables.md` | Final paper table consolidation and manuscript-evidence reconciliation | Completed |
| 16 / P01 | `documentation/INDEX.md` | Final documentation index | Completed |
| 16 / P01 | `documentation/UNRESOLVED_EVIDENCE.md` | Consolidated unresolved-evidence checklist | Completed |
| 17 / P02 | To be created in next phase | Final manuscript writing/reconciliation | Not started in Phase 16 |

---

## 5. Handover stream

Handover documents preserve the one-phase-per-session workflow.

Confirmed examples include:

```text
H03_Phase3_to_Phase4_Handover.md
H04_Phase4_to_Phase5_Handover.md
H05_Phase5_to_Phase6_Handover.md
H06_Phase6_to_Phase7_Handover.md
documentation/handovers/H07_to_08.md
documentation/handovers/H08_to_09.md
documentation/handovers/H09_to_10.md
documentation/handovers/H10_to_11.md
documentation/handovers/H11_to_12.md
documentation/handovers/H12_to_13.md
documentation/handovers/H13_to_14.md
documentation/handovers/H14_to_15.md
documentation/handovers/H15_to_16.md
documentation/handovers/H16_to_17.md
```

Historical naming differences should not be treated as scientific differences.

---

## 6. Canonical scientific source hierarchy

When repository documentation conflicts with older notes, use:

1. latest final manuscript for current official narrative and table placement;
2. FINAL evidence freeze and frozen terminology/index for identities, claims, metrics, and verification limits;
3. direct experiment/package/protocol evidence;
4. completed documentation generated from those frozen sources;
5. older audits, supervisor summaries, and handovers only for chronology/context.

---

## 7. Final reporting hierarchy

```text
F06 = primary proposed method / primary single model / strongest balanced-class model
F08 = complementary final ensemble / strongest ranking model
F03 = strongest isolated imbalance intervention within the tested campaign
E01A = primary external result
E01B = restricted-taxonomy diagnostic
E01C = known–unknown robustness diagnostic
X01/X02 = qualitative post-hoc diagnostics
X03 = not claimed complete without direct package/protocol/metrics
```

---

## 8. Repository naming policy

For new files, use short meaningful filenames rather than `Dxx_...` phase-number names.

Preferred structure:

```text
documentation/
├── INDEX.md
├── UNRESOLVED_EVIDENCE.md
├── experiments/
│   ├── F03_convnext_focal.md
│   ├── F05_convnext_aug200_focal.md
│   ├── F06_convnext_hr384.md
│   ├── F07_convnext_hr384_tta.md
│   ├── F08_convnext_yolo_ensemble.md
│   └── F09_final_method_freeze.md
├── analysis/
│   ├── internal_classwise_error_analysis.md
│   ├── statistical_uncertainty.md
│   ├── external_evaluation.md
│   └── explainability_f06.md
├── consolidation/
│   └── final_publication_tables.md
└── handovers/
    └── ...
```

Legacy D01–D06 artifacts can be migrated later only through an explicit repository-cleanup decision. Their content should not be duplicated or silently changed during manuscript consolidation.
