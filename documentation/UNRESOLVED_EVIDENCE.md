# AQUA20 — Unresolved Evidence Checklist

**Status:** Consolidated after Phase 16 / P01  
**Recommended repository target:** `documentation/UNRESOLVED_EVIDENCE.md`

The items below are intentionally unresolved. They must not be guessed or reconstructed merely to make the manuscript, tables, repository, or reproducibility appendix look complete.

---

## 1. Priority A — claim-blocking if stronger statements are desired

### A1. X03 quantitative Grad-CAM-versus-LIME evidence

Directly missing/not surfaced:

```text
X03 README
X03 protocol lock
Grad-CAM threshold rule
LIME comparison mask rule
mask-resizing rule
IoU definition/results
Dice definition/results
per-image similarity table
group summaries
official X03 figures/package
X03 integrity hashes
```

**Current allowed claim:** X01/X02 support matched qualitative inspection only.

**Do not claim:** quantitative agreement, overlap, IoU/Dice performance, or completed X03.

---

### A2. Direct standalone R09 generated statistical outputs

Not surfaced in the Phase-13 retrieval:

```text
README_R09.md
R09_bootstrap_protocol.json
R09_model_bootstrap_confidence_intervals.csv
R09_pairwise_statistical_comparisons.csv
R09_prediction_source_manifest.csv
R09_*_prediction_disagreements.csv
R09_F06_F08_correctness_switch_summary.csv
```

Also not surfaced there as direct standalone files:

```text
F03/F04/F05/F06/F08 row-level prediction CSVs
```

The R09 script/protocol was inspected, and manuscript-preserved Top-1 interval/McNemar values were documented, but the final generated row-level result artifacts were not directly re-certified.

**Current allowed claim:** use manuscript-preserved paired Top-1 values with the direct-output caveat.

**Still To be verified:** pairwise Macro-F1 intervals, discordant counts, F04/F05 paired statistics, exact model-level bootstrap tables, correctness-switch table.

---

### A3. Final F06 direct per-class rows and raw confusion counts for detailed ranking claims

Phase 12 did not surface the complete final direct F06 classification-report rows and raw confusion-matrix rows during its retrieval.

**Current allowed claim:** support-aware qualitative summary and cautious previously frozen strong/weak-class narrative.

**Do not invent:** exact strongest/weakest rankings, exact class-wise precision/recall/F1 values, or raw confusion counts from plotted figures.

---

## 2. Priority B — evidence-integrity/reproducibility gaps

### B1. Original package hashes / exact package bytes where not collected

Known unresolved examples include:

```text
F06 original ZIP SHA-256
F07 original ZIP SHA-256 / exact ZIP bytes
F08 original ZIP SHA-256 / exact ZIP bytes
F09 original ZIP SHA-256 / exact ZIP bytes / extracted-tree hash / per-file manifest
X01 original ZIP SHA-256
X02 original ZIP SHA-256
definitive final E01 consolidated package SHA-256 / bytes / extracted-tree hash / per-file manifest
```

Where extracted-tree hashes are frozen, retain them as evidence but do not mislabel them as original ZIP hashes.

---

### B2. Immutable dataset revision/content-hash records

Still unresolved:

```text
immutable AQUA20 dataset revision/content hash
immutable external Sea Animals dataset revision/content hash
full preserved decode/duplicate boundary audit
cross-dataset duplicate audit between AQUA20 and external source
```

**Boundary:** do not claim the two datasets are guaranteed duplicate-free without a preserved audit.

---

### B3. Complete run-specific software/hardware/container manifests

Not uniformly preserved for every run:

```text
complete per-stage pip/package freeze
exact scikit-learn/NumPy/pandas/Pillow versions for all stages
NVIDIA driver and complete nvidia-smi records
native cuDNN runtime version per run
immutable Kaggle/container image identifier
CPU/RAM/host configuration
complete worker/backend/autocast state
```

Some stages have verified partial environment records, including P100 and selected Python/PyTorch/torchvision values. Do not generalise one run's environment to all experiments.

---

### B4. Upstream pretrained-weight provenance

For several ConvNeXt stages and YOLO starting checkpoints, immutable upstream file hashes/repository revision records are incomplete.

**Boundary:** distinguish a preserved model identifier/filename from immutable upstream provenance.

---

## 3. Priority C — efficiency/deployment evidence not frozen consistently

The campaign does not contain a standardised cross-model benchmark for:

```text
parameter count
FLOPs
checkpoint/model size under one convention
controlled latency
throughput
peak GPU memory
energy/compute cost
```

**Boundary:** do not create comparative efficiency claims from mixed hardware, mixed tools, or anecdotal timing.

These gaps do not invalidate the frozen predictive results; they limit efficiency/deployment comparisons.

---

## 4. Priority D — statistical generalisation beyond the fixed trained models

Not estimated by the existing R09 fixed-test analysis:

```text
repeated-seed training variability
training-data resampling variability
checkpoint-selection variability across repeated runs
full optimisation uncertainty
```

The existing bootstrap/McNemar analysis is conditional on the already trained/frozen models and the finite 1,612-image test sample.

**Do not claim:** multi-seed statistical robustness or training-run stability.

---

## 5. Priority E — manuscript-reference/comparability items

### E1. 92.68% reference target

The internal campaign target remains:

```text
92.6800% Top-1
```

and was **not beaten**.

Still requiring independent provenance/comparability verification before treating it as a formal external benchmark:

```text
primary citation
dataset split equivalence
class definition equivalence
evaluation protocol equivalence
metric definition equivalence
```

The safe wording is “campaign reference target”, not “state-of-the-art benchmark”, unless these checks are completed.

---

## 6. Priority F — external evaluation integrity/provenance

Although E01A/E01B/E01C scientific values are documented, unresolved items include:

```text
definitive final consolidated E01 package integrity manifest
immutable external dataset revision/content hash
full external licensing/provenance audit
cross-dataset duplicate audit
exact low-level preprocessing semantics if not directly preserved
controlled external inference latency/throughput benchmark
```

Historical GPU/prediction checksum strings may be retained as provenance values but should not be upgraded to newly reverified final-package hashes without the original bytes/checksum command.

---

## 7. Priority G — XAI integrity/runtime fields

Beyond missing X03:

```text
X01 original ZIP SHA-256
X02 original ZIP SHA-256
complete run-local XAI environment manifest
controlled XAI runtime/memory benchmarks
```

Frozen extracted-tree hashes remain valid evidence:

```text
X01: 6c0d3c77820dcea3a9ee75ba00581430f9d34566896867d2ead9844a6f4ea138
X02: 11ca45b02e0f0ab98278c94b39c72fe0d8f655f4879e5f8747ef871b35f36db8
```

---

## 8. Non-gap that must not be “fixed”

The final F09 aggregate table intentionally records:

```text
F01 Top-2 = N/A
F01 Top-3 = N/A
F02 Top-2 = N/A
F02 Top-3 = N/A
```

Older isolated audit evidence may contain some additional Top-k values, but P01 does not back-fill the final F09 table.

This is a **frozen reporting choice**, not an invitation to reconstruct missing cells during manuscript writing.

---

## 9. Publication readiness classification

### Does not block P02 if the manuscript retains current boundaries

```text
- missing package hashes where core metrics/checkpoints are already verified;
- incomplete efficiency benchmarking;
- incomplete full environment manifests;
- repeated-seed variability, provided the paper clearly states single-run/fixed-model limits;
- direct R09 output gaps, provided statistical statements remain explicitly caveated;
- detailed per-class row gaps, provided exact unsupported rows/counts are not claimed.
```

### Blocks any stronger claim

```text
- X03 quantitative agreement without direct X03 evidence;
- exact unsupported class-wise numbers/confusion counts;
- formal multi-seed robustness claims;
- state-of-the-art/benchmark claims based solely on the unverified 92.68 target;
- duplicate-free cross-dataset claims without a preserved duplicate audit;
- efficiency superiority without controlled common benchmarking.
```

---

## 10. Resolution rule

When an unresolved item is later found:

1. verify it against the source hierarchy;
2. record the exact file/path and checksum where applicable;
3. update this checklist from `To be verified` to `Verified`;
4. update only the affected table/claim;
5. do not reopen F09 model selection unless a genuinely controlling scientific correction requires it;
6. preserve an audit trail of the change.
