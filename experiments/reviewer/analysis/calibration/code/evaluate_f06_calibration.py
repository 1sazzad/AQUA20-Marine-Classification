# ============================================================
# AQUA20 REVIEWER CALIBRATION ANALYSIS
# F06 — ConvNeXt-Small HR384 Gentle CB Focal
# Frozen official-test probabilities
# ============================================================

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(
    r"F:\Research\AQUA20-Marine-Classification"
)

SOURCE_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "aqua20_f08_convnext_hr384_yolo26m_prob_ensemble_outputs"
    / "probabilities"
    / "F08_test_probabilities.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "reviewer"
    / "analysis"
    / "calibration"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METRICS_FILE = OUTPUT_DIR / "calibration_metrics.csv"
BINS_FILE = OUTPUT_DIR / "reliability_bins.csv"
FIGURE_FILE = OUTPUT_DIR / "reliability_diagram.png"
SUMMARY_FILE = OUTPUT_DIR / "calibration_summary.md"


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

N_BINS = 15

# Standard equal-width top-label ECE bins.
BIN_EDGES = np.linspace(
    0.0,
    1.0,
    N_BINS + 1,
)


# ------------------------------------------------------------
# SHA-256 helper
# ------------------------------------------------------------

def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as file:

        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


# ------------------------------------------------------------
# Load frozen probability archive
# ------------------------------------------------------------

print("=" * 80)
print("AQUA20 REVIEWER CALIBRATION ANALYSIS")
print("=" * 80)

print()
print("Model  : F06")
print("Method : ConvNeXt-Small HR384 + Gentle CB Focal")
print("Mode   : frozen official-test probability analysis")
print()

assert SOURCE_FILE.exists(), (
    f"Source probability file not found:\n{SOURCE_FILE}"
)

source_sha256 = sha256_file(SOURCE_FILE)

data = np.load(
    SOURCE_FILE,
    allow_pickle=False,
)

required_keys = {
    "targets",
    "f06_probabilities",
    "class_names",
}

missing = required_keys - set(data.files)

assert not missing, (
    f"Missing required NPZ keys: {sorted(missing)}"
)

targets = np.asarray(
    data["targets"],
    dtype=np.int64,
)

probabilities = np.asarray(
    data["f06_probabilities"],
    dtype=np.float64,
)

class_names = np.asarray(
    data["class_names"],
    dtype=str,
)


# ------------------------------------------------------------
# Scientific input verification
# ------------------------------------------------------------

N_SAMPLES = len(targets)
N_CLASSES = len(class_names)

assert targets.shape == (1612,)
assert probabilities.shape == (1612, 20)
assert class_names.shape == (20,)

assert targets.min() >= 0
assert targets.max() < N_CLASSES

assert np.isfinite(probabilities).all()
assert (probabilities >= 0).all()
assert (probabilities <= 1).all()

assert np.allclose(
    probabilities.sum(axis=1),
    1.0,
    atol=1e-5,
)

predictions = probabilities.argmax(axis=1)

confidences = probabilities.max(axis=1)

correct = (
    predictions == targets
).astype(np.float64)

top1_accuracy = float(
    correct.mean()
)

# Frozen F06 official-test Top-1 verification.
assert np.isclose(
    top1_accuracy * 100,
    92.1216,
    atol=0.0001,
), (
    "F06 Top-1 does not match frozen official result."
)


# ------------------------------------------------------------
# 1. Expected Calibration Error
#
# Top-label ECE:
#
# ECE = sum_b (n_b / N) * |acc_b - conf_b|
#
# Equal-width bins over [0, 1].
# ------------------------------------------------------------

bin_indices = np.digitize(
    confidences,
    BIN_EDGES[1:-1],
    right=False,
)

reliability_rows = []

ece = 0.0

for bin_idx in range(N_BINS):

    lower = BIN_EDGES[bin_idx]
    upper = BIN_EDGES[bin_idx + 1]

    mask = (
        bin_indices == bin_idx
    )

    count = int(mask.sum())

    if count > 0:

        mean_confidence = float(
            confidences[mask].mean()
        )

        empirical_accuracy = float(
            correct[mask].mean()
        )

        calibration_gap = abs(
            empirical_accuracy
            - mean_confidence
        )

        weight = count / N_SAMPLES

        contribution = (
            weight * calibration_gap
        )

        ece += contribution

    else:

        mean_confidence = np.nan
        empirical_accuracy = np.nan
        calibration_gap = np.nan
        weight = 0.0
        contribution = 0.0

    reliability_rows.append({
        "bin": bin_idx + 1,
        "lower_bound": lower,
        "upper_bound": upper,
        "count": count,
        "sample_fraction": weight,
        "mean_confidence": mean_confidence,
        "empirical_accuracy": empirical_accuracy,
        "absolute_gap": calibration_gap,
        "ece_contribution": contribution,
    })

ece = float(ece)


# ------------------------------------------------------------
# 2. Negative Log-Likelihood
#
# NLL = -mean(log(p_true))
# ------------------------------------------------------------

true_class_probabilities = probabilities[
    np.arange(N_SAMPLES),
    targets,
]

# Numerical safeguard only.
# No probability modification/tuning is performed.
EPS = np.finfo(np.float64).eps

nll = float(
    -np.mean(
        np.log(
            np.clip(
                true_class_probabilities,
                EPS,
                1.0,
            )
        )
    )
)


# ------------------------------------------------------------
# 3. Multiclass Brier Score
#
# Standard multiclass form:
#
# mean_i sum_k (p_ik - y_ik)^2
#
# Not divided by number of classes.
# ------------------------------------------------------------

one_hot_targets = np.eye(
    N_CLASSES,
    dtype=np.float64,
)[targets]

brier_score = float(
    np.mean(
        np.sum(
            (
                probabilities
                - one_hot_targets
            ) ** 2,
            axis=1,
        )
    )
)


# ------------------------------------------------------------
# Additional descriptive diagnostics
# ------------------------------------------------------------

mean_confidence = float(
    confidences.mean()
)

global_confidence_gap = float(
    mean_confidence
    - top1_accuracy
)


# ------------------------------------------------------------
# Save calibration metrics
# ------------------------------------------------------------

metrics = pd.DataFrame([
    {
        "model": "F06",
        "method": (
            "ConvNeXt-Small HR384 "
            "+ Gentle CB Focal"
        ),
        "evaluation_split": (
            "AQUA20 official test"
        ),
        "test_samples": N_SAMPLES,
        "num_classes": N_CLASSES,
        "ece_bins": N_BINS,
        "top1_accuracy": top1_accuracy,
        "mean_top1_confidence": (
            mean_confidence
        ),
        "confidence_minus_accuracy": (
            global_confidence_gap
        ),
        "ece": ece,
        "nll": nll,
        "multiclass_brier_score": (
            brier_score
        ),
        "source_probability_file": (
            str(SOURCE_FILE)
        ),
        "source_sha256": source_sha256,
    }
])

metrics.to_csv(
    METRICS_FILE,
    index=False,
)


# ------------------------------------------------------------
# Save reliability-bin table
# ------------------------------------------------------------

reliability_df = pd.DataFrame(
    reliability_rows
)

reliability_df.to_csv(
    BINS_FILE,
    index=False,
)

assert (
    reliability_df["count"].sum()
    == N_SAMPLES
)

assert np.isclose(
    reliability_df[
        "ece_contribution"
    ].sum(),
    ece,
)


# ------------------------------------------------------------
# Reliability diagram
# ------------------------------------------------------------

plot_df = reliability_df[
    reliability_df["count"] > 0
].copy()

plt.figure(
    figsize=(7.5, 7.0)
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Perfect calibration",
)

plt.plot(
    plot_df["mean_confidence"],
    plot_df["empirical_accuracy"],
    marker="o",
    linewidth=2,
    label="F06",
)

plt.xlabel(
    "Mean predicted confidence"
)

plt.ylabel(
    "Empirical accuracy"
)

plt.title(
    "F06 Reliability Diagram — AQUA20 Official Test"
)

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------
# Save scientific summary
# ------------------------------------------------------------

summary = f"""# F06 Calibration Analysis

## Scope

Model:

`F06 — ConvNeXt-Small HR384 + Gentle Class-Balanced Focal Loss`

Evaluation set:

`AQUA20 official unchanged test split`

Samples:

`{N_SAMPLES}`

Classes:

`{N_CLASSES}`

No training, checkpoint selection, probability tuning, or post-hoc
calibration was performed. The analysis uses the frozen F06 probability
outputs preserved during the completed F08 inference workflow.

## Input verification

Recomputed Top-1 accuracy:

`{top1_accuracy * 100:.4f}%`

Frozen reported F06 Top-1:

`92.1216%`

Status:

`MATCHED`

Source archive SHA-256:

`{source_sha256}`

## Calibration metrics

Expected Calibration Error (ECE; {N_BINS} equal-width bins):

`{ece:.6f}`

Negative Log-Likelihood (NLL):

`{nll:.6f}`

Multiclass Brier Score:

`{brier_score:.6f}`

Mean Top-1 confidence:

`{mean_confidence:.6f}`

Top-1 accuracy:

`{top1_accuracy:.6f}`

Mean confidence minus accuracy:

`{global_confidence_gap:+.6f}`

## Metric definitions

ECE is the standard top-label expected calibration error using
{N_BINS} equal-width confidence bins over [0, 1].

NLL is the mean negative logarithm of the predicted probability assigned
to the true class.

The multiclass Brier Score is calculated as:

`mean_i sum_k (p_ik - y_ik)^2`

and is not divided by the number of classes.

## Interpretation rule

These values describe the probability calibration of the frozen F06
model on the official test set.

They must not be interpreted as training-seed variability, bootstrap
uncertainty, or evidence from a fitted post-hoc calibration method.

The official test set was used only for final descriptive calibration
evaluation and was not used to tune the model or calibration parameters.
"""

SUMMARY_FILE.write_text(
    summary,
    encoding="utf-8",
)


# ------------------------------------------------------------
# Console report
# ------------------------------------------------------------

print("=" * 80)
print("INPUT VERIFICATION")
print("=" * 80)

print(
    f"Samples                    : {N_SAMPLES}"
)

print(
    f"Classes                    : {N_CLASSES}"
)

print(
    "Probability shape          :",
    probabilities.shape,
)

print(
    f"Recomputed Top-1           : "
    f"{top1_accuracy * 100:.4f}%"
)

print(
    "Frozen Top-1               : "
    "92.1216%"
)

print(
    "Top-1 status               : MATCHED"
)

print()

print("=" * 80)
print("F06 CALIBRATION RESULTS")
print("=" * 80)

print(
    f"ECE ({N_BINS} bins)             : "
    f"{ece:.6f}"
)

print(
    f"NLL                        : "
    f"{nll:.6f}"
)

print(
    f"Multiclass Brier Score     : "
    f"{brier_score:.6f}"
)

print(
    f"Mean Top-1 confidence      : "
    f"{mean_confidence:.6f}"
)

print(
    f"Top-1 accuracy             : "
    f"{top1_accuracy:.6f}"
)

print(
    f"Confidence - Accuracy      : "
    f"{global_confidence_gap:+.6f}"
)

print()

print("=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print("Metrics      :", METRICS_FILE)
print("Bins         :", BINS_FILE)
print("Diagram      :", FIGURE_FILE)
print("Summary      :", SUMMARY_FILE)

print()

print("=" * 80)
print("CALIBRATION ANALYSIS COMPLETE")
print("=" * 80)

print("Training             : NONE")
print("New inference        : NONE")
print("Calibration tuning   : NONE")
print("Official test        : DESCRIPTIVE EVALUATION ONLY")
