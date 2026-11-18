"""Create model-comparison and HMM summary charts from aggregate CSVs."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / "features"
OUT = ROOT / "eda"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def model_comparison():
    rows = read_csv(FEATURES / "group_cv_summary_aligned.csv")
    names = [r["model"] for r in rows]
    x = np.arange(len(rows))
    colors = ["#2563eb" if name == "Logistic Regression" else "#94a3b8" for name in names]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    charts = [
        ("observation_top1_mean", "observation_top1_std", "Observation Top-1", "Accuracy", (0.60, 0.85)),
        ("candidate_macro_f1_mean", "candidate_macro_f1_std", "Candidate macro F1", "Macro F1", (0.65, 0.90)),
        ("candidate_log_loss_mean", "candidate_log_loss_std", "Candidate log loss", "Log loss (lower is better)", None),
    ]
    for ax, (metric, spread, title, ylabel, ylim) in zip(axes, charts):
        values = np.asarray([float(r[metric]) for r in rows])
        errors = np.asarray([float(r[spread]) for r in rows])
        ax.bar(x, values, yerr=errors, capsize=3, color=colors, alpha=.9)
        ax.set_xticks(x, names, rotation=35, ha="right")
        ax.set_title(title, weight="bold")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, axis="y", alpha=.25)
    fig.legend(handles=[Patch(color="#2563eb", label="Selected: Logistic Regression"),
                        Patch(color="#94a3b8", label="Other models")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, .94))
    fig.suptitle("Five-fold trip-grouped model comparison", fontsize=15, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "08_model_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def hmm_comparison():
    rows = read_csv(FEATURES / "student_test_hmm_aggregate.csv")
    values = {r["metric"]: float(r["value"]) for r in rows}
    groups = [
        ("Weighted Top-1\n(all labeled points)", "classifier_weighted_top1_all_route_labeled", "hmm_weighted_top1_all_route_labeled"),
        ("Macro Top-1\n(mean by trip)", "classifier_macro_top1_by_trip_all_route_labeled", "hmm_macro_top1_by_trip_all_route_labeled"),
        ("Path continuity\n(decoded adjacent pairs)", "classifier_mean_continuity", "hmm_mean_continuity"),
    ]
    x = np.arange(len(groups))
    width = .34
    before = [100 * values[a] for _, a, _ in groups]
    after = [100 * values[b] for _, _, b in groups]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, before, width, label="Classifier only", color="#64748b")
    ax.bar(x + width / 2, after, width, label="Classifier + HMM", color="#0f9d8a")
    ax.set_xticks(x, [name for name, _, _ in groups])
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 105)
    ax.set_title("Student-test route decoding summary", weight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "09_hmm_effect.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model_comparison()
    hmm_comparison()
    print(f"Wrote aggregate model and HMM charts to {OUT}")


if __name__ == "__main__":
    main()

