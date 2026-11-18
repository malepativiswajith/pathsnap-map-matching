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
    colors = ["#2563eb" if name == "Extra Trees" else "#94a3b8" for name in names]
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
    fig.legend(handles=[Patch(color="#2563eb", label="Selected family: Extra Trees"),
                        Patch(color="#94a3b8", label="Other models")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, .92))
    fig.suptitle("Five-fold trip-grouped model comparison", fontsize=15, weight="bold", y=.98)
    fig.tight_layout(rect=[0, .12, 1, .84])
    fig.savefig(OUT / "08_model_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def hmm_comparison():
    rows = read_csv(FEATURES / "student_test_model_comparison_aligned.csv")
    selected = next(r for r in rows if r["model"] == "Extra Trees Bayesian tuned")
    groups = [
        ("Weighted Top-1\n(all labeled points)", "classifier_top1_weighted_all_labels", "hmm_top1_weighted_all_labels"),
        ("Macro Top-1\n(mean by trip)", "classifier_top1_macro_by_trip_all_labels", "hmm_top1_macro_by_trip_all_labels"),
        ("Path continuity\n(decoded adjacent pairs)", "classifier_mean_continuity", "hmm_mean_continuity"),
    ]
    x = np.arange(len(groups))
    width = .34
    before = [100 * float(selected[a]) for _, a, _ in groups]
    after = [100 * float(selected[b]) for _, _, b in groups]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, before, width, label="Classifier only", color="#64748b")
    ax.bar(x + width / 2, after, width, label="Classifier + HMM", color="#0f9d8a")
    ax.set_xticks(x, [name for name, _, _ in groups])
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 105)
    ax.set_title("Tuned Extra Trees: student-test HMM effect", weight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "09_hmm_effect.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def student_model_routes():
    rows = read_csv(FEATURES / "student_test_model_comparison_aligned.csv")
    names = [r["model"] for r in rows]
    x = np.arange(len(rows))
    width = .36
    classifier = np.asarray([100 * float(r["classifier_top1_weighted_all_labels"]) for r in rows])
    hmm = np.asarray([100 * float(r["hmm_top1_weighted_all_labels"]) for r in rows])
    hmm_colors = ["#2563eb" if name == "Extra Trees Bayesian tuned" else "#0f9d8a" for name in names]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, classifier, width, label="Classifier only", color="#94a3b8")
    ax.bar(x + width / 2, hmm, width, label="Classifier + HMM", color=hmm_colors)
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_ylabel("Weighted Top-1 across all route-labeled observations (%)")
    ax.set_ylim(65, 88)
    ax.set_title("Student-test route comparison (development split)", weight="bold")
    ax.legend(handles=[Patch(color="#94a3b8", label="Classifier only"),
                       Patch(color="#0f9d8a", label="Classifier + HMM"),
                       Patch(color="#2563eb", label="Selected: tuned Extra Trees + HMM")],
              frameon=False, ncol=3, loc="upper left")
    ax.grid(True, axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "10_student_model_routes.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def student_candidate_metrics():
    rows = read_csv(FEATURES / "student_test_model_comparison_aligned.csv")
    names = [r["model"] for r in rows]
    x = np.arange(len(rows))
    colors = ["#2563eb" if name == "Extra Trees Bayesian tuned" else
              "#0f9d8a" if name == "Hist Gradient Boosting" else "#94a3b8" for name in names]
    f1 = np.asarray([float(r["candidate_macro_f1"]) for r in rows])
    loss = np.asarray([float(r["candidate_log_loss"]) for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].bar(x, f1, color=colors)
    axes[0].set_ylabel("Macro F1 (higher is better)")
    axes[0].set_ylim(.70, .84)
    axes[0].set_title("Candidate classification F1", weight="bold")
    axes[1].bar(x, loss, color=colors)
    axes[1].set_ylabel("Log loss (lower is better)")
    axes[1].set_ylim(0, .70)
    axes[1].set_title("Candidate probability quality", weight="bold")
    for ax in axes:
        ax.set_xticks(x, names, rotation=35, ha="right")
        ax.grid(True, axis="y", alpha=.25)
    fig.legend(handles=[Patch(color="#2563eb", label="Selected route model: tuned Extra Trees"),
                        Patch(color="#0f9d8a", label="Best candidate metrics: Hist Gradient Boosting"),
                        Patch(color="#94a3b8", label="Other models")],
               frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, .98))
    fig.suptitle("Different metrics favor different models", fontsize=14, weight="bold", y=1.04)
    fig.tight_layout(rect=[0, 0, 1, .90])
    fig.savefig(OUT / "11_student_candidate_metrics.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model_comparison()
    hmm_comparison()
    student_model_routes()
    student_candidate_metrics()
    print(f"Wrote aggregate model and HMM charts to {OUT}")


if __name__ == "__main__":
    main()
