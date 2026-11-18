"""Create exploratory plots for the PathSnap trip dataset.

Run from the repository root:
    python deliverable1/eda.py

Outputs are written to artifacts/eda/.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "new" / "dataset"
OUT = ROOT / "artifacts" / "eda"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")


def read_trip(part, trip):
    d = DATA / part / trip
    track = pd.read_csv(d / f"{trip}.track", sep=r"\s+", header=None,
                        names=["lon", "lat", "timestamp"])
    nodes = pd.read_csv(d / f"{trip}.nodes", sep=r"\s+", header=None,
                        names=["lon", "lat"])
    arcs = pd.read_csv(d / f"{trip}.arcs", sep=r"\s+", header=None,
                       names=["from_node", "to_node"])
    route = pd.read_csv(d / f"{trip}.route", sep=r"\s+", header=None,
                        names=["arc_id"])
    return track, nodes, arcs, route


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    rows, tracks, samples = [], [], []
    for part in ["train", "student_test", "evaluator_test"]:
        trips = sorted(p.name for p in (DATA / part).iterdir() if p.is_dir())
        for trip in trips:
            track, nodes, arcs, route = read_trip(part, trip)
            track["timestamp"] = pd.to_numeric(track.timestamp, errors="coerce")
            valid = track.dropna(subset=["lon", "lat", "timestamp"])
            dt = valid.timestamp.diff()
            dlon = valid.lon.diff()
            dlat = valid.lat.diff()
            step_m = np.sqrt((dlon * 111320 * np.cos(np.radians(valid.lat))) ** 2 +
                             (dlat * 111320) ** 2)
            speed = (step_m / dt.replace(0, np.nan)).clip(0, 100)
            rows.append({
                "partition": part, "trip_id": trip,
                "gps_observations": len(track), "valid_observations": len(valid),
                "nodes": len(nodes), "arcs": len(arcs), "route_arcs": len(route),
                "duration_s": valid.timestamp.max() - valid.timestamp.min() if len(valid) else 0,
                "mean_speed_mps": speed.mean(), "max_speed_mps": speed.max(),
                "gaps_gt_1s": int((dt > 1).sum()),
            })
            tracks.append(pd.DataFrame({"partition": part, "trip_id": trip,
                                        "speed_mps": speed.dropna()}))
            if len(samples) < 6 and len(valid):
                samples.append((part, trip, valid.copy()))

    inv = pd.DataFrame(rows)
    motion = pd.concat(tracks, ignore_index=True)
    inv.to_csv(OUT / "eda_trip_summary.csv", index=False)

    # 1. Dataset inventory by partition.
    fig, ax = plt.subplots(figsize=(11, 6))
    long = inv.melt(id_vars="partition", value_vars=["gps_observations", "nodes", "arcs", "route_arcs"],
                    var_name="measure", value_name="count")
    sns.boxplot(data=long, x="measure", y="count", hue="partition", ax=ax, showfliers=False)
    ax.set_yscale("log"); ax.set_title("Trip structure distributions (log scale)")
    ax.set_xlabel(""); ax.set_ylabel("Count")
    save(fig, "01_trip_structure.png")

    # 2. GPS observations per trip.
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.histplot(data=inv, x="gps_observations", hue="partition", bins=25, element="step", ax=ax)
    ax.set_title("GPS observations per trip"); ax.set_xlabel("Observations")
    save(fig, "02_observations_per_trip.png")

    # 3. Motion distributions.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(data=motion, x="speed_mps", hue="partition", bins=50, stat="density",
                 common_norm=False, element="step", ax=axes[0])
    axes[0].set_title("GPS-derived speed"); axes[0].set_xlabel("Speed (m/s)")
    sns.histplot(data=inv, x="duration_s", hue="partition", bins=25, element="step", ax=axes[1])
    axes[1].set_title("Trip duration"); axes[1].set_xlabel("Duration (s)")
    save(fig, "03_motion_distributions.png")

    # 4. Missingness / time gaps.
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.scatterplot(data=inv, x="gps_observations", y="gaps_gt_1s", hue="partition", s=70, ax=ax)
    ax.set_title("Sampling gaps by trip"); ax.set_xlabel("GPS observations"); ax.set_ylabel("Gaps > 1 second")
    save(fig, "04_sampling_gaps.png")

    # 5. Representative GPS trajectories.
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, (part, trip, track) in zip(axes.flat, samples):
        ax.plot(track.lon, track.lat, linewidth=1.2)
        ax.scatter(track.lon.iloc[0], track.lat.iloc[0], c="green", s=45, label="start")
        ax.scatter(track.lon.iloc[-1], track.lat.iloc[-1], c="red", s=45, label="end")
        ax.set_title(f"{part}: {trip}"); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    for ax in axes.flat[len(samples):]: ax.axis("off")
    axes.flat[0].legend(loc="best")
    save(fig, "05_representative_trajectories.png")

    print(f"Wrote {len(inv)} trip summaries and plots to {OUT}")


if __name__ == "__main__":
    main()
