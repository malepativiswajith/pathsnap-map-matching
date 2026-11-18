"""Render a local student-test Folium map with GPS, network, reference and prediction layers.

Run hmm_decode.py first. Maps contain trip coordinates and must stay local.
"""
import argparse
import csv
import os
from pathlib import Path

import folium

ROOT = Path(__file__).resolve().parent


def read_rows(path, width):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            cols = line.split()
            if len(cols) != width:
                raise ValueError(f"{path}:{line_no}: expected {width} columns")
            rows.append(cols)
    return rows


def arc_features(ids, nodes, arcs):
    features = []
    for sequence, aid in enumerate(ids):
        if 0 <= aid < len(arcs):
            u, v = arcs[aid]
            features.append({
                "type": "Feature",
                "properties": {"arc_id": aid, "sequence": sequence + 1},
                "geometry": {"type": "LineString", "coordinates": [nodes[u], nodes[v]]},
            })
    return features


def build_map(title, track, nodes, arcs, reference, predicted=None):
    gps = [[p[1], p[0]] for p in track]
    center = [sum(p[0] for p in gps) / len(gps), sum(p[1] for p in gps) / len(gps)]
    fmap = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap", control_scale=True)

    gps_layer = folium.FeatureGroup(name="Noisy GPS observations", show=True)
    folium.PolyLine(gps, color="#2563eb", weight=3, opacity=.85, tooltip="Observed GPS trace").add_to(gps_layer)
    folium.CircleMarker(gps[0], radius=5, color="#16a34a", fill=True, tooltip="Trip start").add_to(gps_layer)
    folium.CircleMarker(gps[-1], radius=5, color="#7c3aed", fill=True, tooltip="Trip end").add_to(gps_layer)
    gps_layer.add_to(fmap)

    lon_values = [p[0] for p in track]
    lat_values = [p[1] for p in track]
    pad_lon, pad_lat = .005, .005
    bounds = (min(lon_values) - pad_lon, min(lat_values) - pad_lat,
              max(lon_values) + pad_lon, max(lat_values) + pad_lat)
    network_features = []
    for aid, (u, v) in enumerate(arcs):
        a, b = nodes[u], nodes[v]
        if (min(a[0], b[0]) <= bounds[2] and max(a[0], b[0]) >= bounds[0]
                and min(a[1], b[1]) <= bounds[3] and max(a[1], b[1]) >= bounds[1]):
            network_features.append({
                "type": "Feature", "properties": {"arc_id": aid},
                "geometry": {"type": "LineString", "coordinates": [a, b]},
            })
    network = folium.FeatureGroup(name="Road network near trip", show=False)
    if network_features:
        folium.GeoJson(
            {"type": "FeatureCollection", "features": network_features},
            style_function=lambda _f: {"color": "#64748b", "weight": 2, "opacity": .55},
            tooltip=folium.GeoJsonTooltip(fields=["arc_id"], aliases=["Road arc"]),
        ).add_to(network)
    network.add_to(fmap)

    ref_layer = folium.FeatureGroup(name="Reference route", show=False)
    folium.GeoJson(
        {"type": "FeatureCollection", "features": arc_features(reference, nodes, arcs)},
        style_function=lambda _f: {"color": "#16a34a", "weight": 5, "opacity": .85},
        tooltip=folium.GeoJsonTooltip(fields=["arc_id", "sequence"], aliases=["Reference arc", "Route order"]),
    ).add_to(ref_layer)
    ref_layer.add_to(fmap)

    if predicted:
        pred_layer = folium.FeatureGroup(name="Predicted route (HMM/Viterbi)", show=True)
        folium.GeoJson(
            {"type": "FeatureCollection", "features": arc_features(predicted, nodes, arcs)},
            style_function=lambda _f: {"color": "#dc2626", "weight": 5, "opacity": .9},
            tooltip=folium.GeoJsonTooltip(fields=["arc_id", "sequence"], aliases=["Predicted arc", "Route order"]),
        ).add_to(pred_layer)
        pred_layer.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    folium.map.Marker(
        center,
        icon=folium.DivIcon(html=(
            "<div style='font-size:13px;font-weight:700;color:#111827;background:white;"
            "padding:5px 8px;border:1px solid #cbd5e1;border-radius:5px'>" + title + "</div>"
        )),
    ).add_to(fmap)
    return fmap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trip-id", required=True, help="Trip ID from student_test")
    parser.add_argument("--data-root", default=os.getenv("PATHSNAP_DATA_ROOT", str(ROOT / "data")))
    parser.add_argument("--predictions", default=str(ROOT / "features" / "student_test_hmm_predictions_aligned.csv"))
    parser.add_argument("--output", help="HTML output; defaults to reports/maps/<trip-id>.html")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    partition = (data_root / "student_test").resolve()
    trip_dir = (partition / args.trip_id).resolve()
    if trip_dir.parent != partition or not trip_dir.is_dir():
        raise SystemExit(f"Unknown student-test trip: {args.trip_id}")
    track = [[float(x) for x in r] for r in read_rows(trip_dir / f"{args.trip_id}.track", 3)]
    nodes = [[float(x) for x in r] for r in read_rows(trip_dir / f"{args.trip_id}.nodes", 2)]
    arcs = [[int(x) for x in r] for r in read_rows(trip_dir / f"{args.trip_id}.arcs", 2)]
    reference = [int(r[0]) for r in read_rows(trip_dir / f"{args.trip_id}.route", 1)]

    observations = []
    with Path(args.predictions).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["trip_id"] == args.trip_id and row["hmm_arc_id"]:
                observations.append((int(row["obs_idx"]), int(row["hmm_arc_id"])))
    predicted = [aid for _, aid in sorted(observations)]
    if not predicted:
        raise SystemExit(f"No decoded route for {args.trip_id}; run hmm_decode.py first")

    fmap = build_map(f"PathSnap student-test: {args.trip_id}", track, nodes, arcs, reference, predicted)
    out = Path(args.output) if args.output else ROOT / "reports" / "maps" / f"student_test_{args.trip_id}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out))
    print(f"Saved local GPS/network/reference/prediction map to {out}")


if __name__ == "__main__":
    main()
