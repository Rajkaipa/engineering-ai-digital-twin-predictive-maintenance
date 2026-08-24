"""Score the test fleet and emit compact artifacts for the dashboard.

Reads the trained model and operating point, runs the full feature pipeline
on the held-out test set, and writes a lightweight parquet + summary JSON
that the Streamlit app consumes. Keeping SHAP here (rather than in the app)
means the deployed dashboard needs neither the model nor the raw data.
"""
import json

import joblib
import numpy as np
import pandas as pd
import shap

from .config import ARTIFACT_DIR, DATA_DIR, MODEL_DIR
from .evaluate import threshold_decision, total_cost
from .features import align_columns, build_features, last_readout_per_vehicle

TOP_N_DRIVERS = 3

FEATURE_LABELS = {
    "time_step": "Component age (operating hours)",
    "171_0": "Cumulative usage — sensor 171",
    "666_0": "Cumulative usage — sensor 666",
    "427_0": "Cumulative usage — sensor 427",
    "837_0": "Cumulative usage — sensor 837",
    "309_0": "Cumulative usage — sensor 309",
    "835_0": "Cumulative usage — sensor 835",
    "370_0": "Cumulative usage — sensor 370",
    "100_0": "Cumulative usage — sensor 100",
}

def humanise(name: str) -> str:
    """Map raw feature names to language a maintenance engineer can act on."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.endswith("_rate"):
        return f"Usage intensity — sensor {name.split('_')[0]}"
    if name.endswith("_reset"):
        return f"Control-unit reset detected — sensor {name.split('_')[0]}"
    if name.endswith("_share"):
        var, b = name.split("_")[0], name.split("_")[1]
        return f"Operating profile — sensor {var}, band {b}"
    if name.startswith("Spec_"):
        return f"Vehicle configuration ({name.replace('=', ' = ')})"
    return name

def compute_drivers(model, X, features, n=TOP_N_DRIVERS):
    """Top-n SHAP contributors to the class-4 (imminent failure) prediction.

    Returns a readable string per vehicle, e.g.
    "666_0 (+0.184) | Spec_7=Cat6 (+0.091) | 427_0_rate (-0.044)".
    A positive value means the feature pushed failure risk up.
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    sv4 = sv[4] if isinstance(sv, list) else sv[:, :, 4]

    order = np.argsort(-np.abs(sv4), axis=1)[:, :n]
    out = []
    for i, idx in enumerate(order):
        parts = [
            f"{humanise(features[j])} ({'raises' if sv4[i, j] > 0 else 'lowers'} risk)"
            for j in idx
        ]
        out.append(" | ".join(parts))
    return out


def main():
    model = joblib.load(MODEL_DIR / "xgb_final.pkl")
    op = json.load(open(MODEL_DIR / "operating_point.json"))
    features = op["features"]
    threshold = op["threshold"]

    ops = pd.read_csv(DATA_DIR / "test_operational_readouts.csv")
    spec = pd.read_csv(DATA_DIR / "test_specifications.csv")
    labels = pd.read_csv(DATA_DIR / "test_labels.csv")

    df = last_readout_per_vehicle(build_features(ops, spec))
    del ops

    X = align_columns(df, features)
    risk = model.predict_proba(X)[:, 4]

    print("Computing SHAP drivers...")
    drivers = compute_drivers(model, X, features)

    out = pd.DataFrame({
        "vehicle_id": df["vehicle_id"].values,
        "age": df["time_step"].values,
        "risk": risk,
        "health": (100 * (1 - risk)).round(1),
        "flagged": risk >= threshold,
        "drivers": drivers,
    })
    out = (out.merge(spec, on="vehicle_id", how="left")
              .merge(labels, on="vehicle_id", how="left"))

    out["status"] = pd.cut(
        out["risk"],
        bins=[-0.01, threshold * 0.5, threshold, 1.01],
        labels=["Healthy", "Monitor", "Action required"],
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(ARTIFACT_DIR / "fleet_scores.parquet", index=False)

    pred = threshold_decision(risk, threshold)
    baseline = total_cost(out["class_label"], np.zeros(len(out), dtype=int))
    achieved = total_cost(out["class_label"], pred)

    summary = {
        "n_vehicles": int(len(out)),
        "flagged": int(out["flagged"].sum()),
        "baseline_cost": baseline,
        "model_cost": achieved,
        "reduction_pct": round(100 * (1 - achieved / baseline), 1),
        "threshold": float(threshold),
        "class4_total": int((out["class_label"] == 4).sum()),
        "class4_caught": int(((out["class_label"] == 4) & out["flagged"]).sum()),
    }
    json.dump(summary, open(ARTIFACT_DIR / "summary.json", "w"), indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts written to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()