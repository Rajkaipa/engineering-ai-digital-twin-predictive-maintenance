"""Score the test fleet and emit a compact artifact for the dashboard."""
import json
import joblib
import numpy as np
import pandas as pd

from .config import DATA_DIR, MODEL_DIR, ARTIFACT_DIR
from .features import build_features, last_readout_per_vehicle, align_columns
from .evaluate import total_cost, threshold_decision


def main():
    model = joblib.load(MODEL_DIR / "xgb_final.pkl")
    op = json.load(open(MODEL_DIR / "operating_point.json"))
    features, threshold = op["features"], op["threshold"]

    ops  = pd.read_csv(DATA_DIR / "test_operational_readouts.csv")
    spec = pd.read_csv(DATA_DIR / "test_specifications.csv")
    labels = pd.read_csv(DATA_DIR / "test_labels.csv")

    df = last_readout_per_vehicle(build_features(ops, spec))
    del ops

    X = align_columns(df, features)
    proba = model.predict_proba(X)
    risk = proba[:, 4]

    out = pd.DataFrame({
        "vehicle_id": df["vehicle_id"].values,
        "age": df["time_step"].values,
        "risk": risk,
        "health": (100 * (1 - risk)).round(1),
        "flagged": risk >= threshold,
    })
    for c in [c for c in df.columns if c.startswith("Spec_")]:
        pass  # specs are one-hot; store the readable originals instead
    out = out.merge(spec, on="vehicle_id", how="left").merge(labels, on="vehicle_id", how="left")

    out["status"] = pd.cut(out["risk"], bins=[-0.01, threshold * 0.5, threshold, 1.0],
                           labels=["Healthy", "Monitor", "Action required"])

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
        "threshold": threshold,
    }
    json.dump(summary, open(ARTIFACT_DIR / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()