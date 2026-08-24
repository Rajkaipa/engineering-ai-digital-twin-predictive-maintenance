"""Train the imminent-failure model end to end from raw CSVs."""
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .config import (DATA_DIR, MODEL_DIR, COUNTER_COLS, RATE_COLS, RESET_COLS,
                     SHARE_COLS, RANDOM_SEED, VAL_FRACTION)
from .features import build_features, last_readout_per_vehicle
from .evaluate import total_cost, budget_curve

PARAMS = dict(
    objective="multi:softprob", num_class=5,
    max_depth=3, min_child_weight=50, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    n_estimators=2000, early_stopping_rounds=100,
    eval_metric="mlogloss", tree_method="hist",
    random_state=RANDOM_SEED, n_jobs=-1,
)
TARGET_RECALL = 0.90


def split_by_vehicle(df):
    """Split on vehicles, never on rows: readouts within a vehicle are correlated."""
    rng = np.random.default_rng(RANDOM_SEED)
    vehicles = df["vehicle_id"].unique()
    rng.shuffle(vehicles)
    val_ids = set(vehicles[:int(VAL_FRACTION * len(vehicles))])
    is_val = df["vehicle_id"].isin(val_ids)
    return df[~is_val], df[is_val]


def main():
    ops  = pd.read_csv(DATA_DIR / "train_operational_readouts.csv")
    spec = pd.read_csv(DATA_DIR / "train_specifications.csv")
    tte  = pd.read_csv(DATA_DIR / "train_tte.csv")

    df = build_features(ops, spec, tte)
    del ops

    spec_cols = [c for c in df.columns if c.startswith("Spec_")]
    features = COUNTER_COLS + RATE_COLS + RESET_COLS + SHARE_COLS + spec_cols + ["time_step"]

    train_df, val_df = split_by_vehicle(df)

    # Train on last readouts only: matches the evaluation population and
    # raises class-4 density from 0.3% to ~8.5%.
    tr = last_readout_per_vehicle(train_df)
    va = last_readout_per_vehicle(val_df)
    X_tr, y_tr = tr[features], tr["class_label"].values
    X_va, y_va = va[features], va["class_label"].values
    print(f"train {X_tr.shape} | val {X_va.shape}")

    model = xgb.XGBClassifier(**PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    proba = model.predict_proba(X_va)
    curve = budget_curve(proba[:, 4], y_va, budgets=np.arange(0.05, 0.95, 0.05))

    feasible = curve[curve["recall_class4"] >= TARGET_RECALL]
    chosen = feasible.sort_values("alert_rate").iloc[0]

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "xgb_final.pkl")
    with open(MODEL_DIR / "operating_point.json", "w") as f:
        json.dump({
            "threshold": float(chosen["threshold"]),
            "target_recall": TARGET_RECALL,
            "validation": {k: float(chosen[k]) for k in
                           ["alert_rate", "recall_class4", "total_cost"]},
            "features": features,
            "params": {k: v for k, v in PARAMS.items()},
        }, f, indent=2)

    print(curve.round(3).to_string(index=False))
    print(f"\nChosen operating point: alert_rate={chosen['alert_rate']:.0%}, "
          f"recall={chosen['recall_class4']:.3f}, threshold={chosen['threshold']:.6f}")


if __name__ == "__main__":
    main()