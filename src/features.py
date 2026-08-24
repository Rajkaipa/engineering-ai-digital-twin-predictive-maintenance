"""Feature engineering: counter rates, histogram shares, spec encoding, class labels."""
import numpy as np
import pandas as pd

from .config import CLASS_WINDOWS, COUNTER_COLS, HISTOGRAMS, RATE_COLS


def add_counter_features(df: pd.DataFrame) -> pd.DataFrame:
    """Usage rate per time unit, with ECU counter-reset detection.

    Cumulative counters cannot decrease; a negative delta indicates an ECU
    reset that survived the vendor's post-processing. We clip it to zero and
    flag it, since a reset often coincides with a workshop intervention.
    """
    df = df.sort_values(["vehicle_id", "time_step"]).reset_index(drop=True)
    g = df.groupby("vehicle_id")
    df["dt"] = g["time_step"].diff()

    new = {}
    for c in COUNTER_COLS:
        delta = g[c].diff()
        new[f"{c}_reset"] = (delta < 0).astype(int)
        new[f"{c}_rate"] = delta.clip(lower=0) / df["dt"]

    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    df[RATE_COLS] = df[RATE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def add_histogram_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Row-normalise each histogram into a distribution.

    Raw bin counts grow with vehicle age; the *shape* of the distribution
    describes operating conditions independently of accumulated usage.
    Equivalent to a normalised load collective in durability engineering.
    """
    new = {}
    for var, n_bins in HISTOGRAMS.items():
        cols = [f"{var}_{i}" for i in range(n_bins) if f"{var}_{i}" in df.columns]
        total = df[cols].sum(axis=1).replace(0, np.nan)
        shares = df[cols].div(total, axis=0).fillna(0)
        for c in cols:
            new[f"{c}_share"] = shares[c]
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def encode_specifications(spec: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the categorical vehicle specifications."""
    cat_cols = [c for c in spec.columns if c != "vehicle_id"]
    return pd.get_dummies(spec, columns=cat_cols, prefix_sep="=")


def assign_class_labels(df: pd.DataFrame) -> pd.Series:
    """Map time-to-failure onto the five official classes.

    Censored vehicles (never repaired during the study) are class 0 at every
    readout: we know only that they survived to the end of observation.
    """
    ttf = df["length_of_study_time_step"] - df["time_step"]
    w = CLASS_WINDOWS
    return pd.Series(
        np.select(
            [df["in_study_repair"] == 0, ttf > w[0], ttf > w[1], ttf > w[2], ttf > w[3]],
            [0, 0, 1, 2, 3],
            default=4,
        ),
        index=df.index,
        name="class_label",
    )


def build_features(ops, spec, tte=None):
    """Full pipeline. Pass tte for training data; omit it for scoring."""
    df = add_counter_features(ops)
    df = add_histogram_shares(df)
    if tte is not None:
        df = df.merge(tte, on="vehicle_id", how="left")
        df["class_label"] = assign_class_labels(df)
    df = df.merge(encode_specifications(spec), on="vehicle_id", how="left")
    return df.copy()


def align_columns(df, expected):
    """Add missing one-hot columns as zeros so scoring matches training schema."""
    missing = [c for c in expected if c not in df.columns]
    if missing:
        df = pd.concat([df, pd.DataFrame(0, index=df.index, columns=missing)], axis=1)
    return df[expected]


def last_readout_per_vehicle(df):
    """Evaluation and scoring happen at each vehicle's most recent readout."""
    return df.sort_values("time_step").groupby("vehicle_id").tail(1)