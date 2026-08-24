"""Tests for the feature pipeline — the logic that would silently corrupt results."""
import pandas as pd
import pytest

from src.features import (
    add_counter_features,
    add_histogram_shares,
    align_columns,
    assign_class_labels,
    last_readout_per_vehicle,
)


def test_counter_rate_normalises_by_elapsed_time():
    """Sampling is uneven, so rate must divide by dt, not by row count."""
    df = pd.DataFrame({
        "vehicle_id": [1, 1, 1],
        "time_step": [0.0, 2.0, 12.0],
        "171_0": [0.0, 100.0, 600.0],
        **{c: [0.0, 0.0, 0.0] for c in
           ["666_0", "427_0", "837_0", "309_0", "835_0", "370_0", "100_0"]},
    })
    out = add_counter_features(df)
    assert out["171_0_rate"].iloc[1] == pytest.approx(50.0)   # 100 / 2
    assert out["171_0_rate"].iloc[2] == pytest.approx(50.0)   # 500 / 10


def test_counter_reset_is_flagged_and_clipped():
    """A decreasing cumulative counter is an ECU reset, not negative usage."""
    df = pd.DataFrame({
        "vehicle_id": [1, 1],
        "time_step": [0.0, 1.0],
        "171_0": [500.0, 10.0],
        **{c: [0.0, 0.0] for c in
           ["666_0", "427_0", "837_0", "309_0", "835_0", "370_0", "100_0"]},
    })
    out = add_counter_features(df)
    assert out["171_0_reset"].iloc[1] == 1
    assert out["171_0_rate"].iloc[1] >= 0


def test_vehicles_do_not_contaminate_each_other():
    """groupby must scope diffs within a vehicle."""
    df = pd.DataFrame({
        "vehicle_id": [1, 2],
        "time_step": [10.0, 10.0],
        "171_0": [1000.0, 5.0],
        **{c: [0.0, 0.0] for c in
           ["666_0", "427_0", "837_0", "309_0", "835_0", "370_0", "100_0"]},
    })
    out = add_counter_features(df).sort_values("vehicle_id")
    assert (out["171_0_rate"] == 0).all()      # first readout each -> no rate


def test_histogram_shares_sum_to_one():
    df = pd.DataFrame({"vehicle_id": [1], "time_step": [0.0],
                       **{f"167_{i}": [float(i + 1)] for i in range(10)}})
    out = add_histogram_shares(df)
    cols = [f"167_{i}_share" for i in range(10)]
    assert out[cols].sum(axis=1).iloc[0] == pytest.approx(1.0)


def test_empty_histogram_does_not_divide_by_zero():
    df = pd.DataFrame({"vehicle_id": [1], "time_step": [0.0],
                       **{f"167_{i}": [0.0] for i in range(10)}})
    out = add_histogram_shares(df)
    cols = [f"167_{i}_share" for i in range(10)]
    assert out[cols].sum(axis=1).iloc[0] == 0.0
    assert not out[cols].isna().any().any()


def test_class_labels_follow_official_windows():
    df = pd.DataFrame({
        "length_of_study_time_step": [100, 100, 100, 100, 100, 100],
        "time_step":                 [ 10,  60,  80,  90,  97,  50],
        "in_study_repair":           [  1,   1,   1,   1,   1,   0],
    })
    # ttf =                            90,  40,  20,  10,   3,  (censored)
    assert list(assign_class_labels(df)) == [0, 1, 2, 3, 4, 0]


def test_censored_vehicles_are_always_class_zero():
    """We know only that they survived to the end of observation."""
    df = pd.DataFrame({"length_of_study_time_step": [100, 100],
                       "time_step": [1, 99], "in_study_repair": [0, 0]})
    assert list(assign_class_labels(df)) == [0, 0]


def test_align_columns_fills_missing_categories():
    """Test data may lack spec categories seen in training."""
    df = pd.DataFrame({"a": [1.0]})
    out = align_columns(df, ["a", "Spec_9=Cat99"])
    assert list(out.columns) == ["a", "Spec_9=Cat99"]
    assert out["Spec_9=Cat99"].iloc[0] == 0


def test_last_readout_picks_latest_per_vehicle():
    df = pd.DataFrame({"vehicle_id": [1, 1, 2], "time_step": [5.0, 50.0, 7.0]})
    out = last_readout_per_vehicle(df).sort_values("vehicle_id")
    assert list(out["time_step"]) == [50.0, 7.0]