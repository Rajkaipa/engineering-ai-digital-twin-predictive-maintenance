"""Regression guards on the shipped dashboard artifacts."""
import json

import pandas as pd
import pytest

from src.config import ARTIFACT_DIR

pytestmark = pytest.mark.skipif(
    not (ARTIFACT_DIR / "summary.json").exists(),
    reason="artifacts not built",
)


@pytest.fixture(scope="module")
def artifacts():
    fleet = pd.read_parquet(ARTIFACT_DIR / "fleet_scores.parquet")
    summary = json.load(open(ARTIFACT_DIR / "summary.json"))
    return fleet, summary


def test_row_count_matches_summary(artifacts):
    fleet, summary = artifacts
    assert len(fleet) == summary["n_vehicles"]


def test_risk_is_a_valid_probability(artifacts):
    fleet, _ = artifacts
    assert fleet["risk"].between(0, 1).all()


def test_model_beats_do_nothing_baseline(artifacts):
    """Fails the build if a change ever makes the model worse than no action."""
    _, summary = artifacts
    assert summary["model_cost"] < summary["baseline_cost"]


def test_every_vehicle_has_an_explanation(artifacts):
    fleet, _ = artifacts
    assert fleet["drivers"].notna().all()
    assert (fleet["drivers"].str.len() > 0).all()