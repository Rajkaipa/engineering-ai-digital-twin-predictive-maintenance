import numpy as np

from src.config import COST
from src.evaluate import expected_cost_decision, threshold_decision, total_cost


def test_perfect_prediction_costs_nothing():
    y = np.array([0, 1, 2, 3, 4])
    assert total_cost(y, y) == 0


def test_missed_failure_costs_more_than_false_alarm():
    """The 50:1 asymmetry is the whole basis of the decision layer."""
    assert COST[4, 0] == 500
    assert COST[0, 4] == 10
    assert COST[4, 0] > 40 * COST[0, 4]


def test_expected_cost_escalates_on_small_failure_probability():
    """Calibrated 10% risk of class 4 should still trigger action."""
    proba = np.array([[0.90, 0.0, 0.0, 0.0, 0.10]])
    assert expected_cost_decision(proba)[0] == 4


def test_threshold_decision_is_binary():
    out = threshold_decision(np.array([0.01, 0.5]), 0.1)
    assert list(out) == [0, 4]