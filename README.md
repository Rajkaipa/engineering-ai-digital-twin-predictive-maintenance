# Engineering AI Digital Twin — Fleet Predictive Maintenance

Predicting imminent failure of a critical engine component across a fleet of 23,550
heavy-duty trucks, and turning those predictions into an inspection policy a workshop
can actually run.

**Result: 32.8% maintenance cost reduction on held-out test data, catching 97% of
imminent failures.** The operating point was selected on validation data and applied
unchanged to test.

---

## The finding

Four independent optimisation avenues each moved cost by under 4%. Three changes to
the *decision* moved it by double digits.

| Change | Effect on cost |
|---|---|
| Algorithm choice (LightGBM / XGBoost / CatBoost) | < 0.5% |
| Hyperparameter search (58 configurations) | ~2% |
| Engineered temporal features (17 added) | **worse by 3%** |
| Ensembling | **worse by 4%** |
| **Cost-sensitive decision layer** | **−60%** |
| **Matching training to evaluation distribution** | **−54%** |
| **Operating-point selection** | **−16%** |

Without the decision layer, a tuned gradient-boosting model performs almost identically
to predicting "healthy" for every vehicle. Everything that mattered was about the
decision, not the predictor.

## The problem

Component failures are expensive asymmetrically: an unnecessary workshop inspection
costs 10 units; a component failing on the road costs 500. That 50:1 ratio, not
accuracy, is what a maintenance decision must optimise.

The data is heavily imbalanced — 2,272 failures among 23,550 vehicles (9.6%), with
only 0.3% of sensor readouts falling in the imminent-failure window.

## Approach

**Labels from time-to-event records.** Vehicles that never failed are *censored*, not
healthy — we know only that they survived to the end of observation. Treating this
correctly is the difference between a survival problem and a mislabelled
classification problem.

**Features that separate usage from age.** Cumulative counters mostly encode how old a
truck is. Dividing by elapsed time yields *usage intensity*; row-normalising the sensor
histograms yields operating profiles comparable across vehicles regardless of
accumulated mileage — the same transformation as a normalised load collective in
durability engineering.

**Calibrated probabilities, then one cost decision.** An early experiment showed that
applying class weighting *and* a cost-based decision rule double-counts the same
asymmetry: predicted failure probability came out 19× the true rate, and cost rose 48%.
Training unweighted restores calibration; the cost matrix is then applied exactly once,
at decision time.

**Training on the evaluation population.** The model is scored on each vehicle's most
recent readout, but was initially trained on all 1.12M historical readouts. Training
only on last readouts — discarding 98% of the data — halved the cost, because it raised
imminent-failure density from 0.3% to 8.5% and matched the distribution actually being
scored.

## What the model actually learned

Feature attribution and a Cox proportional-hazards model agree: failure hazard is driven
by **vehicle configuration** (one specification category carries 1.69× the baseline
hazard; another is protective at 0.63×) and by **usage intensity** (hazard ratios ~1.20,
p < 0.0001).

Notably, raw cumulative counters show *reduced* hazard once conditioned on age —
survivorship, correctly modelled. Classification obscured this effect entirely; only the
survival framing recovered it.

## Honest limitations

The model ranks risk roughly 3–4× better than random, but cannot isolate a small
high-risk subset — the recall-versus-inspection-budget curve is close to linear with no
knee. Achieving 90%+ recall requires inspecting over half the fleet.

The five-class formulation collapses to binary in practice. Under this cost structure,
over-predicting costs 7–10 regardless of how far you overshoot, so the model learned
"healthy or act now" and never predicts the intermediate windows. It detects that a
vehicle is degrading; it does not resolve *when* within the 48-unit horizon.

With these anonymised sensors, the achievable signal supports **population-level risk
stratification rather than per-vehicle condition monitoring**.

## Operating points

The dashboard exposes inspection capacity as a control, because the right operating
point is a business decision, not a modelling one:

| Inspection budget | Imminent failures caught | Test cost |
|---|---|---|
| 30% | 67% | 34,721 |
| 40% | 75% | 34,906 |
| 55% (published operating point) | 97% | 37,694 |
| Do nothing | 0% | 56,100 |

## Reproducing

```bash
pip install -r requirements.txt
python -m src.train             # trains and selects the operating point
python -m src.build_artifacts   # scores the test fleet, writes dashboard artifacts
streamlit run app/dashboard.py
```

Place the dataset CSVs in `data/` first. Tests: `pytest`.

## Repository

- `src/` — feature pipeline, training, evaluation, artifact generation
- `notebooks/` — exploratory analysis, model comparison, survival analysis
- `app/` — Streamlit dashboard and precomputed artifacts
- `tests/` — 17 tests covering labelling, feature logic, cost metric, artifacts

## Data source

Built on a publicly available multivariate time-series dataset of operational readouts
from a heavy-duty truck fleet, released under CC BY 4.0.
Kharazian et al., *Scientific Data* 12:493 (2025), https://doi.org/10.5878/jvb5-d390
