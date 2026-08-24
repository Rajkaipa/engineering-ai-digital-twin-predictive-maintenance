"""Fleet Digital Twin — predictive maintenance decision support."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ART = Path(__file__).parent / "artifacts"
COST = np.array([[0,7,8,9,10],[200,0,7,8,9],[300,200,0,7,8],
                 [400,300,200,0,7],[500,400,300,200,0]])

st.set_page_config(page_title="Fleet Digital Twin", layout="wide")


@st.cache_data
def load():
    fleet = pd.read_parquet(ART / "fleet_scores.parquet")
    summary = json.load(open(ART / "summary.json"))
    return fleet, summary


fleet, summary = load()

st.title("Fleet Digital Twin — Component X")
st.caption("Predictive maintenance decision support for a heavy-duty truck fleet")

# ---------------- Sidebar: inspection capacity ----------------
st.sidebar.header("Inspection capacity")
budget = st.sidebar.slider(
    "Share of fleet to inspect", 0.05, 0.95,
    value=float(summary["flagged"] / summary["n_vehicles"]), step=0.05,
)
k = int(budget * len(fleet))
threshold = np.sort(fleet["risk"].values)[-k]
flagged = fleet["risk"] >= threshold

pred = np.where(flagged, 4, 0)
truth = fleet["class_label"].values
cm = pd.crosstab(truth, pred).reindex(index=range(5), columns=[0, 4], fill_value=0)
cost = int((pd.crosstab(truth, pred)
            .reindex(index=range(5), columns=range(5), fill_value=0).values * COST).sum())
caught = int(((truth == 4) & flagged).sum())
total4 = int((truth == 4).sum())

# ---------------- Headline metrics ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Trucks inspected", f"{k:,}", f"{budget:.0%} of fleet")
c2.metric("Imminent failures caught", f"{caught}/{total4}", f"{caught/total4:.0%} recall")
c3.metric("Maintenance cost", f"{cost:,}",
          f"{100*(cost/summary['baseline_cost']-1):+.1f}% vs no action")
c4.metric("Missed failures", f"{total4-caught}")

tab1, tab2 = st.tabs(["Fleet worklist", "Vehicle twin"])

# ---------------- Worklist ----------------
with tab1:
    st.subheader("Inspection worklist — highest risk first")
    work = (fleet[flagged].sort_values("risk", ascending=False)
                 [["vehicle_id", "health", "risk", "age", "status", "drivers"]])
    work.columns = ["Vehicle", "Health %", "Risk", "Age", "Status", "Primary drivers"]
    st.dataframe(work, use_container_width=True, hide_index=True, height=460)
    st.download_button("Download worklist (CSV)",
                       work.to_csv(index=False), "worklist.csv", "text/csv")

# ---------------- Single vehicle ----------------
with tab2:
    vid = st.selectbox("Select vehicle",
                       fleet.sort_values("risk", ascending=False)["vehicle_id"].head(200))
    v = fleet[fleet["vehicle_id"] == vid].iloc[0]

    a, b, c = st.columns(3)
    a.metric("Health score", f"{v['health']:.1f}%")
    b.metric("Failure risk", f"{v['risk']:.1%}")
    c.metric("Status", str(v["status"]))

    st.progress(min(max(float(v["health"]) / 100, 0.0), 1.0))

    st.markdown("**Why this vehicle is rated this way**")
    for d in str(v["drivers"]).split(" | "):
        st.write("• " + d)

    st.markdown("**Recommendation**")
    if v["risk"] >= threshold:
        st.error("Schedule inspection of Component X at next workshop visit.")
    else:
        st.success("No action required. Continue monitoring.")