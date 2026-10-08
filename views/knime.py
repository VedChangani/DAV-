import base64

import streamlit as st

from src.knime import KNWF_PATH, load_workflow, scorer_metrics, workflow_svg
from src.ui import get_models, pct

nodes, edges = load_workflow()
metrics = scorer_metrics()

st.title("KNIME Workflow")
st.caption("Everything on this page is read live from the saved KNIME workflow files (`Assignment/workflow.knime` "
           "and each node's `settings.xml`) — not retyped by hand.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Nodes", len(nodes), border=True)
c2.metric("Connections", len(edges), border=True)
c3.metric("Decision-tree accuracy", pct(metrics.get("Accuracy", 0)), border=True)
kappa = metrics.get("Cohen's kappa", 0)
c4.metric("Cohen's κ", f"{kappa:.3f}", border=True)

st.subheader("Workflow canvas")
svg_b64 = base64.b64encode(workflow_svg().encode()).decode()
st.html(f'<div style="background:#fff;border-radius:8px;padding:8px;overflow-x:auto">'
        f'<img src="data:image/svg+xml;base64,{svg_b64}" style="width:100%;min-width:760px"/></div>')

st.subheader("Pipeline stages")
stages = nodes.groupby("Stage", sort=True)
cols = st.columns(len(stages))
for col, (stage, grp) in zip(cols, stages):
    with col.container(border=True, height="stretch"):
        st.markdown(f"**{stage}**")
        st.caption("  \n".join(f"#{r.ID} {r.Node}" for r in grp.itertuples()))

st.subheader("Node configuration")
stage = st.segmented_control("Stage", ["All", *sorted(nodes["Stage"].unique())], default="All",
                             label_visibility="collapsed")
view = nodes if stage in (None, "All") else nodes[nodes["Stage"] == stage]
st.dataframe(view, width="stretch", hide_index=True,
             column_config={"Configuration": st.column_config.TextColumn(width="large")})

st.subheader("Model results: KNIME vs Python")
res = get_models()
replica = res.leaderboard.set_index("Model").loc["Decision Tree (KNIME replica)"]
best = res.leaderboard.iloc[0]
cmp = [
    {"Tool": "KNIME — Decision Tree Learner (#24) + Scorer (#26)", "Accuracy": metrics.get("Accuracy", 0),
     "Cohen's κ": metrics.get("Cohen's kappa", 0)},
    {"Tool": "Python — same tree settings (Gini, min 2 records/node)", "Accuracy": replica["Accuracy"],
     "Cohen's κ": replica["Cohen's κ"]},
    {"Tool": f"Python — best model ({best['Model']})", "Accuracy": best["Accuracy"], "Cohen's κ": best["Cohen's κ"]},
]
st.dataframe(cmp, width="stretch", hide_index=True, column_config={
    "Accuracy": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
    "Cohen's κ": st.column_config.NumberColumn(format="%.3f"),
})
st.caption(
    f"The Python replica lands within {abs(metrics.get('Accuracy', 0) - replica['Accuracy']) * 100:.1f} points "
    "of KNIME (different random 80/20 split), which validates both pipelines. An unpruned single tree overfits; "
    f"an ensemble in Python lifts accuracy to {pct(best['Accuracy'])} — that model powers the predictor."
)

st.download_button("Download KNIME workflow (.knwf)", KNWF_PATH.read_bytes(), KNWF_PATH.name,
                   "application/octet-stream", icon=":material/download:")
st.caption("Import in KNIME Analytics Platform via **File → Import KNIME Workflow…**, then point the CSV Reader "
           "at `data/students_dropout.csv`.")
