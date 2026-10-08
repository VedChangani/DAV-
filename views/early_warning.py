import pandas as pd
import plotly.express as px
import streamlit as st

from src import model as ml
from src.processing import RENAME, engineer, load_raw
from src.ui import RISK_COLORS, RISK_ORDER, get_data, get_models, pct, style

_, df, _ = get_data()
res = get_models()
best = res.models[res.best_name]

st.title("Early-Warning List")
st.caption("Score a whole cohort at once and get a prioritised list of students for advisors to contact. "
           "Use the built-in dataset or upload your own CSV with the same columns.")

source = st.radio("Data source", ["Built-in dataset", "Upload CSV"], horizontal=True, label_visibility="collapsed")
if source == "Upload CSV":
    up = st.file_uploader("CSV with the original UCI column names", type="csv")
    if up is None:
        st.info("Upload a file to score it. The built-in dataset can be downloaded on the Python Processing page "
                "as a template.", icon=":material/upload_file:")
        st.stop()
    try:
        raw = load_raw(up).rename(columns=RENAME)
        cohort = engineer(raw)
        missing = [c for c in ml.FEATURES if c not in cohort.columns]
        if missing:
            raise ValueError(f"missing columns: {', '.join(missing)}")
    except Exception as e:  # noqa: BLE001
        st.error(f"Couldn't score that file: {e}")
        st.stop()
else:
    cohort = df.copy()

scores = ml.predict(best, cohort)
out = pd.concat([cohort, scores], axis=1)
out.insert(0, "student_id", range(1, len(out) + 1))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Students scored", f"{len(out):,}", border=True)
for col, lvl in zip((c2, c3, c4), ("High", "Medium", "Low")):
    n = (out["dropout_risk"] == lvl).sum()
    col.metric(f"{lvl} dropout risk", f"{n:,}  ({n / len(out):.0%})", border=True)

left, right = st.columns([1, 1.4])
with left:
    fig = px.histogram(out, x="p_Dropout", nbins=40, color="dropout_risk", color_discrete_map=RISK_COLORS,
                       category_orders={"dropout_risk": RISK_ORDER}, title="Distribution of dropout probability")
    fig.update_layout(xaxis_tickformat=".0%", xaxis_title=None, yaxis_title="Students", bargap=0.05)
    st.plotly_chart(style(fig, 420).update_traces(marker_line_width=0.5), width="stretch", theme=None)
with right:
    by_course = (out.groupby("course").agg(students=("p_Dropout", "size"), avg_risk=("p_Dropout", "mean"),
                                           high=("dropout_risk", lambda s: (s == "High").sum()))
                 .sort_values("high"))
    fig = px.bar(by_course.reset_index(), x="high", y="course", orientation="h",
                 color_discrete_sequence=[RISK_COLORS["High"]], custom_data=["students", "avg_risk"],
                 title="High-risk students per course")
    fig.update_traces(hovertemplate="%{y}<br>%{x} high-risk of %{customdata[0]}"
                                    "<br>avg. dropout probability %{customdata[1]:.0%}<extra></extra>")
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    fig.update_yaxes(dtick=1, tickfont_size=11)
    st.plotly_chart(style(fig, 420, legend=False), width="stretch", theme=None)

st.subheader("Students to contact")
f1, f2, f3 = st.columns([1, 2, 1])
levels = f1.multiselect("Risk level", RISK_ORDER, default=["High"])
courses = f2.multiselect("Course", sorted(out["course"].unique()), placeholder="All courses")
hide_known = f3.toggle("Hide known dropouts", value=False,
                       help="In the historical data the real outcome is known; hide it to focus on who is still enrolled.")

view = out[out["dropout_risk"].isin(levels or RISK_ORDER)]
if courses:
    view = view[view["course"].isin(courses)]
if hide_known and "target" in view:
    view = view[view["target"] != "Dropout"]
view = view.sort_values("p_Dropout", ascending=False)

cols = ["student_id", "course", "age_at_enrollment", "p_Dropout", "dropout_risk", "academic_risk", "financial_risk",
        "overall_approval_rate", "average_grade", "tuition_fees_up_to_date", "debtor", "scholarship_holder"]
if "target" in view:
    cols.append("target")
st.dataframe(view[cols], width="stretch", hide_index=True, height=420, column_config={
    "p_Dropout": st.column_config.ProgressColumn("Dropout probability", format="percent", min_value=0, max_value=1),
    "overall_approval_rate": st.column_config.NumberColumn("Approval rate", format="percent"),
    "average_grade": st.column_config.NumberColumn("Avg. grade", format="%.1f"),
    "target": "Actual outcome",
    "student_id": "ID", "course": "Course", "age_at_enrollment": "Age", "dropout_risk": "Risk level",
    "academic_risk": "Academic risk", "financial_risk": "Financial risk", "tuition_fees_up_to_date": "Fees paid",
    "debtor": "Debtor", "scholarship_holder": "Scholarship",
})
st.caption(f"{len(view):,} students match.")
if "target" in out and source == "Built-in dataset":
    hi = out[out["dropout_risk"] == "High"]
    st.caption(f"Sanity check on historical data: {pct((hi['target'] == 'Dropout').mean())} of students flagged *High* "
               f"actually dropped out, and the High flag catches "
               f"{pct((hi['target'] == 'Dropout').sum() / (out['target'] == 'Dropout').sum())} of all dropouts. "
               "(In-sample — the model saw 80% of these students during training.)")
st.download_button("Download list (CSV)", view[cols].to_csv(index=False).encode(), "early_warning_list.csv",
                   "text/csv", icon=":material/download:")
