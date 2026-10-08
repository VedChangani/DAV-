import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui import DIVERGING, OUTCOME_COLORS, OUTCOME_ORDER, RISK_ORDER, get_data, pct, style

_, df, _ = get_data()

st.title("Exploratory Analysis")
st.caption("Interactive Python (Plotly) charts. Filters apply to every chart on this page.")

# ---- filters (one row, above the charts) ----
f1, f2, f3, f4, f5 = st.columns([2.2, 1, 1, 1, 1.2])
courses = f1.multiselect("Course", sorted(df["course"].unique()), placeholder="All courses")
gender = f2.selectbox("Gender", ["All", *sorted(df["gender"].unique())])
attendance = f3.selectbox("Attendance", ["All", *sorted(df["daytime_evening_attendance"].unique())])
scholarship = f4.selectbox("Scholarship", ["All", "Yes", "No"])
ages = f5.multiselect("Age group", ["≤20", "21–24", "25–30", "30+"], placeholder="All ages")

d = df
if courses:
    d = d[d["course"].isin(courses)]
if gender != "All":
    d = d[d["gender"] == gender]
if attendance != "All":
    d = d[d["daytime_evening_attendance"] == attendance]
if scholarship != "All":
    d = d[d["scholarship_holder"] == scholarship]
if ages:
    d = d[d["age_group"].isin(ages)]

if d.empty:
    st.warning("No students match these filters.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
diff = ((d["target"] == "Dropout").mean() - (df["target"] == "Dropout").mean()) * 100
k1.metric("Students in view", f"{len(d):,}", border=True)
k2.metric("Dropout rate", pct((d["target"] == "Dropout").mean()),
          delta=f"{diff:+.1f} pts vs all students" if len(d) < len(df) else None,
          delta_color="inverse", border=True)
k3.metric("Avg. grade (0–20)", f"{d['average_grade'].mean():.2f}", border=True)
k4.metric("Avg. approval rate", pct(d["overall_approval_rate"].mean()), border=True)


def outcome_share(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    s = frame.groupby(by)["target"].value_counts(normalize=True).rename("share").reset_index()
    n = frame.groupby(by).size().rename("n").reset_index()
    return s.merge(n, on=by)


# ---- row 1: course ----
share = outcome_share(d, "course")
order = (share[share["target"] == "Dropout"].sort_values("share")["course"].tolist())
order += [c for c in share["course"].unique() if c not in order]
fig = px.bar(share, y="course", x="share", color="target", orientation="h",
             category_orders={"course": order, "target": OUTCOME_ORDER}, color_discrete_map=OUTCOME_COLORS,
             custom_data=["target", "n"], title="Outcome mix by course (sorted by dropout share)")
fig.update_traces(hovertemplate="%{y}<br>%{customdata[0]}: %{x:.1%} of %{customdata[1]} students<extra></extra>")
fig.update_layout(xaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None, barmode="stack")
st.plotly_chart(style(fig, 520), width="stretch", theme=None)

# ---- row 2: financial factors ----
st.subheader("Financial factors")
long = pd.concat([
    outcome_share(d, col).rename(columns={col: "value"}).assign(factor=label)
    for col, label in [("tuition_fees_up_to_date", "Tuition up to date"), ("debtor", "Debtor"),
                       ("scholarship_holder", "Scholarship holder")]
])
fig = px.bar(long, x="value", y="share", color="target", facet_col="factor", custom_data=["target", "n"],
             category_orders={"target": OUTCOME_ORDER, "value": ["Yes", "No"]}, color_discrete_map=OUTCOME_COLORS)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font_size=14))
fig.update_traces(hovertemplate="%{x}<br>%{customdata[0]}: %{y:.1%} of %{customdata[1]}<extra></extra>")
fig.update_yaxes(tickformat=".0%", title=None)
fig.update_xaxes(title=None)
fig = style(fig, 380)
fig.update_layout(legend_y=1.12, margin_t=60)
st.plotly_chart(fig, width="stretch", theme=None)
st.caption("Students behind on tuition or in debt are far more likely to drop out; scholarship holders far less.")

# ---- row 3: academic ----
st.subheader("Academic performance")
c1, c2 = st.columns(2)
with c1:
    g = d.melt(id_vars="target", value_vars=["sem1_grade", "sem2_grade"], var_name="semester", value_name="grade")
    g["semester"] = g["semester"].map({"sem1_grade": "1st semester", "sem2_grade": "2nd semester"})
    fig = px.box(g, x="semester", y="grade", color="target", category_orders={"target": OUTCOME_ORDER},
                 color_discrete_map=OUTCOME_COLORS, title="Semester grades by outcome (0 = no units passed)")
    fig.update_layout(xaxis_title=None, yaxis_title="Grade (0–20)")
    st.plotly_chart(style(fig, 400), width="stretch", theme=None)
with c2:
    sample = d.sample(min(len(d), 2500), random_state=1)
    fig = px.scatter(sample, x="overall_approval_rate", y="average_grade", color="target", opacity=0.55,
                     category_orders={"target": OUTCOME_ORDER}, color_discrete_map=OUTCOME_COLORS,
                     hover_data={"course": True, "total_failed": True},
                     title="Approval rate vs. average grade")
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")))
    fig.add_vline(x=0.5, line_dash="dot", line_color="gray")
    fig.add_hline(y=10, line_dash="dot", line_color="gray")
    fig.add_annotation(x=0.02, y=2, text="High-risk zone", showarrow=False, xanchor="left", font_color="gray")
    fig.update_layout(xaxis_tickformat=".0%", xaxis_title="Units approved / enrolled", yaxis_title="Average grade")
    st.plotly_chart(style(fig, 400), width="stretch", theme=None)

# ---- row 4: demographics + flow ----
c1, c2 = st.columns([1, 1.4])
with c1:
    age = outcome_share(d, "age_group")
    age = age[age["target"] == "Dropout"]
    fig = px.bar(age, x="age_group", y="share", text=age["share"].map(pct), custom_data=["n"],
                 category_orders={"age_group": ["≤20", "21–24", "25–30", "30+"]},
                 color_discrete_sequence=[OUTCOME_COLORS["Dropout"]], title="Dropout rate by age at enrolment")
    fig.update_traces(textposition="outside", cliponaxis=False,
                      hovertemplate="%{x}: %{y:.1%} of %{customdata[0]} students<extra></extra>")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, age["share"].max() * 1.15], xaxis_title=None,
                      yaxis_title=None)
    st.plotly_chart(style(fig, 420, legend=False), width="stretch", theme=None)
with c2:
    flow = d[["scholarship_holder", "financial_risk", "academic_risk", "target"]].copy()
    color_idx = flow["target"].map({o: i for i, o in enumerate(OUTCOME_ORDER)})
    fig = go.Figure(go.Parcats(
        dimensions=[
            dict(values=flow["scholarship_holder"], label="Scholarship", categoryorder="array", categoryarray=["Yes", "No"]),
            dict(values=flow["financial_risk"], label="Financial risk", categoryorder="array", categoryarray=RISK_ORDER),
            dict(values=flow["academic_risk"], label="Academic risk", categoryorder="array", categoryarray=RISK_ORDER),
            dict(values=flow["target"], label="Outcome", categoryorder="array", categoryarray=OUTCOME_ORDER),
        ],
        line=dict(color=color_idx, colorscale=[[0, OUTCOME_COLORS["Graduate"]], [0.5, OUTCOME_COLORS["Enrolled"]],
                                               [1, OUTCOME_COLORS["Dropout"]]], shape="hspline"),
        hoveron="color", hoverinfo="count+probability",
    ))
    fig.update_layout(title="Student pathways: support → risk → outcome")
    st.plotly_chart(style(fig, 420, legend=False), width="stretch", theme=None)

# ---- row 5: correlation ----
st.subheader("What correlates with dropping out?")
num = ["admission_grade", "age_at_enrollment", "sem1_grade", "sem2_grade", "overall_approval_rate",
       "average_grade", "grade_improvement", "total_failed", "total_without_evaluation",
       "unemployment_rate", "inflation_rate", "gdp"]
corr = d[num].assign(
    dropout=(d["target"] == "Dropout").astype(int),
    debtor=(d["debtor"] == "Yes").astype(int),
    fees_overdue=(d["tuition_fees_up_to_date"] == "No").astype(int),
    scholarship=(d["scholarship_holder"] == "Yes").astype(int),
).corr()["dropout"].drop("dropout").sort_values()
fig = px.bar(x=corr.values, y=corr.index, orientation="h", color=corr.values, color_continuous_scale=DIVERGING,
             range_color=[-0.6, 0.6], title="Correlation with dropout (Pearson r)")
fig.update_traces(hovertemplate="%{y}: r = %{x:.2f}<extra></extra>")
fig.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_title=None)
st.plotly_chart(style(fig, 440, legend=False), width="stretch", theme=None)
st.caption("Red bars rise with dropout, blue bars fall with it. Approval rate and grades dominate; "
           "macro-economic indicators barely matter.")
