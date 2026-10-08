import plotly.express as px
import streamlit as st

from src.knime import scorer_metrics
from src.ui import OUTCOME_COLORS, OUTCOME_ORDER, get_data, pct, style

raw, df, _ = get_data()
knime = scorer_metrics()

st.title("🎓 Student Dropout Early-Warning System")
st.markdown(
    "**The problem:** roughly one in three students in this Portuguese higher-education cohort drops out. "
    "Universities usually find out too late — after the student has already left. "
    "This project combines **Python, KNIME and Tableau** to find the academic and financial signals that "
    "predict dropout, and turns them into a **working early-warning tool** that flags at-risk students "
    "after their first year so advisors can intervene."
)

drop_rate = (df["target"] == "Dropout").mean()
grad_rate = (df["target"] == "Graduate").mean()
high_risk = (df["academic_risk"] == "High").mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Students", f"{len(df):,}", border=True)
c2.metric("Dropout rate", pct(drop_rate), border=True)
c3.metric("Graduation rate", pct(grad_rate), border=True)
c4.metric("High academic risk", pct(high_risk), border=True)
c5.metric("KNIME tree accuracy", pct(knime.get("Accuracy", 0)), border=True)

st.subheader("How the tools fit together")
cols = st.columns(4)
steps = [
    (":material/data_object: Python", "Cleaning, feature engineering, EDA, and ML models (scikit-learn).",
     "views/processing.py"),
    (":material/account_tree: KNIME", "Visual, no-code workflow: 28 nodes from CSV to a scored decision tree.",
     "views/knime.py"),
    (":material/dashboard: Tableau", "Two interactive dashboards on outcomes, course performance and risk.",
     "views/tableau.py"),
    (":material/person_search: Streamlit app", "Predicts a student's dropout risk and builds an advisor watch-list.",
     "views/predictor.py"),
]
for col, (title, body, page) in zip(cols, steps):
    with col.container(border=True, height="stretch"):
        st.markdown(f"**{title}**")
        st.caption(body)
        st.page_link(page, label="Open", icon=":material/arrow_forward:")

st.subheader("Key findings")
left, right = st.columns([1, 1.3])

with left:
    counts = df["target"].value_counts().reindex(OUTCOME_ORDER).reset_index()
    fig = px.pie(counts, names="target", values="count", hole=0.6, color="target",
                 color_discrete_map=OUTCOME_COLORS, title="Student outcomes")
    fig.update_traces(textinfo="percent+label", sort=False, marker_line_color="white", marker_line_width=2,
                      hovertemplate="%{label}: %{value:,} students (%{percent})<extra></extra>")
    fig.add_annotation(text=f"<b>{len(df):,}</b><br>students", showarrow=False, font_size=15)
    st.plotly_chart(style(fig, 360, legend=False), width="stretch", theme=None)

with right:
    def rate(mask):
        return (df.loc[mask, "target"] == "Dropout").mean()

    fees_overdue, fees_ok = rate(df["tuition_fees_up_to_date"] == "No"), rate(df["tuition_fees_up_to_date"] == "Yes")
    schol, no_schol = rate(df["scholarship_holder"] == "Yes"), rate(df["scholarship_holder"] == "No")
    hi_ac, lo_ac = rate(df["academic_risk"] == "High"), rate(df["academic_risk"] == "Low")
    old, young = rate(df["age_at_enrollment"] > 30), rate(df["age_at_enrollment"] <= 20)
    by_course = df.groupby("course")["target"].apply(lambda s: (s == "Dropout").mean()).sort_values()

    st.markdown(
        f"""
- **Academic performance is the strongest signal.** Students flagged *High* academic risk drop out
  **{pct(hi_ac)}** of the time vs **{pct(lo_ac)}** for *Low* risk.
- **Money matters.** Students behind on tuition drop out at **{pct(fees_overdue)}**, versus
  **{pct(fees_ok)}** for those up to date.
- **Scholarships protect.** Scholarship holders: **{pct(schol)}** dropout vs **{pct(no_schol)}** without.
- **Mature students are more vulnerable.** Enrolled over 30: **{pct(old)}** dropout vs **{pct(young)}**
  for those 20 or younger.
- **Course matters.** Highest dropout: *{by_course.index[-1]}* (**{pct(by_course.iloc[-1])}**);
  lowest: *{by_course.index[0]}* (**{pct(by_course.iloc[0])}**).
"""
    )
    st.info(
        "**Recommendation:** combine first-year approval rate with tuition status to flag students at the end of "
        "semester 2. That alone captures most dropouts while there's still time to offer tutoring, fee plans "
        "or scholarship support.",
        icon=":material/lightbulb:",
    )
