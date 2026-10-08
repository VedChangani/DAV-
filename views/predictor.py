import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import model as ml
from src.processing import engineer
from src.ui import OUTCOME_COLORS, OUTCOME_ORDER, RISK_COLORS, get_data, get_models, pct, style

_, df, _ = get_data()
res = get_models()
best = res.models[res.best_name]

INPUT_COLS = [
    "gender", "age_at_enrollment", "international", "displaced", "previous_qualification", "admission_grade",
    "course", "debtor", "tuition_fees_up_to_date", "scholarship_holder",
    "sem1_enrolled", "sem1_evaluations", "sem1_approved", "sem1_grade", "sem1_without_evaluations",
    "sem2_enrolled", "sem2_evaluations", "sem2_approved", "sem2_grade", "sem2_without_evaluations",
]
DEFAULTS = df[INPUT_COLS].mode().iloc[0].to_dict() | df[INPUT_COLS].median(numeric_only=True).round(0).to_dict()

st.title("Dropout Risk Predictor")
st.caption(f"Enter a student's profile after their first year. The **{res.best_name}** model estimates the chance "
           "of each outcome, and the KNIME risk rules explain *why*.")

# ---- load an example student into the form ----
def load_example(target: str | None) -> None:
    pool = df if target is None else df[df["target"] == target]
    row = pool.sample(1).iloc[0]
    for c in INPUT_COLS:
        st.session_state[c] = coerce(c, row[c])
    st.session_state["example"] = ({c: st.session_state[c] for c in INPUT_COLS}, row["target"])


FLOAT_COLS = {"admission_grade", "sem1_grade", "sem2_grade"}


def coerce(col, value):
    if col in FLOAT_COLS:
        return round(float(value), 1)
    if pd.api.types.is_numeric_dtype(df[col]):
        return int(value)
    return str(value)


for c in INPUT_COLS:
    st.session_state.setdefault(c, coerce(c, DEFAULTS[c]))

b1, b2, b3, _ = st.columns([1, 1, 1, 2])
b1.button("Random real student", on_click=load_example, args=(None,), icon=":material/shuffle:")
b2.button("Example: dropout", on_click=load_example, args=("Dropout",), icon=":material/person_off:")
b3.button("Example: graduate", on_click=load_example, args=("Graduate",), icon=":material/school:")

opts = lambda col: sorted(df[col].unique())  # noqa: E731

with st.form("student"):
    st.markdown("**Background**")
    c1, c2, c3, c4 = st.columns(4)
    c1.selectbox("Course", opts("course"), key="course")
    c2.selectbox("Gender", opts("gender"), key="gender")
    c3.number_input("Age at enrolment", 17, 70, key="age_at_enrollment")
    c4.number_input("Admission grade (95–190)", 95.0, 190.0, step=0.5, key="admission_grade")
    c1, c2, c3, c4 = st.columns(4)
    c1.selectbox("Previous qualification", opts("previous_qualification"), key="previous_qualification")
    c2.radio("International", ["No", "Yes"], horizontal=True, key="international")
    c3.radio("Displaced (lives away from home)", ["No", "Yes"], horizontal=True, key="displaced")

    st.markdown("**Finances**")
    c1, c2, c3, _ = st.columns(4)
    c1.radio("Tuition fees up to date", ["Yes", "No"], horizontal=True, key="tuition_fees_up_to_date")
    c2.radio("Debtor", ["No", "Yes"], horizontal=True, key="debtor")
    c3.radio("Scholarship holder", ["No", "Yes"], horizontal=True, key="scholarship_holder")

    for sem, label in (("sem1", "1st semester"), ("sem2", "2nd semester")):
        st.markdown(f"**{label} curricular units**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.number_input("Enrolled", 0, 30, key=f"{sem}_enrolled")
        c2.number_input("Evaluations", 0, 50, key=f"{sem}_evaluations")
        c3.number_input("Approved", 0, 30, key=f"{sem}_approved")
        c4.number_input("Average grade (0–20)", 0.0, 20.0, step=0.1, key=f"{sem}_grade")
        c5.number_input("Without evaluation", 0, 20, key=f"{sem}_without_evaluations")

    submitted = st.form_submit_button("Predict risk", type="primary", icon=":material/analytics:")

student = pd.DataFrame([{c: st.session_state[c] for c in INPUT_COLS}])
student = engineer(student)
problems = [f"{s} approved units exceed enrolled units" for s in ("sem1", "sem2")
            if student[f"{s}_approved"].iat[0] > student[f"{s}_enrolled"].iat[0]]
for p in problems:
    st.warning(p.replace("sem1", "1st-semester").replace("sem2", "2nd-semester").capitalize())

pred = ml.predict(best, student).iloc[0]
p_drop = pred["p_Dropout"]
level = pred["dropout_risk"]
s = student.iloc[0]

st.divider()
left, mid, right = st.columns([1.1, 1.1, 1.3])
with left:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p_drop * 100, number=dict(suffix="%", font_size=44),
        title=dict(text=f"Dropout probability<br><span style='font-size:15px'>{level} risk</span>"),
        gauge=dict(
            axis=dict(range=[0, 100], ticksuffix="%"),
            bar=dict(color="#0b0b0b", thickness=0.25),
            steps=[dict(range=[0, 30], color=RISK_COLORS["Low"]), dict(range=[30, 60], color=RISK_COLORS["Medium"]),
                   dict(range=[60, 100], color=RISK_COLORS["High"])],
        ),
    ))
    st.plotly_chart(style(fig, 300, legend=False), width="stretch", theme=None)
    example = st.session_state.get("example")
    if example and example[0] == {c: st.session_state[c] for c in INPUT_COLS}:
        st.caption(f"Real student from the dataset — actual outcome: **{example[1]}**")
with mid:
    probs = pd.DataFrame({"Outcome": OUTCOME_ORDER, "p": [pred[f"p_{o}"] for o in OUTCOME_ORDER]})
    fig = px.bar(probs, x="p", y="Outcome", orientation="h", color="Outcome", color_discrete_map=OUTCOME_COLORS,
                 text=probs["p"].map(pct), title=f"Predicted: {pred['predicted_outcome']}")
    fig.update_traces(textposition="outside", hovertemplate="%{y}: %{x:.1%}<extra></extra>", cliponaxis=False)
    fig.update_layout(xaxis=dict(range=[0, 1.15], tickformat=".0%", title=None), yaxis_title=None)
    st.plotly_chart(style(fig, 300, legend=False), width="stretch", theme=None)
with right:
    st.markdown("**Rule-based segments (KNIME Rule Engine)**")
    a, f = s["academic_risk"], s["financial_risk"]
    icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    st.markdown(f"{icon[a]} Academic risk: **{a}**  \n"
                f"<small>approval rate {pct(s['overall_approval_rate'])}, average grade {s['average_grade']:.1f}</small>",
                unsafe_allow_html=True)
    st.markdown(f"{icon[f]} Financial risk: **{f}**  \n"
                f"<small>debtor: {s['debtor']}, fees up to date: {s['tuition_fees_up_to_date']}</small>",
                unsafe_allow_html=True)
    st.markdown(f"Failed units: **{int(s['total_failed'])}** · Grade change: **{s['grade_improvement']:+.1f}**")

# ---- interventions ----
actions = []
if s["tuition_fees_up_to_date"] == "No":
    actions.append(("Financial office", "Tuition is overdue — offer an instalment plan or emergency fund."))
if s["debtor"] == "Yes":
    actions.append(("Financial counselling", "Student has outstanding debt — refer to financial counselling."))
if s["scholarship_holder"] == "No" and f != "Low":
    actions.append(("Scholarship", "Under financial pressure without a scholarship — check eligibility."))
if s["overall_approval_rate"] < 0.5:
    actions.append(("Academic tutoring", "Passed under half of enrolled units — assign a tutor and review course load."))
elif s["overall_approval_rate"] < 0.7:
    actions.append(("Study support", "Approval rate below 70% — recommend study-skills workshops."))
if s["grade_improvement"] < -1:
    actions.append(("Advisor check-in", "Grades fell between semesters — schedule a one-to-one check-in."))
if s["total_without_evaluation"] > 0:
    actions.append(("Attendance follow-up", "Some units have no evaluation — follow up on missed assessments."))
if s["age_at_enrollment"] > 30:
    actions.append(("Flexible learning", "Mature student — suggest evening or part-time options."))

st.subheader("Recommended interventions")
if not actions:
    st.success("No warning signs. Keep normal monitoring.", icon=":material/check_circle:")
else:
    for who, what in actions:
        st.markdown(f"- **{who}:** {what}")
