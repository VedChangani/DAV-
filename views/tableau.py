import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.config import TABLEAU_DASHBOARDS
from src.ui import get_data, pct

ROOT = Path(__file__).resolve().parent.parent
TWB = ROOT / "tableau" / "student_risk_dashboards.twb"

DASHBOARD = "Student Dropout Risk & Early Warning Dashboard"
KPIS = [
    ("KPI - Total Students", "COUNT([Target])"),
    ("KPI - Dropout Rate", "[Dropout Rate]"),
    ("KPI - High-Risk Students", "[High Risk Students]"),
    ("KPI - Average Grade", "AVG([Average Grade])"),
]
CHARTS = [
    ("Dropout Rate by Academic Risk", "Dropout rate for Low / Medium / High academic risk, labelled with student count"),
    ("Student Outcome by Academic Risk", "Outcome mix (% of students) inside each academic-risk level"),
    ("Dropout by Tuition Status", "Dropout rate for students with fees up to date vs. behind on fees"),
    ("Dropout by Scholarship", "Dropout rate for scholarship holders vs. non-holders"),
    ("Dropout by Age Group", "Dropout rate across age bands ≤20, 21–25, 26–30, 31+"),
    ("Dropout Rate by Course", "Dropout rate for each course"),
]
FILTERS = ["Course", "Academic Risk", "Tuition fees up to date", "Scholarship holder", "Age Group"]
OTHER_SHEETS = [
    ("Student Outcome Distribution", "Count of students by Target (Dropout / Enrolled / Graduate)"),
    ("Performance by Outcome", "Average grade for each outcome group"),
    ("Semester Grade Comparison", "1st vs 2nd semester grades by outcome"),
    ("Academic Performance by Risk", "Average grade and approval rate per academic-risk level"),
    ("Grade Change Among Active Students", "Average grade improvement by course, students active in both semesters"),
    ("Dropout by Debtor Status", "Dropout rate for debtors vs. non-debtors"),
]
CALCS = [
    ("Average Grade", "([1st sem grade] + [2nd sem grade]) / 2"),
    ("Overall Approval Rate", "IF enrolled > 0 THEN approved / enrolled ELSE 0"),
    ("Total Failed Units", "(1st evaluations − 1st approved) + (2nd evaluations − 2nd approved)"),
    ("Grade Improvement", "[2nd sem grade] − [1st sem grade]"),
    ("Academic Risk", 'IF grade < 10 AND approval < 0.5 THEN "High" ELSEIF grade < 12 OR approval < 0.7 THEN "Medium" ELSE "Low"'),
    ("Dropout Rate", 'SUM(IF [Target] = "Dropout" THEN 1 ELSE 0 END) / COUNT([Target])'),
    ("Student Count", "COUNT([Target])"),
    ("High Risk Students", 'SUM(IF [Academic Risk] = "High" THEN 1 ELSE 0 END)'),
    ("Age Group", 'IF age <= 20 THEN "<=20" ELSEIF age <= 25 THEN "21-25" ELSEIF age <= 30 THEN "26-30" ELSE "31+"'),
    ("Active in Two Semesters", 'IF 1st sem enrolled > 0 AND 2nd sem enrolled > 0 THEN "Yes" ELSE "No"'),
]


def normalize(url: str) -> str | None:
    url = url.strip()
    m = re.search(r"public\.tableau\.com/(?:app/profile/[^/]+/viz|views)/([^/?#]+)/([^/?#]+)", url)
    if not m:
        return None
    return f"https://public.tableau.com/views/{m.group(1)}/{m.group(2)}"


def embed(url: str, height: int = 1060) -> None:
    # The dashboard is a fixed 1600 × 1000 layout, plus room for the toolbar
    components.html(
        f"""
        <script type="module" src="https://public.tableau.com/javascripts/api/tableau.embedding.3.latest.min.js"></script>
        <tableau-viz src="{url}" width="100%" height="{height - 20}" toolbar="bottom" hide-tabs device="desktop">
        </tableau-viz>
        """,
        height=height,
        scrolling=False,
    )


def configured_url() -> str:
    url = TABLEAU_DASHBOARDS.get(DASHBOARD, "")
    try:
        url = st.secrets.get("tableau", {}).get(DASHBOARD) or url
    except Exception:
        pass
    return url


st.title("Tableau Dashboard")
st.caption(f"**{DASHBOARD}** — live and interactive on Tableau Public, embedded with the Tableau "
           "Embedding API v3. Hover, filter and click inside it exactly as you would in Tableau.")

_, df, _ = get_data()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total students", f"{len(df):,}", border=True)
k2.metric("Dropout rate", pct((df["target"] == "Dropout").mean()), border=True)
k3.metric("High-risk students", f"{(df['academic_risk'] == 'High').sum():,}", border=True)
k4.metric("Average grade", f"{df['average_grade'].mean():.2f}", border=True)
st.caption("The dashboard's four KPI tiles, recomputed by the Python pipeline. They should match Tableau.")

url = normalize(configured_url() or "")
if not url:
    pasted = st.text_input(
        "Tableau Public link",
        placeholder="https://public.tableau.com/views/<Workbook>/<Dashboard>",
        help="Preview the published dashboard here. To make it permanent, add the link to src/config.py.",
    )
    url = normalize(pasted) if pasted else None
    if pasted and not url:
        st.error("That doesn't look like a Tableau Public view link.")
if url:
    embed(url)
    st.link_button("Open in Tableau Public", url, icon=":material/open_in_new:")
else:
    st.info(
        "The dashboard isn't linked yet. Publish the workbook to Tableau Public, then paste the "
        "dashboard's share link above or into `src/config.py`.",
        icon=":material/link:",
    )

st.subheader("Dashboard layout")
left, right = st.columns([2, 1])
with left:
    st.markdown("**Charts**")
    st.dataframe([{"Sheet": s, "Shows": d} for s, d in CHARTS], width="stretch", hide_index=True)
with right:
    st.markdown("**KPI tiles**")
    st.dataframe([{"Sheet": s, "Measure": m} for s, m in KPIS], width="stretch", hide_index=True)
    st.markdown("**Filters**")
    st.markdown(" · ".join(FILTERS))
with st.expander("Other worksheets in the workbook (not on the dashboard)"):
    st.dataframe([{"Sheet": s, "Shows": d} for s, d in OTHER_SHEETS], width="stretch", hide_index=True)

st.divider()
c1, c2 = st.columns([1.4, 1])
with c1:
    st.subheader("Tableau calculated fields")
    st.caption("The first five use the same logic as the Python pipeline and the KNIME Math Formula / "
               "Rule Engine nodes; the rest drive the dashboard's KPIs and filters.")
    st.dataframe([{"Field": f, "Formula": x} for f, x in CALCS], width="stretch", hide_index=True)
with c2:
    st.subheader("Workbook")
    st.download_button("Download Tableau workbook (.twb)", TWB.read_bytes(), TWB.name,
                       "application/octet-stream", icon=":material/download:")
    with st.expander("How to publish to Tableau Public"):
        st.markdown(
            "1. Open the `.twb` in **Tableau Public Desktop** (free).\n"
            "2. If prompted, point the data source at `data/students_dropout.csv`.\n"
            "3. **File → Save to Tableau Public As…** and sign in.\n"
            "4. On the published page click **Share** and copy the link.\n"
            "5. Paste it into `src/config.py` → `TABLEAU_DASHBOARDS`, commit and push."
        )
