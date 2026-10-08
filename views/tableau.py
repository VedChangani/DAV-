import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.config import TABLEAU_DASHBOARDS

ROOT = Path(__file__).resolve().parent.parent
TWB = ROOT / "tableau" / "student_risk_dashboards.twb"

SHEETS = {
    "Student Academic Performance & Risk Dashboard": [
        ("Student Outcome Distribution", "Count of students by Target (Dropout / Enrolled / Graduate)"),
        ("Performance by Outcome", "Average grade for each outcome group"),
        ("Course Performance", "Average grade by course"),
        ("Academic Risk Distribution", "Students per academic-risk level"),
        ("Semester Grade Comparison", "1st vs 2nd semester grades side by side"),
    ],
    "Academic Risk and Performance Analysis": [
        ("Academic Risk Scatter", "Average grade vs approval rate per student, coloured by risk"),
        ("Risk Category Distribution", "Size of each risk category"),
        ("Average Grade by Risk", "Average grade within Low / Medium / High risk"),
        ("Outcome by Academic Risk", "Outcome mix inside each risk level"),
        ("Grade Improvement Analysis", "Average semester-to-semester grade change by course"),
    ],
}
CALCS = [
    ("Average Grade", "([1st sem grade] + [2nd sem grade]) / 2"),
    ("Overall Approval Rate", "IF enrolled > 0 THEN approved / enrolled ELSE 0"),
    ("Total Failed Units", "(1st evaluations − 1st approved) + (2nd evaluations − 2nd approved)"),
    ("Grade Improvement", "[2nd sem grade] − [1st sem grade]"),
    ("Academic Risk", 'IF grade < 10 AND approval < 0.5 THEN "High" ELSEIF grade < 12 OR approval < 0.7 THEN "Medium" ELSE "Low"'),
]


def normalize(url: str) -> str | None:
    url = url.strip()
    m = re.search(r"public\.tableau\.com/(?:app/profile/[^/]+/viz|views)/([^/?#]+)/([^/?#]+)", url)
    if not m:
        return None
    return f"https://public.tableau.com/views/{m.group(1)}/{m.group(2)}"


def embed(url: str, height: int = 860) -> None:
    components.html(
        f"""
        <script type="module" src="https://public.tableau.com/javascripts/api/tableau.embedding.3.latest.min.js"></script>
        <tableau-viz src="{url}" width="100%" height="{height - 20}" toolbar="bottom" hide-tabs device="desktop">
        </tableau-viz>
        """,
        height=height,
        scrolling=False,
    )


def configured_urls() -> dict:
    urls = dict(TABLEAU_DASHBOARDS)
    try:
        urls.update({k: v for k, v in st.secrets.get("tableau", {}).items() if v})
    except Exception:
        pass
    return urls


st.title("Tableau Dashboards")
st.caption("Live, interactive Tableau Public dashboards embedded with the Tableau Embedding API v3. "
           "Hover, filter and click inside them exactly as you would in Tableau.")

urls = configured_urls()
tabs = st.tabs(list(SHEETS))
for tab, (name, sheets) in zip(tabs, SHEETS.items()):
    with tab:
        url = normalize(urls.get(name, "") or "")
        if not url:
            pasted = st.text_input(
                "Tableau Public link", key=f"url_{name}",
                placeholder="https://public.tableau.com/views/<Workbook>/<Dashboard>",
                help="Preview a published dashboard here. To make it permanent, add the link to src/config.py.",
            )
            url = normalize(pasted) if pasted else None
            if pasted and not url:
                st.error("That doesn't look like a Tableau Public view link.")
        if url:
            embed(url)
            st.link_button("Open in Tableau Public", url, icon=":material/open_in_new:")
        else:
            st.info(
                "This dashboard isn't linked yet. Publish the workbook to Tableau Public, then paste the "
                "dashboard's share link above or into `src/config.py`.",
                icon=":material/link:",
            )
        st.markdown("**Sheets in this dashboard**")
        st.dataframe([{"Sheet": s, "Shows": d} for s, d in sheets], width="stretch", hide_index=True)

st.divider()
c1, c2 = st.columns([1.4, 1])
with c1:
    st.subheader("Tableau calculated fields")
    st.caption("The same logic as the Python pipeline and the KNIME Math Formula / Rule Engine nodes.")
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
