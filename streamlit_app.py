import sys
from pathlib import Path

# Make `src` importable no matter which folder the app is deployed from
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(
    page_title="Student Dropout Early-Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.ui import sidebar_footer  # noqa: E402  (after set_page_config)

pages = {
    "Project": [
        st.Page("views/overview.py", title="Overview", icon=":material/home:", default=True),
    ],
    "Analytics pipeline": [
        st.Page("views/processing.py", title="Python Processing", icon=":material/data_object:"),
        st.Page("views/explore.py", title="Exploratory Analysis", icon=":material/insights:"),
        st.Page("views/tableau.py", title="Tableau Dashboards", icon=":material/dashboard:"),
        st.Page("views/knime.py", title="KNIME Workflow", icon=":material/account_tree:"),
    ],
    "Application": [
        st.Page("views/models.py", title="Model Comparison", icon=":material/model_training:"),
        st.Page("views/predictor.py", title="Dropout Risk Predictor", icon=":material/person_search:"),
        st.Page("views/early_warning.py", title="Early-Warning List", icon=":material/notifications_active:"),
    ],
}

sidebar_footer()
st.navigation(pages).run()
