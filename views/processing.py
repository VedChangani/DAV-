import inspect

import pandas as pd
import plotly.express as px
import streamlit as st

from src import processing
from src.ui import RISK_COLORS, RISK_ORDER, get_data, style

raw, df, log = get_data()

st.title("Python Processing")
st.caption("Pandas pipeline in `src/processing.py`. Each step mirrors a node in the KNIME workflow, so both tools produce the same table.")

tab_raw, tab_quality, tab_pipeline, tab_features, tab_out = st.tabs(
    ["Raw data", "Data quality", "Pipeline log", "Engineered features", "Processed output"]
)

with tab_raw:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{raw.shape[0]:,}", border=True)
    c2.metric("Columns", raw.shape[1], border=True)
    c3.metric("Target classes", raw["Target"].nunique(), border=True)
    st.dataframe(raw.head(200), width="stretch", height=380)
    st.caption("Source: UCI Machine Learning Repository, *Predict Students' Dropout and Academic Success* "
               "(Realinho et al., 2022). Categorical codes were decoded into readable labels.")

with tab_quality:
    profile = pd.DataFrame({
        "dtype": raw.dtypes.astype(str),
        "missing": raw.isna().sum(),
        "unique": raw.nunique(),
        "example": raw.iloc[0].astype(str),
    })
    c1, c2, c3 = st.columns(3)
    c1.metric("Missing cells", int(raw.isna().sum().sum()), border=True)
    c2.metric("Duplicate rows", int(raw.duplicated().sum()), border=True)
    c3.metric("Text / numeric columns",
              f"{(raw.dtypes == 'object').sum() + (raw.dtypes == 'str').sum()} / {raw.select_dtypes('number').shape[1]}",
              border=True)
    st.dataframe(profile, width="stretch", height=420)
    st.markdown("**Issues found and fixed**")
    st.markdown(
        "- Stray tab character in the header `Daytime/evening attendance\\t` → stripped.\n"
        "- Long, space-filled column names → renamed to snake_case (same mapping as KNIME *Column Renamer*).\n"
        "- Students with 0 enrolled units would cause divide-by-zero in approval rates → guarded to 0.\n"
        "- Missing-value and duplicate checks run defensively even though this release of the data is clean."
    )

with tab_pipeline:
    st.dataframe(pd.DataFrame(log), width="stretch", hide_index=True)
    with st.expander("Show the Python code"):
        st.code(inspect.getsource(processing.engineer), language="python")

with tab_features:
    st.dataframe(
        pd.DataFrame(processing.FEATURES, columns=["Feature", "Formula", "Equivalent KNIME node"]),
        width="stretch", hide_index=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        counts = df["academic_risk"].value_counts().reindex(RISK_ORDER).reset_index()
        fig = px.bar(counts, x="academic_risk", y="count", color="academic_risk", color_discrete_map=RISK_COLORS,
                     text="count", title="Academic risk segments")
        fig.update_traces(textposition="outside", hovertemplate="%{x}: %{y:,} students<extra></extra>")
        fig.update_layout(xaxis_title=None, yaxis_title="Students")
        st.plotly_chart(style(fig, 340, legend=False), width="stretch", theme=None)
    with c2:
        counts = df["financial_risk"].value_counts().reindex(RISK_ORDER).reset_index()
        fig = px.bar(counts, x="financial_risk", y="count", color="financial_risk", color_discrete_map=RISK_COLORS,
                     text="count", title="Financial risk segments")
        fig.update_traces(textposition="outside", hovertemplate="%{x}: %{y:,} students<extra></extra>")
        fig.update_layout(xaxis_title=None, yaxis_title="Students")
        st.plotly_chart(style(fig, 340, legend=False), width="stretch", theme=None)

with tab_out:
    st.markdown(f"Final table: **{df.shape[0]:,} rows × {df.shape[1]} columns**. "
                "This is the file fed into Tableau and KNIME.")
    st.dataframe(df.head(200), width="stretch", height=380)
    st.download_button("Download processed CSV", df.to_csv(index=False).encode(), "students_processed.csv",
                       "text/csv", icon=":material/download:")
