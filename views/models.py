import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui import SEQ_BLUE, get_models, style

res = get_models()

st.title("Model Comparison")
st.caption("Four scikit-learn classifiers trained on the same 25 features the KNIME workflow uses, "
           "with a stratified 80/20 split. Target: Dropout / Enrolled / Graduate.")

st.dataframe(res.leaderboard, width="stretch", hide_index=True, column_config={
    "Accuracy": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
    "Cohen's κ": st.column_config.NumberColumn(format="%.3f"),
    "Macro F1": st.column_config.NumberColumn(format="%.3f"),
    "Dropout recall": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1,
                                                      help="Share of real dropouts the model catches"),
})
st.success(f"**{res.best_name}** performs best and is used by the predictor and early-warning list.",
           icon=":material/emoji_events:")

c1, c2 = st.columns(2)
with c1:
    name = st.selectbox("Confusion matrix for", res.leaderboard["Model"], index=0)
    cm = pd.DataFrame(res.confusion[name], index=res.classes, columns=res.classes)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale=SEQ_BLUE, aspect="auto",
                    labels=dict(x="Predicted", y="Actual", color="Students"))
    fig.update_traces(hovertemplate="Actual %{y} → predicted %{x}: %{z} students<extra></extra>")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(style(fig, 380, legend=False), width="stretch", theme=None)
    st.caption("'Enrolled' is hardest to predict — these students are still mid-way and look like either group.")
with c2:
    st.markdown(f"**What drives the {res.best_name} prediction?**")
    imp = res.importance.head(12).iloc[::-1]
    fig = px.bar(imp, x="importance", y="feature", orientation="h", color_discrete_sequence=[SEQ_BLUE[3]])
    fig.update_traces(hovertemplate="%{y}: accuracy drops %{x:.3f} when shuffled<extra></extra>")
    fig.update_layout(xaxis_title="Permutation importance (accuracy drop)", yaxis_title=None)
    st.plotly_chart(style(fig, 420, legend=False), width="stretch", theme=None)
