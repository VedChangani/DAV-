# 🎓 Student Dropout Early-Warning System

**Domain:** Education · **Tools:** Python, KNIME, Tableau, Streamlit

About one in three students in this higher-education cohort drops out. This project finds the academic and
financial signals that predict dropout and turns them into a working early-warning app for student advisors.

## Dataset
UCI *Predict Students' Dropout and Academic Success* (Realinho et al., 2022): 4,424 students, 36 features,
target = Dropout / Enrolled / Graduate. Categorical codes are decoded into readable labels
(`data/students_dropout.csv`).

## What's in the app
| Page | What it shows |
|---|---|
| Overview | Problem, KPIs, how the tools connect, key findings |
| Python Processing | Data-quality report, cleaning steps, 14 engineered features, processed CSV download |
| Exploratory Analysis | Filterable Plotly charts: course, finances, grades, age, risk pathways, correlations |
| Tableau Dashboard | The *Student Dropout Risk & Early Warning* dashboard embedded live (Embedding API v3) |
| KNIME Workflow | The 28-node workflow, parsed live from `Assignment/workflow.knime`; KNIME vs Python accuracy |
| Model Comparison | Decision tree (KNIME replica), logistic regression, random forest, gradient boosting |
| Dropout Risk Predictor | Enter a student → dropout probability, rule-based risk segments, recommended interventions |
| Early-Warning List | Score a whole cohort (or upload a CSV) → a ranked list of students to contact |

## Repository layout
```
streamlit_app.py          entry point (st.navigation)
views/                    one file per page
src/processing.py         pandas pipeline (mirrors the KNIME nodes)
src/model.py              scikit-learn models
src/knime.py              reads the KNIME workflow files
src/config.py             Tableau Public links  ← paste yours here
data/                     dataset
Assignment/               KNIME workflow (source folder)
knime/*.knwf              KNIME workflow export
tableau/*.twb             Tableau workbook
script.py                 original exploratory Python script
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Link the Tableau dashboard
1. Open `tableau/student_risk_dashboards.twb` in Tableau Public Desktop and point the data source at
   `data/students_dropout.csv` if asked.
2. **File → Save to Tableau Public As…**
3. Copy the dashboard's share link into `src/config.py` → `TABLEAU_DASHBOARDS`, then commit and push.

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repository.
2. Go to <https://share.streamlit.io> → **Create app** → pick the repo, branch `main`, main file `streamlit_app.py`.
3. Under **Advanced settings** choose Python 3.12, then **Deploy**.
