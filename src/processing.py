"""Python preprocessing + feature engineering.

Mirrors the KNIME workflow (Assignment/workflow.knime) node-for-node so the
Python results, the KNIME results and the Tableau calculated fields all agree.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "students_dropout.csv"

# Same mapping as KNIME "Column Renamer (#3)"
RENAME = {
    "Marital status": "marital_status",
    "Application mode": "application_mode",
    "Application order": "application_order",
    "Course": "course",
    "Daytime/evening attendance": "daytime_evening_attendance",
    "Previous qualification": "previous_qualification",
    "Previous qualification (grade)": "previous_qualification_grade",
    "Nacionality": "nationality",
    "Mother's qualification": "mothers_qualification",
    "Father's qualification": "fathers_qualification",
    "Mother's occupation": "mothers_occupation",
    "Father's occupation": "fathers_occupation",
    "Admission grade": "admission_grade",
    "Displaced": "displaced",
    "Educational special needs": "educational_special_needs",
    "Debtor": "debtor",
    "Tuition fees up to date": "tuition_fees_up_to_date",
    "Gender": "gender",
    "Scholarship holder": "scholarship_holder",
    "Age at enrollment": "age_at_enrollment",
    "International": "international",
    "Curricular units 1st sem (credited)": "sem1_credited",
    "Curricular units 1st sem (enrolled)": "sem1_enrolled",
    "Curricular units 1st sem (evaluations)": "sem1_evaluations",
    "Curricular units 1st sem (approved)": "sem1_approved",
    "Curricular units 1st sem (grade)": "sem1_grade",
    "Curricular units 1st sem (without evaluations)": "sem1_without_evaluations",
    "Curricular units 2nd sem (credited)": "sem2_credited",
    "Curricular units 2nd sem (enrolled)": "sem2_enrolled",
    "Curricular units 2nd sem (evaluations)": "sem2_evaluations",
    "Curricular units 2nd sem (approved)": "sem2_approved",
    "Curricular units 2nd sem (grade)": "sem2_grade",
    "Curricular units 2nd sem (without evaluations)": "sem2_without_evaluations",
    "Unemployment rate": "unemployment_rate",
    "Inflation rate": "inflation_rate",
    "GDP": "gdp",
    "Target": "target",
}

OUTCOMES = ["Dropout", "Enrolled", "Graduate"]
RISK_LEVELS = ["Low", "Medium", "High"]
# Same bins as the Tableau "Age Group" calculated field
AGE_GROUPS = ["≤20", "21–25", "26–30", "31+"]

# Human-readable description of every engineered column (KNIME node in brackets)
FEATURES = [
    ("total_enrolled", "sem1_enrolled + sem2_enrolled", "Math Formula #6"),
    ("total_approved", "sem1_approved + sem2_approved", "Math Formula #7"),
    ("total_evaluated", "sem1_evaluations + sem2_evaluations", "Math Formula #8"),
    ("overall_approval_rate", "total_approved / total_enrolled (0 if none enrolled)", "Expression #9"),
    ("sem1_approval_rate", "sem1_approved / sem1_enrolled", "Expression #10"),
    ("sem2_approval_rate", "sem2_approved / sem2_enrolled", "Expression #11"),
    ("average_grade", "(sem1_grade + sem2_grade) / 2", "Math Formula #12"),
    ("grade_improvement", "sem2_grade - sem1_grade", "Math Formula #13"),
    ("sem1_failed", "sem1_evaluations - sem1_approved", "Math Formula #14"),
    ("sem2_failed", "sem2_evaluations - sem2_approved", "Math Formula #15"),
    ("total_failed", "sem1_failed + sem2_failed", "Math Formula #16"),
    ("total_without_evaluation", "sem1 + sem2 units without evaluation", "Math Formula #17"),
    ("financial_risk", "High: debtor AND fees overdue · Medium: either · else Low", "Rule Engine #18"),
    ("academic_risk", "High: approval < 50% AND grade < 10 · Medium: approval < 70% OR grade < 12 · else Low", "Rule Engine #19"),
]


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # The source file has a stray tab in "Daytime/evening attendance\t"
    df.columns = [c.strip() for c in df.columns]
    return df


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.where(den > 0)).fillna(0.0)


def financial_risk(debtor: pd.Series, fees_up_to_date: pd.Series) -> pd.Series:
    is_debtor = debtor.eq("Yes")
    overdue = fees_up_to_date.eq("No")
    return pd.Series(
        np.select([is_debtor & overdue, is_debtor | overdue], ["High", "Medium"], "Low"),
        index=debtor.index,
    )


def academic_risk(approval_rate: pd.Series, avg_grade: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [
                (approval_rate < 0.50) & (avg_grade < 10),
                (approval_rate < 0.70) | (avg_grade < 12),
            ],
            ["High", "Medium"],
            "Low",
        ),
        index=approval_rate.index,
    )


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add the KNIME-equivalent engineered features to a snake_case frame."""
    df = df.copy()
    df["total_enrolled"] = df["sem1_enrolled"] + df["sem2_enrolled"]
    df["total_approved"] = df["sem1_approved"] + df["sem2_approved"]
    df["total_evaluated"] = df["sem1_evaluations"] + df["sem2_evaluations"]
    df["overall_approval_rate"] = _safe_div(df["total_approved"], df["total_enrolled"])
    df["sem1_approval_rate"] = _safe_div(df["sem1_approved"], df["sem1_enrolled"])
    df["sem2_approval_rate"] = _safe_div(df["sem2_approved"], df["sem2_enrolled"])
    df["average_grade"] = (df["sem1_grade"] + df["sem2_grade"]) / 2
    df["grade_improvement"] = df["sem2_grade"] - df["sem1_grade"]
    df["sem1_failed"] = df["sem1_evaluations"] - df["sem1_approved"]
    df["sem2_failed"] = df["sem2_evaluations"] - df["sem2_approved"]
    df["total_failed"] = df["sem1_failed"] + df["sem2_failed"]
    df["total_without_evaluation"] = df["sem1_without_evaluations"] + df["sem2_without_evaluations"]
    df["financial_risk"] = financial_risk(df["debtor"], df["tuition_fees_up_to_date"])
    df["academic_risk"] = academic_risk(df["overall_approval_rate"], df["average_grade"])
    df["age_group"] = pd.cut(
        df["age_at_enrollment"],
        bins=[0, 20, 25, 30, np.inf],
        labels=AGE_GROUPS,
    ).astype(str)
    return df


def preprocess(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Full pipeline. Returns the processed frame and a step-by-step log."""
    log: list[dict] = []

    def step(name: str, knime: str, frame: pd.DataFrame, note: str) -> None:
        log.append({"Step": name, "KNIME node": knime, "Rows": len(frame), "Columns": frame.shape[1], "Notes": note})

    df = raw.copy()
    step("Load CSV", "CSV Reader #2", df, "UCI 'Predict Students' Dropout and Academic Success' (labels decoded)")

    df = df.rename(columns=RENAME)
    step("Rename to snake_case", "Column Renamer #3", df, "37 columns renamed for formula-friendly names")

    n_missing = int(df.isna().sum().sum())
    df = df.dropna(subset=["target"])
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    cat_cols = df.select_dtypes(exclude="number").columns
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iat[0])
    step("Handle missing values", "Missing Value #4", df, f"{n_missing} missing cells found; rows without target dropped, numeric→median, text→mode")

    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    step("Remove duplicate rows", "Duplicate Row Filter #5", df, f"{n_dup} duplicates removed")

    df = engineer(df)
    step("Feature engineering", "Math Formula #6–#17, Expression #9–#11", df, "12 academic-performance features derived")
    step("Risk segmentation", "Rule Engine #18, #19", df, "financial_risk and academic_risk (Low/Medium/High) assigned")
    return df, log
