"""Python ML layer: re-trains the KNIME decision tree and compares stronger models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Same inputs as KNIME "Column Filter (#21)"
CATEGORICAL = [
    "gender", "international", "displaced", "previous_qualification", "course",
    "debtor", "tuition_fees_up_to_date", "scholarship_holder", "financial_risk", "academic_risk",
]
NUMERIC = [
    "age_at_enrollment", "admission_grade",
    "sem1_enrolled", "sem1_evaluations", "sem1_approved", "sem1_grade",
    "sem2_enrolled", "sem2_evaluations", "sem2_approved", "sem2_grade",
    "overall_approval_rate", "average_grade", "grade_improvement",
    "total_failed", "total_without_evaluation",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "target"


def _pipeline(estimator, scale: bool = False) -> Pipeline:
    num = StandardScaler() if scale else "passthrough"
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", num, NUMERIC),
    ])
    return Pipeline([("prep", pre), ("model", estimator)])


CANDIDATES = {
    "Decision Tree (KNIME replica)": lambda: _pipeline(
        DecisionTreeClassifier(criterion="gini", min_samples_leaf=2, random_state=42)),
    "Logistic Regression": lambda: _pipeline(
        LogisticRegression(max_iter=2000, class_weight="balanced"), scale=True),
    "Random Forest": lambda: _pipeline(
        RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
                               random_state=42)),
    "Gradient Boosting": lambda: _pipeline(
        HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, random_state=42)),
}


@dataclass
class TrainResult:
    leaderboard: pd.DataFrame
    models: dict
    best_name: str
    confusion: dict
    importance: pd.DataFrame
    classes: list


def train_all(df: pd.DataFrame) -> TrainResult:
    X, y = df[FEATURES], df[TARGET]
    # 80/20 split, same proportion as KNIME "Table Partitioner (#23)"
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    rows, models, confusion = [], {}, {}
    for name, make in CANDIDATES.items():
        m = make().fit(X_tr, y_tr)
        pred = m.predict(X_te)
        models[name] = m
        confusion[name] = confusion_matrix(y_te, pred, labels=m.classes_)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, pred),
            "Cohen's κ": cohen_kappa_score(y_te, pred),
            "Macro F1": f1_score(y_te, pred, average="macro"),
            "Dropout recall": recall_score(y_te, pred, labels=["Dropout"], average="macro"),
        })
    board = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    best = board.iloc[0]["Model"]

    imp = permutation_importance(models[best], X_te, y_te, n_repeats=5, random_state=42)
    importance = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
                  .sort_values("importance", ascending=False).reset_index(drop=True))
    return TrainResult(board, models, best, confusion, importance, list(models[best].classes_))


def predict(model, frame: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba(frame[FEATURES])
    out = pd.DataFrame(proba, columns=[f"p_{c}" for c in model.classes_], index=frame.index)
    out["predicted_outcome"] = np.array(model.classes_)[proba.argmax(axis=1)]
    out["dropout_risk"] = pd.cut(out["p_Dropout"], [-0.01, 0.30, 0.60, 1.0], labels=["Low", "Medium", "High"]).astype(str)
    return out
