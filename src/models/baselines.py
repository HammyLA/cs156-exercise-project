"""
baselines.py
------------
Purpose:
    - Define, train, and evaluate a suite of classical ML baselines:
        Decision Tree, SVM (RBF), Naive Bayes, Random Forest, AdaBoost, XGBoost
    - Compute standard metrics (accuracy, macro-precision/recall/F1)
    - Return a pandas DataFrame of results and each model's confusion matrix.

Notes:
    - We use StandardScaler inside pipelines where it's helpful (SVM/AdaBoost/XGB).
    - XGBoost is optional; if the package isn't installed, we simply skip it.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Try to import XGBoost; if missing, we'll proceed without it.
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False


# ---- small result container --------------------------------------------------

@dataclass
class EvalResult:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    conf_mat: np.ndarray


def _metrics(y_true, y_pred) -> EvalResult:
    """Compute accuracy + macro-averaged precision, recall, F1 + confusion matrix."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    return EvalResult(acc, prec, rec, f1, cm)


def _fit_eval(clf, X_tr, y_tr, X_va, y_va) -> EvalResult:
    """Fit a model and evaluate on a validation set."""
    clf.fit(X_tr, y_tr)
    y_hat = clf.predict(X_va)
    return _metrics(y_va, y_hat)


# ---- model zoo --------------------------------------------------------------

def default_models() -> Dict[str, object]:
    """
    Define the baseline models with reasonable defaults.
    You can tune these later; for now we optimize for getting end-to-end results.
    """
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "SVM_RBF": Pipeline(
            [("scaler", StandardScaler()),
             ("clf", SVC(kernel="rbf", random_state=42))]
        ),
        "NaiveBayes": GaussianNB(),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        "AdaBoost": Pipeline(
            [("scaler", StandardScaler()),
             ("clf", AdaBoostClassifier(random_state=42))]
        ),
    }

    if HAVE_XGB:
        models["XGBoost"] = Pipeline(
            [("scaler", StandardScaler()),
             ("clf", XGBClassifier(
                 n_estimators=300,
                 max_depth=6,
                 learning_rate=0.1,
                 subsample=0.9,
                 colsample_bytree=0.9,
                 reg_lambda=1.0,
                 tree_method="hist",   # fast on CPU
                 random_state=42
             ))]
        )
    return models


# ---- training driver ---------------------------------------------------------

def train_all_models(
    X: np.ndarray, y: np.ndarray, X_val=None, y_val=None
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Train all baseline models and return:
        - results_df: metrics table sorted by F1-macro
        - conf_mats : dict of confusion matrices per model
    If X_val/y_val are not provided, we make an 80/20 stratified split.
    """
    if X_val is None or y_val is None:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    else:
        X_tr, y_tr, X_va, y_va = X, y, X_val, y_val

    rows, cms = [], {}

    for name, model in default_models().items():
        res = _fit_eval(model, X_tr, y_tr, X_va, y_va)
        rows.append(
            {
                "model": name,
                "accuracy": res.accuracy,
                "precision_macro": res.precision_macro,
                "recall_macro": res.recall_macro,
                "f1_macro": res.f1_macro,
            }
        )
        cms[name] = res.conf_mat

    results = (
        pd.DataFrame(rows)
        .sort_values("f1_macro", ascending=False)
        .reset_index(drop=True)
    )
    return results, cms
