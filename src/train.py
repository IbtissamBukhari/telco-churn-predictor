"""Train the churn model and save it with its evaluation metrics.

Reproduces the final model from notebook 05 (Logistic Regression with
balanced class weights, C chosen by 5-fold cross-validation on recall) as a
single scikit-learn pipeline, so the API applies exactly the same
preprocessing that was used in training.

Usage:
    python -m src.train
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import load_clean
from src.preprocessor import FEATURES, build_preprocessor

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

RANDOM_STATE = 42
C_GRID = [0.01, 0.1, 1, 10, 100]


def build_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train(model_path: Path = MODEL_PATH, metrics_path: Path = METRICS_PATH) -> dict:
    df = load_clean()
    X, y = df[FEATURES], df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    search = GridSearchCV(
        build_pipeline(),
        {"model__C": C_GRID},
        cv=5,
        scoring="recall",
    )
    search.fit(X_train, y_train)
    pipeline = search.best_estimator_

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    metrics = {
        "model": "Logistic Regression",
        "C": search.best_params_["model__C"],
        "cv_recall": round(float(search.best_score_), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "churn_rate": round(float(y.mean()), 4),
        "threshold": 0.5,
        "test": {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred), 4),
            "recall": round(recall_score(y_test, pred), 4),
            "f1": round(f1_score(y_test, pred), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    # The mean of each encoded feature in the training data is the baseline
    # used to explain individual predictions (see src/predictor.py).
    encoded_train = pipeline.named_steps["preprocess"].transform(X_train)
    bundle = {
        "pipeline": pipeline,
        "baseline": np.asarray(encoded_train.mean(axis=0)),
        "metrics": metrics,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, indent=2))
    print(f"Model saved to {MODEL_PATH.relative_to(ROOT)}")
