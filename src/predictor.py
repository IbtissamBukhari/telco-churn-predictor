"""Load the trained model and score individual customers."""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessor import FEATURES, consistency_errors, source_feature

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"

HIGH_RISK = 0.70
MEDIUM_RISK = 0.40

LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Has partner",
    "Dependents": "Has dependents",
    "tenure": "Tenure",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
    "MonthlyCharges": "Monthly charges",
}


class InvalidCustomer(ValueError):
    pass


class ChurnPredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Train the model first: python -m src.train"
            )
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.baseline = bundle["baseline"]
        self.metrics = bundle["metrics"]

        preprocess = self.pipeline.named_steps["preprocess"]
        model = self.pipeline.named_steps["model"]
        self.encoded_names = list(preprocess.get_feature_names_out())
        self.coef = model.coef_[0]
        self.intercept = float(model.intercept_[0])

    def predict(self, customer: dict) -> dict:
        errors = consistency_errors(customer)
        if errors:
            raise InvalidCustomer("; ".join(errors))

        row = pd.DataFrame([{col: customer[col] for col in FEATURES}])
        probability = float(self.pipeline.predict_proba(row)[0, 1])
        encoded = self.pipeline.named_steps["preprocess"].transform(row)[0]

        return {
            "churn_probability": round(probability, 4),
            "will_churn": probability >= self.metrics["threshold"],
            "risk_level": risk_level(probability),
            "factors": self.explain(encoded, customer),
            "recommendations": recommendations(customer),
        }

    def explain(self, encoded: np.ndarray, customer: dict, top: int = 6) -> list[dict]:
        """Split the prediction into the contribution of each input.

        For logistic regression the log-odds are a sum of coef x value, so the
        contribution of a feature is exact: coef x (its value - the average
        training customer's value). Positive values push the churn risk up,
        negative values push it down. One-hot columns are summed back into
        the original feature (for example both Contract columns into Contract).
        """
        contributions: dict[str, float] = {}
        for name, coef, value, base in zip(
            self.encoded_names, self.coef, encoded, self.baseline
        ):
            source = source_feature(name)
            contributions[source] = contributions.get(source, 0.0) + coef * (value - base)

        ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return [
            {
                "feature": feature,
                "label": LABELS.get(feature, feature),
                "value": display_value(feature, customer[feature]),
                "impact": round(float(impact), 4),
            }
            for feature, impact in ranked[:top]
            if abs(impact) >= 0.01
        ]


def risk_level(probability: float) -> str:
    if probability >= HIGH_RISK:
        return "high"
    if probability >= MEDIUM_RISK:
        return "medium"
    return "low"


def display_value(feature: str, value) -> str:
    if feature == "tenure":
        return f"{value} months"
    if feature == "MonthlyCharges":
        return f"${value:.2f}"
    if feature == "SeniorCitizen":
        return "Yes" if value else "No"
    return str(value)


def recommendations(customer: dict) -> list[str]:
    recs = []
    if customer["Contract"] == "Month-to-month":
        recs.append("Month-to-month contract: offer a discount to move to an annual plan.")
    if customer["tenure"] < 12:
        recs.append("New customer (under 12 months): this is the critical retention window.")
    if customer["MonthlyCharges"] > 70:
        recs.append(
            f"High monthly bill (${customer['MonthlyCharges']:.2f}): consider a loyalty discount."
        )
    if customer["InternetService"] == "Fiber optic":
        recs.append("Fiber optic customer: check that service quality meets expectations.")
    if customer["SeniorCitizen"] == 1:
        recs.append("Senior citizen: offer a dedicated support programme.")
    if customer["TechSupport"] == "No":
        recs.append("No tech support: offer a free trial to improve the experience.")
    if customer["OnlineSecurity"] == "No":
        recs.append("No online security: recommend the security bundle.")
    if customer["PaymentMethod"] == "Electronic check":
        recs.append("Pays by electronic check: encourage automatic payment.")
    if not recs:
        recs.append("Profile looks stable: keep up regular engagement.")
    return recs


@lru_cache(maxsize=1)
def get_predictor() -> ChurnPredictor:
    return ChurnPredictor()
