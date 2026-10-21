"""Feature definitions and the preprocessing step of the model pipeline.

The encoding matches notebook 04: binary columns become 0/1, multi-category
columns are one-hot encoded with the first category dropped, and tenure and
MonthlyCharges are scaled to 0-1. TotalCharges is not used because it is
almost tenure x MonthlyCharges (correlation with tenure 0.83).

Unlike the notebook, the scaler is part of the pipeline, so it is fitted on
the training split only and the test set stays unseen.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

NUMERIC = ["tenure", "MonthlyCharges"]

# Every categorical feature with its allowed values. The first value of each
# list is the reference category that gets dropped during encoding.
CATEGORIES = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "PaperlessBilling": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}

CATEGORICAL = list(CATEGORIES)
PASSTHROUGH = ["SeniorCitizen"]
FEATURES = NUMERIC + CATEGORICAL + PASSTHROUGH

INTERNET_ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def consistency_errors(customer: dict) -> list[str]:
    """Return the combinations that cannot occur in the real data.

    In the dataset, a customer without internet always has 'No internet
    service' for every internet add-on, and a customer without phone service
    always has 'No phone service' for MultipleLines (and the reverse).
    """
    errors = []
    no_internet = customer["InternetService"] == "No"
    for col in INTERNET_ADDONS:
        if no_internet != (customer[col] == "No internet service"):
            errors.append(
                f"{col} must be 'No internet service' exactly when InternetService is 'No'"
            )
    no_phone = customer["PhoneService"] == "No"
    if no_phone != (customer["MultipleLines"] == "No phone service"):
        errors.append(
            "MultipleLines must be 'No phone service' exactly when PhoneService is 'No'"
        )
    return errors


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", MinMaxScaler(), NUMERIC),
            (
                "cat",
                OneHotEncoder(
                    categories=[CATEGORIES[c] for c in CATEGORICAL],
                    drop="first",
                    sparse_output=False,
                ),
                CATEGORICAL,
            ),
            ("pass", "passthrough", PASSTHROUGH),
        ],
        verbose_feature_names_out=False,
    )


def source_feature(encoded_name: str) -> str:
    """Map an encoded column such as 'Contract_Two year' back to 'Contract'."""
    for col in FEATURES:
        if encoded_name == col or encoded_name.startswith(col + "_"):
            return col
    return encoded_name
