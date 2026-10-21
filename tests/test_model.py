import json

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_clean
from src.predictor import InvalidCustomer, get_predictor
from src.preprocessor import CATEGORIES, FEATURES
from src.train import METRICS_PATH


def test_clean_data_matches_notebooks():
    df = load_clean()
    assert df.shape == (7010, 20)
    assert df["Churn"].sum() == 1857
    assert df.isna().sum().sum() == 0


def test_categories_cover_the_data():
    df = load_clean()
    for col, values in CATEGORIES.items():
        assert set(df[col].unique()) <= set(values), col


def test_metrics_match_committed_results():
    """Retraining must reproduce the numbers reported in the README."""
    committed = json.loads(METRICS_PATH.read_text())
    current = get_predictor().metrics
    assert current["C"] == committed["C"]
    for name, value in committed["test"].items():
        assert current["test"][name] == pytest.approx(value, abs=0.002), name
    assert current["test"]["recall"] >= 0.75


def test_high_and_low_risk_profiles(risky_customer, loyal_customer):
    predictor = get_predictor()
    risky = predictor.predict(risky_customer)
    loyal = predictor.predict(loyal_customer)
    assert risky["risk_level"] == "high" and risky["will_churn"]
    assert loyal["risk_level"] == "low" and not loyal["will_churn"]


def test_factors_explain_the_prediction_exactly(risky_customer):
    """Baseline log-odds plus all contributions must equal the model's log-odds."""
    p = get_predictor()
    encoded = p.pipeline.named_steps["preprocess"].transform(
        pd.DataFrame([risky_customer])[FEATURES]
    )[0]
    contributions = p.coef * (encoded - p.baseline)
    baseline_logit = p.intercept + p.coef @ p.baseline
    prob = 1 / (1 + np.exp(-(baseline_logit + contributions.sum())))
    assert prob == pytest.approx(p.predict(risky_customer)["churn_probability"], abs=1e-4)


def test_contract_and_tenure_drive_risk(risky_customer):
    factors = {f["feature"]: f["impact"] for f in get_predictor().predict(risky_customer)["factors"]}
    assert factors["tenure"] > 0
    assert factors["Contract"] > 0


def test_longer_contract_lowers_risk(risky_customer):
    p = get_predictor()
    monthly = p.predict(risky_customer)["churn_probability"]
    two_year = p.predict({**risky_customer, "Contract": "Two year"})["churn_probability"]
    assert two_year < monthly


def test_inconsistent_internet_addons_rejected(loyal_customer):
    with pytest.raises(InvalidCustomer):
        get_predictor().predict({**loyal_customer, "InternetService": "No"})


def test_inconsistent_phone_rejected(loyal_customer):
    with pytest.raises(InvalidCustomer):
        get_predictor().predict({**loyal_customer, "PhoneService": "No"})
