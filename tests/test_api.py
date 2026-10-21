import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_frontend_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Telco Churn Predictor" in res.text
    assert client.get("/static/app.js").status_code == 200


def test_model_info(client):
    info = client.get("/model-info").json()
    assert info["model"] == "Logistic Regression"
    assert set(info["test"]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_predict(client, risky_customer):
    res = client.post("/predict", json=risky_customer)
    assert res.status_code == 200
    body = res.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] == "high"
    assert body["factors"] and body["recommendations"]


def test_predict_batch(client, risky_customer, loyal_customer):
    res = client.post("/predict/batch", json={"customers": [risky_customer, loyal_customer]})
    assert res.status_code == 200
    levels = [p["risk_level"] for p in res.json()["predictions"]]
    assert levels == ["high", "low"]


@pytest.mark.parametrize(
    "change",
    [
        {"Contract": "Three year"},
        {"tenure": -1},
        {"tenure": 100},
        {"MonthlyCharges": 5},
        {"SeniorCitizen": 2},
        {"InternetService": "No"},
    ],
)
def test_invalid_input_rejected(client, loyal_customer, change):
    assert client.post("/predict", json={**loyal_customer, **change}).status_code == 422


def test_missing_field_rejected(client, loyal_customer):
    del loyal_customer["Contract"]
    assert client.post("/predict", json=loyal_customer).status_code == 422
