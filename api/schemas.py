from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.preprocessor import consistency_errors

YesNo = Literal["Yes", "No"]
AddOn = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = Field(description="1 if the customer is 65 or older")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=72, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: AddOn
    OnlineBackup: AddOn
    DeviceProtection: AddOn
    TechSupport: AddOn
    StreamingTV: AddOn
    StreamingMovies: AddOn
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=18.0, le=120.0, description="Monthly bill in USD")

    @model_validator(mode="after")
    def check_consistency(self):
        errors = consistency_errors(self.model_dump())
        if errors:
            raise ValueError("; ".join(errors))
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 3,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 89.5,
            }
        }
    }


class Factor(BaseModel):
    feature: str
    label: str
    value: str
    impact: float = Field(description="Contribution to the log-odds of churn; positive raises risk")


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_level: Literal["low", "medium", "high"]
    factors: list[Factor]
    recommendations: list[str]


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=500)


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
