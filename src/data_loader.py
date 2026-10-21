"""Download, load and clean the IBM Telco Customer Churn dataset."""

from pathlib import Path
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Same file as the Kaggle dataset "blastchar/telco-customer-churn",
# published by IBM on GitHub so it can be downloaded without a Kaggle account.
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def download_data(path: Path = RAW_PATH, url: str = DATA_URL) -> Path:
    """Download the raw CSV if it is not already present."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset to {path}")
    urllib.request.urlretrieve(url, path)
    return path


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(download_data(path))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning steps from notebook 02.

    - drop customerID (an identifier, not a feature)
    - convert TotalCharges to a number and drop the 11 brand-new customers
      whose TotalCharges is blank
    - drop exact duplicate rows
    - map Churn to 1 (Yes) and 0 (No)
    """
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).drop_duplicates().reset_index(drop=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def load_clean(path: Path = RAW_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


if __name__ == "__main__":
    data = load_clean()
    print(f"Clean dataset: {data.shape[0]} rows, {data.shape[1]} columns")
    print(data["Churn"].value_counts().rename({0: "stayed", 1: "churned"}))
