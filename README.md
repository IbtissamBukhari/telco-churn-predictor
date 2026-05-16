# Telco Customer Churn Prediction

An end-to-end machine learning project focused on predicting customer churn in a telecom company using customer account, billing, and service usage data.

The project covers the complete machine learning workflow, including:

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Model training and evaluation
* Hyperparameter tuning
* Streamlit web application deployment

---

## Live Demo

[Open Application](#)

---

## Business Problem

Customer churn is a major challenge for telecom companies because retaining existing customers is significantly more cost-effective than acquiring new ones.

This project aims to identify customers who are likely to churn so that businesses can take proactive retention actions such as targeted offers, loyalty programs, and improved customer support.

---

## Key Business Insights

* Customers with month-to-month contracts showed the highest churn rate, while long-term contracts were associated with stronger customer retention.
* Most churn occurred during the early months of customer tenure, highlighting the importance of onboarding and early engagement.
* Customers with higher monthly charges were more likely to churn, suggesting pricing sensitivity among high-paying users.
* Fiber optic internet customers showed a higher churn rate despite contributing higher revenue.
* Senior citizen customers demonstrated relatively higher churn behavior, indicating the need for specialized support strategies.

---

## Project Structure

```text
telco-churn-predictor/
├── data/
│   ├── raw/                         
│   └── processed/                   
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_model_building.ipynb
│
├── app/
│   └── app.py                       
│
├── models/
│   ├── churn_model.pkl              
│   ├── scaler.pkl                   
│   └── columns.pkl                  
│
├── outputs/
│   └── figures/                     
│
├── src/                             
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 74.1% |
| Precision | 50.7% |
| Recall    | 76.3% |
| F1 Score  | 60.9% |

Recall was prioritized because failing to identify a customer likely to churn is more costly than generating a false positive prediction.

---

## Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* Conda

---

## Setup Instructions

```bash
# Clone repository
git clone https://github.com/IbtissamBukhari/telco-churn-predictor.git

# Navigate to project folder
cd telco-churn-predictor

# Create environment
conda create -n churn-env python=3.10

# Activate environment
conda activate churn-env

# Install dependencies
pip install -r requirements.txt

# Run Streamlit application
streamlit run app/app.py
```

---

## Dataset

Telco Customer Churn Dataset from Kaggle:

[https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Author

Built by Ibtissam Bukhari
