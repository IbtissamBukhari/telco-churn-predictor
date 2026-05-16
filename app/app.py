# ─────────────────────────────────────────────────────────────
# Customer Churn Predictor — Streamlit Web App
# Author: Ibtissam Bukhari
# ─────────────────────────────────────────────────────────────

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# ── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",

    layout="wide",
    initial_sidebar_state="expanded"
)


# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Source+Sans+3:wght@300;400;600&display=swap');

:root {
    --navy:       #0D1B2A;
    --navy-card:  #112240;
    --navy-light: #1A3258;
    --blue:       #4FC3F7;
    --text:       #E8F0FE;
    --text-muted: #8892A4;
}

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: var(--navy);
    color: var(--text);
}

#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}
header    {visibility: hidden;}

.main .block-container {
    padding: 2rem 2.5rem;
    max-width: 1200px;
}

[data-testid="stSidebar"] {
    background-color: #112240;
    border-right: 1px solid #1A3258;
}
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    base    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model   = joblib.load(os.path.join(base, 'models', 'churn_model.pkl'))
    scaler  = joblib.load(os.path.join(base, 'models', 'scaler.pkl'))
    columns = joblib.load(os.path.join(base, 'models', 'columns.pkl'))
    return model, scaler, columns

model, scaler, columns = load_models()

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem 0;
            border-bottom:1px solid #1A3258; margin-bottom:2rem;">
    <h1 style="font-family:Georgia,serif;
               font-size:2.8rem;
               background:linear-gradient(135deg, #4FC3F7, #818CF8);
               -webkit-background-clip:text;
               -webkit-text-fill-color:transparent;
               margin:0; letter-spacing:2px;">
        📡 CHURN INTELLIGENCE
    </h1>
    <p style="color:#8892A4; font-size:1.1rem; margin-top:0.5rem;">
        AI-powered customer retention predictor — Telco Analytics Platform
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Inputs ────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Rajdhani',sans-serif;
                font-size:1.3rem; font-weight:700;
                color:#4FC3F7; letter-spacing:1px;
                padding:1rem 0 0.5rem 0;
                border-bottom:1px solid #1A3258;
                margin-bottom:1rem;">
        ⚙ CUSTOMER PROFILE
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Demographics**")
    gender     = st.selectbox("Gender", ["Male", "Female"])
    senior     = st.radio("Senior Citizen",  ["No", "Yes"], horizontal=True)
    partner    = st.radio("Has Partner",     ["No", "Yes"], horizontal=True)
    dependents = st.radio("Has Dependents",  ["No", "Yes"], horizontal=True)

    st.markdown("---")
    st.markdown("**Service Details**")
    tenure           = st.slider("Tenure (Months)", 0, 72, 12)
    phone_service    = st.radio("Phone Service", ["No", "Yes"], horizontal=True)
    multiple_lines   = st.selectbox("Multiple Lines",
                                    ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service",
                                    ["DSL", "Fiber optic", "No"])

    st.markdown("---")
    st.markdown("**Online Services**")
    online_security   = st.selectbox("Online Security",
                                     ["No", "Yes", "No internet service"])
    online_backup     = st.selectbox("Online Backup",
                                     ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection",
                                     ["No", "Yes", "No internet service"])
    tech_support      = st.selectbox("Tech Support",
                                     ["No", "Yes", "No internet service"])
    streaming_tv      = st.selectbox("Streaming TV",
                                     ["No", "Yes", "No internet service"])
    streaming_movies  = st.selectbox("Streaming Movies",
                                     ["No", "Yes", "No internet service"])

    st.markdown("---")
    st.markdown("**Billing**")
    contract = st.selectbox("Contract Type",
                            ["Month-to-month", "One year", "Two year"])
    paperless = st.radio("Paperless Billing", ["No", "Yes"], horizontal=True)
    payment   = st.selectbox("Payment Method",
                             ["Electronic check", "Mailed check",
                              "Bank transfer (automatic)",
                              "Credit card (automatic)"])
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)

    st.markdown("---")
    predict_btn = st.button("PREDICT CHURN RISK",
                            use_container_width=True,
                            type="primary")

# ── Prepare Input ─────────────────────────────────────────────
def prepare_input(scaler, columns):
    gender_enc     = 1 if gender == "Male" else 0
    senior_enc     = 1 if senior == "Yes" else 0
    partner_enc    = 1 if partner == "Yes" else 0
    dependents_enc = 1 if dependents == "Yes" else 0
    phone_enc      = 1 if phone_service == "Yes" else 0
    paperless_enc  = 1 if paperless == "Yes" else 0

    scaled         = scaler.transform([[tenure, monthly_charges]])
    tenure_scaled  = scaled[0][0]
    charges_scaled = scaled[0][1]

    input_dict = {
        'gender'          : gender_enc,
        'SeniorCitizen'   : senior_enc,
        'Partner'         : partner_enc,
        'Dependents'      : dependents_enc,
        'tenure'          : tenure_scaled,
        'PhoneService'    : phone_enc,
        'PaperlessBilling': paperless_enc,
        'MonthlyCharges'  : charges_scaled,
    }

    input_dict['MultipleLines_No phone service'] = (
        1 if multiple_lines == "No phone service" else 0)
    input_dict['MultipleLines_Yes'] = (
        1 if multiple_lines == "Yes" else 0)

    input_dict['InternetService_Fiber optic'] = (
        1 if internet_service == "Fiber optic" else 0)
    input_dict['InternetService_No'] = (
        1 if internet_service == "No" else 0)

    input_dict['OnlineSecurity_No internet service'] = (
        1 if online_security == "No internet service" else 0)
    input_dict['OnlineSecurity_Yes'] = (
        1 if online_security == "Yes" else 0)

    input_dict['OnlineBackup_No internet service'] = (
        1 if online_backup == "No internet service" else 0)
    input_dict['OnlineBackup_Yes'] = (
        1 if online_backup == "Yes" else 0)

    input_dict['DeviceProtection_No internet service'] = (
        1 if device_protection == "No internet service" else 0)
    input_dict['DeviceProtection_Yes'] = (
        1 if device_protection == "Yes" else 0)

    input_dict['TechSupport_No internet service'] = (
        1 if tech_support == "No internet service" else 0)
    input_dict['TechSupport_Yes'] = (
        1 if tech_support == "Yes" else 0)

    input_dict['StreamingTV_No internet service'] = (
        1 if streaming_tv == "No internet service" else 0)
    input_dict['StreamingTV_Yes'] = (
        1 if streaming_tv == "Yes" else 0)

    input_dict['StreamingMovies_No internet service'] = (
        1 if streaming_movies == "No internet service" else 0)
    input_dict['StreamingMovies_Yes'] = (
        1 if streaming_movies == "Yes" else 0)

    input_dict['Contract_One year'] = (
        1 if contract == "One year" else 0)
    input_dict['Contract_Two year'] = (
        1 if contract == "Two year" else 0)

    input_dict['PaymentMethod_Credit card (automatic)'] = (
        1 if payment == "Credit card (automatic)" else 0)
    input_dict['PaymentMethod_Electronic check'] = (
        1 if payment == "Electronic check" else 0)
    input_dict['PaymentMethod_Mailed check'] = (
        1 if payment == "Mailed check" else 0)

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=columns, fill_value=0)

    return input_df

# ── Main Content ──────────────────────────────────────────────
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("""
    <div style="font-family:'Rajdhani',sans-serif;
                font-size:1.4rem; font-weight:600;
                color:#4FC3F7; letter-spacing:1px;
                border-bottom:1px solid #1A3258;
                padding-bottom:0.5rem; margin-bottom:1rem;">
        CUSTOMER RISK ANALYSIS
    </div>
    """, unsafe_allow_html=True)

    if predict_btn:
        input_df    = prepare_input(scaler, columns)
        probability = model.predict_proba(input_df)[0][1]
        pct         = round(probability * 100, 1)

        if pct >= 70:
            risk_label  = "HIGH RISK"
            risk_color  = "#FF6B6B"
            risk_icon   = "🔴"
            risk_bg     = "rgba(255,107,107,0.1)"
            risk_border = "#FF6B6B"
        elif pct >= 40:
            risk_label  = "MEDIUM RISK"
            risk_color  = "#FFB347"
            risk_icon   = "🟡"
            risk_bg     = "rgba(255,179,71,0.1)"
            risk_border = "#FFB347"
        else:
            risk_label  = "LOW RISK"
            risk_color  = "#43E97B"
            risk_icon   = "🟢"
            risk_bg     = "rgba(67,233,123,0.1)"
            risk_border = "#43E97B"

        st.markdown(f"""
        <div style="background:{risk_bg};
                    border:1px solid {risk_border};
                    border-radius:16px; padding:2rem;
                    text-align:center; margin:1rem 0;
                    box-shadow:0 0 30px {risk_bg};">
            <div style="font-family:'Rajdhani',sans-serif;
                        font-size:1.3rem; letter-spacing:2px;
                        text-transform:uppercase; margin-bottom:0.5rem;">
                {risk_icon} {risk_label}
            </div>
            <div style="font-family:'Rajdhani',sans-serif;
                        font-size:4rem; font-weight:700;
                        color:{risk_color}; line-height:1;">
                {pct}%
            </div>
            <div style="color:#8892A4; font-size:0.9rem; margin-top:0.5rem;">
                Churn Probability Score
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#1A3258; border-radius:50px;
                    height:12px; margin:1rem 0; overflow:hidden;">
            <div style="width:{pct}%; height:100%; border-radius:50px;
                        background:linear-gradient(90deg,
                        #43E97B,#FFB347,#FF6B6B);">
            </div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div style="background:#112240; border:1px solid #1A3258;
                        border-radius:12px; padding:1rem; text-align:center;">
                <div style="color:#8892A4; font-size:0.75rem;
                            letter-spacing:1px; text-transform:uppercase;">
                    Churn Score</div>
                <div style="font-family:'Rajdhani',sans-serif;
                            font-size:1.8rem; font-weight:700;
                            color:{risk_color};">{pct}%</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div style="background:#112240; border:1px solid #1A3258;
                        border-radius:12px; padding:1rem; text-align:center;">
                <div style="color:#8892A4; font-size:0.75rem;
                            letter-spacing:1px; text-transform:uppercase;">
                    Tenure</div>
                <div style="font-family:'Rajdhani',sans-serif;
                            font-size:1.8rem; font-weight:700;
                            color:#4FC3F7;">{tenure}mo</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div style="background:#112240; border:1px solid #1A3258;
                        border-radius:12px; padding:1rem; text-align:center;">
                <div style="color:#8892A4; font-size:0.75rem;
                            letter-spacing:1px; text-transform:uppercase;">
                    Monthly Bill</div>
                <div style="font-family:'Rajdhani',sans-serif;
                            font-size:1.8rem; font-weight:700;
                            color:#4FC3F7;">${monthly_charges}</div>
            </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#112240; border:1px solid #1A3258;
                    border-radius:16px; padding:3rem; text-align:center;">
            <div style="font-size:3rem;">📡</div>
            <div style="color:#8892A4; font-size:1rem;
                        margin-top:1rem; line-height:1.7;">
                Configure customer profile in the sidebar<br>
                and click PREDICT CHURN RISK
            </div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="font-family:'Rajdhani',sans-serif;
                font-size:1.4rem; font-weight:600;
                color:#4FC3F7; letter-spacing:1px;
                border-bottom:1px solid #1A3258;
                padding-bottom:0.5rem; margin-bottom:1rem;">
        BUSINESS INSIGHTS
    </div>
    """, unsafe_allow_html=True)

    if predict_btn:
        st.markdown("**Risk Factors Detected:**")
        recs = []

        if contract == "Month-to-month":
            recs.append("Month-to-month contract — offer discount for annual upgrade")
        if tenure < 12:
            recs.append("New customer (< 12 months) — critical retention window")
        if monthly_charges > 70:
            recs.append(f"High charges (${monthly_charges}) — consider loyalty discount")
        if internet_service == "Fiber optic":
            recs.append("Fiber optic — ensure service quality meets expectations")
        if senior == "Yes":
            recs.append("Senior citizen — offer dedicated support program")
        if tech_support == "No":
            recs.append("No tech support — offer free trial to improve experience")
        if online_security == "No":
            recs.append("No online security — recommend security bundle")
        if not recs:
            recs.append("Customer profile looks stable — maintain engagement")

        for rec in recs:
            st.markdown(f"""
            <div style="background:#112240;
                        border-left:3px solid #4FC3F7;
                        border-radius:0 12px 12px 0;
                        padding:0.8rem 1rem;
                        margin:0.5rem 0; font-size:0.9rem;">
                💡 {rec}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#112240; border:1px solid #1A3258;
                    border-radius:12px; padding:1rem; margin-top:1rem;">
            <div style="color:#8892A4; font-size:0.75rem;
                        letter-spacing:1px; text-transform:uppercase;">
                Model Information</div>
            <div style="font-size:0.85rem; color:#8892A4;
                        margin-top:0.5rem; line-height:1.7;">
                Algorithm: Logistic Regression<br>
                Recall: 76.3% · Accuracy: 74.1%<br>
                Training samples: 5,608 customers
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#112240; border:1px solid #1A3258;
                    border-radius:12px; padding:1.5rem;">
            <div style="color:#8892A4; font-size:0.75rem;
                        letter-spacing:1px; text-transform:uppercase;">
                About This Tool</div>
            <div style="font-size:0.9rem; color:#8892A4;
                        margin-top:0.5rem; line-height:1.7;">
                This platform uses machine learning to predict
                customer churn probability.<br><br>
                <strong style="color:#4FC3F7">Key Churn Drivers:</strong><br>
                • Month-to-month contracts<br>
                • Low tenure (new customers)<br>
                • High monthly charges<br>
                • Fiber optic service<br>
                • Senior citizens
            </div>
        </div>
        """, unsafe_allow_html=True)
