import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)

# ── Load Artifacts ────────────────────────────────────────────
model        = joblib.load("models/logistic_regression.pkl")
scaler       = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# ── Sidebar — Customer Input ──────────────────────────────────
st.sidebar.header("Enter Customer Details")

def user_input():
    gender           = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior           = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner          = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents       = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    tenure           = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    phone_service    = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines   = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security  = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup    = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protect   = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support     = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv     = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract         = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless        = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment          = st.sidebar.selectbox("Payment Method", [
                            "Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges  = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges    = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, 1500.0)

    # Encode to match training (LabelEncoder order: alphabetical)
    data = {
        "gender":            1 if gender == "Male" else 0,
        "SeniorCitizen":     1 if senior == "Yes" else 0,
        "Partner":           1 if partner == "Yes" else 0,
        "Dependents":        1 if dependents == "Yes" else 0,
        "tenure":            tenure,
        "PhoneService":      1 if phone_service == "Yes" else 0,
        "MultipleLines":     ["No", "No phone service", "Yes"].index(multiple_lines),
        "InternetService":   ["DSL", "Fiber optic", "No"].index(internet_service),
        "OnlineSecurity":    ["No", "No internet service", "Yes"].index(online_security),
        "OnlineBackup":      ["No", "No internet service", "Yes"].index(online_backup),
        "DeviceProtection":  ["No", "No internet service", "Yes"].index(device_protect),
        "TechSupport":       ["No", "No internet service", "Yes"].index(tech_support),
        "StreamingTV":       ["No", "No internet service", "Yes"].index(streaming_tv),
        "StreamingMovies":   ["No", "No internet service", "Yes"].index(streaming_movies),
        "Contract":          ["Month-to-month", "One year", "Two year"].index(contract),
        "PaperlessBilling":  1 if paperless == "Yes" else 0,
        "PaymentMethod":     ["Bank transfer (automatic)", "Credit card (automatic)",
                              "Electronic check", "Mailed check"].index(payment),
        "MonthlyCharges":    monthly_charges,
        "TotalCharges":      total_charges,
    }
    return pd.DataFrame([data])

df_input = user_input()

# ── Main Page ─────────────────────────────────────────────────
st.title("📉 Customer Churn Prediction Dashboard")
st.markdown("Predict whether a customer is likely to churn based on their profile.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Customer Profile")
    st.dataframe(df_input.T.rename(columns={0: "Value"}), use_container_width=True)

with col2:
    st.subheader("Churn Prediction")
    input_scaled = scaler.transform(df_input)
    prob         = model.predict_proba(input_scaled)[0][1]
    prediction   = "🔴 Will Churn" if prob > 0.5 else "🟢 Will Stay"

    st.metric("Prediction", prediction)
    st.metric("Churn Probability", f"{prob:.1%}")
    st.progress(float(prob))

    if prob > 0.7:
        st.error("⚠️ High churn risk! Consider offering a discount or loyalty reward.")
    elif prob > 0.4:
        st.warning("🔔 Moderate risk. Monitor this customer closely.")
    else:
        st.success("✅ Low churn risk. Customer appears satisfied.")

# ── Model Performance Charts ──────────────────────────────────
st.markdown("---")
st.subheader("Model Performance & Explainability")

tab1, tab2, tab3, tab4 = st.tabs([
    "ROC Curves", "Confusion Matrix", "Precision-Recall", "SHAP Feature Importance"
])

with tab1:
    st.image("data/roc_curves.png", use_container_width=True)
with tab2:
    st.image("data/confusion_matrix.png", use_container_width=True)
with tab3:
    st.image("data/precision_recall.png", use_container_width=True)
with tab4:
    st.image("data/shap_top10.png", use_container_width=True)
    st.image("data/shap_summary.png", use_container_width=True)