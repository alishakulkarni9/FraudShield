import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection", layout="wide")

# Load model
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ---------------- SAMPLE DATA ----------------
sample_data = [
    {"features": [0.1]*30, "label": 0},
    {"features": [-1.2]*30, "label": 1},
    {"features": [0.5]*30, "label": 0},
    {"features": [-0.8]*30, "label": 1}
]

# ---------------- UI ----------------
st.title("🛡️ FraudShield - Credit Card Fraud Detection")
st.markdown("AI-powered fraud detection with explainable insights")

st.divider()

# ---------------- MODE SELECTION ----------------
mode = st.radio(
    "Choose Input Method",
    ["Use Sample Transaction", "Enter Transaction Manually"]
)

# ---------------- INPUT HANDLING ----------------
if mode == "Use Sample Transaction":

    index = st.slider("Select Sample Transaction", 0, len(sample_data)-1, 0)
    selected = sample_data[index]

    X = np.array(selected["features"]).reshape(1, -1)
    actual_class = "Fraud" if selected["label"] == 1 else "Legitimate"

    st.caption(f"Sample Transaction ID: {index}")
    st.caption(f"Actual Label: {actual_class}")

else:
    st.subheader("Enter Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input("Transaction Amount", value=100.0)
        v1 = st.number_input("V1", value=0.0)
        v2 = st.number_input("V2", value=0.0)

    with col2:
        v3 = st.number_input("V3", value=0.0)
        v4 = st.number_input("V4", value=0.0)
        v5 = st.number_input("V5", value=0.0)

    # Fill remaining features with 0
    features = [v1, v2, v3, v4, v5] + [0]*25
    X = np.array(features).reshape(1, -1)

# Scale input
X_scaled = scaler.transform(X)

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🔍 Run Fraud Analysis"):

    prob = model.predict_proba(X_scaled)[0][1]
    threshold = 0.05
    pred = 1 if prob > threshold else 0

    st.caption(f"Decision Threshold: {threshold}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prediction")

        if pred == 1:
            st.error("🚨 Fraudulent Transaction")
        else:
            st.success("✅ Legitimate Transaction")

    with col2:
        st.subheader("Confidence Score")
        st.metric("Fraud Probability", f"{prob:.4f}")

    st.divider()

    # ---------------- SHAP ----------------
    st.subheader("📊 Feature Impact Analysis")

    explainer = shap.Explainer(model)
    shap_values = explainer(X_scaled)

    fig, ax = plt.subplots()
    shap.plots.bar(shap_values[0, :, 1], show=False)
    st.pyplot(fig)

    # ---------------- TEXT EXPLANATION ----------------
    values = shap_values[0, :, 1].values
    feature_names = [f"V{i}" for i in range(len(values))]

    top_idx = np.argsort(np.abs(values))[-5:][::-1]

    st.subheader("🔍 Key Influencing Features")

    for i in top_idx:
        impact = values[i]
        st.write(f"{feature_names[i]} contributed to {'higher' if impact > 0 else 'lower'} fraud risk")
