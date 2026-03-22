import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection", layout="wide")

# Load model and scaler
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ---------------- SAMPLE TRANSACTIONS ----------------
# Each entry = (features array, label)
# NOTE: Replace with real samples from your dataset if needed

sample_data = [
    {
        "features": [0.1]*30,
        "label": 0
    },
    {
        "features": [-1.2]*30,
        "label": 1
    },
    {
        "features": [0.5]*30,
        "label": 0
    },
    {
        "features": [-0.8]*30,
        "label": 1
    }
]

# ---------------- UI ----------------
st.title("Credit Card Fraud Detection System")
st.markdown("Machine learning-based detection with explainable AI")

st.divider()

# --- Selection ---
st.subheader("Transaction Selection")

mode = st.selectbox(
    "Select Transaction Type",
    ["Random", "Fraud Only", "Legitimate Only"]
)

# Filter sample data
if mode == "Fraud Only":
    filtered = [d for d in sample_data if d["label"] == 1]
elif mode == "Legitimate Only":
    filtered = [d for d in sample_data if d["label"] == 0]
else:
    filtered = sample_data

index = st.slider("Select Transaction", 0, len(filtered)-1, 0)

selected = filtered[index]

X = np.array(selected["features"]).reshape(1, -1)
X_scaled = scaler.transform(X)

actual_class = "Fraud" if selected["label"] == 1 else "Legitimate"

st.caption(f"Selected Transaction ID: {index}")
st.caption(f"Actual Label: {actual_class}")

st.divider()

# --- Prediction ---
if st.button("Run Analysis"):

    prob = model.predict_proba(X_scaled)[0][1]
    threshold = 0.05
    pred = 1 if prob > threshold else 0

    st.caption(f"Decision Threshold: {threshold}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prediction")

        if pred == 1:
            st.error("Fraudulent Transaction")
        else:
            st.success("Legitimate Transaction")

    with col2:
        st.subheader("Confidence Score")
        st.metric(label="Fraud Probability", value=f"{prob:.4f}")

    st.divider()

    # --- SHAP ---
    st.subheader("Feature Impact Analysis")

    explainer = shap.Explainer(model)
    shap_values = explainer(X_scaled)

    fig, ax = plt.subplots()
    shap.plots.bar(shap_values[0, :, 1], show=False)
    st.pyplot(fig)

    # --- Text Explanation ---
    values = shap_values[0, :, 1].values

    # Dummy feature names (since dataset removed)
    feature_names = [f"V{i}" for i in range(len(values))]

    top_idx = np.argsort(np.abs(values))[-5:][::-1]

    st.subheader("Key Factors Influencing Prediction")

    for i in top_idx:
        impact = values[i]
        st.write(f"{feature_names[i]} contributed to {'higher' if impact > 0 else 'lower'} fraud risk")
