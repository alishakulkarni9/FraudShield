import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection", layout="wide")

# Load assets
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")
df = pd.read_csv("data/creditcard.csv")

# Title
st.title("Credit Card Fraud Detection System")
st.markdown("Machine learning-based detection with explainable AI")

st.divider()

# --- Input Section ---
st.subheader("Transaction Selection")

mode = st.selectbox(
    "Select Transaction Type",
    ["Random", "Fraud Only", "Legitimate Only"]
)

if mode == "Fraud Only":
    df_filtered = df[df["Class"] == 1].reset_index(drop=True)
elif mode == "Legitimate Only":
    df_filtered = df[df["Class"] == 0].reset_index(drop=True)
else:
    df_filtered = df.reset_index(drop=True)

index = st.slider("Select Transaction", 0, len(df_filtered)-1, 0)

row = df_filtered.iloc[index]

X = row.drop("Class").values.reshape(1, -1)
X_scaled = scaler.transform(X)

st.caption(f"Selected Transaction ID: {index}")
actual_class = "Fraud" if row["Class"] == 1 else "Legitimate"
# st.caption(f"Actual Label: {actual_class}")

st.divider()

# --- Prediction ---
if st.button("Run Analysis"):

    # prob = model.predict_proba(X_scaled)[0][1]
    # pred = 1 if prob > 0.3 else 0
    
    prob = model.predict_proba(X_scaled)[0][1]
    threshold = 0.05   # try 0.05 first
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

    # --- SHAP Explanation ---
    st.subheader("Feature Impact Analysis")

    explainer = shap.Explainer(model)
    shap_values = explainer(X_scaled)

    fig, ax = plt.subplots()
    shap.plots.bar(shap_values[0, :, 1], show=False)
    st.pyplot(fig)
    # --- Text Explanation ---
values = shap_values[0, :, 1].values
features = df.drop("Class", axis=1).columns

top_idx = np.argsort(np.abs(values))[-5:][::-1]

st.subheader("Key Factors Influencing Prediction")

for i in top_idx:
    impact = values[i]
    direction = "increased" if impact > 0 else "decreased"

    # st.write(f"- {features[i]} {direction} the fraud probability")
    st.write(f"{features[i]} contributed to {'higher' if impact > 0 else 'lower'} fraud risk")