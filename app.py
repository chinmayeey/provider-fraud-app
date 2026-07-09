import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Provider Fraud Predictor", layout="centered")

# Load the trained model + the exact feature list it expects (saved in the notebook)
model = joblib.load("fraud_model.pkl")
features = joblib.load("model_features.pkl")

st.title("🏥 Healthcare Provider Fraud Predictor")
st.write(
    "Upload a CSV of **provider-level features** (the same columns produced by the "
    "notebook's Stage B feature engineering) to get a fraud prediction for each provider."
)

with st.expander("What columns does this need?"):
    st.write(features)

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Make sure every column the model needs is present before predicting
    missing = [f for f in features if f not in data.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        X = data[features].fillna(0)
        proba = model.predict_proba(X)[:, 1]
        pred = model.predict(X)

        results = data.copy()
        results["FraudProbability"] = proba.round(4)
        results["PredictedFraud"] = pd.Series(pred).map({1: "Yes", 0: "No"})

        st.success(f"Predictions generated for {len(results)} provider(s)!")
        st.dataframe(results)

        st.download_button(
            "Download predictions as CSV",
            results.to_csv(index=False),
            file_name="predictions.csv",
        )
