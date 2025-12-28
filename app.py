import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="Cancer Risk Prediction", layout="wide")

# Load model
with open("cancer_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🧬 Cancer Risk Prediction App")
st.write("Input patient attributes to predict cancer risk.")

# Get feature names used by model
feature_names = model.feature_names_in_

# Collect user inputs
input_data = {}
for feature in feature_names:
    val = st.sidebar.text_input(f"{feature}", "")
    input_data[feature] = val

# Convert inputs to dataframe
if st.button("Predict"):
    df = pd.DataFrame([input_data])
    
    # Try convert numeric columns to float
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except:
            pass

    # Predict
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]

    st.write("### 🔍 Prediction Output")
    st.write(f"**Predicted Class:** {prediction}")
    if model.classes_.shape[0] == 2:
        st.write(f"**Probability of Positive Class:** {probabilities[1]:.2f}")
    else:
        st.write("Class probabilities:", probabilities)

    st.success("Prediction complete!")
