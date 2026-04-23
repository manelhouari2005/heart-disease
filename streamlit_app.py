import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------
# Load Models and Scaler
# --------------------------
log_model = joblib.load("LogisticRegression_model.pkl")
rf_model = joblib.load("RandomForest_model.pkl")
dt_model = joblib.load("DecisionTree_model.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------------
# Streamlit Interface
# --------------------------
st.set_page_config(page_title="❤️ Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.markdown("This app predicts the **probability of heart disease** using three models: "
            "Logistic Regression, Random Forest, and Decision Tree.")

# Sidebar
st.sidebar.header("🔧 Choose Model")
model_choice = st.sidebar.selectbox(
    "Select a model to use:",
    ("Logistic Regression", "Random Forest", "Decision Tree")
)

st.sidebar.info("Upload data or enter manually below.")

# --------------------------
# User Input Form
# --------------------------
st.subheader("🩺 Enter Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)

with col2:
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 70, 210, 150)

with col3:
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

# Collect input into a DataFrame
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]],
                          columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                   'restecg', 'thalach', 'exang', 'oldpeak',
                                   'slope', 'ca', 'thal'])

# --------------------------
# Predict Button
# --------------------------
if st.button("🔍 Predict"):
    # Choose the model
    if model_choice == "Logistic Regression":
        model = log_model
        X_scaled = scaler.transform(input_data)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0][1]
    elif model_choice == "Random Forest":
        model = rf_model
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]
    else:
        model = dt_model
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

    # --------------------------
    # Display Results
    # --------------------------
    st.subheader("🧾 Prediction Result:")
    if pred == 1:
        st.error(f"⚠️ The model predicts **Heart Disease** (Probability: {proba:.2f})")
    else:
        st.success(f"✅ The model predicts **No Heart Disease** (Probability: {proba:.2f})")

    st.caption(f"Model used: **{model_choice}**")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("👩‍⚕️ *Built with Streamlit — Heart Disease Detection Project*")
