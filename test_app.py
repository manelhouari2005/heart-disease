import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('RandomForest_model.pkl')

st.title(" Heart Disease Prediction App")
st.write("Enter your health information below:")

# Collect user input
age = st.number_input("Age", 1, 120, 25)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 0)
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
restecg = st.number_input("Rest ECG (0-2)", 0, 2, 1)
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
oldpeak = st.number_input("ST depression induced by exercise", 0.0, 10.0, 1.0)
slope = st.number_input("Slope (0-2)", 0, 2, 1)
ca = st.number_input("Number of major vessels (0-3)", 0, 3, 0)
thal = st.number_input("Thal (0-3)", 0, 3, 1)

# Prediction
if st.button("Predict"):
    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(user_input)[0]
    if prediction == 1:
        st.success("⚠️ The model predicts a **Heart Disease Risk!**")
    else:
        st.success("✅ The model predicts **No Heart Disease Risk.**")
