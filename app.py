from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load Models and Scaler
log_model = joblib.load("LogisticRegression_model.pkl")
rf_model = joblib.load("RandomForest_model.pkl")
dt_model = joblib.load("DecisionTree_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_choice = data['model']
    features = [
        data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'],
        data['fbs'], data['restecg'], data['thalach'], data['exang'],
        data['oldpeak'], data['slope'], data['ca'], data['thal']
    ]
    input_data = pd.DataFrame([features], columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ])

    if model_choice == "Logistic Regression":
        model = log_model
        X_scaled = scaler.transform(input_data)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0][1]
    elif model_choice == "Random Forest":
        model = rf_model
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]
    else:  # Decision Tree
        model = dt_model
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

    return jsonify({'prediction': int(pred), 'probability': round(float(proba), 2)})


if __name__ == '__main__':
    app.run(debug=True)
