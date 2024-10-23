
# Diabetes Prediction Model

This project demonstrates the creation of a machine learning model for predicting diabetes, using a Logistic Regression algorithm. The model is trained using the Pima Indians Diabetes dataset and deployed via two different approaches:
1. Flask API for serving the model as a REST API.
2. Streamlit application for interactive predictions.

## Project Steps
1. **Data Preprocessing**
2. **Model Training**
3. **Model Persistence with Joblib**
4. **Serving the Model with Flask**
5. **Building a Streamlit Application for User Interaction**

### Libraries Used:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- Flask
- joblib
- streamlit
- dagshub
- mlflow

## 1. Data Preprocessing

The Pima Indians Diabetes dataset contains various health-related attributes, such as pregnancies, glucose level, blood pressure, etc. We will use this data to predict whether a patient has diabetes or not.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load dataset
data = './diabetes.csv'
df = pd.read_csv(data)

# Features and labels
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Handle missing values
fill = SimpleImputer(missing_values=0, strategy="mean")
X_train = fill.fit_transform(X_train)
X_test = fill.fit_transform(X_test)
```

## 2. Model Training

We train a Logistic Regression model on the preprocessed dataset.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Define model parameters
params = {
    "solver": "lbfgs",
    "max_iter": 12,
    "multi_class": "auto",
    "random_state": 123,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Evaluate the model
y_pred = lr.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)
```

## 3. Save the Model Using Joblib

We save the trained model using `joblib` for later use.

```python
import joblib

# Save the model to a file
joblib.dump(lr, 'logistic_regression_model.joblib')
```

## 4. Serving the Model with Flask

### Flask API to Serve the Model

We will create a simple Flask API that loads the saved model and predicts based on input data.

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('logistic_regression_model.joblib')

@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['features']])
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
```

### Run Flask

To run the Flask app, execute:

```bash
python app.py
```

### Make a Prediction Request

To send data for prediction, you can use curl or Postman:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"features": [5,116,74,0,0,25.6,0.201,30]}'
```

## 5. Streamlit Application for Model Deployment

The Streamlit application provides a user-friendly interface for entering feature values and displaying prediction results.

```python
import streamlit as st
import joblib
import numpy as np

# Load the trained Logistic Regression model
model = joblib.load('logistic_regression_model.joblib')

st.title("Diabetes Prediction")

def predict_diabetes(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return int(prediction[0])

# Input fields for each feature
st.sidebar.header('Input Features')
pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=1)
glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=200, value=120)
blood_pressure = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=122, value=70)
skin_thickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=99, value=20)
insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=846, value=80)
bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=67.1, value=25.0)
dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
age = st.sidebar.number_input('Age', min_value=1, max_value=120, value=30)

input_features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]

if st.button("Predict"):
    result = predict_diabetes(input_features)
    
    if result == 1:
        st.success("The model predicts that the patient is likely to have diabetes.")
    else:
        st.success("The model predicts that the patient is unlikely to have diabetes.")
```

### Run Streamlit

To run the Streamlit app:

```bash
streamlit run app.py
```

## Project Structure

```bash
.
├── app.py                      # Flask API or Streamlit app
├── logistic_regression_model.joblib  # Saved model
├── diabetes.csv                # Dataset
└── README.md                   # Project Documentation
```

## Conclusion

This project demonstrates end-to-end machine learning with logistic regression, saving the model, and serving it via Flask and Streamlit. You can further enhance this project by adding more models or improving the user interface in Streamlit.
