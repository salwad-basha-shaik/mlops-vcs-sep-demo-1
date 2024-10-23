import streamlit as st
import joblib
import numpy as np

# Load the trained Logistic Regression model
model = joblib.load('logistic_regression_model.joblib')

# Streamlit app title
st.title("Diabetes Prediction")

# Function to make predictions
def predict_diabetes(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return int(prediction[0])

# Create input fields for each feature
st.sidebar.header('Input Features')
pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=1)
glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=200, value=120)
blood_pressure = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=122, value=70)
skin_thickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=99, value=20)
insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=846, value=80)
bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=67.1, value=25.0)
dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
age = st.sidebar.number_input('Age', min_value=1, max_value=120, value=30)

# Collect input features
input_features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]

# Predict when the button is clicked
if st.button("Predict"):
    result = predict_diabetes(input_features)
    
    if result == 1:
        st.success("The model predicts that the patient is likely to have diabetes.")
    else:
        st.success("The model predicts that the patient is unlikely to have diabetes.")

# Additional description
st.write("""
### Feature Descriptions:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
- **Blood Pressure**: Diastolic blood pressure (mm Hg)
- **Skin Thickness**: Triceps skinfold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **Diabetes Pedigree Function**: Diabetes pedigree function score
- **Age**: Age in years
""")