import streamlit as st
import pandas as pd
import yaml
import mlflow.pyfunc

# Load parameters from the YAML file
with open("/Users/salwad/mlops-vcs-sep-demo-1/mlflow/week-4-assignment-dvc-mlflow/config_params.yaml", "r") as file:
    config = yaml.safe_load(file)
sales_threshold = config['xgboost']["sales_threshold"]

# Set MLflow tracking URI and load the XGBoost model
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
logged_model = 'runs:/e6e7917263494a09b050ea2ee29c8771/XGBoost'  # Replace with your run ID and model name
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Streamlit App UI
st.title("Advertising Sales Prediction")

st.write("Enter the advertising budget for TV, radio, and newspaper to predict sales success.")

# Inputs for TV, Radio, Newspaper spending
tv = st.number_input("TV Advertising Budget", min_value=0.0, max_value=1000.0, value=100.0)
radio = st.number_input("Radio Advertising Budget", min_value=0.0, max_value=1000.0, value=50.0)
newspaper = st.number_input("Newspaper Advertising Budget", min_value=0.0, max_value=1000.0, value=25.0)

# Create a DataFrame from the input
input_data = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'radio', 'newspaper'])

# Predict button
if st.button("Predict"):
    # Make prediction
    prediction = loaded_model.predict(input_data)

    # Show the prediction
    if prediction[0] == 1:
        st.success("Prediction: High sales")
    else:
        st.warning("Prediction: Low sales")