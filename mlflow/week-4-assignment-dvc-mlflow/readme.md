# Advertising Sales Classification Model

This project implements a Logistic Regression model to predict sales based on advertising data. The dataset includes information on TV, radio, and newspaper advertising expenditures, and the model aims to classify whether the sales are high or low based on a specified threshold.

Tools used: Git, DVC, MLFLOW, StreamLit, FLASK, FAST API, 

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Model Training](#model-training)
- [Logging with MLflow](#logging-with-mlflow)
- [How to Run the Code](#how-to-run-the-code)

## Project Overview

The project uses the `pandas` library for data manipulation, `scikit-learn` for implementing the Logistic Regression model, and `mlflow` for tracking experiments and managing models. The primary goal is to predict whether sales will be above or below a certain threshold based on advertising spending.

## Requirements

Make sure to have the following libraries installed:

- pandas
- numpy
- scikit-learn
- mlflow
- PyYAML

You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn mlflow pyyaml
```
---

### Demo-1: Training with advertisement dataset by taking parameters, and then pushing the model to mlflow.

#### Steps to Follow:

1. **Make sure you run mlflow ui in 1st terminal window and remaining below runs has to be run from seperate terminal window and also we have to run it from main ‘mlruns’ folder that you have run your mlflow at first time so you have all experiments history and then you have to run your python file with model experment from that directory then it will work.**

   ```bash
   mlflow ui 
   ```

2. **i have tried with 4 different models and you can run use below commands**
   Navigate to the `mlflow` folder where you have mlruns folder and execute the following steps:

   ```bash
   python week-4-assignment-dvc-mlflow/mlflow_logistic_regression.py
   python week-4-assignment-dvc-mlflow/mlflow_random_forest.py
   python mlflow/week-4-assignment-dvc-mlflow/mlflow_decision_tree.py
   python week-4-assignment-dvc-mlflow/mlflow_xgboost.py
   ```

**After running above 4 models, go to mlflow UI 127.0.0.1:/5000 and compare 4 of them and you will get to see that Decisiontree is having best f1_score with 0.966 so this is the best model out of all 4 models so you can register this model also if you see r2_score XGBoost is having best score with 0.896 so based on discussion we can register.**

---

### Demo-2: whatever we have trained and pushed the model to mlflow, now we can load that model and test with random data and predict the sales.

#### Steps to Follow:

1. **Make sure you run mlflow ui in 1st terminal window and remaining below runs has to be run from seperate terminal window and also we have to run it from main ‘mlruns’ folder that you have run your mlflow at first time so you have all experiments history and then you have to run your python file with model experment from that directory then it will work.**

   ```bash
   mlflow ui 
   ```

2. **i have tried with 2 different models and you can run use below commands**
   Navigate to the `mlflow` folder where you have mlruns folder and execute the following steps:

   ```bash
   python week-4-assignment-dvc-mlflow/mlflow_xgboost_inference_code.py
   python week-4-assignment-dvc-mlflow/mlflow_decisiontree_inference_code.py
   ```

   **if output returns 0 that means it is low sales and 1 means high sales.**

---

### Demo-3: whatever we have trained and pushed the model to mlflow, now we can load that model and test with our own data by takiung 3 inputs from user from UI using streamlit and predict and display high or low sales.

#### Steps to Follow:

1. **Make sure you run mlflow ui in 1st terminal window and remaining below runs has to be run from seperate terminal window and also we have to run it from main ‘mlruns’ folder that you have run your mlflow at first time so you have all experiments history and then you have to run your python file with model experment from that directory then it will work.**

   ```bash
   mlflow ui 
   ```

2. **i have tried with 2 different models and you can run use below commands**
   Navigate to the `mlflow` folder where you have mlruns folder and execute the following steps:

   ```bash
   python week-4-assignment-dvc-mlflow/mlflow_streamlit_taking_inputs_and_test.py
   ```

   **if output returns 0 that means it is low sales and 1 means high sales.**


---

### Demo-4: whatever we have trained and pushed the model to mlflow, now we can load that model and test with our own data by takiung 3 inputs from user from flask and FLASK API using postman or using CURL command and predict and display high or low sales.

#### Steps to Follow:

1. **Make sure you run mlflow ui in 1st terminal window and remaining below runs has to be run from seperate terminal window and also we have to run it from main ‘mlruns’ folder that you have run your mlflow at first time so you have all experiments history and then you have to run your python file with model experment from that directory then it will work.**

   ```bash
   mlflow ui 
   ```

2. **i have tried with 2 different models and you can run use below commands**
   Navigate to the `mlflow` folder where you have mlruns folder and execute the following steps:

   ```bash
   python week-4-assignment-dvc-mlflow/mlflow_flaskapi_taking_inputs_and_test.py
   python week-4-assignment-dvc-mlflow/mlflow_fastapi_taking_inputs_and_test.py
   ```

   **use below input data from postman to test and also output as follows**

    input data with URL :http://127.0.0.1:5001/predict (flask API) and http://127.0.0.1:5002/predict (FAST API)

    {
    "TV": 100.0,
    "radio": 50.0,
    "newspaper": 25.0
    }

    output:
    {
    "prediction": 0,
    "message": "Low sales"
    }

    **use below CURL command to test flask API**
    curl -X POST http://127.0.0.1:5001/predict \-H "Content-Type: application/json" \-d '{"TV": 100.0, "radio": 50.0, "newspaper": 25.0}'
    
    output:
    {
    "message": "Low sales",
    "prediction": 0
    }