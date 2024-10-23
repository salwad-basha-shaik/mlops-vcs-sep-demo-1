
# Logistic Regression Experiment with DagsHub and MLflow

This repository demonstrates how to train a Logistic Regression model on a diabetes dataset, track experiments, and log results using DagsHub and MLflow.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
- [Results](#results)
- [Saving and Reloading the Model](#saving-and-reloading-the-model)

## Prerequisites

- Python 3.x
- MLflow
- DagsHub account
- `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

## Installation

Install the required libraries using the following command:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn dagshub mlflow
```

## Dataset

The dataset used in this project is the diabetes dataset (`diabetes.csv`). Ensure it is placed in the same directory as the code.

```python
import pandas as pd

data = './diabetes.csv'
df = pd.read_csv(data)
print(df.shape)
print(df.head())
```

## Model Training

### 1. Splitting the Data

We split the dataset into training and test sets:

```python
from sklearn.model_selection import train_test_split

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)
```

### 2. Data Preprocessing

We handle missing data by using `SimpleImputer`:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=0, strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.fit_transform(X_test)
```

### 3. Training the Model

We initialize and train the logistic regression model:

```python
from sklearn.linear_model import LogisticRegression

params = {
    "solver": "lbfgs",
    "max_iter": 12,
    "multi_class": "auto",
    "random_state": 123,
}

lr = LogisticRegression(**params)
lr.fit(X_train, y_train)
```

### 4. Model Prediction and Evaluation

We predict and evaluate the model:

```python
y_pred = lr.predict(X_test)

from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred)
print(report)

report_dict = classification_report(y_test, y_pred, output_dict=True)
```

## Experiment Tracking with MLflow

### 1. Initialize DagsHub Integration

First, ensure you have connected your DagsHub repository for MLflow tracking:

```python
import dagshub

dagshub.init(repo_owner='your-username', repo_name='your-repo', mlflow=True)
```

### 2. Log Experiments with MLflow

Start logging the experiment with MLflow:

```python
import mlflow

mlflow.set_experiment("LRexperimentdiabetes")

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics({
        'accuracy': report_dict['accuracy'],
        'recall_class_0': report_dict['0']['recall'],
        'recall_class_1': report_dict['1']['recall'],
        'f1_score_macro': report_dict['macro avg']['f1-score']
    })
```

## Results

You can view the experiment results on DagsHub under your repository's MLflow section.

## Saving and Reloading the Model

To save the trained model:

```python
import pickle

with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr, f)
```

To reload the saved model:

```python
with open('logistic_regression_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

## License

This project is licensed under the MIT License.
