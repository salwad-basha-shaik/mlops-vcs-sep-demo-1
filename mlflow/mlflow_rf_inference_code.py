import numpy as np
import yaml
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
import mlflow
import mlflow.sklearn

# Suppress warnings
warnings.filterwarnings('ignore')

# Load parameters from the YAML file
with open("config_params.yaml", "r") as file:
    config = yaml.safe_load(file)
params = config['random_forest']

# Step 1: Create an imbalanced binary classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=8, 
                           weights=[0.9, 0.1], flip_y=0, random_state=42)

# Verify class distribution
np.unique(y, return_counts=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Set MLflow tracking URI (ensure this matches your MLflow server setup)
mlflow.set_tracking_uri("http://127.0.0.1:5000/")  # Change if your server is elsewhere

data = X_test

import mlflow
logged_model = 'runs:/26a414f9b89e4065af165d0a29636661/Random Forest Classifier'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
y_pred = loaded_model.predict(pd.DataFrame(data))

print(y_pred)

