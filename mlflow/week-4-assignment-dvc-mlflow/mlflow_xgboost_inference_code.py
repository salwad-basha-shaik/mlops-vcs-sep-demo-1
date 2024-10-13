import numpy as np
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import mlflow
import mlflow.pyfunc

# Suppress warnings
warnings.filterwarnings('ignore')

# Load parameters from the YAML file
with open("/Users/salwad/mlops-vcs-sep-demo-1/mlflow/week-4-assignment-dvc-mlflow/config_params.yaml", "r") as file:
    config = yaml.safe_load(file)
params = config['xgboost']

# Load dataset
data = pd.read_csv("/Users/salwad/mlops-vcs-sep-demo-1/mlflow/week-4-assignment-dvc-mlflow/Advertising.csv")
sales_threshold = params.pop("sales_threshold")

# Convert target variable into binary classes based on the threshold
data['SalesBinary'] = (data['sales'] >= sales_threshold).astype(int)
X = data[['TV', 'radio', 'newspaper']]
y = data['SalesBinary']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.pop("test_size"), random_state=params.pop("random_state"))

# Set MLflow tracking URI (ensure this matches your MLflow server setup)
mlflow.set_tracking_uri("http://127.0.0.1:5000/")  # Update this if your server is different

# Load the model from MLflow
logged_model = 'runs:/e6e7917263494a09b050ea2ee29c8771/XGBoost'  # Update with your Decision Tree model run ID and name
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Perform predictions on the test set
y_pred = loaded_model.predict(X_test)

# Print predictions
print("Predictions:", y_pred)