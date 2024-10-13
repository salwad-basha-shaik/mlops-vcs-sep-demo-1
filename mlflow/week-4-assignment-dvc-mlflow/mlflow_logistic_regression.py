import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import warnings
import mlflow
import mlflow.sklearn
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_csv("/Users/salwad/mlops-vcs-sep-demo-1/mlflow/week-4-assignment-dvc-mlflow/Advertising.csv")

# Load parameters from the YAML file
with open("/Users/salwad/mlops-vcs-sep-demo-1/mlflow/week-4-assignment-dvc-mlflow/config_params.yaml", "r") as file:
    config = yaml.safe_load(file)
lr_params = config['logistic_regression']
test_size = lr_params.pop("test_size")
random_state = lr_params.pop("random_state")
sales_threshold = lr_params.pop("sales_threshold")

# Convert target variable into binary classes based on the threshold
data['SalesBinary'] = (data['sales'] >= sales_threshold).astype(int)
X = data[['TV', 'radio', 'newspaper']]
y = data['SalesBinary']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Initialize and train the model
lr = LogisticRegression(**lr_params)
lr.fit(X_train, y_train)

# Make predictions and calculate metrics
y_pred = lr.predict(X_test)
y_pred_proba = lr.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Calculate McFadden's R² (Pseudo R²)
ll_null = log_loss(y_test, [y_test.mean()] * len(y_test))  # Log-loss of a null model
ll_model = log_loss(y_test, y_pred_proba)                  # Log-loss of the fitted model
pseudo_r2 = 1 - (ll_model / ll_null)

# Set MLflow experiment and tracking URI
mlflow.set_experiment("advertising_sales_classification")
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

# Log parameters, metrics, and model to MLflow
with mlflow.start_run(run_name="Logistic Regression model"):
    mlflow.log_params(lr_params)
    mlflow.log_metrics({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'pseudo_r2': pseudo_r2
    })
    
    # Log the model
    mlflow.sklearn.log_model(lr, "Logistic Regression")
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/logistic_regression_model"
    mlflow.register_model(model_uri=model_uri, name="AdvertisingSalesClassificationModel")
    print(f"Model registered in MLflow under name 'AdvertisingSalesClassificationModel'")