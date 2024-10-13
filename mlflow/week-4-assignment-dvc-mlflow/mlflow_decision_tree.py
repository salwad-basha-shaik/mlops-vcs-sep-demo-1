import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
dt_params = config['decision_tree']
test_size = dt_params.pop("test_size")
random_state = dt_params.pop("random_state")
sales_threshold = dt_params.pop("sales_threshold")

# Convert target variable into binary classes based on the threshold
data['SalesBinary'] = (data['sales'] >= sales_threshold).astype(int)
X = data[['TV', 'radio', 'newspaper']]
y = data['SalesBinary']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Initialize and train the model
dt = DecisionTreeClassifier(**dt_params)
dt.fit(X_train, y_train)

# Make predictions and calculate metrics
y_pred = dt.predict(X_test)
y_pred_proba = dt.predict_proba(X_test)[:, 1] if hasattr(dt, "predict_proba") else np.zeros(y_test.shape)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Calculate McFadden's R² (Pseudo R²)
ll_null = log_loss(y_test, [y_test.mean()] * len(y_test))  # Log-loss of a null model
ll_model = log_loss(y_test, y_pred_proba) if np.any(y_pred_proba) else None
pseudo_r2 = 1 - (ll_model / ll_null) if ll_model else None

# Set MLflow experiment and tracking URI
mlflow.set_experiment("advertising_sales_classification")
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

# Log parameters, metrics, and model to MLflow
with mlflow.start_run(run_name="Decision Tree model"):
    mlflow.log_params(dt_params)
    mlflow.log_metrics({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'pseudo_r2': pseudo_r2 if pseudo_r2 else 0.0  # Log as 0 if None
    })
    
    # Log the model
    mlflow.sklearn.log_model(dt, "Decision Tree")
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/decision_tree_model"
    mlflow.register_model(model_uri=model_uri, name="AdvertisingSalesClassificationModel_DT")
    print(f"Model registered in MLflow under name 'AdvertisingSalesClassificationModel_DT'")