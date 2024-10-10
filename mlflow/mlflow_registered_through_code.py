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
rf_params = config['random_forest']

# Step 1: Create an imbalanced binary classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=8, 
                           weights=[0.9, 0.1], flip_y=0, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Initialize and train the model
rf = RandomForestClassifier(**rf_params)
rf.fit(X_train, y_train)

# Make predictions and calculate classification report
y_pred = rf.predict(X_test)
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Set MLflow experiment and tracking URI
mlflow.set_experiment("new_experiments_demo")
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

# Log parameters, metrics, and model to MLflow
with mlflow.start_run():
    mlflow.log_params(rf_params)
    mlflow.log_metrics({
        'accuracy': report_dict['accuracy'],
        'recall_class_0': report_dict['0']['recall'],
        'recall_class_1': report_dict['1']['recall'],
        'f1_score_macro': report_dict['macro avg']['f1-score']
    })
    
    # Register the model
    mlflow.sklearn.log_model(rf, "Random Forest")
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/random_forest_model"
    mlflow.register_model(model_uri=model_uri, name="RandomForestModel")
    print(f"Model registered in MLflow under name 'RandomForestModel'")