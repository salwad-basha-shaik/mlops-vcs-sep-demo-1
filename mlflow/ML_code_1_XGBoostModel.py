import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier  # Importing the XGBoost Classifier

# Suppress warnings
warnings.filterwarnings('ignore')

# Step 1: Create an imbalanced binary classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=8, 
                           weights=[0.9, 0.1], flip_y=0, random_state=42)

# Verify class distribution
np.unique(y, return_counts=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Define the model hyperparameters for XGBoost
params = {
    "objective": "binary:logistic",  # Binary classification
    "eval_metric": "logloss",        # Evaluation metric
    "max_depth": 4,                  # Maximum tree depth
    "learning_rate": 0.1,            # Learning rate
    "n_estimators": 100,             # Number of trees
    "random_state": 8888,
}

# Train the XGBoost model
xgb_model = XGBClassifier(**params)
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb_model.predict(X_test)

# Generate and print classification report
report = classification_report(y_test, y_pred)
print(report)

# Generate classification report as a dictionary for logging
report_dict = classification_report(y_test, y_pred, output_dict=True)

# MLflow setup
mlflow.set_experiment("LRexperiment0611")
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

# Log the experiment in MLflow
with mlflow.start_run(run_name="XGBoost Model"):
    mlflow.log_params(params)
    mlflow.log_metrics({
        'accuracy': report_dict['accuracy'],
        'recall_class_0': report_dict['0']['recall'],
        'recall_class_1': report_dict['1']['recall'],
        'f1_score_macro': report_dict['macro avg']['f1-score']
    })
    mlflow.sklearn.log_model(xgb_model, "XGBoost Classifier")