import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import warnings
import mlflow
import mlflow.sklearn

# Suppress warnings
warnings.filterwarnings('ignore')

# Step 1: Create an imbalanced binary classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=8, 
                           weights=[0.9, 0.1], flip_y=0, random_state=42)

# Check class distribution
print("Class distribution:", np.unique(y, return_counts=True))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 8888,
}

# Train the Logistic Regression model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Generate and print classification report
report = classification_report(y_test, y_pred)
print(report)

# Convert report to dictionary for MLflow logging
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Calculate and log accuracy separately
accuracy = accuracy_score(y_test, y_pred)

# Configure MLflow tracking URI and experiment name
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("LRexperiment0611")

# Start MLflow run
with mlflow.start_run(run_name="Logistic Regression model"):
    # Log model parameters
    mlflow.log_param("model", "Logistic Regression")
    mlflow.log_params(params)
    
    # Log metrics
    mlflow.log_metrics({
        'accuracy': accuracy,
        'recall_class_0': report_dict['0']['recall'],
        'recall_class_1': report_dict['1']['recall'],
        'f1_score_macro': report_dict['macro avg']['f1-score']
    })
    
    # Log the model
    mlflow.sklearn.log_model(lr, "Logistic Regression")  
