import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.metrics import classification_report

data = './diabetes.csv'
df = pd.read_csv(data)
df.shape

df.head()
X = df.drop('Outcome',axis=1) # predictor feature coloumns
y = df.Outcome


X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.20, random_state = 10)

print('Training Set :',len(X_train))
print('Test Set :',len(X_test))
print('Training labels :',len(y_train))
print('Test Labels :',len(y_test))

from sklearn.impute import SimpleImputer
#impute with mean all 0 readings
fill = SimpleImputer(missing_values = 0 , strategy ="mean")#impute with mean all 0 readings

#fill = Imputer(missing_values = 0 , strategy ="mean", axis=0)

X_train = fill.fit_transform(X_train)
X_test = fill.fit_transform(X_test)



# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 20,
    "multi_class": "auto",
    "random_state": 140,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)

report_dict = classification_report(y_test, y_pred, output_dict=True)
report_dict

import joblib

# Save the model
joblib.dump(lr, 'logistic_regression_model.joblib')

# To load the model later
# lr_model = joblib.load('logistic_regression_model.joblib')

import dagshub
dagshub.init(repo_owner='salwad-basha-shaik', repo_name='mlops-vcs-sep-demo-1', mlflow=True)


import mlflow

mlflow.set_experiment("LRexperimentdiabets1")
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics({
        'accuracy': report_dict['accuracy'],
        'recall_class_0': report_dict['0']['recall'],
        'recall_class_1': report_dict['1']['recall'],
        'f1_score_macro': report_dict['macro avg']['f1-score']
    })
    mlflow.sklearn.log_model(lr, "Logistic Regression") 

##########
# below is for the locally running mlflow with 5000 port
from mlflow.tracking import MlflowClient

client = MlflowClient()
all_experiments = client.search_experiments()
print(all_experiments)

############



############ NOT WORKING BELOW CODE.
# below is for the dagshub running mlflow to get list of experiments.
from dagshub import DagsHub
import dagshub

# Initialize DagsHub client
dagshub_client = DagsHub()

# Replace 'your_username' and 'your_repository' with your DagsHub username and repository name
username = 'salwad-basha-shaik'
repository = 'mlops-vcs-sep-demo-1'

# Get the list of experiments
experiments = dagshub_client.get_experiments(username=username, repository=repository)

# Print the experiments
for experiment in experiments:
    print(f"ID: {experiment['id']}, Name: {experiment['name']}")
###############

