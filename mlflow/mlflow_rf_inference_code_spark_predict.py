import os
import numpy as np
import yaml
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
import mlflow
import mlflow.sklearn
from pyspark.sql import SparkSession
from pyspark.sql.functions import struct, col
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')

# Set environment variables for PySpark Python
os.environ["PYSPARK_PYTHON"] = "/opt/anaconda3/bin/python3.12"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/opt/anaconda3/bin/python3.12"

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

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")  # Change if your server is elsewhere

# Initialize SparkSession
spark = SparkSession.builder.appName("MLFlowModelPrediction").getOrCreate()

# Convert X_test to Spark DataFrame
columns = [f"feature_{i}" for i in range(X_test.shape[1])]
X_test_df = spark.createDataFrame(pd.DataFrame(X_test, columns=columns))

# Load the model as a Spark UDF
logged_model = 'runs:/26a414f9b89e4065af165d0a29636661/Random Forest Classifier'
loaded_model_udf = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

# Make predictions on the Spark DataFrame
X_test_df = X_test_df.withColumn("prediction", loaded_model_udf(struct(*[col(f"feature_{i}") for i in range(X_test.shape[1])])))

# Show predictions
X_test_df.select("prediction").show()