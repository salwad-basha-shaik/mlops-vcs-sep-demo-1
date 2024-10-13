import mlflow

experiment_name = "advertising_sales_classification2"
experiment_id = mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)
