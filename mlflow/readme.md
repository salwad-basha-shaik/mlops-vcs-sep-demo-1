# MLFLOW experiments Demos

This repository contains **MLFLOW experiments Demos** to showcase expermiments with different models and push the output to mlflow to compare and register the best model.

---

## MLFLOW experiments Demos

### Demo-1: Generating a synthetic dataset( not realtime data), train with LogisticRegression algorithm and then push the classification report to MLFlow

#### Steps to Follow:

1. **Activate Anaconda**
   After installing Anaconda, activate your environment by running:
   
   ```bash
   conda activate
   ```

2. **(Optional) Create and Activate Virtual Environments**
   
   You can create virtual environments using the following commands:
   
   ```bash
   virtualenv demo1  # Using virtualenv
   python -m venv demo2  # Using venv
   ```

   To activate the virtual environments:
   
   ```bash
   source demo1/bin/activate  # Activating virtualenv
   source demo2/bin/activate  # Activating venv
   ```
3. **Make sure you run mlflow ui in 1st terminal window and remaining below runs has to be run from seperate terminal window**

   ```bash
   mlflow ui 
   ```

4. **Run 1st model with LogisticRegression algorithm**
   Navigate to the `mlflow` folder and execute the following steps:
   
   To run the 1st model with LogisticRegression algorithm

   ```bash
   python mlflow/ML_code_1.py
   ```
---

### Demo-2: Generating a synthetic dataset( not realtime data), train with DecisionTree algorithm and then push the classification report to MLFlow

#### Steps to Follow:

1. **Make sure you run mlflow ui in 1st terminal window and remaining below runs has to be run from seperate terminal window**

   ```bash
   mlflow ui 
   ```

2. **Run 2nd model with DecisionTree algorithm**
   Navigate to the `mlflow` folder and execute the following steps:
   
   To run the 2nd model with DecisionTree algorithm

   ```bash
   python mlflow/ML_code_1_DecisiontreeModel.py
   ```
---

### Demo-3: Generating a synthetic dataset( not realtime data), train with XGBoost algorithm and then push the classification report to MLFlow

#### Steps to Follow:

1. **Make sure you run mlflow ui in 1st terminal window and remaining below runs has to be run from seperate terminal window**

   ```bash
   mlflow ui 
   ```
2. **Install xgboost module**

   ```bash
   pip install xgboost
   ```

3. **Run 3rd model with XGBoost algorithm**
   Navigate to the `mlflow` folder and execute the following steps:
   
   To run the 3rd model with XGBoost algorithm

   ```bash
   python mlflow/ML_code_1_XGBoostModel.py
   ```
---

### Demo-4: Generating a synthetic dataset( not realtime data), train with RandomForest algorithm and then push the classification report to MLFlow

#### Steps to Follow:

1. **Make sure you run mlflow ui in 1st terminal window and remaining below runs has to be run from seperate terminal window**

   ```bash
   mlflow ui 
   ```

2. **Run 4th model with RandomForest algorithm**
   Navigate to the `mlflow` folder and execute the following steps:
   
   To run the 4th model with RandomForest algorithm

   ```bash
   python mlflow/ML_code_1_RandomForestModel.py
   ```

**After running above 4 models, go to mlflow UI 127.0.0.1:/5000 and compare 4 of them and you will get to see that Randomforest is having best f1_score with 0.9409138655462186 so this is the best model out of all 4 models so you can register this model**
---

### Demo-5: Create config_params.yaml file with randomforest and decisiontree parameters to load them in model files and then proceed with the experimenting the model.

#### Steps to Follow:

1. **Make sure you run mlflow ui in 1st terminal window and remaining below runs has to be run from seperate terminal window**

   ```bash
   mlflow ui 
   ```

2. **Run 4th model with RandomForest algorithm but passing the parameters by loading the yaml file**
   Navigate to the `mlflow` folder and execute the following steps:

   ```bash
   python mlflow/ML_code_1_RandomForestModel_yaml_params.py
   ```


---

### Demo-6: Create synthetic dataset and split the dataset and then predict using the inference code. whatever we have experimented in the mlfow with the best model we can go to Artifacts and load this model to use easily and predict.

#### Steps to Follow:

1. **Make sure you run mlflow ui in 1st terminal window and remaining below runs has to be run from seperate terminal window**

   ```bash
   mlflow ui 
   ```

2. **Take the inference code from Decision tree Run in our experiment and add use test your split data to predict**

   Navigate to the `mlflow` folder and execute the following steps:

   ```bash
   python mlflow/mlflow_decisiontree_inference_code.py
   ```

3. **Take the inference code from Randomeforest(best model in our case) and Run in our experiment and use test your split data to predict**
   
   Navigate to the `mlflow` folder and execute the following steps:

   # Make sure you copy the correct Run id.
   # Here we are predicting using the pandas dataframe (case1)

   ```bash
   python mlflow/mlflow_rf_inference_code.py
   ```
4. **Take the inference code from Randomeforest(best model in our case) and Run in our experiment and use test your split data to predict**
   
   Navigate to the `mlflow` folder and execute the following steps:
   
   # Make sure you copy the correct Run id.
   # Here we are predicting using the Spark dataframe (case2)

    **Install pre-requistes modules and JAVA11 in ananconda for spark dataframe to use it to predict**

    Installing pyspark module:

   ```bash
   pip install pyspark
   ```

    Installing JAVA11 in ananconda ( worked for me):

   ```bash
   conda install -c conda-forge openjdk=11
   ```

**Now after installing dependancies you can run below python file to use spark dataframe to predict**

    Run below:

   ```bash
   python mlflow/mlflow_rf_inference_code_spark_predict.py
   ```


---


