from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict_iris(iris: IrisFeatures):
    # Load the trained model
    with open("iris_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Convert input data into a NumPy array for prediction
    test_array = np.array([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]])
    prediction = model.predict(test_array)
    
    # Map numeric prediction to class name
    species = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
    predicted_species = species[int(prediction[0])]
    
    # Return the prediction as a class name
    return {"prediction": predicted_species}

