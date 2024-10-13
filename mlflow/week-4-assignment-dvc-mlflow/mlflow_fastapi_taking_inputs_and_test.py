from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc

# Define the FastAPI app
app = FastAPI()

# Set MLflow tracking URI and load the XGBoost model
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
logged_model = 'runs:/e6e7917263494a09b050ea2ee29c8771/XGBoost'  # Replace with your run ID and model name
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Define request model
class PredictionRequest(BaseModel):
    TV: float
    radio: float
    newspaper: float

@app.post('/predict')
async def predict(request: PredictionRequest):
    # Create a DataFrame from the input
    input_data = pd.DataFrame([[request.TV, request.radio, request.newspaper]], columns=['TV', 'radio', 'newspaper'])

    # Make prediction
    prediction = loaded_model.predict(input_data)

    # Create response
    result = {
        'prediction': int(prediction[0]),
        'message': 'High sales' if prediction[0] == 1 else 'Low sales'
    }
    return result

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=5002)  # Set port to 5002