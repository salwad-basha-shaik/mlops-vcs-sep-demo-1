import requests
import json

# URL of the FastAPI application
url = 'http://127.0.0.1:8000/predict'  # Change the port if necessary

# Sample input data for the Iris model
data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

# Send POST request
response = requests.post(url, json=data)

# Print response
if response.status_code == 200:
    print('Prediction:', response.json())
else:
    print(f'Error: {response.status_code}, {response.text}')