from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('logistic_regression_model.joblib')

@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction API"

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.json

        # Extract the features from the input JSON
        features = np.array([data['features']])

        # Predict using the loaded model
        prediction = model.predict(features)

        # Return the prediction result as a JSON response
        result = {
            'prediction': int(prediction[0])  # Ensure it is an int to avoid serialization issues
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True,  port=5001)