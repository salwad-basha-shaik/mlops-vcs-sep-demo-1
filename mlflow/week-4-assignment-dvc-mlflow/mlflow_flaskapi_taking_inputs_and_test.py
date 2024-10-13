from flask import Flask, request, jsonify
import pandas as pd
import mlflow.pyfunc

app = Flask(__name__)

# Set MLflow tracking URI and load the XGBoost model
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
logged_model = 'runs:/e6e7917263494a09b050ea2ee29c8771/XGBoost'  # Replace with your run ID and model name
loaded_model = mlflow.pyfunc.load_model(logged_model)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json(force=True)
    tv = data.get('TV', 0.0)
    radio = data.get('radio', 0.0)
    newspaper = data.get('newspaper', 0.0)

    # Create a DataFrame from the input
    input_data = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'radio', 'newspaper'])

    # Make prediction
    prediction = loaded_model.predict(input_data)

    # Create response
    result = {
        'prediction': int(prediction[0]),
        'message': 'High sales' if prediction[0] == 1 else 'Low sales'
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Set port to 5001