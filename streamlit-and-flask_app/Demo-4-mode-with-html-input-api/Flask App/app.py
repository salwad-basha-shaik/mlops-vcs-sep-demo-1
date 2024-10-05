from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the area from the request
    data = request.get_json()
    area = data['area']
    
    # Make prediction
    prediction = model.predict([[area]])
    
    # Send back the prediction
    return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
