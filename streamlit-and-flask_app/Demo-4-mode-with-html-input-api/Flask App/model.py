from flask import Flask, request, render_template_string
import pickle

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Simple HTML form to take input and show output
HTML = """
<!doctype html>
<html lang="en">
<head>
    <title>ML Model with Flask</title>
</head>
<body>
    <h2>House Price Prediction</h2>
    <form method="post">
        <label for="area">Enter the area in sq ft:</label>
        <input type="text" id="area" name="area"><br><br>
        <input type="submit" value="Predict Price">
    </form>
    {% if prediction %}
        <h3>Predicted House Price: ${{ prediction }}</h3>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        area = float(request.form['area'])
        prediction = model.predict([[area]])[0]
    return render_template_string(HTML, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
