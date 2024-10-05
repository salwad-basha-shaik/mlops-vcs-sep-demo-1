import numpy as np
import pickle

with open('iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

features = np.array([5.1, 1.0, 1.5, 3])

prediction = model.predict([features])

print(prediction)
