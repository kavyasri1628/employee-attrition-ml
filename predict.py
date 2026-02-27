import pickle
import numpy as np

# Load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Example employee data (must match dataset column order!)
sample = np.array([[41, 2, 1102, 1, 1, 8, 2, 2, 2, 3, 2, 4, 3, 2, 5, 4, 3, 0, 1, 2, 1, 3, 5, 2, 3, 5, 3, 2, 4, 3]])

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

if prediction[0] == 1:
    print("Employee likely to leave")
else:
    print("Employee likely to stay")