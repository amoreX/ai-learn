import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib

# Load model and preprocessor
model = TabNetClassifier()
model.load_model("tabnet_model.zip")
preprocessor = joblib.load("preprocessor.pkl")

# Prepare new input
new_data = pd.DataFrame([{
    "AgeMonths": 10,
    "WeightKg": 6,
    "Vaccinated":1,
    "HealthCondition":0,
    "TimeInShelterDays":0,
    "PreviousOwner":0,
    "PetType": "cat",
    "Breed": "persian",
    "Color": "white",
    "Size": "small",
}])

# Preprocess and predict
processed = preprocessor.transform(new_data)
prediction = model.predict(processed.astype(np.float32))

print(f"Prediction: {prediction[0]}")
