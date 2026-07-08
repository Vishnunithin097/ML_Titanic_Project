import joblib
import pandas as pd

# Load the trained pipeline
pipeline = joblib.load("models/pipeline.pkl")  # Change to .pk1 only if your file is actually named pipeline.pk1

def predict(data: dict):
    df = pd.DataFrame([data])

    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0].tolist()

    return {
        "prediction": int(prediction),
        "probability": {
            "No_Survival": probability[0],
            "Survival": probability[1]
        }
    }