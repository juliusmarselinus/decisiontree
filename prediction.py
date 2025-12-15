import joblib
import pandas as pd

model = joblib.load("decision_tree_model.pkl")
features = joblib.load("features.pkl")


def predict(input_data):
    df = pd.DataFrame([input_data])
    df = df[features]

    prediction = model.predict(df)[0]
    confidence = model.predict_proba(df).max()

    return prediction, confidence
