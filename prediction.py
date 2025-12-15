
import joblib
import pandas as pd


model = joblib.load("decision_tree_model.pkl")
features = joblib.load("features.pkl")




def predict(input_data: dict):
"""
input_data contoh:
{
"Age": 35,
"MonthlyCharges": 70.5,
"Contract": 1
}
"""
df = pd.DataFrame([input_data])
df = df[features]


prediction = model.predict(df)[0]
probability = model.predict_proba(df).max()


return {
"prediction": int(prediction),
"confidence": float(probability)
}
