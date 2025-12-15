import joblib
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "decision_tree_model.pkl"
FEATURES_PATH = "features.pkl"
DATA_PATH = "data/customer_churn_dataset-training-master.csv"


def train_model():
    df = pd.read_csv(DATA_PATH)

    encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = encoder.fit_transform(df[col])

    target = "Churn"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(X.columns), FEATURES_PATH)

    return model, list(X.columns)


# AUTO LOAD / TRAIN
if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
else:
    model, features = train_model()


def predict(input_data):
    df = pd.DataFrame([input_data])
    df = df[features]

    prediction = model.predict(df)[0]
    confidence = model.predict_proba(df).max()

    return prediction, confidence
