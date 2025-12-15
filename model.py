import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib


def train_model(csv_path: str, target_col: str):
# Load data
df = pd.read_csv(csv_path)


# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
df[col] = le.fit_transform(df[col])


X = df.drop(columns=[target_col])
y = df[target_col]


X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)


model = DecisionTreeClassifier(
max_depth=5,
random_state=42
)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")


# Save model
joblib.dump(model, "decision_tree_model.pkl")
joblib.dump(list(X.columns), "features.pkl")


return model




if __name__ == "__main__":
train_model(
csv_path="data/customer_churn_dataset-training-master.csv",
target_col="Churn" # ganti sesuai kolom target
)
