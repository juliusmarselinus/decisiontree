import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib


# Load dataset
df = pd.read_csv("data/customer_churn_dataset-training-master.csv")


# Encode kolom kategorikal
encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
df[col] = encoder.fit_transform(df[col])


# Tentukan fitur & target
target = "Churn" # ganti jika berbeda
X = df.drop(columns=[target])
y = df[target]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)


# Model Decision Tree
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)


# Evaluasi
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"Accuracy: {acc:.2f}")


# Simpan model & fitur
joblib.dump(model, "decision_tree_model.pkl")
joblib.dump(list(X.columns), "features.pkl")


print("Model berhasil disimpan")
