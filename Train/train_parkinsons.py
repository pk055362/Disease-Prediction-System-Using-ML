import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# ✅ Construct absolute paths
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, "../Datasets/parkinsons.csv")
model_path = os.path.join(base_dir, "../models/parkinsons_model.pkl")
scaler_path = os.path.join(base_dir, "../models/parkinsons_scaler.pkl")

# ✅ Load the dataset
df = pd.read_csv(dataset_path)

# ✅ Drop non-numeric columns (e.g., 'name' or ID column)
if "name" in df.columns:
    df = df.drop(columns=["name"])

# ✅ Split dataset into features (X) and target (y)
X = df.drop(columns=["status"])  # 'status' is the target variable
y = df["status"]

# ✅ Convert all features to numeric (ensure no strings)
X = X.apply(pd.to_numeric, errors="coerce")

# ✅ Fill any NaN values with 0
X = X.fillna(0)

# ✅ Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# ✅ Train the model
model.fit(X_train, y_train)

# ✅ Make predictions on the test set
y_pred = model.predict(X_test)

# ✅ Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# ✅ Save the trained model and scaler
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print("✅ Model and scaler saved successfully!")
