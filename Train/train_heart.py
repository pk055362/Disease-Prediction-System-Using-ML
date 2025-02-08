import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the heart disease dataset
df = pd.read_csv('../Datasets/heart.csv')

# Split the dataset into features (X) and target (y)
X = df.drop(columns='target')
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Heart Disease Model Accuracy: {accuracy * 100:.2f}%')

# Save the model and scaler
joblib.dump(model, '../models/heart_model.pkl')    # ✅ Save the model
joblib.dump(scaler, '../models/heart_scaler.pkl')  # ✅ Save the scaler
