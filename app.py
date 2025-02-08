from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__, template_folder="Frontend", static_folder="Frontend/static")

# Load models and scalers
base_dir = os.path.dirname(os.path.abspath(__file__))

def load_model(file_name):
    file_path = os.path.join(base_dir, 'models', file_name)
    if os.path.exists(file_path):
        return joblib.load(file_path)
    else:
        raise FileNotFoundError(f"Error: {file_name} not found in models/ directory.")

heart_model = load_model('heart_model.pkl')
heart_scaler = load_model('heart_scaler.pkl')
diabetes_model = load_model('diabetes_model.pkl')
diabetes_scaler = load_model('diabetes_scaler.pkl')
parkinson_model = load_model('parkinsons_model.pkl')
parkinson_scaler = load_model('parkinsons_scaler.pkl')

# Prediction Functions
def predict_heart_disease(features):
    features_scaled = heart_scaler.transform([features])
    return heart_model.predict(features_scaled)[0]

def predict_diabetes(features):
    features_scaled = diabetes_scaler.transform([features])
    return diabetes_model.predict(features_scaled)[0]

def predict_parkinson(features):
    features_scaled = parkinson_scaler.transform([features])
    return parkinson_model.predict(features_scaled)[0]

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    disease = data['disease']
    features = data['features']

    if disease == 'heart':
        prediction = predict_heart_disease(features)
    elif disease == 'diabetes':
        prediction = predict_diabetes(features)
    elif disease == 'parkinsons':
        prediction = predict_parkinson(features)
    else:
        return jsonify({'error': 'Invalid disease type'}), 400

    result = 'Positive' if prediction == 1 else 'Negative'
    return jsonify(result=result)

if __name__ == "__main__":
    app.run(debug=True)
