import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("rainfall_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


# ------------------------------
# Home Page
# ------------------------------
@app.route('/')
def home():
    return render_template('index.html')


# ------------------------------
# Prediction Route
# ------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        MinTemp = float(request.form['MinTemp'])
        MaxTemp = float(request.form['MaxTemp'])
        Rainfall = float(request.form['Rainfall'])
        WindGustSpeed = float(request.form['WindGustSpeed'])
        Humidity3pm = float(request.form['Humidity3pm'])

        # Arrange features in correct order
        features = np.array([[MinTemp, MaxTemp, Rainfall, WindGustSpeed, Humidity3pm]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        # Result message
        if prediction == 1:
            result = f"🌧️ High chances of rain tomorrow ({probability*100:.2f}%)"
        else:
            result = f"☀️ No chances of rain tomorrow ({(1-probability)*100:.2f}%)"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
