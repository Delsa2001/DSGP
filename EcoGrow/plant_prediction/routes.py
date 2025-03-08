from flask import Flask, request, jsonify, render_template, Blueprint,session
from . import plant_prediction_bp
import joblib
import numpy as np
import os

#plant_prediction_bp = Blueprint('plant_prediction', __name__, template_folder="templates")


# Debugging: Check the current directory and template folder
print("Current directory:", os.getcwd())
print("Templates folder exists:", os.path.exists("templates"))
print("Files inside templates:", os.listdir("templates")
      if os.path.exists("templates") else "Folder not found")

# Initialize Flask app
# app = Flask(__name__, template_folder="templates")

# Load the trained model safely
model_path = "EcoGrow\plant_prediction\plant_prediction_model.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("Model loaded successfully.")
else:
    print("Error: Model file not found!")
    model = None


@plant_prediction_bp.route("/")
def home():
    return render_template("plant_prediction.html")


@plant_prediction_bp.route("/predict", methods=["POST", "GET"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check 'plant_prediction_model.pkl'."})

    try:
        place = session.get("prediction_result", "No prediction made")
        # Get user input from form
        feature1 = float(request.form.get("Temperature", 0))
        feature2 = float(request.form.get("Precipitation", 0))
        feature3 = float(request.form.get("Humidity", 0))
        feature4 = float(request.form.get("Location", 0))
        feature5 = float(request.form.get("Place", 0))

        # Convert to NumPy array (adjust shape for your model)
        features = np.array(
            [[feature1, feature2, feature3, feature4, feature5]])

        # Make a prediction
        prediction = model.predict(features)[0]

        return render_template("plant_prediction.html", prediction=prediction, place = place)

    except Exception as e:
        return jsonify({"error": str(e)})
