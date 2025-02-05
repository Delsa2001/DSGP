from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "model.h5"
model = load_model(MODEL_PATH)

# Class labels
CLASS_LABELS = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Preprocess image
def preprocess_image(image_path, target_size=(225, 225)):
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    return x

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    image_file = request.files['image']
    temp_image_path = "temp_image.jpg"
    image_file.save(temp_image_path)

    try:
        image = preprocess_image(temp_image_path)
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions[0])
        predicted_label = CLASS_LABELS[predicted_class]

        os.remove(temp_image_path)

        return jsonify({
            "predicted_label": predicted_label,
            "probabilities": {
                CLASS_LABELS[i]: float(predictions[0][i]) for i in range(len(CLASS_LABELS))
            }
        })

    except Exception as e:
        os.remove(temp_image_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

