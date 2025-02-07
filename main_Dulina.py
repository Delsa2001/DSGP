from flask import Flask, request, render_template,jsonify
from model_2_Duina import predict, preprocess,model
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

class_labels = ["balcony", "indoor"]

@app.route('/', methods = ['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            prediction, confidence_score = predict(file_path, model)

            return jsonify({
                "prediction": prediction,
                "confidence": confidence_score  
            })
    
    return render_template('page.html')


if __name__ == '__main__':
    app.run(debug=True)

