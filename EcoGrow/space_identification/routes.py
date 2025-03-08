from flask import request, render_template, jsonify,session
from . import space_identification_bp  
from space_identification.components.model_2_Dulina import predict,preprocess, model
import os

@space_identification_bp.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            file_path = os.path.join('./uploads', file.filename)  
            file.save(file_path)
            prediction, confidence_score = predict(file_path, model)

            session["prediction_result"] = prediction
            
            return jsonify({"prediction": prediction}) 

    return render_template('page.html')

