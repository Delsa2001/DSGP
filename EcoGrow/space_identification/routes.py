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

            session["space_result"] = prediction
            
            return jsonify({"prediction": prediction}) 

    return render_template('page.html')

@space_identification_bp.route('/store_result', methods=['POST'])
def store_result():
    data = request.json
    if not data or 'result' not in data:
        return jsonify({'success': False, 'error': 'Invalid data received'}), 400

    session['space_result'] = data['result'] 
    print("Stored in session:", session['space_result'])  
    return jsonify({'success': True})