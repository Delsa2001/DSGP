from flask import Flask, request, render_template
from model_2_Duina import predict, preprocess,model
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

@app.route('/', methods = ['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            prediction = predict(file_path, model)
            return prediction
    
    return render_template('page.html')

if __name__ == '__main__':
    app.run(debug=True)


