from flask import Flask
from space_identification import space_identification_bp  

app = Flask(__name__)

app.register_blueprint(space_identification_bp, url_prefix='/space_identification')

if __name__ == '__main__':
    app.run(debug=True)
