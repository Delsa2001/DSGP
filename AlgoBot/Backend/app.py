from flask import Flask, render_template
from flask_cors import CORS
from blueprints.chatbot import chatbot_bp
from blueprints.plant import plant_bp

app = Flask(__name__, template_folder="static")
CORS(app)  # Enable CORS

# Register Blueprints
app.register_blueprint(chatbot_bp)
app.register_blueprint(plant_bp)

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
