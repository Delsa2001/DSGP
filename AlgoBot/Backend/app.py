from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests

app = Flask(__name__, template_folder=".")
CORS(app)  # Enable CORS for frontend requests

# Define Rasa server URL
RASA_SERVER_URL = "http://localhost:5005/webhooks/rest/webhook"


@app.route("/")
def home():
    return render_template("index.html")  # Serve the chatbot UI


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")

    if not user_message:
        return jsonify({"error": "No message received"}), 400

    # Send message to Rasa
    rasa_response = requests.post(RASA_SERVER_URL, json={"sender": "user", "message": user_message})

    if rasa_response.status_code == 200:
        rasa_reply = rasa_response.json()
        bot_responses = [message.get("text", "I couldn't understand that.") for message in rasa_reply]
        return jsonify({"response": bot_responses})
    else:
        return jsonify({"error": "Error communicating with Rasa"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
