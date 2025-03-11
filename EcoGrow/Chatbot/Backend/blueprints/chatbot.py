from flask import Blueprint, request, jsonify
import requests

chatbot_bp = Blueprint('chatbot', __name__)

RASA_SERVER_URL = "http://localhost:5005/webhooks/rest/webhook"


@chatbot_bp.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message received"}), 400

    rasa_response = requests.post(RASA_SERVER_URL, json={"sender": "user", "message": user_message})

    if rasa_response.status_code == 200:
        rasa_reply = rasa_response.json()
        bot_responses = [message.get("text", "I couldn't understand that.") for message in rasa_reply]
        return jsonify({"response": bot_responses})
    else:
        return jsonify({"error": "Error communicating with Rasa"}), 500
