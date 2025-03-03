#!/bin/bash

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# Start Rasa actions server
nohup rasa run actions --port 5055 &

# Start Rasa chatbot
nohup rasa run --enable-api --cors "*" &

# Start Flask application
nohup python app.py &


#Commands
#rasa run actions
#rasa run --enable-api
#cd Backend
 #flask run