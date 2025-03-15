import os
import sqlite3
from rasa_sdk import Action
from sentence_transformers import SentenceTransformer, util

# Load SBERT model
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Get the correct database path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "plant_diseases.db")
PLANTING_DB_PATH = os.path.join(BASE_DIR, "planting_techniques.db")


class ActionGetPlantInfo(Action):
    def name(self):
        return "action_get_plant_info"

    def run(self, dispatcher, tracker, domain):
        user_query = tracker.latest_message.get('text').strip().lower()

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT disease_name, description, symptoms, treatment FROM plant_diseases")
        diseases = cursor.fetchall()
        conn.close()

        query_embedding = sbert_model.encode(user_query)

        best_match, highest_score = None, 0
        for disease in diseases:
            disease_embedding = sbert_model.encode(disease[0])
            similarity = util.pytorch_cos_sim(query_embedding, disease_embedding).item()
            if similarity > highest_score:
                highest_score, best_match = similarity, disease

        if best_match and highest_score > 0.5:
            if "symptoms" in user_query:
                response = f"Symptoms of {best_match[0]}: {best_match[2]}"
            elif "treatment" in user_query:
                response = f"Treatment for {best_match[0]}: {best_match[3]}"
            elif "description" in user_query:
                response = f"Description of {best_match[0]}: {best_match[1]}"
            else:
                response = f"{best_match[0]} - {best_match[1]}\nSymptoms: {best_match[2]}\nTreatment: {best_match[3]}"
        else:
            response = "Sorry, I couldn't find any exact match."

        dispatcher.utter_message(text=response)
        return []


class ActionGetPlantingTechniques(Action):
    def name(self):
        return "action_get_planting_techniques"

    def run(self, dispatcher, tracker, domain):
        user_query = tracker.latest_message.get('text').strip().lower()

        conn = sqlite3.connect(PLANTING_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT plant_name, light_requirement, water_requirement, soil_type, care_instructions FROM planting_techniques"
        )
        plants = cursor.fetchall()
        conn.close()

        query_embedding = sbert_model.encode(user_query)

        best_match, highest_score = None, 0
        for plant in plants:
            plant_embedding = sbert_model.encode(plant[0])
            similarity = util.pytorch_cos_sim(query_embedding, plant_embedding).item()
            if similarity > highest_score:
                highest_score, best_match = similarity, plant

        if best_match and highest_score > 0.5:
            if "soil" in user_query:
                response = f"The best soil type for {best_match[0]} is: {best_match[3]}"
            elif "water" in user_query or "watering" in user_query:
                response = f"{best_match[0]} requires this amount of water: {best_match[2]}"
            elif "light" in user_query:
                response = f"{best_match[0]} grows best in this light condition: {best_match[1]}"
            elif "care" in user_query:
                response = f"Care instructions for {best_match[0]}: {best_match[4]}"
            else:
                response = f"{best_match[0]} requires {best_match[1]} light, {best_match[2]} water, and prefers {best_match[3]} soil."
        else:
            response = "Sorry, I couldn't find planting instructions for that plant."

        dispatcher.utter_message(text=response)
        return []
