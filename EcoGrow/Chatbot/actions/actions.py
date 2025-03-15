import os
import sqlite3
from rasa_sdk import Action
from sentence_transformers import SentenceTransformer, util

# Load SBERT model
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Get the correct database path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "plant_diseases.db")


class ActionGetPlantInfo(Action):
    def name(self):
        return "action_get_plant_info"

    def run(self, dispatcher, tracker, domain):
        user_query = tracker.latest_message.get('text').strip().lower()

        # Connect using the correct path
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
            response = (
                f"**{best_match[0]}**\n\n"
                f"**Description:** {best_match[1]}\n\n"
                f"**Symptoms:** {best_match[2]}\n\n"
                f"**Treatments:** {best_match[3]}"
            )
        else:
            response = "Sorry, I couldn't find any exact match. Try rephrasing your question or ask about another plant disease."

        dispatcher.utter_message(text=response)
        return []


class ActionGetPlantingTechniques(Action):
    def name(self):
        return "action_get_planting_techniques"

    def run(self, dispatcher, tracker, domain):
        user_query = tracker.latest_message.get('text').strip().lower()
        PLANTING_DB_PATH = os.path.join(BASE_DIR, "planting_techniques.db")

        conn = sqlite3.connect(PLANTING_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT plant_name, light_requirement, water_requirement, soil_type, care_instructions FROM planting_techniques")
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
            response = (
                f"**{best_match[0]} Planting Guide**\n\n"
                f"**Light Requirement:** {best_match[1]}\n\n"
                f"**Watering:** {best_match[2]}\n\n"
                f"**Soil Type:** {best_match[3]}\n\n"
                f"**Care Instructions:** {best_match[4]}"
            )
        else:
            response = "Sorry, I couldn't find planting instructions for that plant. Try asking about another plant."

        dispatcher.utter_message(text=response)
        return []
