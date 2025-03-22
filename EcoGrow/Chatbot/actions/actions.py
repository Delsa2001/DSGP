import os
import sqlite3
import requests
import re
from rasa_sdk import Action
from sentence_transformers import SentenceTransformer, util

sbert_model = SentenceTransformer("./sbert_model")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISEASES_DB_PATH = os.path.join(BASE_DIR, "plant_diseases.db")
PLANTING_DB_PATH = os.path.join(BASE_DIR, "planting_techniques.db")
FLASK_API_URL = "http://localhost:5000/api"

class ActionGetPlantInfo(Action):
    def name(self):
        return "action_get_plant_info"

    def run(self, dispatcher, tracker, domain):
        user_query = tracker.latest_message.get("text").strip().lower()
        KEYWORDS = ["disease", "symptom", "treatment", "cure", "infection", "blight", "spot", "mildew"]
        if not any(keyword in user_query for keyword in KEYWORDS):
            dispatcher.utter_message(text="I can only help with plant diseases or planting techniques.")
            return []

        conn = sqlite3.connect(DISEASES_DB_PATH)
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
            disease_name, description, symptoms, treatment = best_match
            if "symptoms" in user_query:
                response = f"**{disease_name} Symptoms**\n\n{symptoms}"
            elif "treatment" in user_query or "cure" in user_query:
                response = f"**{disease_name} Treatment**\n\n{treatment}"
            elif "description" in user_query:
                response = f"**{disease_name} Description**\n\n{description}"
            else:
                response = (
                    f"**{disease_name} Information**\n\n"
                    f"**Description:** {description}\n\n"
                    f"**Symptoms:** {symptoms}\n\n"
                    f"**Treatment:** {treatment}"
                )
        else:
            fallback_name = re.sub(r"[^\w\s]", "", user_query.split()[-1])
            try:
                r = requests.get(f"{FLASK_API_URL}/plant_disease", params={"name": fallback_name})
                data = r.json()
                if "error" not in data:
                    response = (
                        f"**{data['disease_name']}**\n\n"
                        f"{data.get('description', '')}\n\n"
                        f"Source: {data.get('source', 'N/A')}"
                    )
                else:
                    response = "I couldn't find any info on that disease. Try another one?"
            except:
                response = "Something went wrong fetching external disease info."

        dispatcher.utter_message(text=response)
        return []


class ActionGetPlantingTechniques(Action):
    def name(self):
        return "action_get_planting_techniques"

    def run(self, dispatcher, tracker, domain):
        user_query = tracker.latest_message.get("text").strip().lower()
        KEYWORDS = ["planting", "grow", "soil", "water", "care", "light"]
        if not any(keyword in user_query for keyword in KEYWORDS):
            dispatcher.utter_message(text="Please ask about how to plant or care for a specific plant.")
            return []

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
            plant_name, light, water, soil, care = best_match
            if "soil" in user_query:
                response = f"**{plant_name} Soil Requirement**\n\n{soil}"
            elif "water" in user_query:
                response = f"**{plant_name} Watering Needs**\n\n{water}"
            elif "light" in user_query:
                response = f"**{plant_name} Light Requirement**\n\n{light}"
            elif "care" in user_query:
                response = f"**{plant_name} Care Instructions**\n\n{care}"
            else:
                response = (
                    f"**{plant_name} Planting Guide**\n\n"
                    f"**Light:** {light}\n\n"
                    f"**Water:** {water}\n\n"
                    f"**Soil:** {soil}\n\n"
                    f"**Care:** {care}"
                )
        else:
            fallback_name = re.sub(r"[^\w\s]", "", user_query.split()[-1])
            try:
                r = requests.get(f"{FLASK_API_URL}/planting_techniques", params={"name": fallback_name})
                data = r.json()
                if "error" not in data:
                    response = (
                        f"**{data['plant_name']} (from external source)**\n\n"
                        f"Family: {data.get('family', 'Unknown')}\n"
                        f"Genus: {data.get('genus', 'Unknown')}\n"
                        f"Source: {data.get('source', 'N/A')}"
                    )
                else:
                    response = "I couldn’t find anything for that plant. Try a different name?"
            except:
                response = "Something went wrong while trying to look that up externally."

        dispatcher.utter_message(text=response)
        return []
