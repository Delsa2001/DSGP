import os
import sqlite3
import requests
import re
from rasa_sdk import Action
from sentence_transformers import SentenceTransformer, util

# Set fallback API base URL
FLASK_API_URL = "http://localhost:5000/api"

# Define DB paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISEASES_DB_PATH = os.path.join(BASE_DIR, "plant_diseases.db")
PLANTING_DB_PATH = os.path.join(BASE_DIR, "planting_techniques.db")

# Load SBERT model from root-level /sbert_model folder
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # goes up from /actions
SBERT_MODEL_PATH = os.path.join(PROJECT_ROOT, "sbert_model")
sbert_model = SentenceTransformer(SBERT_MODEL_PATH)

# Helper to sanitize fallback query (removes filler phrases)
def sanitize_for_fallback(text):
    text = text.lower().strip().replace("?", "")
    text = re.sub(
        r"\b(what is|tell me about|information on|how to grow|how do i grow|how to cure|how to treat|watering method for|sunlight required for|care instructions for)\b",
        "",
        text,
    )
    return text.strip()

# ======================
# 🌿 Disease Action
# ======================
class ActionGetPlantInfo(Action):
    def name(self):
        return "action_get_plant_info"

    def run(self, dispatcher, tracker, domain):
        user_query = tracker.latest_message.get("text").strip().lower()
        KEYWORDS = [
            "disease", "symptom", "treatment", "cure", "infection",
            "blight", "spot", "mildew", "virus", "wilt", "rot",
            "fungus", "bacteria", "rust", "leaf", "cause", "causes",
            "reason", "info", "information", "about"
        ]

        if not any(word in user_query for word in KEYWORDS):
            dispatcher.utter_message(text="I can only help with plant diseases or planting techniques.")
            return []

        try:
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
                if "symptom" in user_query:
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
                dispatcher.utter_message(text=response)
                return []

        except Exception as e:
            print(f"[ERROR] Local DB disease lookup failed: {e}")

        # Fallback to Wikipedia
        fallback_name = sanitize_for_fallback(user_query)
        print(f"[INFO] Looking up disease fallback: {fallback_name}")

        try:
            r = requests.get(f"{FLASK_API_URL}/plant_disease", params={"name": fallback_name})
            data = r.json()
            description = data.get("description", "").strip()
            source = data.get("source", "N/A")

            if description:
                response = (
                    f"**{data.get('disease_name', fallback_name.title())} (from external source)**\n\n"
                    f"**Description:** {description}\n\n"
                    f"**Source:** {source}"
                )
            else:
                response = "I couldn't find any info on that disease. Try another one?"
        except Exception as e:
            response = "Something went wrong fetching external disease info."
            print(f"[ERROR] Disease fallback exception: {e}")

        dispatcher.utter_message(text=response)
        return []


# ======================
# 🌱 Planting Techniques Action
# ======================
class ActionGetPlantingTechniques(Action):
    def name(self):
        return "action_get_planting_techniques"

    def run(self, dispatcher, tracker, domain):
        user_query = tracker.latest_message.get("text").strip().lower()
        KEYWORDS = [
            "planting", "grow", "soil", "water", "care",
            "light", "sun", "requirement", "method"
        ]

        if not any(word in user_query for word in KEYWORDS):
            dispatcher.utter_message(
                text="Please ask specifically about planting techniques — such as how to grow, watering needs, soil type, or light requirements."
            )
            return []

        try:
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
                plant_name = best_match[0]
                light_requirement = best_match[1]
                water_requirement = best_match[2]
                soil_type = best_match[3]
                care_instructions = best_match[4]

                if "soil" in user_query:
                    response = f"**{plant_name} Soil Requirement**\n\nThe best soil type for {plant_name} is: {soil_type}"
                elif "water" in user_query or "watering" in user_query:
                    response = f"**{plant_name} Watering Needs**\n\n{plant_name} requires this amount of water: {water_requirement}"
                elif "light" in user_query or "sun" in user_query:
                    response = f"**{plant_name} Light Requirement**\n\n{plant_name} grows best in this light condition: {light_requirement}"
                elif "care" in user_query:
                    response = f"**{plant_name} Care Instructions**\n\n{care_instructions}"
                else:
                    response = (
                        f"**{plant_name} Planting Guide**\n\n"
                        f"**Light Requirement:** {light_requirement}\n\n"
                        f"**Watering Needs:** {water_requirement}\n\n"
                        f"**Soil Type:** {soil_type}\n\n"
                        f"**Care Instructions:** {care_instructions}"
                    )
                dispatcher.utter_message(text=response)
                return []

        except Exception as e:
            print(f"[ERROR] Local DB planting lookup failed: {e}")

        # Fallback to OpenFarm
        fallback_name = sanitize_for_fallback(user_query)
        print(f"[INFO] Looking up planting fallback: {fallback_name}")

        try:
            r = requests.get(f"{FLASK_API_URL}/planting_techniques", params={"name": fallback_name})
            data = r.json()
            if "error" not in data:
                response = (
                    f"**{data.get('plant_name', fallback_name.title())} (from external source)**\n\n"
                    f"**Description:** {data.get('description', 'No info')}\n\n"
                    f"**Sun Requirements:** {data.get('sun_requirements', 'Unknown')}\n\n"
                    f"**Sowing Method:** {data.get('sowing_method', 'Unknown')}\n\n"
                    f"**Source:** {data.get('source', 'N/A')}"
                )
            else:
                response = "I couldn’t find anything for that plant. Try a different name?"
        except Exception as e:
            response = "Something went wrong while trying to look that up externally."
            print(f"[ERROR] Planting fallback exception: {e}")

        dispatcher.utter_message(text=response)
        return []
