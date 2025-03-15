import os
import sqlite3
from rasa_sdk import Action
from sentence_transformers import SentenceTransformer, util

# ✅ Load SBERT from local storage instead of downloading it each time
sbert_model = SentenceTransformer("./sbert_model")  # Uses locally stored model

# ✅ Get the correct database paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISEASES_DB_PATH = os.path.join(BASE_DIR, "plant_diseases.db")
PLANTING_DB_PATH = os.path.join(BASE_DIR, "planting_techniques.db")


class ActionGetPlantInfo(Action):
    def name(self):
        return "action_get_plant_info"

    def run(self, dispatcher, tracker, domain):
        user_query = tracker.latest_message.get('text').strip().lower()

        # ✅ Connect to the plant diseases database
        conn = sqlite3.connect(DISEASES_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT disease_name, description, symptoms, treatment FROM plant_diseases")
        diseases = cursor.fetchall()
        conn.close()

        # ✅ Use SBERT to find the best-matching disease
        query_embedding = sbert_model.encode(user_query)
        best_match, highest_score = None, 0
        for disease in diseases:
            disease_embedding = sbert_model.encode(disease[0])
            similarity = util.pytorch_cos_sim(query_embedding, disease_embedding).item()
            if similarity > highest_score:
                highest_score, best_match = similarity, disease

        if best_match and highest_score > 0.5:
            disease_name = best_match[0]
            description = best_match[1]
            symptoms = best_match[2]
            treatment = best_match[3]

            # ✅ Structured response for specific queries
            if "symptoms" in user_query:
                response = f"**{disease_name} Symptoms**\n\n{symptoms}"
            elif "treatment" in user_query or "cure" in user_query:
                response = f"**{disease_name} Treatment**\n\n{treatment}"
            elif "description" in user_query:
                response = f"**{disease_name} Description**\n\n{description}"
            else:
                # ✅ Fully structured response for general queries
                response = (
                    f"**{disease_name} Information**\n\n"
                    f"**Description:** {description}\n\n"
                    f"**Symptoms:** {symptoms}\n\n"
                    f"**Treatment:** {treatment}"
                )
        else:
            response = "Sorry, I couldn't find any exact match for the disease."

        dispatcher.utter_message(text=response)
        return []


class ActionGetPlantingTechniques(Action):
    def name(self):
        return "action_get_planting_techniques"

    def run(self, dispatcher, tracker, domain):
        user_query = tracker.latest_message.get('text').strip().lower()

        # ✅ Connect to the planting techniques database
        conn = sqlite3.connect(PLANTING_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT plant_name, light_requirement, water_requirement, soil_type, care_instructions FROM planting_techniques"
        )
        plants = cursor.fetchall()
        conn.close()

        # ✅ Use SBERT to find the best-matching plant
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

            # ✅ Structured response for specific queries
            if "soil" in user_query:
                response = f"**{plant_name} Soil Requirement**\n\nThe best soil type for {plant_name} is: {soil_type}"
            elif "water" in user_query or "watering" in user_query:
                response = f"**{plant_name} Watering Needs**\n\n{plant_name} requires this amount of water: {water_requirement}"
            elif "light" in user_query:
                response = f"**{plant_name} Light Requirement**\n\n{plant_name} grows best in this light condition: {light_requirement}"
            elif "care" in user_query:
                response = f"**{plant_name} Care Instructions**\n\n{care_instructions}"
            else:
                # ✅ Fully structured response for general queries
                response = (
                    f"**{plant_name} Planting Guide**\n\n"
                    f"**Light Requirement:** {light_requirement}\n\n"
                    f"**Watering Needs:** {water_requirement}\n\n"
                    f"**Soil Type:** {soil_type}\n\n"
                    f"**Care Instructions:** {care_instructions}"
                )
        else:
            response = "Sorry, I couldn't find planting instructions for that plant."

        dispatcher.utter_message(text=response)
        return []
