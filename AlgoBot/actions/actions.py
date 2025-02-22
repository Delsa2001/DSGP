import sqlite3
from rasa_sdk import Action
from sentence_transformers import SentenceTransformer, util

# Load SBERT model
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class ActionGetPlantInfo(Action):
    def name(self):
        return "action_get_plant_info"

    def run(self, dispatcher, tracker, domain):
        user_query = tracker.latest_message.get('text').strip().lower()
        conn = sqlite3.connect("plant_diseases.db")
        cursor = conn.cursor()
        cursor.execute("SELECT disease_name, description, symptoms, treatment FROM plant_diseases")
        diseases = cursor.fetchall()
        query_embedding = sbert_model.encode(user_query)

        best_match, highest_score = None, 0
        for disease in diseases:
            disease_embedding = sbert_model.encode(disease[0])
            similarity = util.pytorch_cos_sim(query_embedding, disease_embedding).item()
            if similarity > highest_score:
                highest_score, best_match = similarity, disease

        conn.close()

        if best_match and highest_score > 0.5:
            response = (
                f"🌿 **{best_match[0]}** 🌿\n"
                f"📌 *Description:* {best_match[1]}\n"
                f"⚠️ *Symptoms:* {best_match[2]}\n"
                f"💊 *Treatment:* {best_match[3]}\n"
                "\n✅ Hope this helps! Let me know if you need more details."
            )
        else:
            response = "❌ *Sorry, I couldn't find any exact match.*\n💡 Try rephrasing your question or ask about another plant disease."

        dispatcher.utter_message(text=response)
        return []

class ActionGetPlantingTechniques(Action):
    def name(self):
        return "action_get_planting_techniques"

    def run(self, dispatcher, tracker, domain):
        user_query = tracker.latest_message.get('text').strip().lower()
        conn = sqlite3.connect("planting_techniques.db")
        cursor = conn.cursor()
        cursor.execute("SELECT plant_name, light_requirement, water_requirement, soil_type, care_instructions FROM planting_techniques")
        plants = cursor.fetchall()
        query_embedding = sbert_model.encode(user_query)

        best_match, highest_score = None, 0
        for plant in plants:
            plant_embedding = sbert_model.encode(plant[0])
            similarity = util.pytorch_cos_sim(query_embedding, plant_embedding).item()
            if similarity > highest_score:
                highest_score, best_match = similarity, plant

        conn.close()

        if best_match and highest_score > 0.5:
            response = (
                f"🌱 **{best_match[0]} Planting Guide** 🌱\n"
                f"☀️ *Light:* {best_match[1]}\n"
                f"💧 *Watering:* {best_match[2]}\n"
                f"🌱 *Soil:* {best_match[3]}\n"
                f"📖 *Care:* {best_match[4]}\n"
                "\n✅ Hope this helps! Let me know if you need more details."
            )
        else:
            response = "❌ *Sorry, I couldn't find planting instructions for that plant.*\n💡 Try asking about another plant!"

        dispatcher.utter_message(text=response)
        return []
