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

        # Connect to SQLite
        conn = sqlite3.connect("plant_diseases.db")
        cursor = conn.cursor()

        # Fetch all diseases
        cursor.execute("SELECT disease_name, description, symptoms, treatment FROM plant_diseases")
        diseases = cursor.fetchall()

        # Encode user query with SBERT
        query_embedding = sbert_model.encode(user_query)

        best_match = None
        highest_score = 0

        # Find the closest matching disease
        for disease in diseases:
            disease_name = disease[0]
            disease_embedding = sbert_model.encode(disease_name)
            similarity = util.pytorch_cos_sim(query_embedding, disease_embedding).item()

            if similarity > highest_score:
                highest_score = similarity
                best_match = disease

        conn.close()

        if best_match and highest_score > 0.5:  # ✅ Only return if confidence is high
            response = (
                f"🌿 **{best_match[0]}** 🌿\n"
                f"📌 *Description:* {best_match[1]}\n"
                f"⚠️ *Symptoms:* {best_match[2]}\n"
                f"💊 *Treatment:* {best_match[3]}\n"
                "\n✅ Hope this helps! Let me know if you need more details."
            )
        else:
            response = (
                "❌ *Sorry, I couldn't find any exact match.*\n"
                "💡 Try rephrasing your question or ask about another plant disease."
            )

        dispatcher.utter_message(text=response)
        return []
