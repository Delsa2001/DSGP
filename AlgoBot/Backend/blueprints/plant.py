from flask import Blueprint, request, jsonify
import sqlite3
import requests
import os
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

plant_bp = Blueprint('plant', __name__)

# External API URLs
WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
TREFLE_API_TOKEN = os.getenv("TREFLE_API_KEY")  # Securely load the API key
TREFLE_API_PLANT = f"https://trefle.io/api/v1/plants/search?token={TREFLE_API_TOKEN}&q="

def fetch_from_db(query, param):
    """Helper function to fetch data from SQLite"""
    db_path = "data/plant_diseases.db" if "disease" in query else "data/planting_techniques.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query, (param,))
    result = cursor.fetchone()
    conn.close()
    return result

def fetch_from_external_api(query, is_disease=True):
    """Fetch data from an external source if not found in the database"""
    api_url = WIKIPEDIA_API + query if is_disease else TREFLE_API_PLANT + query
    try:
        response = requests.get(api_url, timeout=5)  # Set timeout to prevent hanging requests
        response.raise_for_status()  # Raise exception for HTTP errors (e.g., 404, 500)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from API: {e}")
        return None

@plant_bp.route("/api/plant_disease", methods=["GET"])
def get_plant_disease():
    disease_name = request.args.get("name", "").strip().lower()
    if not disease_name:
        return jsonify({"error": "Disease name is required"}), 400

    disease = fetch_from_db(
        "SELECT disease_name, description, symptoms, treatment FROM plant_diseases WHERE LOWER(disease_name) = ?",
        disease_name
    )

    if disease:
        return jsonify({
            "disease_name": disease[0],
            "description": disease[1],
            "symptoms": disease[2],
            "treatment": disease[3]
        })

    # Fetch from Wikipedia if not found in database
    external_data = fetch_from_external_api(disease_name, is_disease=True)
    if external_data:
        return jsonify({
            "disease_name": disease_name.capitalize(),
            "description": external_data.get("extract", "No description available."),
            "source": external_data.get("content_urls", {}).get("desktop", {}).get("page", "No source available.")
        })

    return jsonify({"error": "Disease not found in database or external sources"}), 404

@plant_bp.route("/api/planting_techniques", methods=["GET"])
def get_planting_techniques():
    plant_name = request.args.get("name", "").strip().lower()
    if not plant_name:
        return jsonify({"error": "Plant name is required"}), 400

    plant = fetch_from_db(
        "SELECT plant_name, light_requirement, water_requirement, soil_type, care_instructions FROM planting_techniques WHERE LOWER(plant_name) = ?",
        plant_name
    )

    if plant:
        return jsonify({
            "plant_name": plant[0],
            "light_requirement": plant[1],
            "water_requirement": plant[2],
            "soil_type": plant[3],
            "care_instructions": plant[4]
        })

    # Fetch from Trefle.io if not found in database
    external_data = fetch_from_external_api(plant_name, is_disease=False)
    if external_data and "data" in external_data and len(external_data["data"]) > 0:
        plant_data = external_data["data"][0]
        return jsonify({
            "plant_name": plant_data.get("common_name", plant_name.capitalize()),
            "scientific_name": plant_data.get("scientific_name", "Unknown"),
            "family": plant_data.get("family", "Unknown"),
            "genus": plant_data.get("genus", "Unknown"),
            "source": plant_data.get("links", {}).get("self", "No source available.")
        })

    return jsonify({"error": "Plant not found in database or external sources"}), 404
