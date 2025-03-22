import os
import sqlite3
import requests
from flask import Blueprint, request, jsonify

plant_bp = Blueprint('plant', __name__)

WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
OPENFARM_API = "https://openfarm.cc/api/v1/crops/?filter="

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DISEASES_DB_PATH = os.path.join(PROJECT_ROOT, "actions", "plant_diseases.db")
PLANTING_DB_PATH = os.path.join(PROJECT_ROOT, "actions", "planting_techniques.db")

def fetch_from_db(db_path, query, param):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query, (param,))
        result = cursor.fetchone()
        conn.close()
        return result
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None

def fetch_from_external_api(query, is_disease=True):
    if is_disease:
        api_url = WIKIPEDIA_API + query
    else:
        api_url = OPENFARM_API + query

    try:
        response = requests.get(api_url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from API: {e}")
        return None

@plant_bp.route("/api/plant_disease", methods=["GET"])
def get_plant_disease():
    disease_name = request.args.get("name", "").strip().lower()
    if not disease_name:
        return jsonify({"error": "Disease name is required"}), 400

    print(f"[INFO] Looking up disease: {disease_name}")
    disease = fetch_from_db(
        DISEASES_DB_PATH,
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

    print(f"[INFO] Looking up planting techniques for: {plant_name}")
    plant = fetch_from_db(
        PLANTING_DB_PATH,
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

    external_data = fetch_from_external_api(plant_name, is_disease=False)
    if external_data and "data" in external_data and len(external_data["data"]) > 0:
        crop = external_data["data"][0]["attributes"]
        return jsonify({
            "plant_name": plant_name.capitalize(),
            "description": crop.get("description", "No description available."),
            "sun_requirements": crop.get("sun_requirements", "Unknown"),
            "sowing_method": crop.get("sowing_method", "Unknown"),
            "source": "https://openfarm.cc/crop/" + plant_name.replace(" ", "-")
        })

    return jsonify({"error": "Plant not found in database or external sources"}), 404
