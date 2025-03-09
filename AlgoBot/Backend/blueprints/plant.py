from flask import Blueprint, request, jsonify
import sqlite3

plant_bp = Blueprint('plant', __name__)

def fetch_from_db(query, param):
    """Helper function to fetch data from SQLite"""
    conn = sqlite3.connect("data/plant_diseases.db" if "disease" in query else "data/planting_techniques.db")
    cursor = conn.cursor()
    cursor.execute(query, (param,))
    result = cursor.fetchone()
    conn.close()
    return result

@plant_bp.route("/api/plant_disease", methods=["GET"])
def get_plant_disease():
    disease_name = request.args.get("name", "").strip().lower()
    if not disease_name:
        return jsonify({"error": "Disease name is required"}), 400

    disease = fetch_from_db("SELECT disease_name, description, symptoms, treatment FROM plant_diseases WHERE LOWER(disease_name) = ?", disease_name)

    if disease:
        return jsonify({
            "disease_name": disease[0],
            "description": disease[1],
            "symptoms": disease[2],
            "treatment": disease[3]
        })
    return jsonify({"error": "Disease not found"}), 404

@plant_bp.route("/api/planting_techniques", methods=["GET"])
def get_planting_techniques():
    plant_name = request.args.get("name", "").strip().lower()
    if not plant_name:
        return jsonify({"error": "Plant name is required"}), 400

    plant = fetch_from_db("SELECT plant_name, light_requirement, water_requirement, soil_type, care_instructions FROM planting_techniques WHERE LOWER(plant_name) = ?", plant_name)

    if plant:
        return jsonify({
            "plant_name": plant[0],
            "light_requirement": plant[1],
            "water_requirement": plant[2],
            "soil_type": plant[3],
            "care_instructions": plant[4]
        })
    return jsonify({"error": "Plant not found"}), 404
