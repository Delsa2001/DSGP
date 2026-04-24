from flask import Flask, jsonify, request, render_template
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE_DIR, "compost.db")

# ── Rule-Based Classifier ─────────────────────────────────────────────────────
# Mirrors rule_classify() in the notebook and classify() in compost_monitor.ino
# Raw sensor units — no model file, no scaler, no ML library needed.

MOISTURE_WET_MAX = 1350
MOISTURE_DRY_MIN = 2100
TEMP_HOT_LOW     = 31.0
TEMP_HOT_HIGH    = 34.0
METHANE_LOW      = 1900
METHANE_HIGH     = 2100

CLASS_LABELS = ['COMPOST_READY', 'TOO_DRY', 'TOO_WET']

def cloud_inference(temperature, moisture, methane):
    """Rule-based compost classifier — mirrors ESP32 logic exactly."""
    moisture = int(moisture)
    methane  = int(methane)

    if moisture <= MOISTURE_WET_MAX:
        result = 'TOO_WET'
    elif moisture >= MOISTURE_DRY_MIN:
        result = 'TOO_DRY'
    elif temperature >= TEMP_HOT_HIGH and methane < METHANE_LOW:
        result = 'TOO_DRY'
    elif temperature <= 26.5 and methane > METHANE_HIGH and moisture < 1600:
        result = 'TOO_WET'
    elif methane >= METHANE_HIGH:
        result = 'COMPOST_READY'
    elif methane < METHANE_LOW:
        result = 'TOO_DRY'
    else:
        result = 'COMPOST_READY'

    # Confidence is always 1.0 — rules are deterministic
    return result, 1.0

# ── Database setup ────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS readings (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT,
            temperature      REAL,
            moisture         REAL,
            methane          REAL,
            result           TEXT,
            cloud_result     TEXT,
            cloud_confidence REAL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS commands (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            command   TEXT DEFAULT 'none',
            timestamp TEXT
        )
    ''')
    c.execute("SELECT COUNT(*) FROM commands")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO commands (command, timestamp) VALUES ('none', ?)",
                  (datetime.now().isoformat(),))
    conn.commit()
    conn.close()

init_db()

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/data', methods=['POST'])
def receive_data():
    data = request.json
    if not data:
        return jsonify({"error": "no data"}), 400

    temperature  = data.get("temperature", 0)
    moisture     = data.get("moisture",    0)
    methane      = data.get("methane",     0)
    esp32_result = data.get("result",      "UNKNOWN")

    cloud_result, cloud_confidence = cloud_inference(temperature, moisture, methane)

    conn = sqlite3.connect(DB)
    c    = conn.cursor()
    c.execute(
        "INSERT INTO readings "
        "(timestamp, temperature, moisture, methane, result, cloud_result, cloud_confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now().isoformat(),
            temperature, moisture, methane,
            esp32_result, cloud_result, cloud_confidence
        )
    )
    conn.commit()
    conn.close()

    return jsonify({
        "status":           "ok",
        "cloud_result":     cloud_result,
        "cloud_confidence": 100.0   # deterministic — always 100%
    })

@app.route('/latest', methods=['GET'])
def latest():
    conn = sqlite3.connect(DB)
    c    = conn.cursor()
    c.execute("SELECT * FROM readings ORDER BY id DESC LIMIT 1")
    row  = c.fetchone()
    conn.close()
    if not row:
        return jsonify({"status": "no data"})
    return jsonify({
        "id":               row[0],
        "timestamp":        row[1],
        "temperature":      row[2],
        "moisture":         row[3],
        "methane":          row[4],
        "result":           row[5],
        "cloud_result":     row[6],
        "cloud_confidence": row[7]
    })

@app.route('/readings', methods=['GET'])
def get_readings():
    conn = sqlite3.connect(DB)
    c    = conn.cursor()
    c.execute("SELECT * FROM readings ORDER BY id DESC LIMIT 20")
    rows = c.fetchall()
    conn.close()
    return jsonify([{
        "id":               r[0],
        "timestamp":        r[1],
        "temperature":      r[2],
        "moisture":         r[3],
        "methane":          r[4],
        "result":           r[5],
        "cloud_result":     r[6],
        "cloud_confidence": r[7]
    } for r in rows])

@app.route('/command', methods=['GET', 'POST'])
def command():
    conn = sqlite3.connect(DB)
    c    = conn.cursor()
    if request.method == 'POST':
        cmd = request.json.get("command", "none")
        c.execute("UPDATE commands SET command=?, timestamp=? WHERE id=1",
                  (cmd, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return jsonify({"status": "command set", "command": cmd})
    c.execute("SELECT command FROM commands WHERE id=1")
    row = c.fetchone()
    conn.close()
    return jsonify({"command": row[0] if row else "none"})

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)