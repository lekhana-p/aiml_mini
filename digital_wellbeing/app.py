"""Flask web app for AI-Based Digital Wellbeing & Screen Addiction Detection System."""

from __future__ import annotations

import os
import pickle
import sqlite3
from typing import Dict

from flask import Flask, redirect, render_template, request, url_for


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

app = Flask(__name__)



def init_db() -> None:
    """Create SQLite table if it does not already exist."""

    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                daily_screen_time REAL NOT NULL,
                phone_unlocks INTEGER NOT NULL,
                social_media_usage REAL NOT NULL,
                night_usage REAL NOT NULL,
                avg_session_length REAL NOT NULL,
                predicted_risk TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()



def load_model():
    """Load trained model from model.pkl."""

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)



def insert_usage_record(data: Dict[str, float], prediction: str) -> None:
    """Save user input and predicted label into SQLite."""

    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO usage_data (
                daily_screen_time,
                phone_unlocks,
                social_media_usage,
                night_usage,
                avg_session_length,
                predicted_risk
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                data["daily_screen_time"],
                data["phone_unlocks"],
                data["social_media_usage"],
                data["night_usage"],
                data["avg_session_length"],
                prediction,
            ),
        )
        connection.commit()



def get_dashboard_stats() -> Dict[str, float]:
    """Fetch simple analytics for dashboard display."""

    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT
                COUNT(*) AS total_entries,
                AVG(daily_screen_time) AS avg_screen_time,
                AVG(phone_unlocks) AS avg_unlocks,
                AVG(social_media_usage) AS avg_social_media,
                AVG(night_usage) AS avg_night_usage
            FROM usage_data
            """
        )
        row = cursor.fetchone()

    total_entries = row[0] or 0
    return {
        "total_entries": total_entries,
        "avg_screen_time": round(row[1] or 0.0, 2),
        "avg_unlocks": round(row[2] or 0.0, 2),
        "avg_social_media": round(row[3] or 0.0, 2),
        "avg_night_usage": round(row[4] or 0.0, 2),
    }




# Initialize database as soon as app starts.
init_db()
@app.route("/")
def index():
    """Render home page with form and analytics."""

    stats = get_dashboard_stats()
    return render_template("index.html", stats=stats)


@app.route("/predict", methods=["POST"])
def predict():
    """Predict addiction risk from form inputs."""

    try:
        user_data = {
            "daily_screen_time": float(request.form["daily_screen_time"]),
            "phone_unlocks": int(request.form["phone_unlocks"]),
            "social_media_usage": float(request.form["social_media_usage"]),
            "night_usage": float(request.form["night_usage"]),
            "avg_session_length": float(request.form["avg_session_length"]),
        }
    except (ValueError, KeyError):
        # If invalid values are submitted, redirect back to form.
        return redirect(url_for("index"))

    model = load_model()
    features = [[
        user_data["daily_screen_time"],
        user_data["phone_unlocks"],
        user_data["social_media_usage"],
        user_data["night_usage"],
        user_data["avg_session_length"],
    ]]
    predicted_risk = model.predict(features)[0]

    insert_usage_record(user_data, predicted_risk)

    stats = get_dashboard_stats()
    return render_template("result.html", risk=predicted_risk, stats=stats)


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
