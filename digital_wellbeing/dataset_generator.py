"""Generate a synthetic dataset for digital wellbeing prediction.

This script creates sample smartphone usage data and labels each row
with an addiction risk level (Low / Medium / High).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


NUM_ROWS = 500
OUTPUT_FILE = "digital_wellbeing/usage_dataset.csv"



def create_addiction_score(df: pd.DataFrame) -> pd.Series:
    """Create a simple weighted risk score from feature columns.

    The formula is intentionally simple and transparent for students.
    """

    # Normalize each feature to roughly 0..1 scale by dividing by a max-like value.
    screen_score = df["daily_screen_time"] / 12.0
    unlock_score = df["phone_unlocks"] / 200.0
    social_score = df["social_media_usage"] / 8.0
    night_score = df["night_usage"] / 6.0
    session_score = df["avg_session_length"] / 40.0

    # Weighted sum: higher values imply higher addiction risk.
    total_score = (
        0.30 * screen_score
        + 0.20 * unlock_score
        + 0.20 * social_score
        + 0.20 * night_score
        + 0.10 * session_score
    )

    # Add a little random noise so dataset is not too perfect.
    noise = np.random.normal(loc=0.0, scale=0.05, size=len(df))
    return total_score + noise



def map_score_to_label(score: float) -> str:
    """Convert continuous score into three classes."""

    if score < 0.40:
        return "Low Addiction Risk"
    if score < 0.70:
        return "Medium Addiction Risk"
    return "High Addiction Risk"



def generate_dataset(num_rows: int = NUM_ROWS, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic smartphone usage dataset."""

    np.random.seed(random_state)

    # Create base feature distributions.
    data = {
        "daily_screen_time": np.round(np.random.uniform(1.0, 12.0, num_rows), 2),
        "phone_unlocks": np.random.randint(20, 220, num_rows),
        "social_media_usage": np.round(np.random.uniform(0.2, 8.0, num_rows), 2),
        "night_usage": np.round(np.random.uniform(0.0, 6.0, num_rows), 2),
        "avg_session_length": np.round(np.random.uniform(2.0, 40.0, num_rows), 2),
    }

    df = pd.DataFrame(data)
    scores = create_addiction_score(df)
    df["addiction_level"] = scores.apply(map_score_to_label)

    return df


if __name__ == "__main__":
    dataset = generate_dataset()
    dataset.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset generated with {len(dataset)} rows: {OUTPUT_FILE}")
    print(dataset.head())
