"""
utils.py
--------
Shared helper functions for AtmosIQ.

These utilities are used by predict.py, explain.py, and streamlit_app.py
so that feature-building and artifact-loading logic lives in exactly one
place instead of being duplicated across modules.
"""

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import (
    BEST_MODEL_PATH,
    BEST_MODEL_NAME_PATH,
    TF_MODEL_PATH,
    SCALER_PATH,
    SEASON_ENCODER_PATH,
    TARGET_ENCODER_PATH,
    ENGINEERED_DATA_PATH,
    TARGET_COLUMN,
)

# ======================================================
# Constants
# ======================================================

# Column order MUST match the columns produced by
# FeatureEngineer.run() (minus the target column). Order matters for
# both the classical sklearn models and the scaled TensorFlow model.
FEATURE_COLUMNS = [
    "precipitation",
    "temp_max",
    "temp_min",
    "wind",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "quarter",
    "week_of_year",
    "month_sin",
    "month_cos",
    "avg_temp",
    "temp_range",
    "season",
]

SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

# Friendly emoji/icons for the Streamlit UI (purely cosmetic).
WEATHER_ICONS = {
    "drizzle": "🌦️",
    "fog": "🌫️",
    "rain": "🌧️",
    "snow": "❄️",
    "sun": "☀️",
}


# ======================================================
# Artifact Loading
# ======================================================

def load_best_model_name() -> str:
    """Read which model won training (Random Forest / Gradient Boosting /
    Neural Network) from disk."""

    if not BEST_MODEL_NAME_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Run the training pipeline "
            "(python main.py) before predicting."
        )

    with open(BEST_MODEL_NAME_PATH, "r") as f:
        return f.read().strip()


def load_best_model():
    """Load whichever model won training, returning (model, model_name,
    is_neural_network)."""

    model_name = load_best_model_name()

    if model_name == "Neural Network":
        model = tf.keras.models.load_model(TF_MODEL_PATH)
        return model, model_name, True

    model = joblib.load(BEST_MODEL_PATH)
    return model, model_name, False


def load_scaler():
    return joblib.load(SCALER_PATH)


def load_season_encoder():
    return joblib.load(SEASON_ENCODER_PATH)


def load_target_encoder():
    return joblib.load(TARGET_ENCODER_PATH)


def load_engineered_data() -> pd.DataFrame:
    """Load the fully engineered dataset (features + encoded target)."""
    return pd.read_csv(ENGINEERED_DATA_PATH)


def load_feature_frame() -> pd.DataFrame:
    """Engineered dataset with only the feature columns (no target)."""
    df = load_engineered_data()
    return df[FEATURE_COLUMNS]


# ======================================================
# Feature Engineering for a Single Prediction
# ======================================================

def build_feature_row(
    date,
    precipitation: float,
    temp_max: float,
    temp_min: float,
    wind: float,
    season_encoder=None,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame of engineered features from raw inputs,
    using the exact same logic as FeatureEngineer so predictions stay
    consistent with training.

    Parameters
    ----------
    date : str | datetime-like
        Date of the observation (e.g. "2026-07-04").
    precipitation, temp_max, temp_min, wind : float
        Raw weather measurements.
    season_encoder : LabelEncoder, optional
        Pass in an already-loaded encoder to avoid reloading it from disk
        on every call (useful in Streamlit where this runs per-interaction).
    """

    date = pd.to_datetime(date)

    if season_encoder is None:
        season_encoder = load_season_encoder()

    month = date.month
    day = date.day
    day_of_week = date.dayofweek
    day_of_year = date.dayofyear
    quarter = date.quarter
    week_of_year = int(date.isocalendar().week)

    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    avg_temp = (temp_max + temp_min) / 2
    temp_range = temp_max - temp_min

    season_label = SEASON_MAP[month]
    season_encoded = int(season_encoder.transform([season_label])[0])

    row = {
        "precipitation": precipitation,
        "temp_max": temp_max,
        "temp_min": temp_min,
        "wind": wind,
        "month": month,
        "day": day,
        "day_of_week": day_of_week,
        "day_of_year": day_of_year,
        "quarter": quarter,
        "week_of_year": week_of_year,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "avg_temp": avg_temp,
        "temp_range": temp_range,
        "season": season_encoded,
    }

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def decode_prediction(encoded_label: int, target_encoder=None) -> str:
    """Convert a numeric class prediction back to its weather-string label."""

    if target_encoder is None:
        target_encoder = load_target_encoder()

    return target_encoder.inverse_transform([encoded_label])[0]


def weather_icon(label: str) -> str:
    return WEATHER_ICONS.get(label, "🌡️")