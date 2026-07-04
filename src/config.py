"""
config.py
---------
Central configuration for the AtmosIQ project.
"""

from pathlib import Path

# ======================================================
# Base Project Directory
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# ======================================================
# Data Directories
# ======================================================

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ======================================================
# Logs
# ======================================================

LOG_DIR = BASE_DIR / "logs"
TENSORBOARD_LOG_DIR = LOG_DIR / "tensorboard"

# ======================================================
# Database
# ======================================================

DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "weather.db"

# ======================================================
# Model Directories
# ======================================================

MODEL_DIR = BASE_DIR / "models"

ENCODER_DIR = MODEL_DIR / "encoders"

TRAINED_MODEL_DIR = MODEL_DIR / "trained"

EVALUATION_DIR = MODEL_DIR / "evaluation"

# ======================================================
# TensorFlow Models
# ======================================================

TF_MODEL_PATH = TRAINED_MODEL_DIR / "weather_nn.keras"

CHECKPOINT_PATH = TRAINED_MODEL_DIR / "best_checkpoint.keras"

# ======================================================
# Classical ML Models
# ======================================================

BEST_MODEL_PATH = TRAINED_MODEL_DIR / "best_model.pkl"

SCALER_PATH = TRAINED_MODEL_DIR / "scaler.pkl"

BEST_MODEL_NAME_PATH = TRAINED_MODEL_DIR / "best_model_name.txt"

# ======================================================
# Encoders
# ======================================================

TARGET_ENCODER_PATH = ENCODER_DIR / "target_encoder.pkl"

SEASON_ENCODER_PATH = ENCODER_DIR / "season_encoder.pkl"

# ======================================================
# Reports
# ======================================================

REPORT_DIR = BASE_DIR / "reports"

FIGURE_DIR = REPORT_DIR / "figures"

METRICS_DIR = REPORT_DIR / "metrics"

# ======================================================
# Dataset Paths
# ======================================================

DATASET_PATH = RAW_DATA_DIR / "weather.csv"

CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "clean_weather.csv"

ENGINEERED_DATA_PATH = PROCESSED_DATA_DIR / "engineered_weather.csv"

# ======================================================
# Evaluation Files
# ======================================================

MODEL_COMPARISON_PATH = METRICS_DIR / "model_comparison.csv"

CLASSIFICATION_REPORT_PATH = (
    METRICS_DIR / "classification_report.txt"
)

CONFUSION_MATRIX_PATH = (
    METRICS_DIR / "confusion_matrix.png"
)

# ======================================================
# Dataset Configuration
# ======================================================

TARGET_COLUMN = "weather"

# ======================================================
# Machine Learning Configuration
# ======================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

VALIDATION_SIZE = 0.10

# ======================================================
# Create Required Directories Automatically
# ======================================================

DIRECTORIES = [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DATABASE_DIR,
    MODEL_DIR,
    ENCODER_DIR,
    TRAINED_MODEL_DIR,
    EVALUATION_DIR,
    REPORT_DIR,
    FIGURE_DIR,
    METRICS_DIR,
    LOG_DIR,
    TENSORBOARD_LOG_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)