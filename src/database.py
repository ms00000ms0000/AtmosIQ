"""
database.py
-----------
Database layer for AtmosIQ.

Responsibilities
----------------
1. Connect to SQLite
2. Import CSV dataset
3. Load weather data
4. Execute SQL queries
5. Save prediction history
6. Save model evaluation metrics
"""

import pandas as pd
from sqlalchemy import create_engine, text
from src.config import DATABASE_PATH, DATASET_PATH


DATABASE_URL = f"sqlite:///{DATABASE_PATH}"


class DatabaseManager:
    """Handles all database operations."""

    def __init__(self):
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(DATABASE_URL, echo=False)

    # --------------------------------------------------
    # Create required tables
    # --------------------------------------------------

    def create_tables(self):

        with self.engine.begin() as conn:

            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS prediction_history(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction TEXT NOT NULL,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))

            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS model_results(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))

    # --------------------------------------------------
    # Import Dataset
    # --------------------------------------------------

    def import_csv(self):

        if not DATASET_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{DATASET_PATH}"
            )

        df = pd.read_csv(DATASET_PATH)

        if df.empty:
            raise ValueError("Dataset is empty.")

        df.to_sql(
            "weather_data",
            self.engine,
            if_exists="replace",
            index=False
        )

        print(f" Imported {len(df)} rows into weather_data")

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------

    def load_data(self):

        query = "SELECT * FROM weather_data"

        return pd.read_sql(query, self.engine)

    # --------------------------------------------------
    # Execute Custom SQL
    # --------------------------------------------------

    def execute_query(self, query):

        return pd.read_sql(query, self.engine)

    # --------------------------------------------------
    # Save Prediction
    # --------------------------------------------------

    def save_prediction(self, prediction, confidence):

        with self.engine.begin() as conn:

            conn.execute(
                text("""
                    INSERT INTO prediction_history
                    (prediction, confidence)
                    VALUES (:prediction, :confidence)
                """),
                {
                    "prediction": prediction,
                    "confidence": confidence
                }
            )

    # --------------------------------------------------
    # Save Model Result
    # --------------------------------------------------

    def save_model_result(
        self,
        model_name,
        accuracy,
        precision,
        recall,
        f1
    ):

        with self.engine.begin() as conn:

            conn.execute(
                text("""
                    INSERT INTO model_results
                    (
                        model_name,
                        accuracy,
                        precision_score,
                        recall_score,
                        f1_score
                    )
                    VALUES
                    (
                        :model_name,
                        :accuracy,
                        :precision,
                        :recall,
                        :f1
                    )
                """),
                {
                    "model_name": model_name,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }
            )

    # --------------------------------------------------
    # Retrieve Prediction History
    # --------------------------------------------------

    def get_prediction_history(self):

        return pd.read_sql(
            "SELECT * FROM prediction_history ORDER BY created_at DESC",
            self.engine
        )

    # --------------------------------------------------
    # Retrieve Model Results
    # --------------------------------------------------

    def get_model_results(self):

        return pd.read_sql(
            "SELECT * FROM model_results",
            self.engine
        )

    # --------------------------------------------------
    # Close Connection
    # --------------------------------------------------

    def close(self):

        self.engine.dispose()