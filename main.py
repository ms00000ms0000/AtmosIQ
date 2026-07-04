"""
main.py
-------
Main pipeline for AtmosIQ.
"""

from src.database import DatabaseManager
from src.preprocess import DataPreprocessor
from src.eda import EDA
from src.feature_engineering import FeatureEngineer
from src.train import ModelTrainer


def main():

    print("\n" + "=" * 65)
    print("          AtmosIQ - Weather Intelligence Platform")
    print("=" * 65)

    # ==================================================
    # Database Setup
    # ==================================================

    db = DatabaseManager()

    print("\n[1/5] Setting up database...")

    db.create_tables()

    db.import_csv()

    # ==================================================
    # Data Preprocessing
    # ==================================================

    print("\n[2/5] Running preprocessing pipeline...")

    preprocessor = DataPreprocessor()

    clean_df = preprocessor.run_pipeline()

    # ==================================================
    # Exploratory Data Analysis
    # ==================================================

    print("\n[3/5] Performing Exploratory Data Analysis...")

    eda = EDA(clean_df)

    eda.run()

    # ==================================================
    # Feature Engineering
    # ==================================================

    print("\n[4/5] Creating engineered features...")

    feature_engineer = FeatureEngineer()

    engineered_df = feature_engineer.run()

    # ==================================================
    # Model Training
    # ==================================================

    print("\n[5/5] Training Machine Learning Models...")

    trainer = ModelTrainer()

    trainer.run()

    # ==================================================
    # Summary
    # ==================================================

    print("\n" + "=" * 65)
    print("             PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print(f"\nFinal Dataset Shape : {engineered_df.shape}")

    print("\nEngineered Dataset Preview:\n")

    print(engineered_df.head())

    db.close()


if __name__ == "__main__":
    main()