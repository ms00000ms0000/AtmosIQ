"""
train.py
--------
Model Training Engine for AtmosIQ.
"""

import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)

from src.database import DatabaseManager

from src.config import (
    ENGINEERED_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    VALIDATION_SIZE,
    RANDOM_STATE,
    MODEL_COMPARISON_PATH,
    BEST_MODEL_PATH,
    BEST_MODEL_NAME_PATH,
    TF_MODEL_PATH,
    CHECKPOINT_PATH,
    SCALER_PATH,
    TENSORBOARD_LOG_DIR,
)


class ModelTrainer:

    def __init__(self):
        self.db = DatabaseManager()
        self.scaler = StandardScaler()
        self.results = []
        self.best_model = None
        self.best_accuracy = 0
        self.best_model_name = None

    # --------------------------------------------------
    # Load & Split Data
    # --------------------------------------------------
    def load_data(self):
        df = pd.read_csv(ENGINEERED_DATA_PATH)

        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        return train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

    # --------------------------------------------------
    # Neural Network Architecture
    # --------------------------------------------------
    def build_neural_network(self, input_dim, output_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(output_dim, activation="softmax"),
        ])

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    # --------------------------------------------------
    # Generic Evaluator for Sklearn Models
    # --------------------------------------------------
    def evaluate_sklearn(self, model, name, X_train, X_test, y_train, y_test):
        start = time.time()
        model.fit(X_train, y_train)
        training_time = round(time.time() - start, 3)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
        recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

        self.results.append({
            "Model": name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1 Score": round(f1, 4),
            "Training Time (s)": training_time,
        })

        self.db.save_model_result(name, accuracy, precision, recall, f1)

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = model
            self.best_model_name = name

        print(f"{name:<25} Accuracy : {accuracy:.4f}")

    # --------------------------------------------------
    # Random Forest
    # --------------------------------------------------
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
        )
        self.evaluate_sklearn(model, "Random Forest", X_train, X_test, y_train, y_test)

    # --------------------------------------------------
    # Gradient Boosting
    # --------------------------------------------------
    def train_gradient_boosting(self, X_train, X_test, y_train, y_test):
        model = GradientBoostingClassifier(random_state=RANDOM_STATE)
        self.evaluate_sklearn(model, "Gradient Boosting", X_train, X_test, y_train, y_test)

    # --------------------------------------------------
    # TensorFlow Neural Network
    # --------------------------------------------------
    def train_tensorflow(self, X_train, X_test, y_train, y_test):
        model = self.build_neural_network(
            X_train.shape[1],
            len(pd.unique(y_train)),
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                filepath=CHECKPOINT_PATH,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=0,
            ),
            TensorBoard(
                log_dir=TENSORBOARD_LOG_DIR,
                histogram_freq=1,
            ),
        ]

        start = time.time()
        model.fit(
            X_train,
            y_train,
            validation_split=VALIDATION_SIZE,
            epochs=150,
            batch_size=32,
            callbacks=callbacks,
            verbose=0,
        )
        train_time = round(time.time() - start, 3)

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)

        precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
        recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

        self.results.append({
            "Model": "Neural Network",
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1 Score": round(f1, 4),
            "Training Time (s)": train_time,
        })

        self.db.save_model_result("Neural Network", accuracy, precision, recall, f1)

        # TensorFlow models are saved in their own format, not via joblib.
        model.save(TF_MODEL_PATH)

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            # best_model stays a classical-model placeholder (None) because the
            # TF model is already persisted above via model.save(); save_best_model()
            # only handles joblib-picklable sklearn models.
            self.best_model = None
            self.best_model_name = "Neural Network"

        print(f"{'Neural Network':<25} Accuracy : {accuracy:.4f}")

    # --------------------------------------------------
    # Train All Models
    # --------------------------------------------------
    def train(self):
        X_train, X_test, y_train, y_test = self.load_data()

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        joblib.dump(self.scaler, SCALER_PATH)

        # Tree-based models don't need scaled features.
        self.train_random_forest(X_train, X_test, y_train, y_test)
        self.train_gradient_boosting(X_train, X_test, y_train, y_test)

        # Neural network trains on scaled features.
        self.train_tensorflow(X_train_scaled, X_test_scaled, y_train, y_test)

    # --------------------------------------------------
    # Save Best Classical ML Model
    # --------------------------------------------------
    def save_best_model(self):
        if self.best_model is not None:
            joblib.dump(self.best_model, BEST_MODEL_PATH)

            with open(BEST_MODEL_NAME_PATH, "w") as file:
                file.write(self.best_model_name or type(self.best_model).__name__)

            print("\nBest classical ML model saved.")
        elif self.best_model_name == "Neural Network":
            # Best model is the TF network; record its name so downstream
            # code (e.g. Streamlit app) knows to load TF_MODEL_PATH instead.
            with open(BEST_MODEL_NAME_PATH, "w") as file:
                file.write("Neural Network")

            print("\nBest model is the Neural Network (already saved to TF_MODEL_PATH).")

    # --------------------------------------------------
    # Save Comparison Results
    # --------------------------------------------------
    def save_results(self):
        results = pd.DataFrame(self.results)
        results.sort_values(by="Accuracy", ascending=False, inplace=True)
        results.to_csv(MODEL_COMPARISON_PATH, index=False)

        print("\nModel comparison saved.")
        print("\n")
        print(results)

    # --------------------------------------------------
    # Main Pipeline
    # --------------------------------------------------
    def run(self):
        print("\n========== MODEL TRAINING ==========\n")

        self.train()
        self.save_best_model()
        self.save_results()

        print("\nTraining Completed Successfully.")


# --------------------------------------------------------
# Standalone Execution
# --------------------------------------------------------
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()