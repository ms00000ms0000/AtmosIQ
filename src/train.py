"""
train.py
--------
Model Training Engine for AtmosIQ.

Methodology notes (why this file looks the way it does):

1. The dataset is small (1461 rows) and imbalanced across weather classes
   (e.g. 'snow' is only 26 rows vs 'rain'/'sun' at ~640 each). A single
   80/20 train-test split on data this size is noisy — whichever 292 rows
   land in the test set has an outsized effect on the reported score. So
   every classical model is compared using Stratified K-Fold Cross-
   Validation (5 folds), and the model with the best *mean* performance
   across folds is trusted, not the model that got lucky on one split.

2. Macro-F1 (unweighted average of per-class F1) is used as the primary
   selection metric instead of raw accuracy. With 'rain' and 'sun' making
   up ~88% of the data, a model can hit high accuracy while completely
   ignoring 'snow' or 'drizzle' — macro-F1 penalizes that.

3. Five models are compared on purpose, not just one "obvious winner":
     - Random Forest        (bagged trees, class_weight='balanced')
     - Extra Trees          (more randomized bagged trees, often
                              generalizes better than RF on small/noisy
                              tabular data, class_weight='balanced')
     - Gradient Boosting    (classic boosting, balanced via sample_weight)
     - HistGradient Boosting(sklearn's modern, faster boosting
                              implementation, class_weight='balanced')
     - Neural Network       (kept as a documented baseline — see note below)
   The point isn't to always crown the same "favorite" model; it's to
   demonstrate an honest, reproducible comparison and let the numbers
   decide. Whichever model wins the mean CV macro-F1 is promoted.

4. The Neural Network is NOT put through 5-fold CV (retraining a network
   5x per run is expensive) and is not eligible to "win" the comparison.
   On a dataset this size, tree ensembles are expected to outperform deep
   learning — this is a well-known result for small/medium tabular data,
   not a bug — so its purpose here is to demonstrate that conclusion
   empirically as a documented baseline, not to compete for production.
"""

import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
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

CV_FOLDS = 5


class ModelTrainer:

    def __init__(self):
        self.db = DatabaseManager()
        self.scaler = StandardScaler()
        self.results = []
        self.best_score = -1.0          # ranked by mean CV macro-F1
        self.best_model = None
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
    # Cross-Validated Evaluation for Sklearn Models
    # --------------------------------------------------
    def evaluate_sklearn(self, model, name, X_train, X_test, y_train, y_test,
                          fit_params=None):
        """
        Runs Stratified K-Fold CV on the training data for a robust
        performance estimate, then fits the model once on the full
        training split for the held-out test metrics (used for reporting,
        SHAP, and as the artifact that gets saved to disk).
        """
        fit_params = fit_params or {}

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "f1_macro"],
            n_jobs=-1,
        )

        cv_accuracy_mean = cv_results["test_accuracy"].mean()
        cv_accuracy_std = cv_results["test_accuracy"].std()
        cv_f1_macro_mean = cv_results["test_f1_macro"].mean()
        cv_f1_macro_std = cv_results["test_f1_macro"].std()

        # Final fit on the full training split (held-out test metrics,
        # and this is the fitted model instance that may get saved/served).
        start = time.time()
        model.fit(X_train, y_train, **fit_params)
        training_time = round(time.time() - start, 3)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
        recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
        f1_macro = f1_score(y_test, predictions, average="macro", zero_division=0)

        self.results.append({
            "Model": name,
            "CV Accuracy (mean)": round(cv_accuracy_mean, 4),
            "CV Accuracy (std)": round(cv_accuracy_std, 4),
            "CV F1-Macro (mean)": round(cv_f1_macro_mean, 4),
            "CV F1-Macro (std)": round(cv_f1_macro_std, 4),
            "Test Accuracy": round(accuracy, 4),
            "Test Precision": round(precision, 4),
            "Test Recall": round(recall, 4),
            "Test F1 (weighted)": round(f1, 4),
            "Test F1 (macro)": round(f1_macro, 4),
            "Training Time (s)": training_time,
        })

        self.db.save_model_result(name, accuracy, precision, recall, f1)

        # Model selection is based on mean CV macro-F1, not a single
        # held-out accuracy number — this is robust to a small, imbalanced
        # dataset where a single split is noisy and accuracy alone hides
        # poor performance on minority classes.
        if cv_f1_macro_mean > self.best_score:
            self.best_score = cv_f1_macro_mean
            self.best_model = model
            self.best_model_name = name

        print(
            f"{name:<25} CV F1-Macro : {cv_f1_macro_mean:.4f} (+/- {cv_f1_macro_std:.4f})"
            f"   |   Test Accuracy : {accuracy:.4f}"
        )

    # --------------------------------------------------
    # Random Forest
    # --------------------------------------------------
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        self.evaluate_sklearn(model, "Random Forest", X_train, X_test, y_train, y_test)

    # --------------------------------------------------
    # Extra Trees
    # --------------------------------------------------
    def train_extra_trees(self, X_train, X_test, y_train, y_test):
        """
        Extremely Randomized Trees: like Random Forest, but both the
        feature subset AND the split threshold at each node are chosen
        randomly (RF only randomizes the feature subset). The extra
        randomness increases bias slightly but often reduces variance
        more, which tends to help on small, noisy tabular datasets like
        this one.
        """
        model = ExtraTreesClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        self.evaluate_sklearn(model, "Extra Trees", X_train, X_test, y_train, y_test)

    # --------------------------------------------------
    # Gradient Boosting
    # --------------------------------------------------
    def train_gradient_boosting(self, X_train, X_test, y_train, y_test):
        model = GradientBoostingClassifier(random_state=RANDOM_STATE)

        # GradientBoostingClassifier has no class_weight param, so minority
        # classes are up-weighted manually via sample_weight instead.
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

        self.evaluate_sklearn(
            model, "Gradient Boosting", X_train, X_test, y_train, y_test,
            fit_params={"sample_weight": sample_weight},
        )

    # --------------------------------------------------
    # Histogram-Based Gradient Boosting
    # --------------------------------------------------
    def train_hist_gradient_boosting(self, X_train, X_test, y_train, y_test):
        """
        Scikit-learn's modern, histogram-binned boosting implementation
        (the same family of algorithm as LightGBM). It's typically faster
        and often stronger than the classic GradientBoostingClassifier,
        and it supports class_weight='balanced' natively.
        """
        model = HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            max_iter=300,
        )
        self.evaluate_sklearn(
            model, "Hist Gradient Boosting", X_train, X_test, y_train, y_test
        )

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
        ]

        try:
            import tensorboard  # noqa: F401

            callbacks.append(
                TensorBoard(log_dir=TENSORBOARD_LOG_DIR, histogram_freq=1)
            )
        except ImportError:
            print(
                "  [!] 'tensorboard' package not found — skipping TensorBoard "
                "logging. Install it with: pip install tensorboard"
            )

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
        f1_macro = f1_score(y_test, predictions, average="macro", zero_division=0)

        # The NN is deliberately NOT run through K-Fold CV: retraining a
        # neural net 5x is expensive, and its role here is to serve as a
        # documented baseline, not to compete for production selection.
        self.results.append({
            "Model": "Neural Network",
            "CV Accuracy (mean)": None,
            "CV Accuracy (std)": None,
            "CV F1-Macro (mean)": None,
            "CV F1-Macro (std)": None,
            "Test Accuracy": round(accuracy, 4),
            "Test Precision": round(precision, 4),
            "Test Recall": round(recall, 4),
            "Test F1 (weighted)": round(f1, 4),
            "Test F1 (macro)": round(f1_macro, 4),
            "Training Time (s)": train_time,
        })

        self.db.save_model_result("Neural Network", accuracy, precision, recall, f1)

        # TensorFlow models are saved in their own format, not via joblib.
        model.save(TF_MODEL_PATH)

        print(
            f"{'Neural Network':<25} (baseline, no CV)            "
            f"|   Test Accuracy : {accuracy:.4f}"
        )

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
        self.train_extra_trees(X_train, X_test, y_train, y_test)
        self.train_gradient_boosting(X_train, X_test, y_train, y_test)
        self.train_hist_gradient_boosting(X_train, X_test, y_train, y_test)

        # Neural network trains on scaled features.
        self.train_tensorflow(X_train_scaled, X_test_scaled, y_train, y_test)

    # --------------------------------------------------
    # Save Best Classical ML Model
    # --------------------------------------------------
    def save_best_model(self):
        if self.best_model is not None:
            joblib.dump(self.best_model, BEST_MODEL_PATH)

            with open(BEST_MODEL_NAME_PATH, "w") as file:
                file.write(self.best_model_name)

            print(
                f"\nBest model: {self.best_model_name} "
                f"(mean CV F1-Macro = {self.best_score:.4f}) — saved."
            )

    # --------------------------------------------------
    # Save Comparison Results
    # --------------------------------------------------
    def save_results(self):
        results = pd.DataFrame(self.results)
        results.sort_values(by="Test Accuracy", ascending=False, inplace=True)
        results.to_csv(MODEL_COMPARISON_PATH, index=False)

        print("\nModel comparison saved.")
        print("\n")
        print(results.to_string(index=False))

    # --------------------------------------------------
    # Main Pipeline
    # --------------------------------------------------
    def run(self):
        print("\n========== MODEL TRAINING ==========\n")
        print(f"Using {CV_FOLDS}-fold Stratified Cross-Validation for model selection.")
        print("Selection metric: mean CV macro-F1 (robust to class imbalance).\n")

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