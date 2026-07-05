"""
train.py
--------
Model Training Engine for AtmosIQ.

Methodology notes (why this file looks the way it does):

1. The dataset is small (1461 rows) and severely imbalanced across weather
   classes: rain=641, sun=640, fog=101, drizzle=53, snow=26. Left alone,
   every model learns to just predict rain/sun and barely ever predicts
   snow/drizzle/fog — accuracy looks fine because those two classes
   dominate the data, but the model is nearly useless for the minority
   weather types.

2. SMOTE (Synthetic Minority Over-sampling Technique) is used to fix this
   at the data level: it generates synthetic examples of the minority
   classes (interpolating between real neighbors) until every class has
   as many training examples as the majority class. This is a stronger
   fix than class_weight='balanced' — class weighting only reweights the
   loss function, it doesn't give the model more actual examples of what
   snow/drizzle look like to learn from.

3. SMOTE is applied INSIDE each cross-validation fold (via an imblearn
   Pipeline), never before the train/test split or before the CV split.
   Oversampling before splitting would leak synthetic-derived information
   between folds/test set and produce misleadingly optimistic scores.
   With an imblearn Pipeline, cross_validate() and the final .fit() both
   correctly resample only the training portion each time; the held-out
   test set and validation folds always stay untouched, real data.

4. A single 80/20 train-test split is still noisy on a dataset this size,
   so classical models are compared using Stratified 5-Fold Cross-
   Validation, and macro-F1 (which treats every class equally regardless
   of size) is the primary selection metric — not raw accuracy, which
   would still reward a model that ignores snow entirely.

5. Five models are compared: Random Forest, Extra Trees, Gradient
   Boosting, Hist Gradient Boosting (all wrapped with SMOTE), plus a
   Neural Network kept as a documented baseline (SMOTE applied once to
   its training split; not run through 5-fold CV since retraining a
   network 5x per run is expensive, and it isn't eligible to "win").
"""

import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

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

# The smallest class (snow) has only ~26 rows total, and even less inside
# a single CV fold's training portion. SMOTE's k_neighbors must be smaller
# than the minority class count in whatever data it's fit on, so a
# conservative value is used instead of the sklearn default of 5.
SMOTE_K_NEIGHBORS = 3


def make_smote_pipeline(model):
    """Wrap a classifier so SMOTE oversampling runs only on the training
    fold during .fit(), and is a no-op during .predict()/.predict_proba()."""
    return ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K_NEIGHBORS)),
        ("clf", model),
    ])


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
    def evaluate_sklearn(self, pipeline, name, X_train, X_test, y_train, y_test):
        """
        pipeline : an imblearn Pipeline of [SMOTE, classifier]. SMOTE only
        activates during .fit()/cross_validate() training folds; it's
        automatically skipped during .predict()/.predict_proba(), so the
        held-out test set and CV validation folds are always evaluated on
        real, untouched data.
        """

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        cv_results = cross_validate(
            pipeline,
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

        # Final fit on the full training split (SMOTE resamples only this
        # training data internally; X_test/y_test are never touched).
        start = time.time()
        pipeline.fit(X_train, y_train)
        training_time = round(time.time() - start, 3)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
        recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
        f1_macro = f1_score(y_test, predictions, average="macro", zero_division=0)

        # Per-class recall makes the SMOTE fix visible: are minority
        # classes (snow, drizzle) actually being predicted now?
        per_class_recall = recall_score(y_test, predictions, average=None, zero_division=0)

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

        if cv_f1_macro_mean > self.best_score:
            self.best_score = cv_f1_macro_mean
            self.best_model = pipeline
            self.best_model_name = name

        print(
            f"{name:<25} CV F1-Macro : {cv_f1_macro_mean:.4f} (+/- {cv_f1_macro_std:.4f})"
            f"   |   Test Accuracy : {accuracy:.4f}"
        )
        print(f"    Per-class recall (drizzle, fog, rain, snow, sun): "
              f"{np.round(per_class_recall, 3).tolist()}")

    # --------------------------------------------------
    # Random Forest
    # --------------------------------------------------
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
        pipeline = make_smote_pipeline(model)
        self.evaluate_sklearn(pipeline, "Random Forest", X_train, X_test, y_train, y_test)

    # --------------------------------------------------
    # Extra Trees
    # --------------------------------------------------
    def train_extra_trees(self, X_train, X_test, y_train, y_test):
        model = ExtraTreesClassifier(n_estimators=400, random_state=RANDOM_STATE)
        pipeline = make_smote_pipeline(model)
        self.evaluate_sklearn(pipeline, "Extra Trees", X_train, X_test, y_train, y_test)

    # --------------------------------------------------
    # Gradient Boosting
    # --------------------------------------------------
    def train_gradient_boosting(self, X_train, X_test, y_train, y_test):
        model = GradientBoostingClassifier(random_state=RANDOM_STATE)
        pipeline = make_smote_pipeline(model)
        self.evaluate_sklearn(pipeline, "Gradient Boosting", X_train, X_test, y_train, y_test)

    # --------------------------------------------------
    # Histogram-Based Gradient Boosting
    # --------------------------------------------------
    def train_hist_gradient_boosting(self, X_train, X_test, y_train, y_test):
        model = HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_iter=300)
        pipeline = make_smote_pipeline(model)
        self.evaluate_sklearn(
            pipeline, "Hist Gradient Boosting", X_train, X_test, y_train, y_test
        )

    # --------------------------------------------------
    # TensorFlow Neural Network
    # --------------------------------------------------
    def train_tensorflow(self, X_train, X_test, y_train, y_test):

        # SMOTE is applied once here (not per-CV-fold — the NN isn't run
        # through K-Fold CV at all, see module docstring) directly on the
        # scaled training split, so the network also gets balanced classes
        # to learn from.
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K_NEIGHBORS)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model = self.build_neural_network(
            X_train_res.shape[1],
            len(pd.unique(y_train_res)),
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
            X_train_res,
            y_train_res,
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
        per_class_recall = recall_score(y_test, predictions, average=None, zero_division=0)

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

        model.save(TF_MODEL_PATH)

        print(
            f"{'Neural Network':<25} (baseline, no CV)            "
            f"|   Test Accuracy : {accuracy:.4f}"
        )
        print(f"    Per-class recall (drizzle, fog, rain, snow, sun): "
              f"{np.round(per_class_recall, 3).tolist()}")

    # --------------------------------------------------
    # Train All Models
    # --------------------------------------------------
    def train(self):
        X_train, X_test, y_train, y_test = self.load_data()

        print("Class counts in training split BEFORE SMOTE:")
        print(y_train.value_counts().sort_index().to_dict())
        print()

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
            # self.best_model is the full imblearn Pipeline (SMOTE + the
            # winning classifier). SMOTE is a no-op at prediction time, so
            # saving/loading the whole pipeline is safe and transparent to
            # predict.py / explain.py.
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
        print("Class imbalance fix: SMOTE oversampling (applied inside each fold).")
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