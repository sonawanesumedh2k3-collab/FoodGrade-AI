"""
FoodGrade AI V6 - Machine Learning

Simple Random Forest demonstration for Health Grade prediction.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "data" / "models"
MODEL_PATH = MODEL_DIR / "health_grade_model.pkl"

FEATURES = [
    "calories",
    "sugar",
    "total_fat",
    "saturated_fat",
    "protein",
    "fibre",
    "salt",
]


# ============================================================
# PROTOTYPE DATASET
# ============================================================

def get_training_data():

    data = [

        # A
        [280, 2, 4, 1, 14, 11, 0.2, "A"],
        [290, 3, 5, 1, 15, 12, 0.2, "A"],
        [300, 3, 5, 1, 13, 10, 0.3, "A"],
        [310, 4, 6, 1, 14, 9, 0.3, "A"],
        [320, 4, 6, 2, 15, 10, 0.3, "A"],

        # B
        [340, 6, 7, 2, 12, 8, 0.4, "B"],
        [350, 7, 8, 2, 11, 8, 0.4, "B"],
        [360, 8, 8, 2, 12, 7, 0.5, "B"],
        [370, 8, 9, 3, 11, 7, 0.5, "B"],
        [380, 9, 9, 3, 10, 6, 0.6, "B"],

        # C
        [390, 11, 11, 4, 9, 5, 0.7, "C"],
        [400, 12, 12, 4, 8, 5, 0.7, "C"],
        [410, 13, 13, 5, 8, 4, 0.8, "C"],
        [420, 14, 14, 5, 7, 4, 0.9, "C"],
        [430, 15, 15, 6, 7, 3, 0.9, "C"],

        # D
        [450, 18, 18, 7, 6, 2, 1.1, "D"],
        [460, 19, 19, 7, 5, 2, 1.2, "D"],
        [470, 21, 20, 8, 5, 2, 1.2, "D"],
        [480, 22, 21, 9, 4, 1, 1.3, "D"],
        [490, 24, 22, 9, 4, 1, 1.4, "D"],

        # E
        [510, 28, 25, 11, 3, 1, 1.7, "E"],
        [520, 30, 26, 12, 3, 1, 1.8, "E"],
        [540, 32, 28, 13, 2, 1, 2.0, "E"],
        [560, 35, 30, 15, 2, 0, 2.2, "E"],
        [580, 38, 32, 16, 1, 0, 2.5, "E"],
    ]

    columns = FEATURES + ["health_grade"]

    return pd.DataFrame(
        data,
        columns=columns
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_health_model():

    df = get_training_data()

    X = df[FEATURES]

    y = df["health_grade"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=8
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    labels = [
        "A",
        "B",
        "C",
        "D",
        "E"
    ]

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    # Feature importance
    feature_importance = []

    for feature, importance in zip(
        FEATURES,
        model.feature_importances_
    ):

        feature_importance.append(
            {
                "feature": feature,
                "importance": round(
                    float(importance) * 100,
                    2
                )
            }
        )

    feature_importance.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    # Save model
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    return {
        "dataset_size": len(df),

        "training_size": len(X_train),

        "testing_size": len(X_test),

        "accuracy": round(
            accuracy * 100,
            2
        ),

        "classification_report": report,

        "confusion_matrix": matrix.tolist(),

        "labels": labels,

        "feature_importance":
            feature_importance,

        "model_path":
            str(MODEL_PATH)
    }


# ============================================================
# PREDICTION
# ============================================================

def predict_health_grade(product):

    if not MODEL_PATH.exists():

        train_health_model()

    model = joblib.load(
        MODEL_PATH
    )

    values = []

    for feature in FEATURES:

        try:

            value = float(
                product.get(
                    feature,
                    0
                )
            )

        except (
            ValueError,
            TypeError
        ):

            value = 0.0

        values.append(
            value
        )

    dataframe = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    grade = model.predict(
        dataframe
    )[0]

    return {
        "grade": str(grade)
    }