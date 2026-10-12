"""
FoodGrade AI V6 - Machine Learning

Random Forest classifier for Health Grade prediction.

Changes (Phase 2):
- Empirical training: DB products + OFF CSV, with synthetic fallback.
- Model cached on disk; only retrained when train_health_model() is called.
- load_model_metrics() returns cached metrics from disk (no retraining).
- Cross-validation scores included in training output.
- save_metrics / load_metrics helpers for persistent JSON cache.
"""

from __future__ import annotations

import json
import sqlite3
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "data" / "models"
MODEL_PATH = MODEL_DIR / "health_grade_model.pkl"
METRICS_PATH = MODEL_DIR / "health_grade_metrics.json"

DB_PATH = BASE_DIR / "database" / "foodgrade.db"

FEATURES = [
    "calories",
    "sugar",
    "total_fat",
    "saturated_fat",
    "protein",
    "fibre",
    "salt",
]

GRADE_ORDER = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}


# ============================================================
# SYNTHETIC PROTOTYPE DATASET
# ============================================================

def _get_synthetic_data() -> list[list]:
    """25 hand-crafted rows — used as fallback when real data is sparse."""
    return [
        # A — high fibre, high protein, low sugar
        [280, 2,  4,  1,  14, 11, 0.2, "A"],
        [290, 3,  5,  1,  15, 12, 0.2, "A"],
        [300, 3,  5,  1,  13, 10, 0.3, "A"],
        [310, 4,  6,  1,  14,  9, 0.3, "A"],
        [320, 4,  6,  2,  15, 10, 0.3, "A"],
        # B
        [340, 6,  7,  2,  12,  8, 0.4, "B"],
        [350, 7,  8,  2,  11,  8, 0.4, "B"],
        [360, 8,  8,  2,  12,  7, 0.5, "B"],
        [370, 8,  9,  3,  11,  7, 0.5, "B"],
        [380, 9,  9,  3,  10,  6, 0.6, "B"],
        # C
        [390, 11, 11,  4,  9,  5, 0.7, "C"],
        [400, 12, 12,  4,  8,  5, 0.7, "C"],
        [410, 13, 13,  5,  8,  4, 0.8, "C"],
        [420, 14, 14,  5,  7,  4, 0.9, "C"],
        [430, 15, 15,  6,  7,  3, 0.9, "C"],
        # D
        [450, 18, 18,  7,  6,  2, 1.1, "D"],
        [460, 19, 19,  7,  5,  2, 1.2, "D"],
        [470, 21, 20,  8,  5,  2, 1.2, "D"],
        [480, 22, 21,  9,  4,  1, 1.3, "D"],
        [490, 24, 22,  9,  4,  1, 1.4, "D"],
        # E
        [510, 28, 25, 11,  3,  1, 1.7, "E"],
        [520, 30, 26, 12,  3,  1, 1.8, "E"],
        [540, 32, 28, 13,  2,  1, 2.0, "E"],
        [560, 35, 30, 15,  2,  0, 2.2, "E"],
        [580, 38, 32, 16,  1,  0, 2.5, "E"],
    ]


# ============================================================
# REAL / EMPIRICAL TRAINING DATA LOADER
# ============================================================

def _load_db_products() -> pd.DataFrame:
    """Load all products from the local SQLite DB that have a valid health_grade."""
    rows: list[dict] = []
    if not DB_PATH.exists():
        return pd.DataFrame(columns=FEATURES + ["health_grade"])

    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT calories, sugar, total_fat, saturated_fat,
                   protein, fibre, salt, health_grade
            FROM products
            WHERE health_grade IS NOT NULL
              AND health_grade IN ('A','B','C','D','E')
            """
        )
        for row in cursor.fetchall():
            rows.append(dict(row))
        conn.close()
    except Exception:
        pass

    if not rows:
        return pd.DataFrame(columns=FEATURES + ["health_grade"])

    df = pd.DataFrame(rows)

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES)
    return df[FEATURES + ["health_grade"]]


def _load_csv_products() -> pd.DataFrame:
    """Load the latest OFF CSV snapshot if it exists."""
    latest_csv = BASE_DIR / "data" / "processed" / "foodgrade_india_latest.csv"
    if not latest_csv.exists():
        return pd.DataFrame(columns=FEATURES + ["health_grade"])

    try:
        df = pd.read_csv(str(latest_csv), encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame(columns=FEATURES + ["health_grade"])

    # health_grade may be missing in the CSV — derive it from the grading engine
    if "health_grade" not in df.columns:
        try:
            from services.grading_engine import calculate_foodgrade
            grades = []
            for _, row in df.iterrows():
                try:
                    result = calculate_foodgrade(row.to_dict())
                    grades.append(result.get("health_grade", "C"))
                except Exception:
                    grades.append("C")
            df["health_grade"] = grades
        except Exception:
            return pd.DataFrame(columns=FEATURES + ["health_grade"])

    # Keep only labelled rows with valid grades.
    df = df[df["health_grade"].isin(["A", "B", "C", "D", "E"])].copy()

    # Convert feature values to numeric without replacing missing values with zero.
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Exclude rows with missing or invalid nutrition features.
    df = df.dropna(subset=FEATURES)

    return df[FEATURES + ["health_grade"]]


def get_real_training_data() -> pd.DataFrame:
    """
    Build training dataframe from real data sources:
    1. Local SQLite DB products
    2. Latest OFF India CSV (if present)
    3. Synthetic fallback rows (always included to maintain grade coverage)

    Returns combined, deduplicated dataframe.
    """
    db_df = _load_db_products()
    csv_df = _load_csv_products()

    # Always include synthetic data so every grade class has representation
    synthetic_cols = FEATURES + ["health_grade"]
    synth_df = pd.DataFrame(_get_synthetic_data(), columns=synthetic_cols)

    # Filter out empty frames before concat to avoid pandas FutureWarning
    frames = [f for f in [db_df, csv_df, synth_df] if not f.empty]
    combined = pd.concat(frames, ignore_index=True)

    # Drop rows where all features are zero (degenerate entries)
    feature_sum = combined[FEATURES].sum(axis=1)
    combined = combined[feature_sum > 0].copy()
    combined = combined.drop_duplicates(subset=FEATURES).reset_index(drop=True)

    return combined


# legacy alias kept for backward compat
def get_training_data() -> pd.DataFrame:
    return get_real_training_data()


# ============================================================
# METRICS PERSISTENCE
# ============================================================

def _save_metrics(metrics: dict[str, Any]) -> None:
    """Persist metrics dict as JSON alongside the model file."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # confusion_matrix may be ndarray — convert to list
    safe = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            safe[k] = v.tolist()
        else:
            safe[k] = v
    METRICS_PATH.write_text(
        json.dumps(safe, indent=2, default=str), encoding="utf-8"
    )


def _load_metrics_from_disk() -> dict[str, Any] | None:
    """Load cached metrics from JSON; returns None if missing or corrupt."""
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_model_metrics() -> dict[str, Any]:
    """
    Return cached model metrics from disk.

    If the model has never been trained, trains it first so the page
    always has something to display.  Subsequent page loads just read
    from the JSON file — no sklearn work at all.
    """
    cached = _load_metrics_from_disk()
    if cached is not None and MODEL_PATH.exists():
        cached["model_state"] = "cached"
        return cached

    # First-time bootstrap — train once and cache
    result = train_health_model()
    result["model_state"] = "just_trained"
    return result


# ============================================================
# TRAIN MODEL
# ============================================================

def train_health_model() -> dict[str, Any]:
    """
    Train (or retrain) the Random Forest on available real + synthetic data.
    Saves both the model pkl and a metrics JSON for fast page loads.
    """
    df = get_real_training_data()

    X = df[FEATURES]
    y = df["health_grade"]

    labels = ["A", "B", "C", "D", "E"]

    # Guard: need at least 2 distinct classes and enough rows to split
    unique_classes = y.nunique()
    min_rows_needed = max(unique_classes * 2, 10)

    if len(df) < min_rows_needed or unique_classes < 2:
        # Return a minimal error result without crashing
        return {
            "dataset_size": len(df),
            "training_size": 0,
            "testing_size": 0,
            "accuracy": 0.0,
            "classification_report": {},
            "confusion_matrix": [[0]*5 for _ in range(5)],
            "labels": labels,
            "feature_importance": [],
            "cross_val_scores": [],
            "cross_val_mean": 0.0,
            "cross_val_std": 0.0,
            "model_path": str(MODEL_PATH),
            "model_state": "error_insufficient_data",
            "data_sources": {
                "db_rows": len(_load_db_products()),
                "csv_rows": len(_load_csv_products()),
                "synthetic_rows": 25,
            },
        }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=8,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test, predictions,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(y_test, predictions, labels=labels)

    # Cross-validation (5-fold, suppress warnings when folds lack a class)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_folds = min(5, len(df) // max(unique_classes, 1))
        cv_folds = max(cv_folds, 2)
        cv_scores = cross_val_score(
            RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8),
            X, y, cv=cv_folds, scoring="accuracy"
        )

    # Feature importance
    feature_importance = sorted(
        [
            {"feature": f, "importance": round(float(imp) * 100, 2)}
            for f, imp in zip(FEATURES, model.feature_importances_)
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    class_distribution = {label: int((df["health_grade"] == label).sum()) for label in labels}

    limitations = [
        "Surrogate Model: Approximates the deterministic 6-dimension FoodGrade scoring rules; not independent clinical validation.",
        "Feature Scope: Evaluates only reported macronutrients and sodium; micronutrients, additives, and processing level are not modeled.",
        "Class Distribution: Real Indian packaged foods from Open Food Facts exhibit skew toward middle and lower grades (C/D).",
        "Label Origin: Training targets reflect FoodGrade V2.5 scoring thresholds applied to label values rather than subjective human ratings."
    ]

    result = {
        "dataset_size": len(df),
        "training_size": len(X_train),
        "testing_size": len(X_test),
        "accuracy": round(accuracy * 100, 2),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "labels": labels,
        "class_distribution": class_distribution,
        "limitations": limitations,
        "target_label_policy": "FoodGrade V2.5 algorithmic profile applied to nutrition facts",
        "feature_importance": feature_importance,
        "cross_val_scores": [round(s * 100, 2) for s in cv_scores.tolist()],
        "cross_val_mean": round(float(cv_scores.mean()) * 100, 2),
        "cross_val_std": round(float(cv_scores.std()) * 100, 2),
        "model_path": str(MODEL_PATH),
        "model_state": "trained",
        "data_sources": {
            "db_rows": len(_load_db_products()),
            "csv_rows": len(_load_csv_products()),
            "synthetic_rows": 25,
        },
    }

    _save_metrics(result)
    return result


# ============================================================
# PREDICTION
# ============================================================

def predict_health_grade(
    product: dict[str, Any],
    rule_grade: str | None = None
) -> dict[str, Any]:
    """Predict health grade for a single product dict and compare with rule-based grade."""
    if not MODEL_PATH.exists():
        train_health_model()

    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        # Fallback if pickle is corrupt or incompatible
        train_health_model()
        model = joblib.load(MODEL_PATH)

    values = []
    for feature in FEATURES:

        values = []

        for feature in FEATURES:
            raw_value = product.get(feature)

            if raw_value is None or raw_value == "":
                return {
                    "success": False,
                    "error": f"Missing nutrition value: {feature}",
                }

            try:
                value = float(raw_value)
            except (ValueError, TypeError):
                return {
                    "success": False,
                    "error": f"Invalid nutrition value: {feature}",
                }

            if not np.isfinite(value) or value < 0:
                return {
                    "success": False,
                    "error": f"Invalid nutrition value: {feature}",
                }

            values.append(value)

    dataframe = pd.DataFrame([values], columns=FEATURES)
    grade = str(model.predict(dataframe)[0])

    # Also get probability distribution
    confidence: dict[str, float] = {}
    try:
        proba = model.predict_proba(dataframe)[0]
        classes = model.classes_
        confidence = {str(c): round(float(p) * 100, 1) for c, p in zip(classes, proba)}
    except Exception:
        confidence = {}

    top_prob = max(confidence.values()) if confidence else 0.0

    aligned = (grade == str(rule_grade).upper()) if rule_grade else None
    if rule_grade:
        rule_g = str(rule_grade).upper()
        if grade == rule_g:
            explanation = f"ML model concurs with the deterministic {rule_g} grade ({top_prob}% confidence)."
        else:
            explanation = (
                f"ML model predicted grade {grade} ({top_prob}% confidence) while the "
                f"authoritative 6-dimension score assigned grade {rule_g}. "
                "The deterministic rule-based grade remains the primary evaluation."
            )
    else:
        explanation = f"Random Forest predicted grade {grade} with {top_prob}% estimated confidence."

    return {
        "grade": grade,
        "confidence": confidence,
        "top_confidence": top_prob,
        "aligned_with_rules": aligned,
        "comparison_note": explanation,
        "features_used": {f: v for f, v in zip(FEATURES, values)},
    }