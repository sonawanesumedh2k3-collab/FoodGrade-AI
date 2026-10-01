"""
FoodGrade AI V7
Testing and Validation Module
"""

import sqlite3
from pathlib import Path
import importlib


BASE_DIR = Path(__file__).resolve().parents[1]

DB_PATH = BASE_DIR / "database" / "foodgrade.db"
MODEL_PATH = BASE_DIR / "data" / "models" / "health_grade_model.pkl"


def run_tests():

    tests = []

    # ========================================================
    # 1. DATABASE TEST
    # ========================================================

    try:

        if DB_PATH.exists():

            conn = sqlite3.connect(
                DB_PATH
            )

            cursor = conn.cursor()

            cursor.execute(
                "SELECT COUNT(*) FROM products"
            )

            count = cursor.fetchone()[0]

            conn.close()

            tests.append({
                "name": "Database Connectivity",
                "status": "PASS",
                "details": f"{count} product records available"
            })

        else:

            tests.append({
                "name": "Database Connectivity",
                "status": "FAIL",
                "details": "Database file not found"
            })

    except Exception as exc:

        tests.append({
            "name": "Database Connectivity",
            "status": "FAIL",
            "details": str(exc)
        })


    # ========================================================
    # 2. V3 NLP TEST
    # ========================================================

    try:

        importlib.import_module(
            "services.v3_nlp"
        )

        importlib.import_module(
            "services.v3_extractor"
        )

        importlib.import_module(
            "services.v3_ranker"
        )

        tests.append({
            "name": "V3 Information Extraction & NLP",
            "status": "PASS",
            "details": "V3 NLP modules loaded successfully"
        })

    except Exception as exc:

        tests.append({
            "name": "V3 Information Extraction & NLP",
            "status": "FAIL",
            "details": str(exc)
        })


    # ========================================================
    # 3. V4 WEB RETRIEVAL
    # ========================================================

    try:

        importlib.import_module(
            "services.web_retriever"
        )

        importlib.import_module(
            "services.web_parser"
        )

        importlib.import_module(
            "services.scrapy_runner"
        )

        tests.append({
            "name": "V4 Web Retrieval & Crawling",
            "status": "PASS",
            "details": "BeautifulSoup/Scrapy modules loaded successfully"
        })

    except Exception as exc:

        tests.append({
            "name": "V4 Web Retrieval & Crawling",
            "status": "FAIL",
            "details": str(exc)
        })


    # ========================================================
    # 4. V5 OCR
    # ========================================================

    try:

        from services.v5_ocr import perform_ocr

        import pytesseract

        version = pytesseract.get_tesseract_version()

        tests.append({
            "name": "V5 OCR",
            "status": "PASS",
            "details": f"Tesseract {version} detected"
        })

    except Exception as exc:

        tests.append({
            "name": "V5 OCR",
            "status": "FAIL",
            "details": str(exc)
        })


    # ========================================================
    # 5. CONTENT-BASED RECOMMENDATION
    # ========================================================

    try:

        from services.v5_recommender import (
            recommend_similar_products
        )

        tests.append({
            "name": "Content-Based Recommendation",
            "status": "PASS",
            "details": "TF-IDF and cosine similarity module loaded"
        })

    except Exception as exc:

        tests.append({
            "name": "Content-Based Recommendation",
            "status": "FAIL",
            "details": str(exc)
        })


    # ========================================================
    # 6. V6 MACHINE LEARNING
    # ========================================================

    try:

        from services.v6_ml import (
            train_health_model
        )

        result = train_health_model()

        tests.append({
            "name": "V6 Machine Learning",
            "status": "PASS",
            "details": (
                f"Random Forest trained successfully "
                f"with {result['dataset_size']} samples; "
                f"accuracy {result['accuracy']}%"
            )
        })

    except Exception as exc:

        tests.append({
            "name": "V6 Machine Learning",
            "status": "FAIL",
            "details": str(exc)
        })


    # ========================================================
    # SUMMARY
    # ========================================================

    passed = sum(
        1
        for test in tests
        if test["status"] == "PASS"
    )

    failed = sum(
        1
        for test in tests
        if test["status"] == "FAIL"
    )

    total = len(tests)

    return {
        "tests": tests,
        "passed": passed,
        "failed": failed,
        "total": total
    }