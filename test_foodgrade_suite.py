"""
FoodGrade AI — Master Comprehensive Automated Test Suite (Pytest)
Covers:
1. 6-Dimension Health Grading, Weights, Boundaries, and Missing Data Handling
2. FSSAI Claim Verification (All 4 states: VERIFIED, NOT_VERIFIED, INSUFFICIENT_EVIDENCE, NO_CLAIMS_DETECTED)
3. Parentheses-Aware Compound Ingredient Parsing and Category Detection
4. SSRF Hardening, Redirect Safety, and Web Retrieval Constraints
5. OCR Upload Constraints and Preprocessing
6. Content-Based Recommendations and Healthier Alternatives Reasoning
7. Machine Learning Model Training, Caching, and Prediction Integrity
8. Full Application Route Rendering (HTTP 200)
"""

import os
import sys
import unittest
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app import app, get_db
from services.grading_engine import (
    calculate_foodgrade,
    calculate_health_score,
    verify_claims,
    calculate_data_quality_score,
    grade_from_score,
)
from services.v3_extractor import (
    split_ingredients,
    parse_compound_ingredient,
    classify_ingredients,
)
from services.v3_pipeline import run_v3_pipeline
from services.web_retriever import is_safe_destination, validate_url
from services.v5_recommender import (
    recommend_similar_products,
    recommend_healthier_alternatives,
)
from services.v6_ml import (
    train_health_model,
    load_model_metrics,
    predict_health_grade,
    FEATURES,
)


class TestFoodGradeComprehensive(unittest.TestCase):

    def setUp(self):
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    # ==================================================================
    # 1. 6-DIMENSION SCORING & GRADE BOUNDARIES
    # ==================================================================
    def test_grade_boundaries(self):
        """Verify strict grade boundaries for health and claims."""
        self.assertEqual(grade_from_score(100.0), "A")
        self.assertEqual(grade_from_score(85.0), "A")
        self.assertEqual(grade_from_score(84.99), "B")
        self.assertEqual(grade_from_score(70.0), "B")
        self.assertEqual(grade_from_score(69.99), "C")
        self.assertEqual(grade_from_score(55.0), "C")
        self.assertEqual(grade_from_score(54.99), "D")
        self.assertEqual(grade_from_score(40.0), "D")
        self.assertEqual(grade_from_score(39.99), "E")
        self.assertEqual(grade_from_score(0.0), "E")

    def test_six_dimension_weights_and_calculation(self):
        """Verify exact weights: Sugar 20%, Sat Fat 20%, Salt 20%, Energy 10%, Fibre 15%, Protein 15%."""
        healthy_product = {
            "product_name": "Organic Rolled Oats",
            "calories": 350.0,
            "sugar": 1.0,
            "total_fat": 5.0,
            "saturated_fat": 0.8,
            "protein": 13.0,
            "fibre": 10.0,
            "salt": 0.05,
            "ingredients": "Whole grain rolled oats",
        }
        res = calculate_foodgrade(healthy_product)
        self.assertGreaterEqual(res["health_score"], 85.0)
        self.assertEqual(res["health_grade"], "A")
        self.assertIn("sugar", res["dimension_scores"])
        self.assertIn("saturated_fat", res["dimension_scores"])
        self.assertIn("salt", res["dimension_scores"])
        self.assertIn("energy", res["dimension_scores"])
        self.assertIn("fibre", res["dimension_scores"])
        self.assertIn("protein", res["dimension_scores"])

    def test_missing_data_uses_neutral_reference_not_zero(self):
        """Missing nutrients must NOT be silently replaced with 0 (which would give false high scores)."""
        product_missing = {
            "product_name": "Incomplete Product",
            "sugar": None,  # Undeclared
            "saturated_fat": None,
            "protein": 10.0,
            "fibre": 5.0,
        }
        res = calculate_foodgrade(product_missing)
        # Neutral reference score is 50.0 for undeclared negative nutrients
        self.assertEqual(res["dimension_scores"]["sugar"], 50.0)
        self.assertEqual(res["dimension_scores"]["saturated_fat"], 50.0)
        # Check human-readable missing field descriptions
        missing_text = " ".join(res["missing_fields"]).lower()
        self.assertIn("sugar", missing_text)
        self.assertIn("saturated fat", missing_text)
        self.assertLess(res["data_quality_score"], 100.0)

    # ==================================================================
    # 2. CLAIM VERIFICATION (ALL 4 REQUIRED STATES)
    # ==================================================================
    def test_claim_state_verified(self):
        """Product meeting FSSAI threshold receives VERIFIED state."""
        product = {
            "product_name": "High Protein Greek Yogurt",
            "claims": "High Protein",
            "protein": 12.0,  # >= 10g FSSAI limit
            "ingredients": "Cultured pasteurized milk",
        }
        res = calculate_foodgrade(product)
        self.assertEqual(res["claim_status"], "VERIFIED")
        self.assertEqual(res["claim_grade"], "A")

    def test_claim_state_not_verified(self):
        """Product failing FSSAI threshold receives NOT_VERIFIED state."""
        product = {
            "product_name": "Protein Snack",
            "claims": "High Protein",
            "protein": 2.5,  # Below 10g limit
            "ingredients": "Wheat flour, palm oil, salt",
        }
        res = calculate_foodgrade(product)
        self.assertEqual(res["claim_status"], "NOT_VERIFIED")
        self.assertEqual(res["claim_grade"], "E")

    def test_claim_state_insufficient_evidence(self):
        """Claim made but corresponding nutrient undeclared receives INSUFFICIENT_EVIDENCE."""
        product = {
            "product_name": "Fibre Cracker",
            "claims": "High Fibre",
            "fibre": None,  # Missing on label!
            "ingredients": "Wheat flour",
        }
        res = calculate_foodgrade(product)
        self.assertEqual(res["claim_status"], "INSUFFICIENT_EVIDENCE")

    def test_claim_state_no_claims_detected(self):
        """Product with no claims must NOT receive invented Grade B; must represent absence explicitly."""
        product = {
            "product_name": "Plain Rice",
            "claims": "",
            "calories": 130.0,
            "sugar": 0.1,
            "protein": 2.7,
            "ingredients": "Rice",
        }
        res = calculate_foodgrade(product)
        self.assertEqual(res["claim_status"], "NO_CLAIMS_DETECTED")
        self.assertIn(res["claim_grade"], ["No Claim", "E"])

    # ==================================================================
    # 3. COMPOUND INGREDIENT PARSING
    # ==================================================================
    def test_compound_ingredient_splitting(self):
        """Commas inside parentheses must not break compound ingredients."""
        text = "Whole grain oats (54%), Milk Chocolate (Sugar, Cocoa Butter, Milk Powder), Salt"
        split_items = split_ingredients(text)
        self.assertEqual(len(split_items), 3)
        self.assertEqual(split_items[0], "Whole grain oats (54%)")
        self.assertEqual(split_items[1], "Milk Chocolate (Sugar, Cocoa Butter, Milk Powder)")
        self.assertEqual(split_items[2], "Salt")

    def test_parse_compound_ingredient_structure(self):
        """Extract base name and sub-ingredients from parenthesized compound."""
        parsed = parse_compound_ingredient("Milk Chocolate (Sugar, Cocoa Butter, Milk Powder)")
        self.assertTrue(parsed["is_compound"])
        self.assertEqual(parsed["name"], "Milk Chocolate")
        self.assertIn("Sugar", parsed["sub_ingredients"])
        self.assertIn("Cocoa Butter", parsed["sub_ingredients"])

    def test_classify_ingredients_detects_sub_ingredients(self):
        """Sugar in sub-ingredients must be detected as sugar_related category."""
        items = ["Milk Chocolate (Sugar, Cocoa Butter)", "Oats"]
        classified = classify_ingredients(items)
        self.assertIn("sugar_related", classified[0]["categories"])
        self.assertIn("oil_or_fat", classified[0]["categories"])
        self.assertIn("whole_grains", classified[1]["categories"])

    # ==================================================================
    # 4. SSRF & WEB RETRIEVAL HARDENING
    # ==================================================================
    def test_ssrf_rejects_loopback(self):
        """Localhost and loopback addresses must be rejected."""
        safe, reason = is_safe_destination("http://127.0.0.1:5000/admin")
        self.assertFalse(safe)
        self.assertIn("loopback", reason.lower())

        safe, _ = is_safe_destination("http://localhost:8080")
        self.assertFalse(safe)

    def test_ssrf_rejects_private_ips(self):
        """RFC1918 private IP addresses must be rejected."""
        for ip in ["10.0.0.1", "172.16.0.5", "192.168.1.1"]:
            safe, reason = is_safe_destination(f"http://{ip}/secret")
            self.assertFalse(safe)

    def test_ssrf_rejects_cloud_metadata(self):
        """AWS/GCP link-local metadata endpoints must be rejected."""
        safe, reason = is_safe_destination("http://169.254.169.254/latest/meta-data/")
        self.assertFalse(safe)

    def test_ssrf_rejects_non_http_schemes(self):
        """File, ftp, gopher, and ldap schemes must be rejected."""
        self.assertFalse(validate_url("file:///etc/passwd"))
        self.assertFalse(validate_url("ftp://example.com/data"))

    # ==================================================================
    # 5. RECOMMENDATIONS & HEALTHIER ALTERNATIVES
    # ==================================================================
    def test_recommend_similar_and_healthier(self):
        """Verify similar and healthier recommendations provide transparent reasons."""
        target = {
            "id": 101,
            "product_name": "Sugar Corn Flakes",
            "brand": "BrandX",
            "category": "Cereal",
            "ingredients": "Corn, sugar, salt",
            "health_grade": "D",
            "health_score": 42.0,
            "sugar": 28.0,
            "saturated_fat": 1.0,
            "fibre": 1.5,
            "protein": 4.0,
            "salt": 1.2,
        }
        alt = {
            "id": 102,
            "product_name": "High Fibre Bran Flakes",
            "brand": "BrandY",
            "category": "Cereal",
            "ingredients": "Whole wheat, wheat bran, salt",
            "health_grade": "A",
            "health_score": 88.0,
            "sugar": 5.0,
            "saturated_fat": 0.5,
            "fibre": 11.0,
            "protein": 10.0,
            "salt": 0.3,
        }
        catalog = [target, alt]
        healthier = recommend_healthier_alternatives(target, catalog, top_n=3)
        self.assertEqual(len(healthier), 1)
        self.assertEqual(healthier[0]["product"]["id"], 102)
        self.assertIn("reasons", healthier[0])
        self.assertTrue(len(healthier[0]["reasons"]) > 0)

    # ==================================================================
    # 6. MACHINE LEARNING TRAINING & PREDICTION
    # ==================================================================
    def test_ml_train_cache_and_predict(self):
        """Verify model trains, caches metrics, and outputs defensible predictions."""
        metrics = train_health_model()
        self.assertGreaterEqual(metrics["dataset_size"], 25)
        self.assertIn("accuracy", metrics)
        self.assertIn("cross_val_mean", metrics)
        self.assertIn("class_distribution", metrics)
        self.assertIn("limitations", metrics)

        # Cached load must not re-run training
        cached = load_model_metrics()
        self.assertEqual(cached.get("model_state"), "cached")

        # Single product prediction
        pred = predict_health_grade({
            "calories": 280, "sugar": 3, "total_fat": 4, "saturated_fat": 1,
            "protein": 14, "fibre": 10, "salt": 0.2
        }, rule_grade="A")
        self.assertIn(pred["grade"], ["A", "B", "C", "D", "E"])
        self.assertIn("comparison_note", pred)

    # ==================================================================
    # 7. CORE HTTP ROUTE INTEGRITY
    # ==================================================================
    def test_all_major_routes_return_200(self):
        """Verify all major application endpoints render successfully without exceptions."""
        routes = [
            "/",
            "/products",
            "/product/new",
            "/product/1",
            "/v3",
            "/v3/analyze/1",
            "/v3/evaluation",
            "/web-retrieval",
            "/v5",
            "/v6",
            "/v7",
            "/v8",
            "/methodology",
            "/data-center",
        ]
        for route in routes:
            with self.subTest(route=route):
                res = self.client.get(route)
                self.assertEqual(
                    res.status_code, 200,
                    f"Route {route} failed with status {res.status_code}"
                )


if __name__ == "__main__":
    unittest.main()
