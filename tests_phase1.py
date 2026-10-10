"""
FoodGrade AI — Phase 1 Verification Test Suite
Tests Core Intelligence Upgrades:
1. Secure filename & OCR upload route
2. Six-dimension health scoring (Sugar 20%, Sat Fat 20%, Sodium 20%, Energy 10%, Fibre 15%, Protein 15%)
3. V3 NLP Claim Verification producing VERIFIED, NOT_VERIFIED, INSUFFICIENT_EVIDENCE, NO_CLAIMS_DETECTED
4. Dynamic Data Completeness / Confidence scoring (missing != 0)
5. Full route regression test
"""

import io
import unittest
from app import app, get_db
from services.grading_engine import (
    calculate_foodgrade,
    calculate_health_score,
    verify_claims,
    calculate_data_quality_score,
    grade_from_score
)
from services.v3_pipeline import run_v3_pipeline


class TestPhase1CoreIntelligence(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    # ------------------------------------------------------------------
    # 1. OCR UPLOAD & SECURE_FILENAME TEST
    # ------------------------------------------------------------------
    def test_secure_filename_and_ocr_upload(self):
        """Verify secure_filename import is active and /v5/ocr processes images."""
        with open("data/ocr/demo.webp", "rb") as f:
            data = {
                "image": (io.BytesIO(f.read()), "demo.webp")
            }
            res = self.client.post("/v5/ocr", data=data, content_type="multipart/form-data")
            self.assertEqual(res.status_code, 200)
            html = res.get_data(as_text=True)
            self.assertTrue("OCR Extracted Text" in html or "Food Product" in html)

    # ------------------------------------------------------------------
    # 2. SIX-DIMENSION HEALTH SCORING (V2.5 METHODOLOGY)
    # ------------------------------------------------------------------
    def test_six_dimension_weights_and_grades(self):
        """Verify 6-dimension weights and score calculations."""
        # Product A: Excellent nutrition profile (High fibre, high protein, low sugar, low sat fat, low salt)
        healthy_product = {
            "product_name": "Nutritious Bran Cereal",
            "brand": "HealthCorp",
            "category": "Cereal",
            "ingredients": "Whole wheat, bran, oats, salt",
            "claims": "High Fibre, High Protein",
            "calories": 250.0,
            "sugar": 3.0,
            "total_fat": 2.0,
            "saturated_fat": 0.5,
            "protein": 14.0,
            "fibre": 10.0,
            "salt": 0.2,
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

        # Product B: Unhealthy ultra-processed snack (High sugar, high sat fat, high salt, high calorie, low fibre)
        junk_product = {
            "product_name": "Sugar Rush Cookies",
            "brand": "SweetCo",
            "category": "Biscuits",
            "ingredients": "Sugar, refined wheat flour, palm oil, salt",
            "claims": "",
            "calories": 520.0,
            "sugar": 38.0,
            "total_fat": 26.0,
            "saturated_fat": 15.0,
            "protein": 3.0,
            "fibre": 0.5,
            "salt": 1.8,
        }
        res_junk = calculate_foodgrade(junk_product)
        self.assertLess(res_junk["health_score"], 50.0)
        self.assertIn(res_junk["health_grade"], ["D", "E"])

    # ------------------------------------------------------------------
    # 3. CLAIM VERIFICATION (ALL 4 REQUIRED STATES)
    # ------------------------------------------------------------------
    def test_claim_verified_state(self):
        """Test VERIFIED claim state."""
        product = {
            "product_name": "Oatmeal",
            "claims": "High Fibre",
            "fibre": 8.0,
            "ingredients": "Whole grain rolled oats",
        }
        res = calculate_foodgrade(product)
        self.assertEqual(res["claim_status"], "VERIFIED")
        self.assertEqual(res["claim_score"], 100.0)
        self.assertEqual(res["claim_grade"], "A")
        self.assertTrue(any("[VERIFIED]" in r for r in res["claim_reasons"]))

    def test_claim_not_verified_state(self):
        """Test NOT_VERIFIED claim state (false advertising / insufficient level)."""
        product = {
            "product_name": "Protein Bar",
            "claims": "High Protein, Low Sugar",
            "protein": 4.0,  # Below FSSAI 10g threshold
            "sugar": 22.0,   # Above FSSAI 5g threshold
            "ingredients": "Dates, oats, whey",
        }
        res = calculate_foodgrade(product)
        self.assertEqual(res["claim_status"], "NOT_VERIFIED")
        self.assertEqual(res["claim_score"], 0.0)
        self.assertEqual(res["claim_grade"], "E")
        self.assertTrue(any("[NOT VERIFIED]" in r for r in res["claim_reasons"]))

    def test_claim_no_added_sugar_conflict(self):
        """Test No Added Sugar claim conflicting with sugar in ingredients."""
        product = {
            "product_name": "Fruit Drink",
            "claims": "No Added Sugar",
            "ingredients": "Water, apple juice concentrate, sugar, citric acid",
            "sugar": 12.0,
        }
        res = calculate_foodgrade(product)
        self.assertEqual(res["claim_status"], "NOT_VERIFIED")
        self.assertTrue(any("Added sugar ingredients detected" in r for r in res["claim_reasons"]))

    def test_claim_insufficient_evidence_state(self):
        """Test INSUFFICIENT_EVIDENCE claim state when required nutrient is undeclared."""
        product = {
            "product_name": "Mystery Bar",
            "claims": "High Fibre",
            "fibre": None,  # Undeclared on label
            "ingredients": "Oats, crisps",
        }
        res = calculate_foodgrade(product)
        self.assertEqual(res["claim_status"], "INSUFFICIENT_EVIDENCE")
        self.assertTrue(any("[INSUFFICIENT EVIDENCE]" in r for r in res["claim_reasons"]))

    def test_claim_no_claims_detected_state(self):
        """Test NO_CLAIMS_DETECTED state when product has no claims."""
        product = {
            "product_name": "Plain White Bread",
            "claims": "",
            "calories": 260.0,
            "sugar": 4.0,
            "protein": 7.0,
            "ingredients": "Wheat flour, water, yeast, salt",
        }
        res = calculate_foodgrade(product)
        self.assertEqual(res["claim_status"], "NO_CLAIMS_DETECTED")
        self.assertEqual(res["claim_grade"], "No Claim")
        self.assertEqual(res["claim_score"], 0.0)

    # ------------------------------------------------------------------
    # 4. DYNAMIC DATA COMPLETENESS / CONFIDENCE (MISSING != 0)
    # ------------------------------------------------------------------
    def test_missing_data_not_silently_zero(self):
        """Verify that missing negative nutrients are NOT treated as 0g (which would unearnedly award 100)."""
        # Product with explicit 0g sugar
        declared_zero_sugar = {
            "product_name": "Pure Green Tea",
            "sugar": 0.0,
            "saturated_fat": 0.0,
            "salt": 0.0,
            "calories": 5.0,
            "fibre": 0.0,
            "protein": 0.0,
            "ingredients": "Green tea leaves",
        }
        res_zero = calculate_foodgrade(declared_zero_sugar)
        self.assertEqual(res_zero["dimension_scores"]["sugar"], 100.0)

        # Product with undeclared / missing sugar
        missing_sugar = {
            "product_name": "Incomplete Label Snack",
            "sugar": None,  # Missing!
            "saturated_fat": 0.0,
            "salt": 0.0,
            "calories": 200.0,
            "fibre": 0.0,
            "protein": 0.0,
            "ingredients": "Puffed wheat",
        }
        res_missing = calculate_foodgrade(missing_sugar)
        # Must NOT be 100.0! Must use neutral reference (50.0)
        self.assertEqual(res_missing["dimension_scores"]["sugar"], 50.0)
        self.assertTrue(any("Total sugar not declared" in r for r in res_missing["health_reasons"]))
        # Data quality score must be penalized
        self.assertLess(res_missing["data_quality_score"], res_zero["data_quality_score"])

    # ------------------------------------------------------------------
    # 5. V3 PIPELINE INTEGRATION
    # ------------------------------------------------------------------
    def test_v3_pipeline_integration(self):
        """Verify run_v3_pipeline executes full NLP extraction and passes into FoodGrade engine."""
        product = {
            "id": 1,
            "product_name": "Oats & Fibre Cereal",
            "brand": "Demo",
            "category": "Cereal",
            "claims": "High Fibre",
            "ingredients": "Whole grain oats, wheat bran, salt",
            "calories": 360.0,
            "sugar": 8.0,
            "total_fat": 6.0,
            "saturated_fat": 1.0,
            "protein": 11.0,
            "fibre": 9.0,
            "salt": 0.5,
        }
        result = run_v3_pipeline(product)
        self.assertIn("information_extraction", result)
        self.assertIn("nlp", result)
        self.assertIn("ranking", result)
        self.assertIn("foodgrade", result)
        fg = result["foodgrade"]
        self.assertEqual(fg["claim_status"], "VERIFIED")
        self.assertEqual(fg["claim_grade"], "A")
        self.assertEqual(fg["health_grade"], "A")
        self.assertIn("claim_details", fg["v3_evidence"])

    # ------------------------------------------------------------------
    # 6. ROUTE REGRESSION TESTS
    # ------------------------------------------------------------------
    def test_all_existing_routes_render_200(self):
        """Verify all core routes return HTTP 200."""
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
                self.assertEqual(res.status_code, 200, f"Route {route} failed with status {res.status_code}")


if __name__ == "__main__":
    unittest.main()
