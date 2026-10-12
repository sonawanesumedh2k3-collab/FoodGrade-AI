"""
test_ocr_integration.py — Integration and regression test suite for OCR integration.
Tests:
1. Base64 payload encoding and MIME type formatting sent to OCR service.
2. Successful OCR response normalization into FoodGrade AI product schema.
3. Correct handling of all 9 Lay's regression nutrition values without corruption.
4. Preservation of zero cholesterol (0.0 mg) and zero trans fat (0.0 g).
5. Preservation of missing nutrients as None (not silently converted to zero).
6. Serving size, ingredients, and FSSAI license number extraction and preservation.
7. OCR service connection failure (OCRConnectionError) handling.
8. OCR HTTP error (500) and malformed non-JSON response handling.
9. Invalid or malformed response fields handling.
10. Invalid file extension and file size rejection in /v5/ocr route.
11. Existing grading engine execution and separate claim/health grades.
12. Application route stability and graceful recovery when OCR service is down.
"""

import io
import json
import unittest
from unittest.mock import MagicMock, patch

import requests

from app import app
from services.grading_engine import calculate_foodgrade
from services.v5_ocr import (
    DEFAULT_OCR_URL,
    OCRConnectionError,
    OCRResponseError,
    call_ocr_service,
    extract_product_from_text,
    normalize_ocr_response,
    perform_ocr,
)


class TestOCRServiceIntegration(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    # 1. Correct base64 request and MIME type
    @patch("requests.post")
    def test_call_ocr_service_request_format(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "nutrition": {"energy_kcal": 200.0},
            "serving_size": 30.0,
            "ingredients": "Oats",
            "fssai": "10014064000435",
            "confidence": "high",
            "source": "snippet",
        }
        mock_post.return_value = mock_resp

        fake_bytes = b"\x89PNG\r\n\x1a\nfakeimagebytes"
        res = call_ocr_service(fake_bytes, mime_type="image/png", service_url="http://127.0.0.1:8001")

        self.assertEqual(res["confidence"], "high")
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://127.0.0.1:8001/extract/image")
        self.assertIn("image", kwargs["json"])
        self.assertEqual(kwargs["json"]["mime_type"], "image/png")

    # 2. Response normalization into schema
    def test_normalize_ocr_response_schema(self):
        sample_data = {
            "nutrition": {
                "energy_kcal": 536.0,
                "protein_g": 6.6,
                "carbohydrates_g": 53.6,
                "sugar_g": 0.6,
                "total_fat_g": 34.2,
                "saturated_fat_g": 15.2,
                "trans_fat_g": 0.1,
                "cholesterol_mg": 0.0,
                "sodium_mg": 670.0,
            },
            "serving_size": 20.0,
            "ingredients": "Potatoes, Edible Vegetable Oil, Salt",
            "fssai": "10014064000435",
            "confidence": "high",
            "source": "snippet",
        }
        product = normalize_ocr_response(sample_data, raw_text="Lay's Potato Chips")

        # product_name is extracted from raw_text lines; for a single line like
        # "Lay's Potato Chips" the scorer returns the full text (brand not stripped).
        self.assertIsInstance(product["product_name"], str)
        self.assertTrue(len(product["product_name"]) > 0)
        # Brand detected from known-brand list
        self.assertEqual(product["brand"], "Lay's")
        self.assertEqual(product["serving_size"], 20.0)
        self.assertEqual(product["fssai"], "10014064000435")
        self.assertEqual(product["ocr_confidence"], "high")
        self.assertEqual(product["source"], "OCR")

    # 3. All 9 Lay's regression nutrition values preserved
    def test_all_nine_lays_regression_values_preserved(self):
        sample_data = {
            "nutrition": {
                "energy_kcal": 536.0,
                "protein_g": 6.6,
                "carbohydrates_g": 53.6,
                "sugar_g": 0.6,
                "total_fat_g": 34.2,
                "saturated_fat_g": 15.2,
                "trans_fat_g": 0.1,
                "cholesterol_mg": 0.0,
                "sodium_mg": 670.0,
            },
            "serving_size": 20.0,
            "ingredients": "Potatoes, Oil, Salt",
            "fssai": "10014064000435",
            "confidence": "high",
        }
        product = normalize_ocr_response(sample_data)
        raw = product["raw_nutrition"]

        self.assertEqual(raw["energy_kcal"], 536.0)
        self.assertEqual(raw["protein_g"], 6.6)
        self.assertEqual(raw["carbohydrates_g"], 53.6)
        self.assertEqual(raw["sugar_g"], 0.6)
        self.assertEqual(raw["total_fat_g"], 34.2)
        self.assertEqual(raw["saturated_fat_g"], 15.2)
        self.assertEqual(raw["trans_fat_g"], 0.1)
        self.assertEqual(raw["cholesterol_mg"], 0.0)
        self.assertEqual(raw["sodium_mg"], 670.0)

    # 4. Preservation of zero cholesterol and zero trans fat
    def test_preservation_of_genuine_zeroes(self):
        sample_data = {
            "nutrition": {
                "energy_kcal": 100.0,
                "cholesterol_mg": 0.0,
                "trans_fat_g": 0.0,
            },
            "serving_size": 10.0,
            "ingredients": "Tea",
        }
        product = normalize_ocr_response(sample_data)
        raw = product["raw_nutrition"]

        self.assertIsNotNone(raw["cholesterol_mg"])
        self.assertEqual(raw["cholesterol_mg"], 0.0)
        self.assertIsNotNone(raw["trans_fat_g"])
        self.assertEqual(raw["trans_fat_g"], 0.0)

    # 5. Preservation of missing values as missing (None)
    def test_preservation_of_missing_nutrients(self):
        sample_data = {
            "nutrition": {
                "energy_kcal": 150.0,
                # protein, fibre, sat_fat are omitted
            },
            "serving_size": None,
            "ingredients": None,
        }
        product = normalize_ocr_response(sample_data)
        raw = product["raw_nutrition"]

        self.assertIsNone(raw["protein_g"])
        self.assertIsNone(raw["dietary_fiber_g"])
        self.assertIsNone(raw["saturated_fat_g"])
        self.assertIn("Protein not detected on label", product["nutrition_warnings"])
        self.assertIn("Protein", product["missing_nutrients"])

    # 6. Serving size, ingredients, FSSAI preservation
    def test_metadata_fields_preservation(self):
        sample_data = {
            "nutrition": {"energy_kcal": 250.0},
            "serving_size": 25.5,
            "ingredients": "Whole Wheat Flour, Water, Yeast, Salt",
            "fssai": "10015011000123",
            "confidence": "high",
        }
        product = normalize_ocr_response(sample_data)

        self.assertEqual(product["serving_size"], 25.5)
        self.assertEqual(product["ingredients"], "Whole Wheat Flour, Water, Yeast, Salt")
        self.assertEqual(product["fssai"], "10015011000123")

    # 7. OCR service connection failure
    @patch("requests.post")
    def test_ocr_connection_failure(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        with self.assertRaises(OCRConnectionError) as ctx:
            call_ocr_service(b"fakebytes", service_url="http://127.0.0.1:8001")
        self.assertIn("Could not connect to OCR service", str(ctx.exception))

    # 8. OCR HTTP error (500) and non-JSON response
    @patch("requests.post")
    def test_ocr_http_error_and_malformed_json(self, mock_post):
        # 500 error
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_post.return_value = mock_resp

        with self.assertRaises(OCRResponseError) as ctx:
            call_ocr_service(b"fakebytes")
        self.assertIn("returned error status 500", str(ctx.exception))

        # 200 but not valid JSON
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("Invalid JSON")
        with self.assertRaises(OCRResponseError) as ctx:
            call_ocr_service(b"fakebytes")
        self.assertIn("invalid non-JSON response", str(ctx.exception))

    # 9. Invalid or malformed response fields
    def test_malformed_response_fields_handled_gracefully(self):
        malformed = {
            "nutrition": "not a dict",
            "serving_size": "invalid_number",
            "ingredients": None,
            "fssai": 12345,
            "confidence": None,
        }
        product = normalize_ocr_response(malformed)
        self.assertIsInstance(product, dict)
        self.assertIsNone(product["serving_size"])
        self.assertEqual(product["fssai"], "12345")
        self.assertEqual(product["ocr_confidence"], "medium")

    # 10. File upload validation (unsupported extension & size)
    def test_invalid_file_extension_rejected(self):
        data = {
            "image": (io.BytesIO(b"fake pdf content"), "document.pdf")
        }
        res = self.client.post("/v5/ocr", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 302)
        with self.client.session_transaction() as sess:
            flashes = [f[1] for f in sess.get("_flashes", [])]
            self.assertTrue(any("PNG, JPG, JPEG or WEBP" in f for f in flashes))

    def test_oversized_file_rejected(self):
        # 11 MB of zeros
        large_bytes = b"0" * (11 * 1024 * 1024)
        data = {
            "image": (io.BytesIO(large_bytes), "huge.png")
        }
        res = self.client.post("/v5/ocr", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 302)
        with self.client.session_transaction() as sess:
            flashes = [f[1] for f in sess.get("_flashes", [])]
            self.assertTrue(any("Maximum allowed size is 10 MB" in f for f in flashes))

    # 11. Grading integration without changing existing grading rules
    def test_grading_integration_with_extracted_product(self):
        ocr_product = {
            "product_name": "Lay's Classic Salted",
            "brand": "Lay's",
            "category": "Snacks",
            "ingredients": "Potatoes, Edible Vegetable Oil, Iodised Salt",
            "claims": "",
            "calories": 536.0,
            "sugar": 0.6,
            "total_fat": 34.2,
            "saturated_fat": 15.2,
            "protein": 6.6,
            "fibre": 0.0,
            "salt": 1.675,
            "sodium": 670.0,
        }
        res = calculate_foodgrade(ocr_product)
        self.assertIn("health_grade", res)
        self.assertIn("claim_grade", res)
        # Separate evaluation
        self.assertNotEqual(res["health_grade"], "")
        self.assertEqual(res["claim_status"], "NO_CLAIMS_DETECTED")
        self.assertIn("dimension_scores", res)
        self.assertEqual(len(res["dimension_scores"]), 6)

    # 12. Graceful UI behavior when OCR service is down and no fallback
    @patch("app.perform_ocr")
    def test_graceful_flash_when_service_unavailable(self, mock_perform_ocr):
        """The app must catch OCRConnectionError from perform_ocr and flash a clear message."""
        mock_perform_ocr.side_effect = OCRConnectionError("Service down")

        data = {
            "image": (io.BytesIO(b"\x89PNG\r\n\x1a\nfake"), "sample.png")
        }
        res = self.client.post("/v5/ocr", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 302)
        with self.client.session_transaction() as sess:
            flashes = [f[1] for f in sess.get("_flashes", [])]
            self.assertTrue(
                any("Standalone OCR Service Unavailable" in f for f in flashes),
                f"Expected 'Standalone OCR Service Unavailable' in flashes, got: {flashes}"
            )


if __name__ == "__main__":
    unittest.main()
