from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# V2 uses the currently documented structured-search endpoint.
# Open Food Facts currently documents API v2 as deprecated but still supported;
# v3 is current, while structured search is documented under v2.
SEARCH_URL = "https://world.openfoodfacts.org/api/v2/search"
USER_AGENT = "FoodGradeAI/2.5 (educational project)"

FIELDS = [
    "code",
    "product_name",
    "brands",
    "categories",
    "ingredients_text",
    "labels",
    "labels_tags",
    "countries",
    "countries_tags",
    "nutriments",
    "image_front_url",
    "image_ingredients_url",
    "image_nutrition_url",
]

CSV_COLUMNS = [
    "barcode", "product_name", "brand", "category", "ingredients", "claims",
    "calories", "sugar", "total_fat", "saturated_fat", "protein", "fibre", "salt",
    "source", "source_url", "image_url", "data_quality_score", "imported_at"
]


def _num(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return round(float(value), 3)
    except (TypeError, ValueError):
        return 0.0


def _tags_to_text(value: Any) -> str:
    if isinstance(value, list):
        parts = []
        for item in value:
            text = str(item)
            if ":" in text:
                text = text.split(":", 1)[1]
            text = text.replace("-", " ").strip()
            if text:
                parts.append(text)
        return ", ".join(dict.fromkeys(parts))
    return str(value or "").strip()


def normalize_product(product: dict[str, Any]) -> dict[str, Any]:
    nutriments = product.get("nutriments") or {}
    ingredients = str(product.get("ingredients_text") or "").strip()
    claims = str(product.get("labels") or "").strip()
    if not claims:
        claims = _tags_to_text(product.get("labels_tags"))

    row = {
        "barcode": str(product.get("code") or "").strip(),
        "product_name": str(product.get("product_name") or "").strip(),
        "brand": str(product.get("brands") or "").strip(),
        "category": str(product.get("categories") or "").split(",")[0].strip(),
        "ingredients": ingredients,
        "claims": claims,
        "calories": _num(nutriments.get("energy-kcal_100g")),
        "sugar": _num(nutriments.get("sugars_100g")),
        "total_fat": _num(nutriments.get("fat_100g")),
        "saturated_fat": _num(nutriments.get("saturated-fat_100g")),
        "protein": _num(nutriments.get("proteins_100g")),
        "fibre": _num(nutriments.get("fiber_100g")),
        "salt": _num(nutriments.get("salt_100g")),
        "source": "Open Food Facts",
        "source_url": f"https://world.openfoodfacts.org/product/{str(product.get('code') or '').strip()}",
        "image_url": str(product.get("image_front_url") or "").strip(),
        "data_quality_score": 0,
        "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    checks = [
        bool(row["product_name"]), bool(row["ingredients"]), bool(row["brand"]),
        bool(row["category"]), row["calories"] > 0, row["sugar"] > 0 or row["protein"] > 0,
        bool(row["barcode"]),
    ]
    row["data_quality_score"] = round(sum(checks) / len(checks) * 100, 1)
    return row


def fetch_indian_products(limit: int = 25, timeout: int = 30) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch a small, structured sample of products associated with India.

    This deliberately limits requests/results for a student project and avoids
    bulk scraping. The caller controls the requested page size.
    """
    limit = max(1, min(int(limit), 100))
    params = {
        "countries_tags_en": "india",
        "page": 1,
        "page_size": limit,
        "fields": ",".join(FIELDS),
        "sort_by": "popularity_key",
    }
    headers = {"User-Agent": USER_AGENT}

    # Open Food Facts documents a global HTTP 503 response when its
    # infrastructure/rate limits are exceeded. Retry a few times with a
    # delay instead of immediately sending repeated requests.
    response = None
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(
                SEARCH_URL,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            if response.status_code not in (429, 503):
                response.raise_for_status()
                break

            retry_after = response.headers.get("Retry-After")
            try:
                delay = min(float(retry_after), 30.0) if retry_after else 4.0 * (2 ** attempt)
            except ValueError:
                delay = 4.0 * (2 ** attempt)

            last_error = f"Open Food Facts returned HTTP {response.status_code}"
            if attempt < 2:
                time.sleep(delay)
        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt < 2:
                time.sleep(4.0 * (2 ** attempt))

    if response is None:
        raise RuntimeError(f"Open Food Facts request failed after retries: {last_error}")

    if response.status_code in (429, 503):
        raise RuntimeError(
            "Open Food Facts is temporarily unavailable (HTTP "
            f"{response.status_code}). Please wait a minute and try the import again."
        )

    response.raise_for_status()
    payload = response.json()
    products = payload.get("products", [])
    normalized = [normalize_product(p) for p in products]
    normalized = [p for p in normalized if p["product_name"]]
    return normalized, payload


def save_dataset(rows: list[dict[str, Any]], payload: dict[str, Any]) -> dict[str, str]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_path = RAW_DIR / f"openfoodfacts_india_{stamp}.json"
    csv_path = PROCESSED_DIR / f"foodgrade_india_{stamp}.csv"
    latest_csv = PROCESSED_DIR / "foodgrade_india_latest.csv"

    raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    latest_csv.write_text(csv_path.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
    return {"raw": str(raw_path), "csv": str(csv_path), "latest_csv": str(latest_csv)}
