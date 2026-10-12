"""
FoodGrade AI V5 - OCR + food-label extraction.

Integration with standalone FastAPI PaddleOCR/PaddleX service:
    Default endpoint: http://127.0.0.1:8001/extract/image (configured via FOODGRADE_OCR_URL)

Public API preserved:
    perform_ocr(image_path) -> text
    extract_product_from_text(text, rows=None) -> product dictionary
    perform_ocr_and_extract(image_path) -> (text, product, rows)
    extract_from_ocr_service(image_path) -> dict (direct call to OCR service)

Important:
    OCR-derived values are estimates. Missing or ambiguous nutrition values
    are preserved as None internally, distinguishing missing fields from genuine 0.0.
    For legacy database / UI fields expecting numbers, numeric values are preserved.
"""
import base64
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

# Configurable standalone OCR service URL
DEFAULT_OCR_URL = "http://127.0.0.1:8001"
OCR_SERVICE_URL = os.environ.get("FOODGRADE_OCR_URL", DEFAULT_OCR_URL).rstrip("/")

# Tesseract paths preserved for local fallback/testing
import pytesseract
from pytesseract import Output

TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]
for _path in TESSERACT_PATHS:
    if Path(_path).exists():
        pytesseract.pytesseract.tesseract_cmd = _path
        break

_LAST_OCR_DATA = {"text": "", "rows": []}


class OCRError(Exception):
    """Base exception for OCR service errors."""
    pass


class OCRConnectionError(OCRError):
    """Raised when the standalone OCR service cannot be reached."""
    pass


class OCRResponseError(OCRError):
    """Raised when the standalone OCR service returns an HTTP error or malformed payload."""
    pass


def normalize(text):
    value = str(text or "").lower().replace("’", "'").replace("‘", "'")
    value = value.replace("totalfat", "total fat")
    value = value.replace("saturatedfat", "saturated fat")
    value = value.replace("totalsugars", "total sugars")
    value = value.replace("dietaryfibre", "dietary fibre")
    value = value.replace("dietaryfiber", "dietary fiber")
    value = value.replace("per100g", "per 100 g")
    value = value.replace("per100ml", "per 100 ml")
    value = re.sub(r"[^a-z0-9.%']+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def extract_numbers(text):
    return [float(x) for x in re.findall(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)", str(text or ""))]


def prepare_image(image):
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
    image = image.convert("RGB")
    image = ImageEnhance.Contrast(image).enhance(1.35)
    image = ImageEnhance.Sharpness(image).enhance(1.4)
    return image.filter(ImageFilter.SHARPEN)


def build_rows(data):
    grouped = {}
    for i, raw in enumerate(data.get("text", [])):
        word_text = str(raw).strip()
        if not word_text:
            continue
        try:
            conf = float(data.get("conf", [0] * len(data["text"]))[i])
        except (ValueError, TypeError, IndexError):
            conf = 0
        if conf < 5:
            continue
        key = (
            data["block_num"][i], data["par_num"][i], data["line_num"][i]
        )
        word = {
            "text": word_text,
            "x": int(data["left"][i]), "y": int(data["top"][i]),
            "width": int(data["width"][i]), "height": int(data["height"][i]),
            "confidence": conf,
        }
        grouped.setdefault(key, []).append(word)
    result = []
    for words in grouped.values():
        words.sort(key=lambda w: w["x"])
        result.append({
            "words": words,
            "y": min(w["y"] for w in words),
            "text": " ".join(w["text"] for w in words),
        })
    return sorted(result, key=lambda r: r["y"])


def identify_nutrient(text):
    value = normalize(text)
    if "saturated fat" in value or "saturat ed fat" in value:
        return "saturated_fat"
    if "total fat" in value:
        return "total_fat"
    if "dietary fibre" in value or "dietary fiber" in value:
        return "fibre"
    if "total sugars" in value or "total sugar" in value:
        return "sugar"
    if re.search(r"\bsugars?\b", value):
        return "sugar"
    if "protein" in value:
        return "protein"
    if "energy" in value or "calories" in value or "kcal" in value:
        return "calories"
    if "sodium" in value:
        return "sodium"
    if re.search(r"\bsalt\b", value):
        return "salt"
    if re.search(r"\bfib(re|er)\b", value):
        return "fibre"
    return None


def nutrition_section_start(rows):
    for i, row in enumerate(rows):
        v = normalize(row.get("text", ""))
        if any(s in v for s in (
            "nutritional information", "nutrition information",
            "nutrition facts", "nutritional facts"
        )):
            return i
    return 0


def _numbers_in_row(row):
    found = []
    for word in row.get("words", []):
        raw = str(word.get("text", "")).strip()
        for match in re.finditer(r"(?<!\d)\d+(?:\.\d+)?(?!\d)", raw):
            found.append({
                "value": float(match.group()),
                "x": int(word.get("x", 0)),
                "text": raw,
            })
    found.sort(key=lambda n: n["x"])
    return found


def extract_values_from_row(row, nutrient=None):
    return _numbers_in_row(row)


def _label_and_values(row):
    raw = str(row.get("text", ""))
    nutrient = identify_nutrient(raw)
    nums = _numbers_in_row(row)
    return nutrient, nums


def extract_nutrition(rows):
    result = {
        "calories": None, "sugar": None, "total_fat": None,
        "saturated_fat": None, "protein": None, "fibre": None,
        "salt": None, "sodium": None, "serving_basis": "100g",
        "nutrition_warnings": [],
    }
    start = nutrition_section_start(rows)
    nutrition_rows = rows[start:]

    for row in nutrition_rows:
        nutrient, nums = _label_and_values(row)
        if not nutrient or not nums:
            continue
        selected = nums[0]["value"]
        if nutrient == "calories":
            if 1 <= selected <= 900:
                result["calories"] = selected
        elif nutrient == "sodium":
            if 0 <= selected <= 10000:
                result["sodium"] = selected
        elif nutrient == "salt":
            if 0 <= selected <= 100:
                result["salt"] = selected
        elif nutrient in ("sugar", "total_fat", "saturated_fat", "protein", "fibre"):
            if 0 <= selected <= 100:
                result[nutrient] = selected

    joined = normalize(" ".join(r.get("text", "") for r in nutrition_rows))
    if "per serving" in joined or "per serve" in joined or "serving size" in joined:
        result["serving_basis"] = "per serving"
    elif "per 100g" in joined or "per 100 g" in joined or "per 100ml" in joined:
        result["serving_basis"] = "per 100g"

    for key in ("calories", "sugar", "total_fat", "saturated_fat", "protein", "fibre", "salt"):
        if result[key] is None:
            result["nutrition_warnings"].append(f"{key} not confidently detected")
    if result["sodium"] is not None and result["salt"] is None:
        result["salt"] = round(result["sodium"] * 2.5 / 1000, 3)
        result["nutrition_warnings"].append("Salt estimated from sodium; verify label units")
    return result


def find_heading_index(lines, headings):
    for i, line in enumerate(lines):
        value = normalize(line)
        for heading in headings:
            target = normalize(heading)
            if value == target or value.startswith(target + " "):
                return i
    return None


def extract_ingredients(lines):
    start = find_heading_index(lines, ["ingredients", "ingredient"])
    if start is None:
        return ""
    collected = []
    stop_words = [
        "claims", "storage", "storage instructions", "best before",
        "nutritional information", "nutrition information", "nutrition facts",
        "manufactured by", "marketed by", "net weight", "netweight",
    ]
    for line in lines[start + 1:]:
        value = normalize(line)
        if any(value == normalize(w) or value.startswith(normalize(w) + " ") for w in stop_words):
            break
        collected.append(line)
    return " ".join(collected).strip(" :-.,|")


def extract_claims(lines):
    start = find_heading_index(lines, ["claims", "claim"])
    if start is None:
        return ""
    collected = []
    stop_words = ["storage", "storage instructions", "best before", "ingredients"]
    for line in lines[start + 1:]:
        value = normalize(line)
        if any(value == normalize(w) or value.startswith(normalize(w) + " ") for w in stop_words):
            break
        collected.append(line)
    return " ".join(collected).strip(" :-.,|")


def extract_product_name(lines):
    ignored_phrases = (
        "nutritional information", "nutrition information", "nutrition facts",
        "ingredients", "ingredient", "nutrition", "claims", "storage",
        "per serve", "per serving", "per 100 g", "per 100g",
        "mrp", "net weight", "netweight",
        "best before", "consumer services", "feedback", "lic no", "licence no",
        "manufactured by", "marketed by", "fssai", "village", "punjab",
        "india holdings", "write to", "images are for illustration",
        "cholesterol", "saturated fat", "total fat", "total sugars",
        "protein", "trans fat", "energy", "calories", "sodium", "salt",
        # Additional: catches synthesized nutrition-key lines leaked from OCR service
        "carbohydrates", "dietary fibre", "dietary fiber", "added sugars",
        "serving size", "_g:", "_mg:", "_kcal:",
        # Catches Tesseract fallback / no-detection messages
        "detected", "no nutrition",
    )
    # Pre-compiled patterns to reject before scoring
    _nut_key_re = re.compile(r'^\w+_[a-z]+\s*:\s*[\d.]+', re.IGNORECASE)
    # Matches numeric-table rows: brackets/pipes around digits (single or multiple)
    _table_noise_re = re.compile(r'[\[\]\|]\s*[\d.]+\s*[\[\]\|]|[\[\]\|=]{2,}')
    candidates = []
    for i, line in enumerate(lines):
        value = line.strip(" :|,.-")
        low = normalize(value)
        if not value or len(value) > 90:
            continue
        if any(phrase in low for phrase in ignored_phrases):
            continue
        if re.search(r"\b(mrp|net ?weight|lic(ence)? no|consumer services|qutab enclave)\b", low):
            continue
        # Reject synthesized nutrition keys (e.g. "carbohydrates_g: 53.6")
        if _nut_key_re.search(value):
            continue
        # Reject table separator noise (e.g. "[ 536 [ 107 | - J")
        if _table_noise_re.search(value):
            continue
        words = re.findall(r"[A-Za-z][A-Za-z''-]*", value)
        if not (2 <= len(words) <= 8):
            continue
        if any(w in low for w in ("chips", "crisps", "biscuit", "cookie", "noodles",
                                  "cereal", "chocolate", "juice", "milk", "drink",
                                  "wafer", "potato snack", "lays", "lay's")):
            score = 10
        else:
            score = 0
        score += max(0, 5 - i * 0.15)
        if len(value) <= 55:
            score += 1
        candidates.append((score, i, value))
    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1]))
        return candidates[0][2]
    return "Food Product"


# ---------------------------------------------------------------------------
# Product-name validation
# ---------------------------------------------------------------------------

# Matches synthesized nutrition-key format: word_word: number
_NUTRITION_KEY_RE = re.compile(r'\b\w+_\w+\s*:\s*[\d.]+', re.IGNORECASE)
# Matches numeric-table rows: bracket/pipe around numbers, or 2+ consecutive symbols
_TABLE_NOISE_RE = re.compile(r'[\[\]\|]\s*[\d.]+\s*[\[\]\|]|[\[\]\|=]{2,}')
# Known placeholder / junk fallback strings (lower-cased for comparison)
_JUNK_NAMES: frozenset = frozenset({
    "food product", "ocr product", "leo acts",
    "no nutrition or ingredient text detected on label.",
    "no nutrition or ingredient text detected on label",
})


def is_valid_product_name(name: str) -> bool:
    """Return True if *name* is a plausible food product name; False if junk/malformed.

    Rules (any failure → invalid):
    1. Must be non-empty after stripping whitespace.
    2. Must not be a known placeholder / junk fallback string.
    3. Must not match the nutrition-key format (e.g. ``carbohydrates_g: 53.6``).
    4. Must not consist mainly of table-separator symbols (``[ 536 [ 107``).
    5. Must contain at least 3 alphabetic characters.
    6. Must not have more than 35 % non-alphanumeric, non-space characters (symbol noise).
    """
    if not name or not name.strip():
        return False
    stripped = name.strip()
    if stripped.lower() in _JUNK_NAMES:
        return False
    if _NUTRITION_KEY_RE.search(stripped):
        return False
    if _TABLE_NOISE_RE.search(stripped):
        return False
    alpha_chars = sum(c.isalpha() for c in stripped)
    if alpha_chars < 3:
        return False
    non_alnum = sum(not c.isalnum() and c != ' ' for c in stripped)
    if len(stripped) > 0 and non_alnum / len(stripped) > 0.35:
        return False
    return True



def extract_brand(lines):
    known_brands = [
        "Lay's", "Lays", "Lay’s", "PepsiCo", "Doritos", "Kurkure", "Cheetos",
        "Uncle Chipps", "Too Yumm", "Saffola", "Parle", "Britannia", "Nestle",
        "Kellogg", "Amul", "Haldiram", "ITC", "Patanjali", "Mother Dairy",
        "Aashirvaad", "Tata", "Fortune", "MTR",
    ]
    for line in lines:
        low = line.lower().replace("’", "'")
        for brand in known_brands:
            if brand.lower().replace("’", "'") in low:
                if brand.lower() in ("pepsico",) and "lays" in low:
                    continue
                return "Lay's" if brand.lower() in ("lays", "lay's", "lay’s") else brand
    return ""


def extract_category(product_name, ingredients):
    value = (str(product_name) + " " + str(ingredients)).lower()
    if any(w in value for w in ("oat", "cereal", "granola", "muesli")):
        return "Cereal"
    if any(w in value for w in ("biscuit", "cookie", "cracker")):
        return "Biscuits"
    if any(w in value for w in ("snack", "chips", "crisps", "wafer", "potato")):
        return "Snacks"
    if "protein" in value:
        return "Nutrition"
    return "Food"


# ==============================================================================
# STANDALONE FASTAPI OCR INTEGRATION LAYER
# ==============================================================================

def call_ocr_service(image_bytes: bytes, mime_type: str = "image/png", service_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Call the standalone FastAPI OCR service at POST /extract/image with base64 payload.
    Returns the parsed JSON response dictionary.
    
    Raises:
        OCRConnectionError: When server cannot be reached or times out.
        OCRResponseError: When server returns 4xx/5xx HTTP error or malformed JSON.
    """
    base_url = (service_url or os.environ.get("FOODGRADE_OCR_URL", DEFAULT_OCR_URL)).rstrip("/")
    endpoint = f"{base_url}/extract/image"
    b64_str = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "image": b64_str,
        "mime_type": mime_type,
    }

    try:
        resp = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=25,
        )
    except requests.exceptions.ConnectionError as err:
        raise OCRConnectionError(
            f"Could not connect to OCR service at {endpoint}. "
            "Please ensure the standalone OCR service is running on port 8001."
        ) from err
    except requests.exceptions.Timeout as err:
        raise OCRConnectionError(
            f"OCR service at {endpoint} timed out after 25 seconds."
        ) from err
    except requests.exceptions.RequestException as err:
        raise OCRConnectionError(
            f"Network error communicating with OCR service at {endpoint}: {err}"
        ) from err

    if resp.status_code != 200:
        raise OCRResponseError(
            f"OCR service returned error status {resp.status_code}: {resp.text}"
        )

    try:
        data = resp.json()
    except Exception as err:
        raise OCRResponseError(
            f"OCR service returned invalid non-JSON response: {resp.text[:200]}"
        ) from err

    if not isinstance(data, dict):
        raise OCRResponseError("OCR service returned malformed response (expected JSON object).")

    return data


def normalize_ocr_response(ocr_data: Dict[str, Any], raw_text: str = "") -> Dict[str, Any]:
    """
    Normalize the standalone FastAPI OCR response into FoodGrade AI product schema.
    
    Preserves:
    - Real 0.0 values (e.g. cholesterol_mg: 0.0, trans_fat_g: 0.0)
    - Missing values as None (does not invent zeros)
    - Detailed nutrition dictionary
    - Serving size, ingredients, FSSAI license number, confidence, warnings
    """
    raw_nutrition = ocr_data.get("nutrition")
    if not isinstance(raw_nutrition, dict):
        raw_nutrition = {}

    # Extract all nutrients preserving exact values and None for missing
    def _val(k: str) -> Optional[float]:
        v = raw_nutrition.get(k)
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    energy = _val("energy_kcal")
    protein = _val("protein_g")
    carbs = _val("carbohydrates_g")
    sugar = _val("sugar_g")
    added_sugar = _val("added_sugar_g")
    fat = _val("total_fat_g")
    sat_fat = _val("saturated_fat_g")
    trans_fat = _val("trans_fat_g")
    cholesterol = _val("cholesterol_mg")
    sodium = _val("sodium_mg")
    fibre = _val("dietary_fiber_g")

    # Salt estimation if sodium provided: sodium (mg) * 2.5 / 1000 = salt (g)
    salt = None
    if sodium is not None:
        salt = round(sodium * 2.5 / 1000.0, 3)

    # Detailed nutrition structure
    detailed_nutrition = {
        "energy_kcal": energy,
        "protein_g": protein,
        "carbohydrates_g": carbs,
        "sugar_g": sugar,
        "added_sugar_g": added_sugar,
        "total_fat_g": fat,
        "saturated_fat_g": sat_fat,
        "trans_fat_g": trans_fat,
        "cholesterol_mg": cholesterol,
        "sodium_mg": sodium,
        "dietary_fiber_g": fibre,
        "salt_g": salt,
    }

    # Tracking missing fields / warnings
    nutrition_warnings = []
    missing_nutrients = []
    for k, v in [
        ("Energy (calories)", energy),
        ("Protein", protein),
        ("Total Carbohydrate", carbs),
        ("Total Sugar", sugar),
        ("Total Fat", fat),
        ("Saturated Fat", sat_fat),
        ("Sodium / Salt", sodium),
    ]:
        if v is None:
            nutrition_warnings.append(f"{k} not detected on label")
            missing_nutrients.append(k)

    serving_size = ocr_data.get("serving_size")
    if serving_size is not None:
        try:
            serving_size = float(serving_size)
        except (ValueError, TypeError):
            serving_size = None

    ingredients = str(ocr_data.get("ingredients") or "").strip()
    fssai = str(ocr_data.get("fssai") or "").strip() or None
    confidence = str(ocr_data.get("confidence") or "medium").lower()

    # Product text analysis for name, brand, category, claims
    # Filter out synthesized nutrition-key lines (e.g. "energy_kcal: 536.0") that
    # perform_ocr() injects into synth_text — these must not pollute name extraction.
    _synth_nut_re = re.compile(r'^\w+_[a-z]+\s*:\s*[\d.]+', re.IGNORECASE)
    all_lines = [line.strip() for line in str(raw_text or "").splitlines() if line.strip()]
    name_lines = [l for l in all_lines if not _synth_nut_re.match(l)]
    product_name = extract_product_name(name_lines) if name_lines else "OCR Product"
    brand = extract_brand(all_lines) if all_lines else ""
    category = extract_category(product_name, ingredients)
    claims = extract_claims(all_lines) if all_lines else ""

    product_dict = {
        "product_name": product_name,
        "brand": brand,
        "category": category,
        "ingredients": ingredients,
        "claims": claims,

        # Standard FoodGrade AI legacy fields:
        # Provide numeric values for existing database / UI compatibility
        "calories": energy if energy is not None else 0.0,
        "sugar": sugar if sugar is not None else 0.0,
        "total_fat": fat if fat is not None else 0.0,
        "saturated_fat": sat_fat if sat_fat is not None else 0.0,
        "protein": protein if protein is not None else 0.0,
        "fibre": fibre if fibre is not None else 0.0,
        "salt": salt if salt is not None else 0.0,
        "sodium": sodium,

        # Exact uncoerced nutrient values
        "raw_nutrition": detailed_nutrition,
        "missing_nutrients": missing_nutrients,

        # Label packaging metadata
        "serving_size": serving_size,
        "serving_basis": "100g",
        "fssai": fssai,
        "ocr_confidence": confidence,
        "nutrition_warnings": nutrition_warnings,

        "source": "OCR",
        "source_url": "",
        "image_url": "",
    }
    return product_dict


def perform_ocr(image_path: str) -> str:
    """
    Perform OCR on the image. First tries the high-accuracy standalone FastAPI service
    if available; falls back to local Tesseract OCR if configured.
    """
    global _LAST_OCR_DATA
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    # Determine MIME type
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        ext = path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(ext, "image/jpeg")

    with open(path, "rb") as f:
        image_bytes = f.read()

    # Attempt standalone OCR service call
    try:
        service_data = call_ocr_service(image_bytes, mime_type=mime_type)
        # Synthesize a formatted OCR text representation from the service output
        nut_lines = []
        nut = service_data.get("nutrition") or {}
        for k, v in nut.items():
            nut_lines.append(f"{k}: {v}")
        
        text_parts = []
        if nut_lines:
            text_parts.append("NUTRITIONAL INFORMATION (Per 100g):\n" + "\n".join(nut_lines))
        if service_data.get("serving_size"):
            text_parts.append(f"Serving Size: {service_data['serving_size']} g")
        if service_data.get("ingredients"):
            text_parts.append(f"INGREDIENTS: {service_data['ingredients']}")
        if service_data.get("fssai"):
            text_parts.append(f"FSSAI Lic. No.: {service_data['fssai']}")

        synth_text = "\n\n".join(text_parts).strip()
        if not synth_text:
            # If the service detected no structured fields, check if local Tesseract finds text (e.g. watermark/general text)
            with Image.open(path) as im:
                image = prepare_image(im)
            best_rows = []
            best_text_len = -1
            for psm in (6, 11, 4):
                try:
                    data = pytesseract.image_to_data(
                        image, config=f"--oem 3 --psm {psm}", output_type=Output.DICT
                    )
                    rows = build_rows(data)
                    text_len = sum(len(r["text"]) for r in rows)
                    if text_len > best_text_len:
                        best_rows, best_text_len = rows, text_len
                except Exception:
                    break
            tess_text = "\n".join(r["text"] for r in best_rows if r["text"].strip())
            synth_text = tess_text if tess_text else "No nutrition or ingredient text detected on label."

        _LAST_OCR_DATA = {
            "text": synth_text,
            "rows": [],
            "service_data": service_data,
        }
        return synth_text

    except (OCRConnectionError, OCRResponseError) as err:
        logger.warning(f"[v5_ocr] Standalone OCR service call failed: {err}. Falling back to Tesseract.")
        # Local Tesseract fallback
        with Image.open(path) as im:
            image = prepare_image(im)
        best_rows = []
        best_text_len = -1
        for psm in (6, 11, 4):
            try:
                data = pytesseract.image_to_data(
                    image, config=f"--oem 3 --psm {psm}", output_type=Output.DICT
                )
                rows = build_rows(data)
                text_len = sum(len(r["text"]) for r in rows)
                if text_len > best_text_len:
                    best_rows, best_text_len = rows, text_len
            except Exception as t_err:
                logger.error(f"[v5_ocr] Tesseract failed: {t_err}")
                break

        text = "\n".join(r["text"] for r in best_rows if r["text"].strip())
        _LAST_OCR_DATA = {"text": text, "rows": best_rows, "service_data": None}
        if not text:
            # Re-raise the connection error so caller knows why extraction failed
            raise err
        return text


def extract_product_from_text(text: str, rows: Optional[list] = None) -> Dict[str, Any]:
    """
    Extract product metadata and nutrition dictionary from OCR text or last OCR data.
    """
    global _LAST_OCR_DATA
    service_data = _LAST_OCR_DATA.get("service_data")
    if service_data:
        # Use high-accuracy structured service data directly
        return normalize_ocr_response(service_data, raw_text=text)

    # Fallback to row-based / regex extraction
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if rows is None:
        rows = _LAST_OCR_DATA.get("rows", [])
    nutrition = extract_nutrition(rows) if rows else {
        "calories": None, "sugar": None, "total_fat": None,
        "saturated_fat": None, "protein": None, "fibre": None,
        "salt": None, "sodium": None, "serving_basis": "100g",
        "nutrition_warnings": ["No positional OCR data available"],
    }
    product_name = extract_product_name(lines)
    ingredients = extract_ingredients(lines)
    claims = extract_claims(lines)

    output = {}
    for key in ("calories", "sugar", "total_fat", "saturated_fat", "protein", "fibre", "salt"):
        output[key] = nutrition.get(key) if nutrition.get(key) is not None else 0.0

    output.update({
        "product_name": product_name,
        "brand": extract_brand(lines),
        "category": extract_category(product_name, ingredients),
        "ingredients": ingredients,
        "claims": claims,
        "sodium": nutrition.get("sodium"),
        "serving_basis": nutrition.get("serving_basis", "100g"),
        "nutrition_warnings": nutrition.get("nutrition_warnings", []),
        "source": "OCR",
        "source_url": "",
        "image_url": "",
    })
    return output


def perform_ocr_and_extract(image_path: str) -> Tuple[str, Dict[str, Any], list]:
    """
    Direct OCR processing pipeline returning (text, product_dict, rows).
    """
    text = perform_ocr(image_path)
    rows = _LAST_OCR_DATA.get("rows", [])
    product = extract_product_from_text(text, rows=rows)
    return text, product, rows
