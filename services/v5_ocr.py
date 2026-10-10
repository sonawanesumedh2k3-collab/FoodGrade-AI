"""
FoodGrade AI V5
OCR + Food Label Extraction

Stable architecture:

Image
  ↓
Tesseract OCR with coordinates
  ↓
Visual rows
  ↓
Nutrition table detection
  ↓
Nutrient + numeric column detection
  ↓
Structured product information
"""

import re
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
from pytesseract import Output


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

for path in TESSERACT_PATHS:

    if Path(path).exists():

        pytesseract.pytesseract.tesseract_cmd = path

        break


# ============================================================
# GLOBAL OCR DATA
# ============================================================

# The Flask application already calls:
#
#     text = perform_ocr(...)
#
# followed by:
#
#     extract_product_from_text(text)
#
# Therefore we keep the structured OCR information here so
# the second function can use coordinates as well as text.

_LAST_OCR_DATA = {
    "text": "",
    "rows": [],
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):

    value = str(text).lower()

    replacements = {
        "totalfat": "total fat",
        "saturatedfat": "saturated fat",
        "totalsugars": "total sugars",
        "dietaryfibre": "dietary fibre",
        "dietaryfiber": "dietary fiber",
    }

    for old, new in replacements.items():

        value = value.replace(
            old,
            new
        )

    value = re.sub(
        r"[^a-z0-9.%]+",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def extract_numbers(text):

    matches = re.findall(
        r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)",
        str(text)
    )

    result = []

    for value in matches:

        try:
            result.append(
                float(value)
            )

        except ValueError:
            pass

    return result


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(image):
    # Auto-orient based on camera EXIF data (rotation handling)
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    image = image.convert("RGB")
    image = ImageEnhance.Contrast(image).enhance(1.30)
    image = ImageEnhance.Sharpness(image).enhance(1.35)
    image = image.filter(ImageFilter.SHARPEN)

    return image


# ============================================================
# GROUP TESSERACT WORDS USING NATIVE LINE INFORMATION
# ============================================================

def build_rows(data):

    rows = {}

    total = len(
        data["text"]
    )

    for i in range(total):

        text = str(
            data["text"][i]
        ).strip()

        if not text:
            continue

        try:

            confidence = float(
                data["conf"][i]
            )

        except (
            ValueError,
            TypeError
        ):

            confidence = 0

        if confidence < 10:
            continue

        key = (
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i],
        )

        word = {
            "text": text,

            "x": int(
                data["left"][i]
            ),

            "y": int(
                data["top"][i]
            ),

            "width": int(
                data["width"][i]
            ),

            "height": int(
                data["height"][i]
            ),

            "confidence": confidence,
        }

        rows.setdefault(
            key,
            []
        ).append(
            word
        )

    result = []

    for words in rows.values():

        words.sort(
            key=lambda item:
            item["x"]
        )

        if not words:
            continue

        result.append(
            {
                "words": words,

                "y": min(
                    word["y"]
                    for word in words
                ),

                "text": " ".join(
                    word["text"]
                    for word in words
                ),
            }
        )

    result.sort(
        key=lambda row:
        row["y"]
    )

    return result


# ============================================================
# OCR
# ============================================================

def perform_ocr(image_path):

    global _LAST_OCR_DATA

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    image = Image.open(
        image_path
    )

    image = prepare_image(
        image
    )

    # --------------------------------------------------------
    # Primary OCR
    # --------------------------------------------------------

    data = pytesseract.image_to_data(
        image,
        config="--psm 6",
        output_type=Output.DICT
    )

    rows = build_rows(
        data
    )

    # --------------------------------------------------------
    # Create readable OCR text.
    # --------------------------------------------------------

    text_lines = []

    for row in rows:

        line = row["text"]

        line = re.sub(
            r"\s+",
            " ",
            line
        ).strip()

        if line:

            text_lines.append(
                line
            )

    text = "\n".join(
        text_lines
    )

    _LAST_OCR_DATA = {
        "text": text,
        "rows": rows,
    }

    return text


# ============================================================
# NUTRIENT IDENTIFICATION
# ============================================================

NUTRIENT_PATTERNS = {

    "calories": [
        "energy",
    ],

    "protein": [
        "protein",
    ],

    "total_fat": [
        "total fat",
    ],

    "saturated_fat": [
        "saturated fat",
    ],

    "sugar": [
        "total sugars",
        "total sugar",
        "sugars",
        "sugar",
    ],

    "fibre": [
        "dietary fibre",
        "dietary fiber",
        "fibre",
        "fiber",
    ],

    "salt": [
        "salt",
    ],
}


def identify_nutrient(
    text
):

    value = normalize(
        text
    )

    # Check specific terms first.

    if "saturated fat" in value:

        return "saturated_fat"

    if "total fat" in value:

        return "total_fat"

    if (
        "dietary fibre" in value
        or "dietary fiber" in value
    ):

        return "fibre"

    if (
        "total sugars" in value
        or "total sugar" in value
    ):

        return "sugar"

    if "protein" in value:

        return "protein"

    if "energy" in value:

        return "calories"

    if "salt" in value:

        return "salt"

    if (
        "fibre" in value
        or "fiber" in value
    ):

        return "fibre"

    if "sugar" in value:

        return "sugar"

    return None


# ============================================================
# FIND NUTRITION SECTION
# ============================================================

def nutrition_section_start(
    rows
):

    for index, row in enumerate(
        rows
    ):

        value = normalize(
            row["text"]
        )

        if (
            "nutritional information"
            in value
            or "nutrition information"
            in value
            or "nutrition facts"
            in value
        ):

            return index

    return 0


# ============================================================
# EXTRACT NUTRITION ROW
# ============================================================

def extract_values_from_row(
    row,
    nutrient
):
    """
    Extract numeric values from a single OCR row.

    Example:

        Protein (g) 13.2 5.3

    becomes:

        [13.2, 5.3]

    We then select 13.2 because it is the first
    numeric value after the nutrient label.
    """

    words = row["words"]

    numeric_words = []

    for word in words:

        values = extract_numbers(
            word["text"]
        )

        if not values:
            continue

        for value in values:

            numeric_words.append(
                {
                    "value": value,
                    "x": word["x"],
                    "text": word["text"],
                }
            )

    if not numeric_words:

        return []

    numeric_words.sort(
        key=lambda item:
        item["x"]
    )

    return numeric_words


# ============================================================
# NUTRITION EXTRACTION
# ============================================================

def extract_nutrition(
    rows
):

    result = {

        "calories": 0.0,

        "sugar": 0.0,

        "total_fat": 0.0,

        "saturated_fat": 0.0,

        "protein": 0.0,

        "fibre": 0.0,

        "salt": 0.0,
    }

    start = nutrition_section_start(
        rows
    )

    # --------------------------------------------------------
    # Search only inside the nutrition section.
    # --------------------------------------------------------

    nutrition_rows = rows[
        start:
    ]

    for row in nutrition_rows:

        row_text = row["text"]

        nutrient = identify_nutrient(
            row_text
        )

        if nutrient is None:

            continue

        values = extract_values_from_row(
            row,
            nutrient
        )

        if not values:

            continue

        # ----------------------------------------------------
        # First numeric value = per 100 g.
        # ----------------------------------------------------

        selected = values[0][
            "value"
        ]

        # ----------------------------------------------------
        # Sanity checking.
        # ----------------------------------------------------

        if nutrient == "calories":
            if 0 <= selected <= 900:
                result[nutrient] = selected
        else:
            if 0 <= selected <= 100:
                result[nutrient] = selected

    # Detect declared serving basis
    serving_basis = "100g"
    for r in nutrition_rows:
        t = str(r.get("text", "")).lower()
        if "per serving" in t or "serving size" in t:
            serving_basis = "per serving"
            break
        elif "per pack" in t:
            serving_basis = "per pack"
            break
        elif "per 100g" in t or "per 100 g" in t or "per 100ml" in t:
            serving_basis = "per 100g"
            break

    result["serving_basis"] = serving_basis
    return result


# ============================================================
# SECTION FINDER
# ============================================================

def find_heading_index(
    lines,
    headings
):

    for index, line in enumerate(
        lines
    ):

        value = normalize(
            line
        )

        for heading in headings:

            target = normalize(
                heading
            )

            if (
                value == target
                or value.startswith(
                    target + " "
                )
            ):

                return index

    return None


# ============================================================
# INGREDIENTS
# ============================================================

def extract_ingredients(
    lines
):

    start = find_heading_index(
        lines,
        [
            "ingredients",
            "ingredient",
        ]
    )

    if start is None:

        return ""

    collected = []

    stop_words = [
        "claims",
        "storage",
        "storage instructions",
        "best before",
        "nutritional information",
        "nutrition information",
    ]

    for line in lines[
        start + 1:
    ]:

        value = normalize(
            line
        )

        if any(
            value == normalize(word)
            or value.startswith(
                normalize(word) + " "
            )
            for word in stop_words
        ):

            break

        collected.append(
            line
        )

    return " ".join(
        collected
    ).strip(
        " :-.,"
    )


# ============================================================
# CLAIMS
# ============================================================

def extract_claims(
    lines
):

    start = find_heading_index(
        lines,
        [
            "claims",
            "claim",
        ]
    )

    if start is None:

        return ""

    collected = []

    stop_words = [
        "storage",
        "storage instructions",
        "best before",
    ]

    for line in lines[
        start + 1:
    ]:

        value = normalize(
            line
        )

        if any(
            value == normalize(word)
            or value.startswith(
                normalize(word) + " "
            )
            for word in stop_words
        ):

            break

        collected.append(
            line
        )

    return " ".join(
        collected
    ).strip(
        " :-.,"
    )


# ============================================================
# PRODUCT NAME
# ============================================================

def extract_product_name(
    lines
):

    food_words = [
        "oat",
        "oats",
        "cereal",
        "biscuit",
        "cookie",
        "cracker",
        "snack",
        "chips",
        "noodles",
        "pasta",
        "bread",
        "milk",
        "juice",
        "drink",
        "protein",
        "granola",
        "muesli",
        "flour",
        "rice",
        "chickpea",
        "bar",
        "chocolate",
    ]

    ignored = [
        "nutritional information",
        "nutrition information",
        "nutrition facts",
        "ingredients",
        "claims",
        "storage",
        "per serve",
        "per serving",
        "per 100 g",
        "per 100g",
    ]

    candidates = []

    for line in lines:

        value = line.strip()

        lower = value.lower()

        if not value:
            continue

        if any(
            item in lower
            for item in ignored
        ):

            continue

        # Do not use nutrition rows.
        if identify_nutrient(
            value
        ):

            continue

        if any(
            word in lower
            for word in food_words
        ):

            score = 0

            if (
                "oat" in lower
                or "oats" in lower
            ):

                score += 5

            if "cereal" in lower:

                score += 4

            if 2 <= len(
                value.split()
            ) <= 8:

                score += 2

            if len(value) <= 70:

                score += 1

            candidates.append(
                (
                    score,
                    value
                )
            )

    if candidates:

        candidates.sort(
            key=lambda item:
            item[0],
            reverse=True
        )

        return candidates[0][1]

    return "Food Product"


# ============================================================
# BRAND
# ============================================================

def extract_brand(
    lines
):

    known_brands = [
        "Saffola",
        "Parle",
        "Britannia",
        "Nestle",
        "Kellogg",
        "Amul",
        "Haldiram",
        "PepsiCo",
        "ITC",
        "Patanjali",
        "Mother Dairy",
        "Aashirvaad",
        "Tata",
        "Fortune",
        "MTR",
    ]

    for line in lines:

        lower = line.lower()

        for brand in known_brands:

            if brand.lower() in lower:

                return brand

    # Never guess a brand.
    return ""


# ============================================================
# CATEGORY
# ============================================================

def extract_category(
    product_name,
    ingredients
):

    value = (
        str(product_name)
        + " "
        + str(ingredients)
    ).lower()

    if any(
        word in value
        for word in [
            "oat",
            "oats",
            "cereal",
            "granola",
            "muesli",
        ]
    ):

        return "Cereal"

    if any(
        word in value
        for word in [
            "biscuit",
            "cookie",
            "cracker",
        ]
    ):

        return "Biscuits"

    if any(
        word in value
        for word in [
            "snack",
            "chips",
        ]
    ):

        return "Snacks"

    if "protein" in value:

        return "Nutrition"

    return "Food"


# ============================================================
# MAIN EXTRACTION
# ============================================================

def extract_product_from_text(
    text,
    rows=None
):

    global _LAST_OCR_DATA

    lines = [
        line.strip()
        for line in str(
            text
        ).splitlines()
        if line.strip()
    ]

    # --------------------------------------------------------
    # Use positional OCR rows if provided or available.
    # --------------------------------------------------------

    if rows is None:
        rows = _LAST_OCR_DATA.get(
            "rows",
            []
        )

    if rows:

        nutrition = extract_nutrition(
            rows
        )

    else:

        nutrition = {
            "calories": 0.0,
            "sugar": 0.0,
            "total_fat": 0.0,
            "saturated_fat": 0.0,
            "protein": 0.0,
            "fibre": 0.0,
            "salt": 0.0,
            "serving_basis": "100g",
        }

    product_name = extract_product_name(
        lines
    )

    ingredients = extract_ingredients(
        lines
    )

    claims = extract_claims(
        lines
    )

    return {

        "product_name":
            product_name,

        "brand":
            extract_brand(
                lines
            ),

        "category":
            extract_category(
                product_name,
                ingredients
            ),

        "ingredients":
            ingredients,

        "claims":
            claims,

        "calories":
            nutrition["calories"],

        "sugar":
            nutrition["sugar"],

        "total_fat":
            nutrition["total_fat"],

        "saturated_fat":
            nutrition["saturated_fat"],

        "protein":
            nutrition["protein"],

        "fibre":
            nutrition["fibre"],

        "salt":
            nutrition["salt"],

        "serving_basis":
            nutrition.get("serving_basis", "100g"),

        "source":
            "OCR",

        "source_url":
            "",

        "image_url":
            "",
    }


def perform_ocr_and_extract(image_path):
    """
    Direct OCR processing avoiding reliance on mutable module state.
    Returns (text, product_dict, rows).
    """
    text = perform_ocr(image_path)
    rows = _LAST_OCR_DATA.get("rows", [])
    product = extract_product_from_text(text, rows=rows)
    return text, product, rows