import re
from datetime import datetime

from services.web_parser import (
    soup_from_html,
    extract_jsonld,
    find_product_jsonld,
    page_title,
    visible_text,
    clean_text,
)


# ============================================================
# CLAIMS
# ============================================================

CLAIM_TERMS = [
    "high protein",
    "high fibre",
    "high fiber",
    "source of protein",
    "source of fibre",
    "source of fiber",
    "low sugar",
    "no added sugar",
    "sugar free",
    "sugar-free",
    "low fat",
    "fat free",
    "fat-free",
    "low calorie",
    "reduced sugar",
    "reduced fat",
    "whole grain",
    "whole wheat",
    "organic",
    "natural",
]


# ============================================================
# NUTRITION PATTERNS
# ============================================================

NUTRIENT_PATTERNS = {

    "calories": [
        r"(?:energy|calories|calorie)"
        r"\s*(?:per\s*(?:100\s*g|serving))?"
        r"\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(?:kcal|cal)",

        r"(\d+(?:\.\d+)?)\s*kcal",
    ],

    "sugar": [
        r"(?:total\s+)?sugars?"
        r"\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g",

        r"sugars?"
        r".{0,20}?(\d+(?:\.\d+)?)\s*g",
    ],

    "total_fat": [
        r"(?:total\s+)?fat"
        r"\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g",

        r"total\s+fat"
        r".{0,20}?(\d+(?:\.\d+)?)\s*g",
    ],

    "saturated_fat": [
        r"satur(?:ated|ates)\s+fat"
        r"\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g",

        r"satur(?:ated|ates)\s+fat"
        r".{0,20}?(\d+(?:\.\d+)?)\s*g",
    ],

    "protein": [
        r"protein"
        r"\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g",

        r"protein"
        r".{0,20}?(\d+(?:\.\d+)?)\s*g",
    ],

    "fibre": [
        r"(?:dietary\s+)?fib(?:re|er)"
        r"\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g",

        r"(?:dietary\s+)?fib(?:re|er)"
        r".{0,20}?(\d+(?:\.\d+)?)\s*g",
    ],

    "salt": [
        r"salt"
        r"\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g",

        r"salt"
        r".{0,20}?(\d+(?:\.\d+)?)\s*g",
    ],
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def _first_number(text, patterns):
    """
    Return the first numeric value matching any pattern.
    """
    text = text or ""

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.I | re.S
        )

        if match:
            try:
                return float(match.group(1))
            except (ValueError, TypeError):
                pass

    return 0.0


def _clean_value(value):
    """
    Convert arbitrary values to clean text.
    """
    if value is None:
        return ""

    if isinstance(value, list):
        value = ", ".join(
            str(item)
            for item in value
            if item is not None
        )

    if isinstance(value, dict):
        value = value.get("name", "")

    return clean_text(value)


# ============================================================
# JSON-LD HELPERS
# ============================================================

def _find_nested_product(records):
    """
    Search JSON-LD records recursively for a Product object.

    Handles:
    - Product
    - @graph
    - nested dictionaries
    - nested lists
    """

    def search(obj):

        if isinstance(obj, dict):

            obj_type = obj.get("@type", "")

            if isinstance(obj_type, list):
                types = [
                    str(x).lower()
                    for x in obj_type
                ]
            else:
                types = [
                    str(obj_type).lower()
                ]

            if "product" in types:
                return obj

            for value in obj.values():

                result = search(value)

                if result:
                    return result

        elif isinstance(obj, list):

            for item in obj:

                result = search(item)

                if result:
                    return result

        return None

    return search(records)


# ============================================================
# BRAND EXTRACTION
# ============================================================

def _extract_brand(soup, product, title):
    """
    Try multiple sources to identify the brand.
    """

    # 1. JSON-LD
    brand = product.get("brand", "")

    if isinstance(brand, dict):
        brand = brand.get("name", "")

    brand = _clean_value(brand)

    if brand:
        return brand

    # 2. Meta tags
    meta_selectors = [
        ("meta", {"property": "product:brand"}),
        ("meta", {"name": "brand"}),
        ("meta", {"property": "og:brand"}),
    ]

    for tag_name, attrs in meta_selectors:

        tag = soup.find(
            tag_name,
            attrs=attrs
        )

        if tag and tag.get("content"):
            value = clean_text(
                tag.get("content")
            )

            if value:
                return value

    # 3. HTML elements whose class/id contains brand
    elements = soup.find_all(
        attrs={
            "class": re.compile(
                r"brand",
                re.I
            )
        }
    )

    for element in elements[:10]:

        value = clean_text(
            element.get_text(" ")
        )

        if value and len(value) < 100:
            return value

    # 4. Title fallback
    #
    # Example:
    # KURKURE | Pepsnacks
    #
    # We avoid blindly assuming the second part is always
    # the brand. This is only a weak fallback.
    if title and "|" in title:

        parts = [
            clean_text(x)
            for x in title.split("|")
            if clean_text(x)
        ]

        if len(parts) >= 2:

            possible = parts[-1]

            if (
                possible
                and len(possible) <= 80
            ):
                return possible

    return ""


# ============================================================
# CATEGORY EXTRACTION
# ============================================================

def _extract_category(soup, product):

    category = _clean_value(
        product.get("category", "")
    )

    if category:
        return category

    # Meta category
    meta_candidates = [
        soup.find(
            "meta",
            attrs={"name": "category"}
        ),
        soup.find(
            "meta",
            attrs={"property": "product:category"}
        ),
    ]

    for tag in meta_candidates:

        if tag and tag.get("content"):

            value = clean_text(
                tag.get("content")
            )

            if value:
                return value

    # Breadcrumb fallback
    breadcrumb = soup.find(
        attrs={
            "class": re.compile(
                r"breadcrumb",
                re.I
            )
        }
    )

    if breadcrumb:

        links = breadcrumb.find_all(
            ["a", "span", "li"]
        )

        values = [
            clean_text(x.get_text(" "))
            for x in links
        ]

        values = [
            x for x in values
            if x
        ]

        if values:
            return values[-1]

    return ""


# ============================================================
# PRODUCT NAME EXTRACTION
# ============================================================

def _extract_product_name(soup, product, title):

    # 1. JSON-LD
    name = _clean_value(
        product.get("name", "")
    )

    if name:
        return name

    # 2. OpenGraph title
    og_title = soup.find(
        "meta",
        attrs={
            "property": "og:title"
        }
    )

    if og_title and og_title.get("content"):

        value = clean_text(
            og_title.get("content")
        )

        if value:
            return value

    # 3. Product heading
    product_heading = soup.find(
        attrs={
            "class": re.compile(
                r"(product[-_ ]?)?title|product[-_ ]?name",
                re.I
            )
        }
    )

    if product_heading:

        value = clean_text(
            product_heading.get_text(" ")
        )

        if value and len(value) < 300:
            return value

    # 4. H1
    h1 = soup.find("h1")

    if h1:

        value = clean_text(
            h1.get_text(" ")
        )

        if value:
            return value

    # 5. Page title
    if title:
        return clean_text(title)

    return "Web Product"


# ============================================================
# INGREDIENT EXTRACTION
# ============================================================

def _extract_ingredients_from_elements(soup):

    # Look for elements where class/id contains
    # ingredient / ingredients.
    elements = soup.find_all(
        attrs={
            "class": re.compile(
                r"ingredient",
                re.I
            )
        }
    )

    elements += soup.find_all(
        attrs={
            "id": re.compile(
                r"ingredient",
                re.I
            )
        }
    )

    for element in elements[:30]:

        value = clean_text(
            element.get_text(" ")
        )

        if not value:
            continue

        # Remove heading-only values.
        if value.lower() in {
            "ingredient",
            "ingredients"
        }:
            continue

        # Avoid returning tiny unrelated strings.
        if len(value) >= 15:

            # If text begins with "Ingredients:",
            # remove the label.
            value = re.sub(
                r"^ingredients?\s*[:\-]\s*",
                "",
                value,
                flags=re.I
            )

            return value[:3000]

    return ""


def _ingredients_from_text(text):

    if not text:
        return ""

    patterns = [

        r"ingredients?\s*[:\-]\s*(.+?)"
        r"(?=\s+(?:nutrition|nutritional information|"
        r"allergen|allergens|contains|storage|"
        r"directions|preparation|manufacturer|"
        r"product information)\b|$)",

        r"ingredients?\s+(.+?)"
        r"(?=\s+(?:nutrition|nutritional information|"
        r"allergen|allergens|contains|storage|"
        r"directions|preparation|manufacturer)\b|$)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.I | re.S
        )

        if match:

            value = clean_text(
                match.group(1)
            )

            if len(value) >= 10:
                return value[:3000]

    return ""


def _extract_ingredients(soup, product, text):

    # 1. JSON-LD
    ingredients = _clean_value(
        product.get("ingredients", "")
        or product.get("ingredient", "")
    )

    if ingredients:
        return ingredients[:3000]

    # 2. HTML elements
    ingredients = _extract_ingredients_from_elements(
        soup
    )

    if ingredients:
        return ingredients

    # 3. Visible text
    ingredients = _ingredients_from_text(
        text
    )

    return ingredients


# ============================================================
# CLAIM EXTRACTION
# ============================================================

def _claims_from_text(text):

    lower = (text or "").lower()

    found = []

    for term in CLAIM_TERMS:

        if term in lower:

            # Keep one normalized spelling.
            normalized = term.replace(
                "sugar-free",
                "sugar free"
            ).replace(
                "fat-free",
                "fat free"
            )

            found.append(
                normalized.title()
            )

    # Remove duplicates while preserving order.
    return ", ".join(
        dict.fromkeys(found)
    )


def _extract_claims(soup, product, text):

    description = _clean_value(
        product.get("description", "")
    )

    # Meta description can contain product claims.
    meta_description = soup.find(
        "meta",
        attrs={
            "name": "description"
        }
    )

    meta_text = ""

    if meta_description:
        meta_text = clean_text(
            meta_description.get("content", "")
        )

    combined = " ".join([
        description,
        meta_text,
        text or ""
    ])

    return _claims_from_text(
        combined
    )


# ============================================================
# NUTRITION FROM HTML TABLES
# ============================================================

def _extract_nutrition_from_tables(soup):

    nutrition = {}

    # Search all tables.
    for table in soup.find_all("table"):

        rows = table.find_all("tr")

        for row in rows:

            cells = row.find_all(
                ["th", "td"]
            )

            if len(cells) < 2:
                continue

            label = clean_text(
                cells[0].get_text(" ")
            ).lower()

            value = clean_text(
                cells[1].get_text(" ")
            )

            number_match = re.search(
                r"(\d+(?:\.\d+)?)",
                value
            )

            if not number_match:
                continue

            number = float(
                number_match.group(1)
            )

            if "energy" in label or "calorie" in label:
                nutrition["calories"] = number

            elif (
                "sugar" in label
                and "added" not in label
            ):
                nutrition["sugar"] = number

            elif (
                "saturated" in label
                and "fat" in label
            ):
                nutrition["saturated_fat"] = number

            elif (
                "total fat" in label
                or label == "fat"
                or label.endswith(" fat")
            ):
                nutrition["total_fat"] = number

            elif "protein" in label:
                nutrition["protein"] = number

            elif (
                "fibre" in label
                or "fiber" in label
            ):
                nutrition["fibre"] = number

            elif "salt" in label:
                nutrition["salt"] = number

    return nutrition


# ============================================================
# NUTRITION FROM JSON-LD
# ============================================================

def _extract_jsonld_nutrition(product):

    nutrition = {}

    # Correct Schema.org field:
    # nutritionInformation
    info = product.get(
        "nutritionInformation"
    )

    # Support the old/custom field too.
    if not info:
        info = product.get(
            "nutrition"
        )

    if not isinstance(info, dict):
        return nutrition

    mappings = {

        "calories": [
            "calories",
            "energy"
        ],

        "sugar": [
            "sugarContent",
            "sugars"
        ],

        "total_fat": [
            "fatContent",
            "fat"
        ],

        "saturated_fat": [
            "saturatedFatContent",
            "saturatedFat"
        ],

        "protein": [
            "proteinContent",
            "protein"
        ],

        "fibre": [
            "fiberContent",
            "fibreContent",
            "fibre"
        ],

        "salt": [
            "saltContent",
            "salt"
        ],
    }

    for field, keys in mappings.items():

        for key in keys:

            if key not in info:
                continue

            value = info[key]

            if isinstance(value, dict):
                value = (
                    value.get("value")
                    or value.get("amount")
                )

            match = re.search(
                r"\d+(?:\.\d+)?",
                str(value)
            )

            if match:

                nutrition[field] = float(
                    match.group(0)
                )

                break

    return nutrition


# ============================================================
# IMAGE EXTRACTION
# ============================================================

def _extract_image(soup, product):

    image = product.get(
        "image",
        ""
    )

    if isinstance(image, list):

        image = (
            image[0]
            if image
            else ""
        )

    image = _clean_value(
        image
    )

    if image:
        return image

    og_image = soup.find(
        "meta",
        attrs={
            "property": "og:image"
        }
    )

    if og_image and og_image.get("content"):

        return clean_text(
            og_image.get("content")
        )

    return ""


# ============================================================
# MAIN EXTRACTION FUNCTION
# ============================================================

def extract_product_from_html(
    url,
    html,
    headers=None
):
    """
    Extract product information from a webpage.

    Extraction priority:

    1. JSON-LD / Schema.org Product
    2. Meta tags
    3. Product-specific HTML elements
    4. Nutrition tables
    5. Visible page text
    6. Page title

    Output is normalized to the format
    expected by the existing FoodGrade/V3 pipeline.
    """

    soup = soup_from_html(
        html
    )

    # Extract JSON-LD BEFORE visible_text()
    # because visible_text() removes script tags.
    records = extract_jsonld(
        soup
    )

    product = (
        _find_nested_product(records)
        or find_product_jsonld(records)
        or {}
    )

    title = page_title(
        soup
    )

    # Visible page text.
    text = visible_text(
        soup
    )

    # --------------------------------------------------------
    # BASIC PRODUCT INFORMATION
    # --------------------------------------------------------

    product_name = _extract_product_name(
        soup,
        product,
        title
    )

    brand = _extract_brand(
        soup,
        product,
        title
    )

    category = _extract_category(
        soup,
        product
    )

    ingredients = _extract_ingredients(
        soup,
        product,
        text
    )

    claims = _extract_claims(
        soup,
        product,
        text
    )

    # --------------------------------------------------------
    # NUTRITION
    # --------------------------------------------------------

    jsonld_nutrition = (
        _extract_jsonld_nutrition(
            product
        )
    )

    table_nutrition = (
        _extract_nutrition_from_tables(
            soup
        )
    )

    nutrition = {}

    # JSON-LD gets priority.
    nutrition.update(
        table_nutrition
    )

    nutrition.update(
        jsonld_nutrition
    )

    # Text fallback for anything still missing.
    for field, patterns in NUTRIENT_PATTERNS.items():

        if field not in nutrition:

            value = _first_number(
                text,
                patterns
            )

            if value:
                nutrition[field] = value

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    extraction_sources = []

    if product:
        extraction_sources.append(
            "JSON-LD"
        )

    if ingredients:
        extraction_sources.append(
            "Ingredients HTML/text"
        )

    if nutrition:
        extraction_sources.append(
            "Nutrition HTML/text"
        )

    if claims:
        extraction_sources.append(
            "Claims/page text"
        )

    if not extraction_sources:
        extraction_sources.append(
            "Visible-page fallback"
        )

    result = {

        "product_name": product_name,

        "brand": brand,

        "category": category,

        "ingredients": ingredients,

        "claims": claims,

        "calories": nutrition.get(
            "calories",
            0.0
        ),

        "sugar": nutrition.get(
            "sugar",
            0.0
        ),

        "total_fat": nutrition.get(
            "total_fat",
            0.0
        ),

        "saturated_fat": nutrition.get(
            "saturated_fat",
            0.0
        ),

        "protein": nutrition.get(
            "protein",
            0.0
        ),

        "fibre": nutrition.get(
            "fibre",
            0.0
        ),

        "salt": nutrition.get(
            "salt",
            0.0
        ),

        "source": "Web Retrieval",

        "source_url": url,

        "image_url": _extract_image(
            soup,
            product
        ),

        "retrieved_at":
            datetime.utcnow().isoformat()
            + "Z",

        "page_title": title,

        "extraction_method":
            " + ".join(
                extraction_sources
            ),
    }

    return result