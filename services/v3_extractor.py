"""
FoodGrade AI V3.1
Information Extraction

Extracts:
- Food claims
- Ingredients
- Ingredient categories
- Nutrition entities

This is a transparent rule-based extraction layer.
Later V3 stages will add NLP and language-model processing.
"""


import re


CLAIM_PATTERNS = {
    "high_protein": [
        "high protein",
        "protein rich",
        "protein-rich",
        "rich in protein",
    ],
    "high_fibre": [
        "high fibre",
        "high fiber",
        "rich in fibre",
        "rich in fiber",
    ],
    "low_sugar": [
        "low sugar",
        "sugar free",
        "sugar-free",
        "no added sugar",
    ],
    "low_fat": [
        "low fat",
        "fat free",
        "fat-free",
    ],
    "organic": [
        "organic",
    ],
    "natural": [
        "natural",
        "all natural",
    ],
}


INGREDIENT_CATEGORIES = {
    "sugar_related": [
        "sugar",
        "glucose syrup",
        "fructose",
        "sucrose",
        "maltose",
        "dextrose",
        "corn syrup",
    ],
    "oil_or_fat": [
        "palm oil",
        "vegetable oil",
        "sunflower oil",
        "soybean oil",
        "coconut oil",
        "butter",
        "shortening",
        "margarine",
    ],
    "protein_sources": [
        "whey protein",
        "milk protein",
        "soy protein",
        "pea protein",
        "egg",
        "chickpea",
        "lentil",
        "beans",
    ],
    "whole_grains": [
        "whole wheat",
        "whole grain",
        "oats",
        "oat",
        "wheat bran",
        "brown rice",
    ],
    "salt_or_sodium": [
        "salt",
        "sodium",
    ],
    "preservatives": [
        "preservative",
        "sodium benzoate",
        "potassium sorbate",
        "sorbic acid",
    ],
    "sweeteners": [
        "aspartame",
        "sucralose",
        "stevia",
        "saccharin",
        "acesulfame",
        "erythritol",
        "sorbitol",
    ],
}


def clean_text(text):
    """Normalize text for extraction."""
    if not text:
        return ""

    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)

    return text


def extract_claims(claim_text):
    """
    Extract and normalize claims from the product claim field.
    """

    text = clean_text(claim_text)

    if not text:
        return []

    extracted = []

    for normalized_name, patterns in CLAIM_PATTERNS.items():

        for pattern in patterns:

            if pattern in text:

                extracted.append({
                    "text": pattern,
                    "normalized": normalized_name,
                    "type": "nutritional"
                    if normalized_name in {
                        "high_protein",
                        "high_fibre",
                        "low_sugar",
                        "low_fat",
                    }
                    else "product_attribute"
                })

                break

    return extracted


def split_ingredients(ingredient_text):
    """
    Split ingredient text into individual ingredients.
    """

    if not ingredient_text:
        return []

    parts = re.split(r",|;|\n", str(ingredient_text))

    ingredients = []

    for part in parts:

        cleaned = re.sub(r"\([^)]*\)", "", part)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if cleaned:
            ingredients.append(cleaned)

    return ingredients


def classify_ingredients(ingredients):
    """
    Identify important ingredient categories.
    """

    detected = []

    for ingredient in ingredients:

        normalized = clean_text(ingredient)

        categories = []

        for category, keywords in INGREDIENT_CATEGORIES.items():

            for keyword in keywords:

                if keyword in normalized:
                    categories.append(category)
                    break

        detected.append({
            "ingredient": ingredient,
            "categories": categories
        })

    return detected


def extract_nutrition(product):
    """
    Convert database nutrition fields into structured nutrition entities.
    """

    fields = [
        ("calories", "energy", "kcal"),
        ("sugar", "sugar", "g"),
        ("total_fat", "total_fat", "g"),
        ("saturated_fat", "saturated_fat", "g"),
        ("protein", "protein", "g"),
        ("fibre", "fibre", "g"),
        ("salt", "salt", "g"),
    ]

    nutrition = []

    for field, normalized_name, unit in fields:

        value = product[field]

        nutrition.append({
            "name": normalized_name,
            "value": value,
            "unit": unit
        })

    return nutrition


def extract_product_information(product):
    """
    Main V3.1 extraction function.
    """

    claims = extract_claims(product["claims"])

    ingredients = split_ingredients(product["ingredients"])

    classified_ingredients = classify_ingredients(ingredients)

    nutrition = extract_nutrition(product)

    return {
        "claims": claims,
        "ingredients": classified_ingredients,
        "nutrition": nutrition,
    }