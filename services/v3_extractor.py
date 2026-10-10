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


from __future__ import annotations

import re
from typing import Any


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


def parse_compound_ingredient(ingredient_str: str) -> dict[str, Any]:
    """
    Parse a compound ingredient into base name and its sub-ingredients.
    E.g. 'Milk Chocolate (Sugar, Cocoa Butter, Milk Powder)' ->
    {'name': 'Milk Chocolate', 'sub_ingredients': ['Sugar', 'Cocoa Butter', 'Milk Powder']}
    """
    cleaned = ingredient_str.strip()
    match = re.search(r"^(.*?)\s*\((.*?)\)$", cleaned)
    if match:
        base_name = match.group(1).strip()
        sub_text = match.group(2).strip()
        sub_items = [s.strip() for s in re.split(r",|;", sub_text) if s.strip()]
        return {
            "name": base_name or cleaned,
            "raw": cleaned,
            "is_compound": True,
            "sub_ingredients": sub_items,
        }
    return {
        "name": cleaned,
        "raw": cleaned,
        "is_compound": False,
        "sub_ingredients": [],
    }


def split_ingredients(ingredient_text: str) -> list[str]:
    """
    Split ingredient text into individual ingredients, respecting parentheses
    and compound ingredients (e.g. 'Milk Chocolate (Sugar, Cocoa Butter, Milk Powder)').
    """
    if not ingredient_text:
        return []

    text = str(ingredient_text).strip()
    ingredients: list[str] = []
    current: list[str] = []
    depth = 0

    for char in text:
        if char in "([{":
            depth += 1
            current.append(char)
        elif char in ")]}":
            depth = max(0, depth - 1)
            current.append(char)
        elif char in ",;\n" and depth == 0:
            item = "".join(current).strip()
            if item:
                ingredients.append(item)
            current = []
        else:
            current.append(char)

    if current:
        item = "".join(current).strip()
        if item:
            ingredients.append(item)

    return ingredients


def classify_ingredients(ingredients: list[str]) -> list[dict[str, Any]]:
    """
    Identify important ingredient categories, inspecting both base names
    and parenthesized compound sub-ingredients.
    """
    detected = []

    for ingredient in ingredients:
        parsed = parse_compound_ingredient(ingredient)
        # Search both raw string and sub-ingredients
        texts_to_check = [ingredient, parsed["name"]] + parsed["sub_ingredients"]
        all_normalized = " ".join(clean_text(t) for t in texts_to_check)

        categories = []
        for category, keywords in INGREDIENT_CATEGORIES.items():
            for keyword in keywords:
                if keyword in all_normalized:
                    categories.append(category)
                    break

        detected.append({
            "ingredient": ingredient,
            "name": parsed["name"],
            "is_compound": parsed["is_compound"],
            "sub_ingredients": parsed["sub_ingredients"],
            "categories": categories
        })

    return detected


def _get_val(product, key, default=""):
    if hasattr(product, "get"):
        return product.get(key, default)
    try:
        val = product[key]
        return val if val is not None else default
    except (KeyError, IndexError, TypeError):
        return default


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

        value = _get_val(product, field, 0.0)

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

    claims = extract_claims(_get_val(product, "claims", ""))

    ingredients = split_ingredients(_get_val(product, "ingredients", ""))

    classified_ingredients = classify_ingredients(ingredients)

    nutrition = extract_nutrition(product)

    return {
        "claims": claims,
        "ingredients": classified_ingredients,
        "nutrition": nutrition,
    }