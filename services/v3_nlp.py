"""
FoodGrade AI V3.2
NLP / Text Normalization Layer

This module normalizes food-related claims and ingredients.

V3.2 is intentionally lightweight and explainable.
It does not require a large external language model.
"""

import re
from difflib import SequenceMatcher


CLAIM_SYNONYMS = {
    "high_protein": [
        "high protein",
        "protein rich",
        "protein-rich",
        "rich in protein",
        "rich protein",
        "good source of protein",
    ],

    "high_fibre": [
        "high fibre",
        "high fiber",
        "rich in fibre",
        "rich in fiber",
        "fibre rich",
        "fiber rich",
        "rich source of fibre",
    ],

    "low_sugar": [
        "low sugar",
        "low in sugar",
    ],

    "no_added_sugar": [
        "no added sugar",
        "without added sugar",
        "no sugar added",
    ],

    "sugar_free": [
        "sugar free",
        "sugar-free",
        "free from sugar",
    ],

    "low_fat": [
        "low fat",
        "low-fat",
        "low in fat",
    ],

    "fat_free": [
        "fat free",
        "fat-free",
        "free from fat",
    ],

    "organic": [
        "organic",
        "organically produced",
    ],

    "natural": [
        "natural",
        "all natural",
        "100% natural",
    ],
}


INGREDIENT_SYNONYMS = {
    "sugar": [
        "sugar",
        "cane sugar",
        "brown sugar",
        "white sugar",
        "sucrose",
        "dextrose",
        "glucose",
        "fructose",
        "maltose",
    ],

    "palm_oil": [
        "palm oil",
        "palmolein",
        "palm olein",
    ],

    "wheat": [
        "wheat",
        "wheat flour",
        "whole wheat",
        "whole wheat flour",
    ],

    "oats": [
        "oat",
        "oats",
        "oat flour",
        "oatmeal",
    ],

    "salt": [
        "salt",
        "sodium chloride",
    ],

    "whey_protein": [
        "whey protein",
        "whey protein concentrate",
        "whey protein isolate",
    ],

    "soy_protein": [
        "soy protein",
        "soya protein",
        "soy protein isolate",
    ],

    "milk": [
        "milk",
        "milk solids",
        "milk powder",
        "skimmed milk powder",
    ],

    "preservative": [
        "preservative",
        "sodium benzoate",
        "potassium sorbate",
        "sorbic acid",
    ],
}


def normalize_text(text):
    """Basic text normalization."""

    if text is None:
        return ""

    text = str(text).lower().strip()

    # Replace multiple spaces with one space
    text = re.sub(r"\s+", " ", text)

    return text


def similarity(text_a, text_b):
    """Return basic text similarity between 0 and 1."""

    return SequenceMatcher(
        None,
        normalize_text(text_a),
        normalize_text(text_b)
    ).ratio()


def normalize_claim(claim_text):
    """
    Normalize one claim into a standard concept.
    """

    original = str(claim_text).strip()
    normalized = normalize_text(original)

    if not normalized:
        return {
            "original": original,
            "normalized": None,
            "confidence": 0.0
        }

    # Exact / phrase matching
    for concept, phrases in CLAIM_SYNONYMS.items():

        for phrase in phrases:

            if normalized == phrase:

                return {
                    "original": original,
                    "normalized": concept,
                    "confidence": 1.0
                }

            if phrase in normalized:

                return {
                    "original": original,
                    "normalized": concept,
                    "confidence": 0.95
                }

    # Basic similarity matching
    best_concept = None
    best_score = 0.0

    for concept, phrases in CLAIM_SYNONYMS.items():

        for phrase in phrases:

            score = similarity(normalized, phrase)

            if score > best_score:
                best_score = score
                best_concept = concept

    if best_score >= 0.80:

        return {
            "original": original,
            "normalized": best_concept,
            "confidence": round(best_score, 2)
        }

    return {
        "original": original,
        "normalized": "unknown",
        "confidence": round(best_score, 2)
    }


def normalize_ingredient(ingredient_text):
    """
    Normalize an ingredient into a standard concept.
    """

    original = str(ingredient_text).strip()
    normalized = normalize_text(original)

    if not normalized:
        return {
            "original": original,
            "normalized": None,
            "confidence": 0.0
        }

    for concept, phrases in INGREDIENT_SYNONYMS.items():

        for phrase in phrases:

            if phrase in normalized:

                return {
                    "original": original,
                    "normalized": concept,
                    "confidence": 0.95
                }

    return {
        "original": original,
        "normalized": "other",
        "confidence": 0.0
    }


def process_claims(claims):
    """
    Normalize all extracted claims from V3.1.
    """

    results = []

    for claim in claims:

        # V3.1 returns a dictionary containing 'text'
        if isinstance(claim, dict):
            text = claim.get("text", "")
        else:
            text = claim

        result = normalize_claim(text)

        # Preserve V3.1 classification
        if isinstance(claim, dict):
            result["type"] = claim.get(
                "type",
                "unknown"
            )

        results.append(result)

    return results


def process_ingredients(ingredients):
    """
    Normalize all extracted ingredients from V3.1.
    """

    results = []

    for item in ingredients:

        if isinstance(item, dict):

            ingredient = item.get(
                "ingredient",
                ""
            )

            categories = item.get(
                "categories",
                []
            )

        else:

            ingredient = item
            categories = []

        result = normalize_ingredient(
            ingredient
        )

        result["categories"] = categories

        results.append(result)

    return results


def run_nlp(extraction_result):
    """
    Main V3.2 NLP function.
    """

    claims = extraction_result.get(
        "claims",
        []
    )

    ingredients = extraction_result.get(
        "ingredients",
        []
    )

    normalized_claims = process_claims(
        claims
    )

    normalized_ingredients = process_ingredients(
        ingredients
    )

    return {
        "status": "implemented",
        "normalized_claims": normalized_claims,
        "normalized_ingredients": normalized_ingredients
    }