"""
FoodGrade AI V3.3
Claim and Ingredient Ranking

The ranking score represents downstream relevance for FoodGrade
analysis. It is NOT a health score and NOT a claim-verification score.
"""


CLAIM_PRIORITY = {
    "high_protein": 0.95,
    "high_fibre": 0.95,
    "low_sugar": 0.95,
    "no_added_sugar": 0.95,
    "sugar_free": 0.95,
    "low_fat": 0.90,
    "fat_free": 0.90,
    "organic": 0.70,
    "natural": 0.65,
    "unknown": 0.40,
}


INGREDIENT_PRIORITY = {
    "sugar": 0.95,
    "salt": 0.90,
    "palm_oil": 0.90,
    "whey_protein": 0.85,
    "soy_protein": 0.85,
    "milk": 0.75,
    "preservative": 0.80,
    "wheat": 0.70,
    "oats": 0.70,
    "other": 0.40,
}


def rank_claims(normalized_claims):
    """
    Rank normalized claims according to downstream relevance.
    """

    ranked = []

    for claim in normalized_claims:

        concept = claim.get(
            "normalized",
            "unknown"
        )

        base_score = CLAIM_PRIORITY.get(
            concept,
            0.40
        )

        confidence = claim.get(
            "confidence",
            0.0
        )

        # Combine relevance with NLP matching confidence.
        priority_score = round(
            base_score * confidence,
            2
        )

        ranked.append({
            "original": claim.get(
                "original",
                ""
            ),
            "normalized": concept,
            "confidence": confidence,
            "priority_score": priority_score,
            "reason": "Relevant to FoodGrade claim analysis"
        })

    ranked.sort(
        key=lambda x: x["priority_score"],
        reverse=True
    )

    # Add ranking position
    for position, item in enumerate(
        ranked,
        start=1
    ):
        item["rank"] = position

    return ranked


def rank_ingredients(normalized_ingredients):
    """
    Rank normalized ingredients according to downstream relevance.
    """

    ranked = []

    for ingredient in normalized_ingredients:

        concept = ingredient.get(
            "normalized",
            "other"
        )

        base_score = INGREDIENT_PRIORITY.get(
            concept,
            0.40
        )

        confidence = ingredient.get(
            "confidence",
            0.0
        )

        priority_score = round(
            base_score * confidence,
            2
        )

        ranked.append({
            "original": ingredient.get(
                "original",
                ""
            ),
            "normalized": concept,
            "confidence": confidence,
            "priority_score": priority_score,
            "categories": ingredient.get(
                "categories",
                []
            ),
            "reason": "Relevant to FoodGrade ingredient analysis"
        })

    ranked.sort(
        key=lambda x: x["priority_score"],
        reverse=True
    )

    for position, item in enumerate(
        ranked,
        start=1
    ):
        item["rank"] = position

    return ranked


def run_ranking(nlp_result):
    """
    Main V3.3 ranking function.
    """

    claims = nlp_result.get(
        "normalized_claims",
        []
    )

    ingredients = nlp_result.get(
        "normalized_ingredients",
        []
    )

    ranked_claims = rank_claims(
        claims
    )

    ranked_ingredients = rank_ingredients(
        ingredients
    )

    return {
        "status": "implemented",
        "ranked_claims": ranked_claims,
        "ranked_ingredients": ranked_ingredients
    }