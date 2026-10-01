"""
FoodGrade AI V3.4
FoodGrade Integration Layer

This module connects the V3 information-processing pipeline
with the existing FoodGrade grading engine.

Important:
The existing grading engine remains unchanged.
V3 evidence is attached to the existing FoodGrade result.

The current grading methodology is still prototype-based.
Final methodology will be established separately.
"""

from services.grading_engine import calculate_foodgrade


def integrate_foodgrade(product, pipeline_result):
    """
    Connect V3 pipeline output with the existing FoodGrade engine.

    Parameters
    ----------
    product : dict-like
        Product record from SQLite.

    pipeline_result : dict
        Output from V3.1, V3.2 and V3.3.

    Returns
    -------
    dict
        Existing FoodGrade result plus V3 evidence.
    """

    # -----------------------------------------
    # Existing FoodGrade calculation
    # -----------------------------------------
    # Convert SQLite Row to a normal Python dictionary

    product_data = {
        key: product[key]
        for key in product.keys()
    }

    foodgrade_result = calculate_foodgrade(product_data)
    # -----------------------------------------
    # V3 evidence
    # -----------------------------------------

    extraction = pipeline_result.get(
        "information_extraction",
        {}
    )

    nlp = pipeline_result.get(
        "nlp",
        {}
    )

    ranking = pipeline_result.get(
        "ranking",
        {}
    )

    # -----------------------------------------
    # Evidence summary
    # -----------------------------------------

    evidence = {
        "extracted_claim_count": len(
            extraction.get("claims", [])
        ),

        "extracted_ingredient_count": len(
            extraction.get("ingredients", [])
        ),

        "normalized_claim_count": len(
            nlp.get("normalized_claims", [])
        ),

        "normalized_ingredient_count": len(
            nlp.get("normalized_ingredients", [])
        ),

        "ranked_claim_count": len(
            ranking.get("ranked_claims", [])
        ),

        "ranked_ingredient_count": len(
            ranking.get("ranked_ingredients", [])
        ),

        "top_claims": ranking.get(
            "ranked_claims",
            []
        )[:3],

        "top_ingredients": ranking.get(
            "ranked_ingredients",
            []
        )[:5]
    }

    return {
        "health_score": foodgrade_result[
            "health_score"
        ],

        "health_grade": foodgrade_result[
            "health_grade"
        ],

        "claim_score": foodgrade_result[
            "claim_score"
        ],

        "claim_grade": foodgrade_result[
            "claim_grade"
        ],

        "health_reasons": foodgrade_result[
            "health_reasons"
        ],

        "claim_reasons": foodgrade_result[
            "claim_reasons"
        ],

        "v3_evidence": evidence,

        "methodology_status":
            "Current prototype FoodGrade methodology"
    }