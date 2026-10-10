"""
FoodGrade AI V3.4
FoodGrade Integration Layer

This module connects the V3 information-processing pipeline
with the FoodGrade V2.5 grading and verification engine.

V3 evidence (extracted, normalized and ranked claims/ingredients)
directly feeds into the FSSAI claim verification and six-dimension
health scoring calculations.
"""

from services.grading_engine import calculate_foodgrade


def integrate_foodgrade(product, pipeline_result):
    """
    Connect V3 pipeline output with the FoodGrade V2.5 engine.

    Parameters
    ----------
    product : dict-like
        Product record from SQLite.

    pipeline_result : dict
        Output from V3.1, V3.2 and V3.3.

    Returns
    -------
    dict
        FoodGrade result plus V3 evidence.
    """

    # -----------------------------------------
    # Product data dictionary
    # -----------------------------------------
    if hasattr(product, "keys"):
        product_data = {
            key: product[key]
            for key in product.keys()
        }
    else:
        product_data = dict(product)

    # Calculate FoodGrade assessment passing V3 pipeline results
    foodgrade_result = calculate_foodgrade(
        product_data,
        pipeline_result=pipeline_result
    )

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
        )[:5],

        "claim_status": foodgrade_result.get(
            "claim_status",
            "NO_CLAIMS_DETECTED"
        ),

        "claim_details": foodgrade_result.get(
            "claim_details",
            []
        ),

        "data_quality_score": foodgrade_result.get(
            "data_quality_score",
            0.0
        ),

        "dimension_scores": foodgrade_result.get(
            "dimension_scores",
            {}
        ),

        "missing_fields": foodgrade_result.get(
            "missing_fields",
            []
        )
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

        "claim_status": foodgrade_result.get(
            "claim_status",
            "NO_CLAIMS_DETECTED"
        ),

        "claim_details": foodgrade_result.get(
            "claim_details",
            []
        ),

        "data_quality_score": foodgrade_result.get(
            "data_quality_score",
            0.0
        ),

        "dimension_scores": foodgrade_result.get(
            "dimension_scores",
            {}
        ),

        "missing_fields": foodgrade_result.get(
            "missing_fields",
            []
        ),

        "health_reasons": foodgrade_result[
            "health_reasons"
        ],

        "claim_reasons": foodgrade_result[
            "claim_reasons"
        ],

        "v3_evidence": evidence,

        "methodology_status":
            "FoodGrade V2.5 FSSAI-aligned research methodology"
    }