"""
FoodGrade AI V3 Pipeline

V3.1 = Information Extraction
V3.2 = NLP / Normalization
V3.3 = Ranking
V3.4 = FoodGrade Integration
"""

from services.v3_extractor import extract_product_information
from services.v3_nlp import run_nlp
from services.v3_ranker import run_ranking
from services.v3_foodgrade import integrate_foodgrade


def run_v3_pipeline(product):

    # --------------------------------
    # V3.1 - Information Extraction
    # --------------------------------

    extraction_result = extract_product_information(
        product
    )

    # --------------------------------
    # V3.2 - NLP / Normalization
    # --------------------------------

    nlp_result = run_nlp(
        extraction_result
    )

    # --------------------------------
    # V3.3 - Ranking
    # --------------------------------

    ranking_result = run_ranking(
        nlp_result
    )

    # --------------------------------
    # Temporary combined result
    # --------------------------------

    pipeline_result = {
        "product": product,

        "information_extraction":
            extraction_result,

        "nlp":
            nlp_result,

        "ranking":
            ranking_result
    }

    # --------------------------------
    # V3.4 - FoodGrade Integration
    # --------------------------------

    foodgrade_result = integrate_foodgrade(
        product,
        pipeline_result
    )

    pipeline_result[
        "foodgrade"
    ] = foodgrade_result

    return pipeline_result