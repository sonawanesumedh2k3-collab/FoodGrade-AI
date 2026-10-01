"""
FoodGrade AI V5
Content-Based Recommendation
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_product_text(product):

    fields = [
        product.get(
            "product_name",
            ""
        ),

        product.get(
            "brand",
            ""
        ),

        product.get(
            "category",
            ""
        ),

        product.get(
            "ingredients",
            ""
        ),

        product.get(
            "claims",
            ""
        ),
    ]

    return " ".join(
        str(value)
        for value in fields
        if value
    ).strip()


def normalize_name(name):

    return " ".join(
        str(name)
        .lower()
        .strip()
        .split()
    )


def is_same_product(
    target,
    candidate
):

    target_name = normalize_name(
        target.get(
            "product_name",
            ""
        )
    )

    candidate_name = normalize_name(
        candidate.get(
            "product_name",
            ""
        )
    )

    if (
        target_name
        and candidate_name
        and target_name == candidate_name
    ):
        return True

    return False


def recommend_similar_products(
    target_product,
    products,
    top_n=5
):

    if not products:
        return []

    target_text = build_product_text(
        target_product
    )

    if not target_text:
        return []

    usable_products = []

    seen_names = set()

    for product in products:

        # ----------------------------------------------------
        # Remove current product.
        # ----------------------------------------------------

        if is_same_product(
            target_product,
            product
        ):
            continue

        product_name = normalize_name(
            product.get(
                "product_name",
                ""
            )
        )

        # ----------------------------------------------------
        # Remove duplicate products.
        # ----------------------------------------------------

        if product_name:

            if product_name in seen_names:
                continue

            seen_names.add(
                product_name
            )

        product_text = build_product_text(
            product
        )

        if not product_text:
            continue

        usable_products.append(
            product
        )

    if not usable_products:
        return []

    corpus = [
        target_text
    ]

    for product in usable_products:

        corpus.append(
            build_product_text(
                product
            )
        )

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(
        corpus
    )

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:]
    )[0]

    ranked = []

    for product, score in zip(
        usable_products,
        similarity
    ):

        ranked.append(
            {
                "product": product,

                "similarity": round(
                    float(score) * 100,
                    2
                ),
            }
        )

    ranked.sort(
        key=lambda item:
        item["similarity"],
        reverse=True
    )

    return ranked[:top_n]