"""
FoodGrade AI V5
Content-Based + Health-Aware Recommendation Engine

Phase 2 changes:
- Added recommend_healthier_alternatives() — same category, better health grade.
- Added empty-vocabulary guard in TF-IDF to prevent ValueError on sparse text.
- Added category-match helper.
- recommend_similar_products() unchanged for backward compatibility.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# GRADE ORDER — A is best (5), E is worst (1)
# ============================================================

GRADE_ORDER: dict[str, int] = {
    "A": 5, "B": 4, "C": 3, "D": 2, "E": 1
}


# ============================================================
# TEXT HELPERS
# ============================================================

def build_product_text(product):

    fields = [
        product.get("product_name", ""),
        product.get("brand", ""),
        product.get("category", ""),
        product.get("ingredients", ""),
        product.get("claims", ""),
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


def normalize_category(category: str) -> str:
    return str(category or "").lower().strip()


def is_same_product(target, candidate):

    target_name = normalize_name(
        target.get("product_name", "")
    )

    candidate_name = normalize_name(
        candidate.get("product_name", "")
    )

    if (
        target_name
        and candidate_name
        and target_name == candidate_name
    ):
        return True

    # Also match by DB id if both products are from the DB
    tid = target.get("id")
    cid = candidate.get("id")
    if tid and cid and tid == cid:
        return True

    return False


def _same_category(target, candidate) -> bool:
    """True when category strings share at least one significant word."""
    t_cat = normalize_category(target.get("category", ""))
    c_cat = normalize_category(candidate.get("category", ""))
    if not t_cat or not c_cat:
        return False
    # Full match
    if t_cat == c_cat:
        return True
    # At least one content word in common (ignore very short tokens)
    t_words = {w for w in t_cat.split() if len(w) > 2}
    c_words = {w for w in c_cat.split() if len(w) > 2}
    return bool(t_words & c_words)


def _grade_rank(product) -> int:
    grade = str(product.get("health_grade", "") or "").upper().strip()
    return GRADE_ORDER.get(grade, 0)


def _health_score(product) -> float:
    try:
        return float(product.get("health_score", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


# ============================================================
# SIMILAR PRODUCTS  (TF-IDF cosine similarity)
# ============================================================

def recommend_similar_products(
    target_product,
    products,
    top_n=5
):
    if not products:
        return []

    target_text = build_product_text(target_product)

    if not target_text:
        return []

    usable_products = []
    seen_names = set()

    for product in products:

        if is_same_product(target_product, product):
            continue

        product_name = normalize_name(
            product.get("product_name", "")
        )

        if product_name:
            if product_name in seen_names:
                continue
            seen_names.add(product_name)

        product_text = build_product_text(product)

        if not product_text:
            continue

        usable_products.append(product)

    if not usable_products:
        return []

    corpus = [target_text] + [
        build_product_text(p) for p in usable_products
    ]

    # Guard: empty vocabulary raises ValueError
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # Fallback: plain vectorizer without stop-word list
        try:
            vectorizer = TfidfVectorizer()
            matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            return []

    similarity = cosine_similarity(matrix[0:1], matrix[1:])[0]

    ranked = []
    for product, score in zip(usable_products, similarity):
        reasons = []
        if _same_category(target_product, product):
            cat = product.get("category")
            if cat:
                reasons.append(f"Category: {cat}")
        t_brand = str(target_product.get("brand") or "").strip().lower()
        c_brand = str(product.get("brand") or "").strip().lower()
        if t_brand and c_brand and t_brand == c_brand:
            reasons.append(f"Brand: {product.get('brand')}")
        sim_pct = round(float(score) * 100, 1)
        reasons.append(f"{sim_pct}% content similarity")

        ranked.append(
            {
                "product": product,
                "similarity": round(float(score) * 100, 2),
                "reasons": reasons,
                "reasons_text": " · ".join(reasons),
            }
        )

    ranked.sort(key=lambda item: item["similarity"], reverse=True)

    return ranked[:top_n]


# ============================================================
# HEALTH REASON GENERATOR
# ============================================================

def _generate_health_reasons(target: dict, candidate: dict) -> list[str]:
    """Provide specific nutritional justification for why candidate is healthier."""
    reasons = []
    t_grade = str(target.get("health_grade", "E")).upper()
    c_grade = str(candidate.get("health_grade", "E")).upper()
    if GRADE_ORDER.get(c_grade, 0) > GRADE_ORDER.get(t_grade, 0):
        reasons.append(f"Better Grade ({c_grade} vs {t_grade})")

    def _val(d, k):
        try:
            return float(d.get(k, 0) or 0)
        except Exception:
            return 0.0

    t_sugar, c_sugar = _val(target, "sugar"), _val(candidate, "sugar")
    if c_sugar < t_sugar and (t_sugar - c_sugar) >= 1.0:
        reasons.append(f"-{round(t_sugar - c_sugar, 1)}g sugar")

    t_sat, c_sat = _val(target, "saturated_fat"), _val(candidate, "saturated_fat")
    if c_sat < t_sat and (t_sat - c_sat) >= 0.5:
        reasons.append(f"-{round(t_sat - c_sat, 1)}g sat. fat")

    t_salt, c_salt = _val(target, "salt"), _val(candidate, "salt")
    if c_salt < t_salt and (t_salt - c_salt) >= 0.1:
        reasons.append(f"-{round(t_salt - c_salt, 1)}g salt")

    t_fib, c_fib = _val(target, "fibre"), _val(candidate, "fibre")
    if c_fib > t_fib and (c_fib - t_fib) >= 1.0:
        reasons.append(f"+{round(c_fib - t_fib, 1)}g fibre")

    t_prot, c_prot = _val(target, "protein"), _val(candidate, "protein")
    if c_prot > t_prot and (c_prot - t_prot) >= 1.5:
        reasons.append(f"+{round(c_prot - t_prot, 1)}g protein")

    if not reasons:
        t_score = _health_score(target)
        c_score = _health_score(candidate)
        reasons.append(f"+{round(c_score - t_score, 1)} pts higher score")

    return reasons


# ============================================================
# HEALTHIER ALTERNATIVES  (same category, better grade)
# ============================================================

def recommend_healthier_alternatives(
    target_product,
    products,
    top_n=3
):
    """
    Return up to `top_n` products that are:
    - In the same or similar category as `target_product`
    - Not the same product
    - Have a strictly better health grade, OR equal grade but higher health_score

    Results are sorted: best health grade first, then highest health_score.
    """
    if not products:
        return []

    target_grade_rank = _grade_rank(target_product)
    target_score = _health_score(target_product)

    candidates = []
    seen_names: set[str] = set()

    for product in products:

        if is_same_product(target_product, product):
            continue

        product_name = normalize_name(product.get("product_name", ""))
        if product_name in seen_names:
            continue
        seen_names.add(product_name)

        cand_rank = _grade_rank(product)
        cand_score = _health_score(product)

        # Must be at least as good grade, AND better score (or better grade)
        if cand_rank < target_grade_rank:
            continue
        if cand_rank == target_grade_rank and cand_score <= target_score:
            continue

        # Prefer same category; accept any category if pool is tiny
        is_cat_match = _same_category(target_product, product)
        reasons = _generate_health_reasons(target_product, product)

        candidates.append({
            "product": product,
            "health_grade": product.get("health_grade", "E"),
            "health_score": cand_score,
            "grade_rank": cand_rank,
            "category_match": is_cat_match,
            "reasons": reasons,
            "reasons_text": " · ".join(reasons),
        })

    if not candidates:
        return []

    # Sort: category-match first, then grade rank desc, then score desc
    candidates.sort(
        key=lambda x: (x["category_match"], x["grade_rank"], x["health_score"]),
        reverse=True,
    )

    return [
        {
            "product": c["product"],
            "health_grade": c["health_grade"],
            "health_score": c["health_score"],
            "reasons": c["reasons"],
            "reasons_text": c["reasons_text"],
        }
        for c in candidates[:top_n]
    ]