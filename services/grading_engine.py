import re


def grade_from_score(score):
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "E"


def clamp(value, low=0, high=100):
    return max(low, min(high, value))


def calculate_health_score(p):
    """
    V1 prototype scoring only.
    This is intentionally NOT the final FoodGrade methodology.
    The final thresholds/weights will be established in V2.5
    using documented nutritional/regulatory sources.
    """
    score = 60.0

    # Positive factors
    if p["protein"] >= 10:
        score += 10
    elif p["protein"] >= 5:
        score += 5

    if p["fibre"] >= 6:
        score += 12
    elif p["fibre"] >= 3:
        score += 6

    # Negative factors
    if p["sugar"] > 20:
        score -= 18
    elif p["sugar"] > 10:
        score -= 9

    if p["saturated_fat"] > 8:
        score -= 15
    elif p["saturated_fat"] > 5:
        score -= 8

    if p["salt"] > 1:
        score -= 12
    elif p["salt"] > 0.5:
        score -= 5

    if p["calories"] > 500:
        score -= 10
    elif p["calories"] > 400:
        score -= 5

    return clamp(round(score, 1))


def _contains_any(text, terms):
    lower = text.lower()
    return any(term in lower for term in terms)


def calculate_claim_score(p):
    """
    V1 prototype claim checker.
    It intentionally uses simple keyword/rule checks.
    Advanced claim verification will be added later.
    """
    claims = p.get("claims", "") or ""
    ingredients = p.get("ingredients", "") or ""
    score = 70.0
    reasons = []

    if not claims:
        return 70.0, reasons

    claim_text = claims.lower()
    ingredient_text = ingredients.lower()

    if "high protein" in claim_text:
        if p["protein"] >= 10:
            score += 20
            reasons.append("High Protein claim is supported by the current protein value.")
        else:
            score -= 25
            reasons.append("High Protein claim is not supported by the current protein value.")

    if "high fibre" in claim_text or "high fiber" in claim_text:
        if p["fibre"] >= 6:
            score += 15
            reasons.append("High Fibre claim is supported by the current fibre value.")
        else:
            score -= 20
            reasons.append("High Fibre claim is not supported by the current fibre value.")

    if "no added sugar" in claim_text:
        added_sugars = ["sugar", "glucose syrup", "fructose", "maltose", "corn syrup"]
        if _contains_any(ingredient_text, added_sugars):
            score -= 35
            reasons.append("No Added Sugar claim conflicts with the current ingredient text.")
        else:
            score += 15
            reasons.append("No obvious added-sugar ingredient was detected.")

    if "low sugar" in claim_text:
        if p["sugar"] <= 5:
            score += 15
            reasons.append("Low Sugar claim is consistent with the current sugar value.")
        else:
            score -= 20
            reasons.append("Low Sugar claim is not consistent with the current sugar value.")

    return clamp(round(score, 1)), reasons


def calculate_foodgrade(p):
    health_score = calculate_health_score(p)
    health_grade = grade_from_score(health_score)

    claim_score, claim_reasons = calculate_claim_score(p)
    claim_grade = grade_from_score(claim_score)

    health_reasons = []
    if p["sugar"] > 20:
        health_reasons.append("High sugar reduced the prototype health score.")
    elif p["sugar"] <= 5:
        health_reasons.append("Lower sugar helped the prototype health score.")

    if p["fibre"] >= 6:
        health_reasons.append("Higher fibre improved the prototype health score.")

    if p["protein"] >= 10:
        health_reasons.append("Higher protein improved the prototype health score.")

    if p["saturated_fat"] > 8:
        health_reasons.append("Higher saturated fat reduced the prototype health score.")

    if p["salt"] > 1:
        health_reasons.append("Higher salt reduced the prototype health score.")

    return {
        "health_score": health_score,
        "health_grade": health_grade,
        "claim_score": claim_score,
        "claim_grade": claim_grade,
        "health_reasons": health_reasons,
        "claim_reasons": claim_reasons,
    }
