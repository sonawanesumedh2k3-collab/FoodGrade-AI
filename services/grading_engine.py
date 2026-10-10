"""
FoodGrade AI — Core Grading & Verification Engine
Version: FoodGrade V2.5 Methodology Implementation

Produces two independent, unmerged evaluations aligned with FSSAI standards:
1. Health Grade (A–E): A transparent nutrition-profile score based on six dimensions:
   - Total sugar: 20% (Lower is better)
   - Saturated fat: 20% (Lower is better)
   - Sodium/salt: 20% (Lower is better)
   - Energy: 10% (Lower is preferable)
   - Dietary fibre: 15% (Higher is better)
   - Protein: 15% (Higher is better)
   Reference bands:
     85–100: A
     70–84.99: B
     55–69.99: C
     40–54.99: D
     0–39.99: E

2. Claim Grade (A–E) & FSSAI Verification Status:
   Evaluated against FSSAI Food Safety and Standards (Advertising and Claims) Regulations, 2018.
   Claim statuses:
   - VERIFIED: product data satisfies applicable FSSAI criterion
   - NOT_VERIFIED: product data does not satisfy criterion
   - INSUFFICIENT_EVIDENCE: required nutrition/ingredient information is missing
   - NO_CLAIMS_DETECTED: no marketing or nutrition claims detected

3. Dynamic Data Completeness & Confidence Scoring:
   Missing information is NOT silently converted into zero.
"""

from __future__ import annotations

import re
from typing import Any

from services.v3_extractor import extract_claims as v3_extract_claims
from services.v3_nlp import normalize_claim as v3_normalize_claim


# ============================================================
# HELPER FUNCTIONS & THRESHOLDS
# ============================================================

def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Clamp value between low and high bounds."""
    return max(low, min(high, float(value)))


def grade_from_score(score: float, is_claim: bool = False) -> str:
    """
    Convert a 0–100 score into an A–E grade band according to
    FoodGrade V2.5 methodology.
    """
    score = float(score)
    if is_claim:
        # Section 5: Claim Grade Bands
        if score >= 90.0:
            return "A"
        if score >= 75.0:
            return "B"
        if score >= 50.0:
            return "C"
        if score >= 25.0:
            return "D"
        return "E"
    else:
        # Section 6: Health Grade Bands
        if score >= 85.0:
            return "A"
        if score >= 70.0:
            return "B"
        if score >= 55.0:
            return "C"
        if score >= 40.0:
            return "D"
        return "E"


def _safe_float(val: Any) -> float | None:
    """Return float value or None if missing/unparseable."""
    if val is None or val == "":
        return None
    try:
        f = float(val)
        return f if f >= 0 else 0.0
    except (ValueError, TypeError):
        return None


def _to_dict(product: Any) -> dict[str, Any]:
    """Ensure product mapping is accessible as a standard Python dictionary."""
    if isinstance(product, dict):
        return product
    if hasattr(product, "keys"):
        return {k: product[k] for k in product.keys()}
    return dict(product)


# ============================================================
# DATA QUALITY & COMPLETENESS SCORING (Section 7)
# ============================================================

def calculate_data_quality_score(product: dict[str, Any]) -> tuple[float, list[str]]:
    """
    Calculate dynamic data completeness and confidence score (0–100).
    Missing information must not silently become zero.

    Returns:
        (score, missing_fields)
    """
    product = _to_dict(product)
    points = 0.0
    missing_fields = []

    # 1. Product Identity (25 points)
    if str(product.get("product_name") or "").strip():
        points += 10.0
    else:
        missing_fields.append("product_name")

    if str(product.get("brand") or "").strip():
        points += 5.0

    if str(product.get("category") or "").strip():
        points += 5.0

    if str(product.get("barcode") or "").strip() or str(product.get("source_url") or "").strip():
        points += 5.0

    # 2. Ingredient Transparency (15 points)
    ingredients_text = str(product.get("ingredients") or "").strip()
    if ingredients_text and ingredients_text.lower() not in {"not provided", "none"}:
        points += 15.0
    else:
        missing_fields.append("ingredients")

    # 3. Core Nutrition Declarations (60 points — 10 points per core nutrient)
    nutrition_keys = [
        ("calories", "energy"),
        ("sugar", "total sugar"),
        ("total_fat", "total fat"),
        ("saturated_fat", "saturated fat"),
        ("protein", "protein"),
        ("salt", "salt / sodium"),
    ]

    for key, label in nutrition_keys:
        raw_val = product.get(key)
        val = _safe_float(raw_val)
        if val is not None:
            # For energy, 0 kcal in solid food is typically an omission
            if key == "calories" and val == 0.0:
                missing_fields.append(label)
            else:
                points += 10.0
        else:
            missing_fields.append(label)

    return round(clamp(points, 0.0, 100.0), 1), missing_fields


# ============================================================
# SIX-DIMENSION HEALTH SCORING ENGINE (Section 6)
# ============================================================

def calculate_health_score(product: dict[str, Any]) -> tuple[float, str, list[str], dict[str, float]]:
    """
    FoodGrade V2.5 Research Scoring Model:
    Six nutrition dimensions:
    - Total sugar: 20% (Lower is better)
    - Saturated fat: 20% (Lower is better)
    - Sodium/salt: 20% (Lower is better)
    - Energy: 10% (Lower is generally preferable)
    - Dietary fibre: 15% (Higher is better)
    - Protein: 15% (Higher is better)

    Missing-data rule:
    Missing negative nutrients use neutral score (50.0) with an explanation notice,
    rather than unearned 100 points.

    Returns:
        (health_score, health_grade, health_reasons, dimension_scores)
    """
    product = _to_dict(product)
    health_reasons = []
    dimension_scores = {}

    # 1. Total Sugar (20% weight, lower is better)
    # FSSAI Low Sugar <= 5g/100g. High threshold >= 45g/100g.
    sugar_val = _safe_float(product.get("sugar"))
    if sugar_val is None:
        sugar_score = 50.0
        health_reasons.append("Total sugar not declared on label (neutral reference applied).")
    else:
        sugar_score = clamp(100.0 - (sugar_val / 45.0) * 100.0)
        if sugar_val <= 5.0:
            health_reasons.append(f"Low sugar ({sugar_val}g/100g) positively contributed to health score.")
        elif sugar_val > 22.5:
            health_reasons.append(f"High sugar content ({sugar_val}g/100g) significantly reduced health score.")
        elif sugar_val > 10.0:
            health_reasons.append(f"Moderate sugar ({sugar_val}g/100g) reduced health score.")
    dimension_scores["sugar"] = round(sugar_score, 1)

    # 2. Saturated Fat (20% weight, lower is better)
    # FSSAI Low Saturated Fat <= 1.5g/100g. High threshold >= 18g/100g.
    sat_val = _safe_float(product.get("saturated_fat"))
    if sat_val is None:
        sat_score = 50.0
        health_reasons.append("Saturated fat not declared on label (neutral reference applied).")
    else:
        sat_score = clamp(100.0 - (sat_val / 18.0) * 100.0)
        if sat_val <= 1.5:
            health_reasons.append(f"Low saturated fat ({sat_val}g/100g) positively contributed to score.")
        elif sat_val > 8.0:
            health_reasons.append(f"High saturated fat ({sat_val}g/100g) reduced health score.")
    dimension_scores["saturated_fat"] = round(sat_score, 1)

    # 3. Sodium / Salt (20% weight, lower is better)
    # FSSAI Low Sodium <= 0.12g sodium (approx <= 0.3g salt). High salt >= 2.5g/100g.
    salt_val = _safe_float(product.get("salt"))
    if salt_val is None:
        salt_score = 50.0
        health_reasons.append("Salt/sodium not declared on label (neutral reference applied).")
    else:
        salt_score = clamp(100.0 - (salt_val / 2.5) * 100.0)
        if salt_val <= 0.3:
            health_reasons.append(f"Low salt/sodium ({salt_val}g/100g) positively contributed to score.")
        elif salt_val > 1.25:
            health_reasons.append(f"Elevated salt/sodium ({salt_val}g/100g) reduced health score.")
    dimension_scores["salt"] = round(salt_score, 1)

    # 4. Energy (10% weight, lower is generally preferable)
    # Range 50 to 550 kcal/100g.
    cal_val = _safe_float(product.get("calories"))
    if cal_val is None or cal_val == 0.0:
        energy_score = 50.0
        health_reasons.append("Energy not declared on label (neutral reference applied).")
    else:
        energy_score = clamp(100.0 - ((cal_val - 50.0) / 500.0) * 100.0)
        if cal_val <= 200.0:
            health_reasons.append(f"Moderate caloric density ({cal_val} kcal/100g).")
        elif cal_val > 450.0:
            health_reasons.append(f"High caloric density ({cal_val} kcal/100g) reduced health score.")
    dimension_scores["energy"] = round(energy_score, 1)

    # 5. Dietary Fibre (15% weight, higher is better)
    # FSSAI High Fibre >= 6g/100g. Source of fibre >= 3g/100g.
    fibre_val = _safe_float(product.get("fibre"))
    if fibre_val is None:
        fibre_score = 0.0
        health_reasons.append("Dietary fibre not declared on label (no positive score added).")
    else:
        fibre_score = clamp((fibre_val / 6.0) * 100.0)
        if fibre_val >= 6.0:
            health_reasons.append(f"High dietary fibre ({fibre_val}g/100g) significantly improved score.")
        elif fibre_val >= 3.0:
            health_reasons.append(f"Contains dietary fibre ({fibre_val}g/100g, source of fibre).")
    dimension_scores["fibre"] = round(fibre_score, 1)

    # 6. Protein (15% weight, higher is better)
    # FSSAI High Protein >= 10g/100g. Source of protein >= 5g/100g.
    protein_val = _safe_float(product.get("protein"))
    if protein_val is None:
        protein_score = 0.0
        health_reasons.append("Protein not declared on label (no positive score added).")
    else:
        protein_score = clamp((protein_val / 10.0) * 100.0)
        if protein_val >= 10.0:
            health_reasons.append(f"High protein ({protein_val}g/100g meets FSSAI high-protein threshold) improved score.")
        elif protein_val >= 5.0:
            health_reasons.append(f"Contains protein ({protein_val}g/100g, source of protein).")
    dimension_scores["protein"] = round(protein_score, 1)

    # Weighted calculation
    health_score = (
        0.20 * sugar_score
        + 0.20 * sat_score
        + 0.20 * salt_score
        + 0.10 * energy_score
        + 0.15 * fibre_score
        + 0.15 * protein_score
    )

    health_score = round(clamp(health_score, 0.0, 100.0), 1)
    health_grade = grade_from_score(health_score, is_claim=False)

    return health_score, health_grade, health_reasons, dimension_scores


# ============================================================
# FSSAI CLAIM VERIFICATION ENGINE (Sections 4 & 5)
# ============================================================

def verify_claims(
    product: dict[str, Any],
    pipeline_result: dict[str, Any] | None = None
) -> tuple[float, str, str, list[str], list[dict[str, Any]]]:
    """
    Connect V3 NLP extraction & normalization to FSSAI Claim Verification.

    Produces:
    - VERIFIED: available product data satisfies documented criterion
    - NOT_VERIFIED: available product data does not satisfy criterion
    - INSUFFICIENT_EVIDENCE: required nutrition/ingredient data is missing
    - NO_CLAIMS_DETECTED: no claims detected on product

    Returns:
        (claim_score, claim_grade, overall_status, claim_reasons, claim_details)
    """
    product = _to_dict(product)
    raw_claims = str(product.get("claims") or "").strip()
    ingredients_text = str(product.get("ingredients") or "").lower().strip()

    # Extract claims using V3 extraction/NLP if not already provided in pipeline_result
    extracted_claims = []
    if pipeline_result and "ranking" in pipeline_result and "ranked_claims" in pipeline_result["ranking"]:
        extracted_claims = pipeline_result["ranking"]["ranked_claims"]
    elif pipeline_result and "nlp" in pipeline_result and "normalized_claims" in pipeline_result["nlp"]:
        extracted_claims = pipeline_result["nlp"]["normalized_claims"]
    else:
        raw_items = v3_extract_claims(raw_claims)
        for item in raw_items:
            norm = v3_normalize_claim(item.get("text", ""))
            extracted_claims.append(norm)

    # If no structured claims were extracted from raw text, try splitting by comma or semicolon
    if not extracted_claims and raw_claims:
        for chunk in re.split(r"[,;\n|]", raw_claims):
            chunk = chunk.strip()
            if chunk:
                norm = v3_normalize_claim(chunk)
                if norm.get("normalized") and norm.get("normalized") != "unknown":
                    extracted_claims.append(norm)

    if not extracted_claims:
        return 0.0, "No Claim", "NO_CLAIMS_DETECTED", ["No claims detected on product label."], []

    claim_details = []
    claim_reasons = []
    evaluable_scores = []
    has_insufficient = False

    sugar_val = _safe_float(product.get("sugar"))
    sat_val = _safe_float(product.get("saturated_fat"))
    fat_val = _safe_float(product.get("total_fat"))
    protein_val = _safe_float(product.get("protein"))
    fibre_val = _safe_float(product.get("fibre"))
    salt_val = _safe_float(product.get("salt"))

    # FSSAI Added Sugars keyword list
    added_sugars_list = [
        "sugar", "glucose syrup", "fructose", "sucrose", "maltose",
        "dextrose", "corn syrup", "invert sugar", "honey", "jaggery"
    ]

    for item in extracted_claims:
        concept = item.get("normalized") or "unknown"
        original = item.get("original") or item.get("text") or concept
        status = "INSUFFICIENT_EVIDENCE"
        evidence = ""
        rule = ""

        if concept == "high_protein":
            rule = "FSSAI Claim Regs 2018: High Protein requires >= 20% RDA (approx >= 10g/100g for solids)."
            if protein_val is None:
                status = "INSUFFICIENT_EVIDENCE"
                evidence = "Protein value not declared on label to verify High Protein claim."
            elif protein_val >= 10.0:
                status = "VERIFIED"
                evidence = f"Protein content ({protein_val}g/100g) satisfies FSSAI High Protein threshold (>= 10g/100g)."
            else:
                status = "NOT_VERIFIED"
                evidence = f"Protein content ({protein_val}g/100g) is below FSSAI High Protein threshold (>= 10g/100g)."

        elif concept == "high_fibre":
            rule = "FSSAI Claim Regs 2018: High Fibre requires >= 6g/100g."
            if fibre_val is None:
                status = "INSUFFICIENT_EVIDENCE"
                evidence = "Dietary fibre not declared on label to verify High Fibre claim."
            elif fibre_val >= 6.0:
                status = "VERIFIED"
                evidence = f"Dietary fibre ({fibre_val}g/100g) satisfies FSSAI High Fibre threshold (>= 6g/100g)."
            elif fibre_val >= 3.0:
                status = "NOT_VERIFIED"
                evidence = f"Dietary fibre ({fibre_val}g/100g) meets 'Source of Fibre' (>= 3g) but not 'High Fibre' (>= 6g)."
            else:
                status = "NOT_VERIFIED"
                evidence = f"Dietary fibre ({fibre_val}g/100g) is below FSSAI High Fibre threshold (>= 6g/100g)."

        elif concept == "low_sugar":
            rule = "FSSAI Claim Regs 2018: Low Sugar requires <= 5g/100g for solids."
            if sugar_val is None:
                status = "INSUFFICIENT_EVIDENCE"
                evidence = "Total sugar not declared on label to verify Low Sugar claim."
            elif sugar_val <= 5.0:
                status = "VERIFIED"
                evidence = f"Total sugar ({sugar_val}g/100g) satisfies FSSAI Low Sugar limit (<= 5g/100g)."
            else:
                status = "NOT_VERIFIED"
                evidence = f"Total sugar ({sugar_val}g/100g) exceeds FSSAI Low Sugar limit (<= 5g/100g)."

        elif concept == "sugar_free":
            rule = "FSSAI Claim Regs 2018: Sugar-Free requires <= 0.5g/100g."
            if sugar_val is None:
                status = "INSUFFICIENT_EVIDENCE"
                evidence = "Total sugar not declared on label to verify Sugar-Free claim."
            elif sugar_val <= 0.5:
                status = "VERIFIED"
                evidence = f"Total sugar ({sugar_val}g/100g) satisfies FSSAI Sugar-Free limit (<= 0.5g/100g)."
            else:
                status = "NOT_VERIFIED"
                evidence = f"Total sugar ({sugar_val}g/100g) exceeds FSSAI Sugar-Free limit (<= 0.5g/100g)."

        elif concept == "no_added_sugar":
            rule = "FSSAI Claim Regs 2018: No Added Sugar requires absence of added sugars in processing."
            if not ingredients_text or ingredients_text in {"not provided", "none"}:
                status = "INSUFFICIENT_EVIDENCE"
                evidence = "Ingredient list missing on label to verify No Added Sugar claim."
            else:
                found_sugars = [s for s in added_sugars_list if s in ingredients_text]
                if found_sugars:
                    status = "NOT_VERIFIED"
                    evidence = f"Added sugar ingredients detected: {', '.join(found_sugars)}."
                else:
                    status = "VERIFIED"
                    evidence = "No added sugar ingredients identified in ingredient text."

        elif concept == "low_fat":
            rule = "FSSAI Claim Regs 2018: Low Fat requires <= 3g/100g for solids."
            if fat_val is None:
                status = "INSUFFICIENT_EVIDENCE"
                evidence = "Total fat not declared on label to verify Low Fat claim."
            elif fat_val <= 3.0:
                status = "VERIFIED"
                evidence = f"Total fat ({fat_val}g/100g) satisfies FSSAI Low Fat limit (<= 3g/100g)."
            else:
                status = "NOT_VERIFIED"
                evidence = f"Total fat ({fat_val}g/100g) exceeds FSSAI Low Fat limit (<= 3g/100g)."

        elif concept == "fat_free":
            rule = "FSSAI Claim Regs 2018: Fat-Free requires <= 0.5g/100g."
            if fat_val is None:
                status = "INSUFFICIENT_EVIDENCE"
                evidence = "Total fat not declared on label to verify Fat-Free claim."
            elif fat_val <= 0.5:
                status = "VERIFIED"
                evidence = f"Total fat ({fat_val}g/100g) satisfies FSSAI Fat-Free limit (<= 0.5g/100g)."
            else:
                status = "NOT_VERIFIED"
                evidence = f"Total fat ({fat_val}g/100g) exceeds FSSAI Fat-Free limit (<= 0.5g/100g)."

        elif concept == "low_saturated_fat":
            rule = "FSSAI Claim Regs 2018: Low Saturated Fat requires <= 1.5g/100g."
            if sat_val is None:
                status = "INSUFFICIENT_EVIDENCE"
                evidence = "Saturated fat not declared on label to verify Low Saturated Fat claim."
            elif sat_val <= 1.5:
                status = "VERIFIED"
                evidence = f"Saturated fat ({sat_val}g/100g) satisfies FSSAI Low Saturated Fat limit (<= 1.5g/100g)."
            else:
                status = "NOT_VERIFIED"
                evidence = f"Saturated fat ({sat_val}g/100g) exceeds FSSAI Low Saturated Fat limit (<= 1.5g/100g)."

        elif concept in {"low_sodium", "low_salt"}:
            rule = "FSSAI Claim Regs 2018: Low Sodium requires <= 0.12g sodium/100g (approx <= 0.3g salt)."
            if salt_val is None:
                status = "INSUFFICIENT_EVIDENCE"
                evidence = "Salt/sodium not declared on label to verify Low Sodium claim."
            elif salt_val <= 0.3:
                status = "VERIFIED"
                evidence = f"Salt level ({salt_val}g/100g) satisfies FSSAI Low Sodium limit."
            else:
                status = "NOT_VERIFIED"
                evidence = f"Salt level ({salt_val}g/100g) exceeds FSSAI Low Sodium limit (<= 0.3g salt/100g)."

        elif concept in {"organic", "natural"}:
            rule = "FSSAI Organic Food Regulations: Requires third-party accreditation (Jaivik Bharat)."
            status = "INSUFFICIENT_EVIDENCE"
            evidence = f"{concept.title()} claim requires independent regulatory certification not inferable from label text."

        else:
            status = "INSUFFICIENT_EVIDENCE"
            evidence = f"Claim '{original}' requires domain verification."
            rule = "FSSAI General Claim Regulations: Requires supporting documentation."

        if status == "VERIFIED":
            evaluable_scores.append(100.0)
            claim_reasons.append(f"[VERIFIED] {original}: {evidence}")
        elif status == "NOT_VERIFIED":
            evaluable_scores.append(0.0)
            claim_reasons.append(f"[NOT VERIFIED] {original}: {evidence}")
        else:
            has_insufficient = True
            claim_reasons.append(f"[INSUFFICIENT EVIDENCE] {original}: {evidence}")

        claim_details.append({
            "original": original,
            "concept": concept,
            "status": status,
            "evidence": evidence,
            "rule": rule,
        })

    # Calculate overall Claim Score & Grade (Section 5)
    if evaluable_scores:
        claim_score = round(sum(evaluable_scores) / len(evaluable_scores), 1)
        claim_grade = grade_from_score(claim_score, is_claim=True)
        if all(s == 100.0 for s in evaluable_scores) and not has_insufficient:
            overall_status = "VERIFIED"
        elif any(s == 0.0 for s in evaluable_scores):
            overall_status = "NOT_VERIFIED"
        else:
            overall_status = "VERIFIED" if claim_score >= 75.0 else "NOT_VERIFIED"
    else:
        # All claims were insufficient evidence
        claim_score = 0.0
        claim_grade = "E"
        overall_status = "INSUFFICIENT_EVIDENCE"

    return claim_score, claim_grade, overall_status, claim_reasons, claim_details


# ============================================================
# COMPATIBILITY WRAPPERS
# ============================================================

def calculate_claim_score(product: dict[str, Any]) -> tuple[float, list[str]]:
    """
    Backward-compatible wrapper for existing calculate_claim_score callers.
    """
    score, grade, status, reasons, _ = verify_claims(product)
    return score, reasons


# ============================================================
# MAIN FOODGRADE CALCULATION ENTRYPOINT
# ============================================================

def calculate_foodgrade(
    product: dict[str, Any],
    pipeline_result: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Main FoodGrade V2.5 scoring and verification engine.

    Produces two separate, unmerged evaluations:
    - Health Grade (A–E)
    - Claim Grade (A–E) + FSSAI Claim Status (VERIFIED, NOT_VERIFIED,
      INSUFFICIENT_EVIDENCE, NO_CLAIMS_DETECTED)
    - Dynamic Data Quality Score (0–100)
    """
    product = _to_dict(product)
    # 1. Health Score (6 dimensions)
    health_score, health_grade, health_reasons, dimension_scores = calculate_health_score(product)

    # 2. Claim Verification (V3 NLP connected to FSSAI)
    claim_score, claim_grade, claim_status, claim_reasons, claim_details = verify_claims(
        product,
        pipeline_result=pipeline_result
    )

    # 3. Dynamic Data Completeness & Confidence Score
    data_quality_score, missing_fields = calculate_data_quality_score(product)

    # Adjust confidence if evaluable claims had missing evidence
    if claim_status == "INSUFFICIENT_EVIDENCE" and data_quality_score > 30:
        data_quality_score = round(data_quality_score * 0.85, 1)

    return {
        "health_score": health_score,
        "health_grade": health_grade,
        "claim_score": claim_score,
        "claim_grade": claim_grade,
        "health_reasons": health_reasons,
        "claim_reasons": claim_reasons,
        "data_quality_score": data_quality_score,
        "claim_status": claim_status,
        "claim_details": claim_details,
        "dimension_scores": dimension_scores,
        "missing_fields": missing_fields,
        "methodology_status": "FoodGrade V2.5 research methodology",
    }
