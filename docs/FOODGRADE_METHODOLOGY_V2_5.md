# FoodGrade AI V2.5 — Research Methodology

## 1. Purpose

V2.5 freezes the **research methodology** before we add NLP, web crawling, OCR, recommendations, and machine learning.

FoodGrade AI produces two separate outputs:

1. **Claim Grade (A–E):** How well the product's stated claims are supported by the available label data and the documented FSSAI claim criteria.
2. **Health Grade (A–E):** A transparent nutrition-profile score based on nutrients available in the product data.

These two grades must not be merged. A claim can be accurate even when a product is not nutritionally strong, and a product can have a reasonable nutrition profile without making any marketing claim.

> **Important:** FoodGrade AI is an academic research prototype. Its A–E health grade is not an official FSSAI rating, medical recommendation, or diagnosis.

## 2. Evidence hierarchy

For Indian packaged foods, the project uses the following evidence order:

1. Product nutrition/ingredient information from the product label or a trusted product database.
2. FSSAI Food Safety and Standards (Advertising and Claims) Regulations, 2018 for claim criteria.
3. FSSAI Food Safety and Standards (Labelling and Display) Regulations, 2020 for label/nutrition fields.
4. WHO healthy-diet guidance for health-context interpretation.
5. Research literature for later NLP/ML features and validation.

## 3. Nutrition data basis

The preferred comparison basis is **per 100 g for solid foods** or **per 100 ml for liquid foods**, because FSSAI nutrition-claim criteria are expressed on these bases for many nutrients.

Core fields:

- Energy (kcal)
- Total sugar (g)
- Total fat (g)
- Saturated fat (g)
- Sodium/salt
- Protein (g)
- Dietary fibre (g)

Where the source gives sodium, it should be retained as sodium. Where only salt is available, the dataset keeps the source value and records the unit clearly rather than silently treating salt as sodium.

## 4. Claim verification methodology

Claims are first normalized into categories such as:

- Low sugar / sugar-free
- Low fat / fat-free
- Low saturated fat
- Low sodium / sodium-free
- High/source protein
- High/source fibre
- High/source vitamin/mineral
- No added sugar
- No added salt
- Other claims requiring evidence

Each detected claim receives one of these labels:

- **Verified** — available product data satisfies the applicable documented criterion.
- **Not verified** — available product data does not satisfy the criterion.
- **Insufficient evidence** — required nutrition/ingredient information is missing.
- **Requires domain review** — claim depends on evidence not safely inferable from the available label fields.

### Important FSSAI examples

FSSAI's claim schedule includes, among others:

- Low sugars: not more than 5 g/100 g for solids or 2.5 g/100 ml for liquids.
- High protein: 20% of RDA per 100 g for solids, 10% per 100 ml for liquids, or 10% per 100 kcal.
- High fibre: at least 6 g/100 g or 3 g/100 kcal.
- Low sodium: not more than 0.12 g sodium/100 g or 100 ml.
- Low saturated fat: not more than 1.5 g/100 g for solids or 0.75 g/100 ml for liquids, with the additional energy condition specified by FSSAI.

These are **claim-use criteria**, not a universal definition of a healthy food.

## 5. Claim Grade

The claim score is calculated from the claims that can actually be evaluated.

For each evaluable claim:

- Verified = 100 points
- Not verified = 0 points
- Insufficient evidence = excluded from the denominator but lowers the product's evidence-confidence indicator
- Requires domain review = excluded from the automatic score and sent to the review queue

Automatic claim grade:

| Score | Grade |
|---:|:---:|
| 90–100 | A |
| 75–89.99 | B |
| 50–74.99 | C |
| 25–49.99 | D |
| 0–24.99 | E |

If no claim is detected, the system should display **“No claim detected”** alongside the product rather than pretending that absence of a claim is proof of claim accuracy.

## 6. Health Grade — research scoring model

The Health Grade is deliberately separate from claim verification.

The first research version uses six nutrition dimensions:

| Dimension | Weight | Direction |
|---|---:|---|
| Total sugar | 20% | Lower is better |
| Saturated fat | 20% | Lower is better |
| Sodium/salt | 20% | Lower is better |
| Energy | 10% | Lower is generally preferable within the model |
| Dietary fibre | 15% | Higher is better |
| Protein | 15% | Higher is better |

Each dimension is normalized to a 0–100 component score using documented reference ranges and the product dataset. The weighted components are combined into a 0–100 **FoodGrade Health Score**.

The A–E bands are:

| Health Score | Grade |
|---:|:---:|
| 85–100 | A |
| 70–84.99 | B |
| 55–69.99 | C |
| 40–54.99 | D |
| 0–39.99 | E |

These bands are **FoodGrade research labels**, not official FSSAI grades.

### Why the model is not just “less sugar = healthy”

The model combines negative and positive nutrition dimensions. This reduces the risk of giving a high grade based on a single nutrient or marketing claim.

FSSAI itself states that food described as part of a healthy/balanced diet should not be based on selective consideration of only one or a few aspects of the food. WHO guidance also emphasizes limiting free sugars and sodium while maintaining an overall healthy dietary pattern.

## 7. Missing-data rules

Missing information must not be silently converted into zero.

- Missing sugar ≠ 0 sugar.
- Missing saturated fat ≠ 0 saturated fat.
- Missing sodium/salt ≠ 0 sodium/salt.
- Missing protein/fibre reduces confidence in the positive-factor portion.
- Missing ingredients reduces confidence in ingredient-based claim verification.

Each product therefore needs a **data quality / confidence indicator** in addition to the A–E grade.

## 8. Dataset labeling workflow

The research dataset will be labelled in this order:

1. Clean product name, category, ingredients and nutrition fields.
2. Detect and normalize claims.
3. Verify each claim against the applicable criterion.
4. Record evidence and reason.
5. Calculate claim score/grade.
6. Calculate nutrition components.
7. Calculate health score/grade.
8. Record missing-data/confidence information.
9. Keep the original source and source URL for traceability.

## 9. Features planned for ML in later versions

### Nutrition features

- energy_kcal
- sugar_g
- total_fat_g
- saturated_fat_g
- protein_g
- fibre_g
- sodium_or_salt

### Text features

- ingredient text length
- claim text length
- normalized claim categories
- ingredient keyword counts
- additive/processing-related terms
- whole-grain/legume/nut/fruit/vegetable indicators where detectable

### Source/data features

- data quality score
- source
- country
- category
- missing-field count

The ML model will learn from the labelled dataset only after the methodology and labels have been reviewed.

## 10. What V2.5 does not claim

FoodGrade AI does **not**:

- diagnose medical conditions;
- determine whether a food is medically suitable for an individual;
- certify FSSAI compliance;
- replace laboratory testing;
- treat an ingredient keyword as proof of healthiness;
- treat an accurate marketing claim as proof that the entire product is healthy.

## 11. Primary references

- FSSAI, Food Safety and Standards (Advertising and Claims) Regulations, 2018.
- FSSAI, Food Safety and Standards (Labelling and Display) Regulations, 2020 and amendments.
- WHO, Healthy Diet guidance.

The project should re-check the current FSSAI amendment status before final submission because regulations can be amended over time.
