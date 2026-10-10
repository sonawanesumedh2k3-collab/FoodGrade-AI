# FoodGrade AI: Intelligent Food Product Quality Assessment System

FoodGrade AI is an academic research prototype and information-retrieval system that evaluates packaged food products through two independent, transparent evaluations:

1. **Health Grade (A–E):** Assesses the nutritional profile of a product using a transparent 6-dimension scoring model (Total Sugar 20%, Saturated Fat 20%, Sodium 20%, Energy 10%, Dietary Fibre 15%, Protein 15%).
2. **Claim Accuracy Grade (A–E):** Evaluates marketing and nutritional claims against statutory FSSAI (Food Safety and Standards Authority of India) Advertising and Claims Regulations (2018).

These assessments remain unmerged: a product can have accurate claims without being nutritionally healthy, or be healthy without making marketing claims.

---

## 1. System Architecture & Development Stages

- **V1–V2: Application Core & Data Collection:**
  - Flask web application, dashboard, and SQLite catalog.
  - Manual product entry and Open Food Facts Indian dataset integration.
  - Transparent data completeness indicators and missing data handling.

- **V3: Information Extraction, NLP, and Ranking:**
  - Extraction of food claims, ingredients, and nutritional entities.
  - Parentheses-aware compound ingredient parsing preserving sub-ingredients.
  - Text normalization, ingredient category classification, and relevance ranking.
  - Direct integration into the FoodGrade V2.5 grading engine.

- **V4: Web Retrieval & Crawling:**
  - Single product URL fetching with Schema.org / JSON-LD parsing.
  - Controlled Scrapy crawling obeying `robots.txt` and domain boundaries.
  - **Security Hardening:** Server-Side Request Forgery (SSRF) defense prohibiting access to loopback, private RFC1918, link-local, and cloud metadata endpoints.
  - **Cookie Protection:** Server-side persistence of crawl outputs in `data/web/` preventing client-side cookie overflow.

- **V5: OCR & Recommendation Engine:**
  - Optical Character Recognition (OCR) using Pillow and Tesseract.
  - Image preprocessing with camera EXIF rotation correction and contrast optimization.
  - Structured extraction of nutrition tables, ingredients, and claims.
  - Hybrid content-based recommendations: TF-IDF lexical profile + category matching + nutritional distance.
  - Healthier alternatives recommendation with transparent, granular reasons (e.g., `-8.5g sugar`, `+4.2g fibre`).

- **V6: Machine Learning (Random Forest Surrogate):**
  - Supervised Random Forest classifier trained on real empirical Indian food records (SQLite DB + Open Food Facts) and reference prototype anchors.
  - Stratified train/test split, 5-fold cross-validation, feature importance, and confusion matrix.
  - Model and metrics persistence via Joblib and JSON disk caching (no retraining on page reload).
  - Explicit surrogate comparison: explains whether ML prediction agrees or diverges from the deterministic 6-dimension score.

- **V7: Interactive System Diagnostics & Testing:**
  - Automated test runner verifying database connectivity, NLP pipelines, web parsers, OCR engines, recommender, and ML models.

- **V8: Project Overview & Academic Documentation:**
  - Comprehensive problem statement, objectives, architecture, workflow, and research limitations.

---

## 2. Environment Setup

### Requirements
- Windows 10/11
- Python 3.12 or 3.13
- Tesseract OCR (installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`)

### Virtual Environment Setup
Ensure the existing virtual environment `.venv` is activated:
```powershell
# In PowerShell:
.venv\Scripts\Activate.ps1

# Or run directly using the environment's python:
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 3. Launching the Application

### Option A: PowerShell / Command Line
```powershell
& .venv\Scripts\python.exe app.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

### Option B: Batch Scripts
Double-click `start.bat` from Windows Explorer.

---

## 4. Running the Automated Test Suite

All tests can be executed using `pytest` inside the virtual environment:

```powershell
# Run the master comprehensive test suite:
& .venv\Scripts\python.exe -m pytest test_foodgrade_suite.py -v

# Run all test suites including baseline tests:
& .venv\Scripts\python.exe -m pytest tests_phase1.py test_foodgrade_suite.py -v
```

All 27 automated tests and 28 subtests run in isolation without modifying the primary database.

---

## 5. Scoring & Claim Verification Methodology

### Health Score Weights (0–100)
| Dimension | Weight | Target Direction | FSSAI Reference Standard |
|---|---:|---|---|
| Total Sugar | 20% | Lower is better | &le; 5g/100g low sugar; &gt; 22.5g high penalty |
| Saturated Fat | 20% | Lower is better | &le; 1.5g/100g low sat fat; &gt; 10g penalty |
| Sodium / Salt | 20% | Lower is better | &le; 0.3g/100g low sodium; &gt; 1.5g penalty |
| Energy (Calories) | 10% | Lower is preferable | Baseline 150 kcal/100g |
| Dietary Fibre | 15% | Higher is better | &ge; 3g source; &ge; 6g high fibre bonus |
| Protein | 15% | Higher is better | &ge; 5g source; &ge; 10g high protein bonus |

### Health Grade Thresholds
- **A:** 85.0 – 100.0
- **B:** 70.0 – 84.99
- **C:** 55.0 – 69.99
- **D:** 40.0 – 54.99
- **E:** 0.0 – 39.99

### Missing Data Policy
Missing or undeclared nutritional fields are **not silently converted to zero**. Incomplete labels receive a neutral reference score (50.0/100) for unmeasured dimensions, and the data completeness score is penalized accordingly.

### Claim States
- `VERIFIED`: The product data fully satisfies applicable statutory FSSAI limits.
- `NOT_VERIFIED`: Product data contradicts or fails statutory FSSAI limits.
- `INSUFFICIENT_EVIDENCE`: A claim was made, but the corresponding nutrient was undeclared.
- `NO_CLAIMS_DETECTED`: No marketing or nutritional claims detected on the label.

---

## 6. Research Disclaimers & Known Limitations

1. **Academic Research Prototype:** FoodGrade AI is developed for academic demonstration and research. It does not constitute clinical, dietary, or medical advice.
2. **Surrogate ML Model:** The Random Forest model in V6 serves as an algorithmic surrogate trained to approximate the deterministic 6-dimension rules; it is not independently trained on human health outcomes.
3. **Data Completeness:** Product evaluations rely on declared package labels and Open Food Facts crowd-sourced entries, which may contain omissions or reporting variations.
