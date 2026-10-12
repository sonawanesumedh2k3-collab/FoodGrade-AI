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
  - **Standalone PaddleOCR/PaddleX Service Integration:** High-accuracy nutrition-label OCR via a separate FastAPI service (`nutrition-label-ocr-master`) running on port 8001. Extracts energy, protein, carbohydrates, sugars, fat, saturated fat, trans fat, cholesterol, sodium, serving size, ingredients, and FSSAI licence number per 100 g.
  - Graceful Tesseract fallback when the standalone service is unavailable (with a clear actionable flash message directing the user to start it).
  - Real zero values (e.g., cholesterol 0 mg) preserved and distinguished from missing values (stored as `None`, never silently coerced to zero).
  - Image preprocessing with camera EXIF rotation correction and contrast optimisation.
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
- Tesseract OCR (installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`) — used as fallback when the standalone OCR service is unavailable
- **Standalone OCR Service** (`nutrition-label-ocr-master\`) with its own separate Python virtual environment (PaddleOCR/PaddleX); see Section 3 for startup instructions

### FoodGrade AI Virtual Environment Setup
Ensure the existing virtual environment `.venv` is activated:
```powershell
# In PowerShell:
.venv\Scripts\Activate.ps1

# Or run directly using the environment's python:
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 3. Launching the Application

FoodGrade AI uses **two separate processes**. The OCR service must be started first for full nutrition-label extraction accuracy. The Flask app runs independently and falls back to Tesseract if the OCR service is unavailable.

### Step 1 — Start the Standalone OCR Service

Open a **dedicated PowerShell terminal** in `nutrition-label-ocr-master\`:

```powershell
cd "..\nutrition-label-ocr-master"
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
& ".\.venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8001
```

Wait until you see: `Uvicorn running on http://127.0.0.1:8001`

- Health check: `GET http://127.0.0.1:8001/health`
- Swagger docs: `http://127.0.0.1:8001/docs`

> **Keep this terminal open** while using the OCR feature in FoodGrade AI.

### Step 2 — Start FoodGrade AI (Flask)

Open a **second PowerShell terminal** in `FoodGrade_AI_Main\`:

```powershell
& ".\.venv\Scripts\python.exe" app.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Configuration

| Variable | Default | Description |
|---|---|---|
| `FOODGRADE_OCR_URL` | `http://127.0.0.1:8001` | Base URL of the standalone OCR service |

To use a custom OCR service URL:
```powershell
$env:FOODGRADE_OCR_URL = "http://192.168.1.100:8001"
& ".\.venv\Scripts\python.exe" app.py
```

### Option: Batch Script
Double-click `start.bat` from Windows Explorer (starts Flask only; start the OCR service separately).

---

## 4. Troubleshooting

### OCR Service Not Available
If you upload an image and see **"Standalone OCR Service Unavailable"**:
1. Open a new PowerShell terminal in `nutrition-label-ocr-master\`.
2. Run: `$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"` then `& ".\.venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8001`
3. Wait for `Application startup complete.`
4. Retry the image upload.

FoodGrade AI will automatically fall back to Tesseract OCR while the service is down. Tesseract produces less accurate extraction for complex nutrition tables.

### Port 8001 Already in Use
```powershell
# Find and kill the process using port 8001:
netstat -ano | findstr :8001
taskkill /PID <pid> /F
```



## 5. Running the Automated Test Suite

All tests are run via `unittest` inside the FoodGrade AI virtual environment:

```powershell
# Comprehensive suite (24 tests):
& ".\.venv\Scripts\python.exe" -m unittest test_foodgrade_suite.py -v

# Core intelligence + phase 1 tests (10 tests):
& ".\.venv\Scripts\python.exe" -m unittest tests_phase1.py -v

# OCR integration suite (13 mocked tests — no live OCR service required):
& ".\.venv\Scripts\python.exe" -m unittest test_ocr_integration.py -v

# Phase 2 ML + recommender tests (script-mode):
& ".\.venv\Scripts\python.exe" tests_phase2.py

# Run suites together:
& ".\.venv\Scripts\python.exe" -m unittest test_foodgrade_suite.py tests_phase1.py test_ocr_integration.py -v
```

| Test File | Tests | Scope |
|---|---:|---|
| `test_foodgrade_suite.py` | 24 | Grading engine, SSRF, OFF normalisation, ML, routes |
| `tests_phase1.py` | 10 | Claim states, 6-dimension weights, NLP pipeline |
| `test_ocr_integration.py` | 13 | OCR service integration, normalisation, error handling |
| `tests_phase2.py` | 7 | ML training, caching, prediction, recommender |
| **Total** | **54** | All run without modifying the primary database |

---


## 6. Scoring & Claim Verification Methodology

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

## 7. Research Disclaimers & Known Limitations

1. **Academic Research Prototype:** FoodGrade AI is developed for academic demonstration and research. It does not constitute clinical, dietary, or medical advice.
2. **Surrogate ML Model:** The Random Forest model in V6 serves as an algorithmic surrogate trained to approximate the deterministic 6-dimension rules; it is not independently trained on human health outcomes.
3. **Data Completeness:** Product evaluations rely on declared package labels and Open Food Facts crowd-sourced entries, which may contain omissions or reporting variations.
4. **OCR Accuracy:** Nutrition values extracted via OCR are estimates. Verify critical values against the physical product label.
