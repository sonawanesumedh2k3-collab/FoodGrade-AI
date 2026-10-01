# FoodGrade AI - Main Development Project

This is the single, continuing FoodGrade AI project. It is based on the working V2.5.1 build and is intended to be extended step-by-step instead of creating a new Python environment for every version.

## Current baseline

- Flask web application
- SQLite local database
- Manual product entry/search
- Product dashboard and product details
- Open Food Facts import with retry handling
- Raw/processed data folders
- Data Center
- FoodGrade research methodology page
- Prototype Claim Grade A-E
- Prototype Health Grade A-E
- Requirements file

## One environment for the whole project

Create **one** virtual environment in this folder:

```text
FoodGrade_AI_Main/
└── .venv/
```

Do not create a new `.venv` for each future FoodGrade stage. We will continue adding features to this project.

## First-time setup (recommended)

On Windows, double-click:

```text
setup.bat
```

It creates `.venv` once and installs the packages from `requirements.txt`.

Or use PyCharm:

1. Open the `FoodGrade_AI_Main` folder.
2. Let PyCharm detect/use `.venv` if it already exists.
3. Set the interpreter to:
   `FoodGrade_AI_Main\.venv\Scripts\python.exe`
4. Do **not** create another environment for future FoodGrade stages.

## Start the project

After the first setup, double-click:

```text
start.bat
```

Or in PyCharm run:

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Future development plan

We will continue this same project folder:

- V3.1 - Information Extraction
- V3.2 - Language Model / NLP processing
- V3.3 - Claim and ingredient ranking
- V3.4 - Connect NLP to FoodGrade assessment
- V4 - Web retrieval / crawling
- V5 - OCR and content-based recommendation
- V6 - ML training and evaluation
- V7 - Testing and refinement
- V8 - Documentation, PPT and final packaging

These stages are development milestones, not separate installations.

## Important

The current A-E grading implementation is still a research/prototype implementation based on the documented FoodGrade methodology. It will be refined and evaluated as later stages are developed.
