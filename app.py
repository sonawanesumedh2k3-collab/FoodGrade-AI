from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session
)
from werkzeug.utils import secure_filename

import json
import sqlite3
from pathlib import Path

from services.grading_engine import calculate_foodgrade
from services.openfoodfacts import (
    fetch_indian_products,
    save_dataset
)
from services.v3_pipeline import run_v3_pipeline
from services.web_retriever import fetch_product_url
from services.scrapy_runner import run_crawler

from services.v5_ocr import (
    perform_ocr,
    extract_product_from_text,
    is_valid_product_name,
    OCRConnectionError,
    OCRResponseError,
)
from services.v6_ml import (
    train_health_model,
    load_model_metrics,
    predict_health_grade
)
from services.v5_recommender import (
    recommend_similar_products,
    recommend_healthier_alternatives
)

from services.v7_testing import run_tests

# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = (
    BASE_DIR
    / "database"
    / "foodgrade.db"
)


app = Flask(__name__)

app.secret_key = (
    "foodgrade-v4-development-key"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db():

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    conn = sqlite3.connect(
        DB_PATH
    )

    conn.row_factory = sqlite3.Row

    return conn


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_db():

    conn = get_db()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS products (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            barcode TEXT UNIQUE,

            product_name TEXT NOT NULL,

            brand TEXT,

            category TEXT,

            ingredients TEXT,

            claims TEXT,

            calories REAL DEFAULT 0,

            sugar REAL DEFAULT 0,

            total_fat REAL DEFAULT 0,

            saturated_fat REAL DEFAULT 0,

            protein REAL DEFAULT 0,

            fibre REAL DEFAULT 0,

            salt REAL DEFAULT 0,

            health_score REAL DEFAULT 0,

            health_grade TEXT DEFAULT 'E',

            claim_score REAL DEFAULT 0,

            claim_grade TEXT DEFAULT 'E',

            source TEXT DEFAULT 'Manual',

            source_url TEXT,

            image_url TEXT,

            data_quality_score REAL DEFAULT 0,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()

    conn.close()


# ============================================================
# SAMPLE PRODUCTS
# ============================================================

def seed_sample_products():

    conn = get_db()

    count = conn.execute(
        "SELECT COUNT(*) FROM products"
    ).fetchone()[0]

    if count == 0:

        samples = [

            {
                "product_name":
                    "Oats & Fibre Breakfast Cereal",

                "brand":
                    "Demo Foods",

                "category":
                    "Cereal",

                "ingredients":
                    "Whole grain oats, wheat bran, raisins, almonds, sugar, salt",

                "claims":
                    "High Fibre",

                "calories":
                    360,

                "sugar":
                    8,

                "total_fat":
                    6,

                "saturated_fat":
                    1,

                "protein":
                    11,

                "fibre":
                    9,

                "salt":
                    0.5,

                "source":
                    "Demo"
            },


            {
                "product_name":
                    "Chocolate Cream Biscuits",

                "brand":
                    "Demo Foods",

                "category":
                    "Biscuits",

                "ingredients":
                    "Wheat flour, sugar, palm oil, cocoa powder, glucose syrup, salt",

                "claims":
                    "Tasty, Energy Rich",

                "calories":
                    500,

                "sugar":
                    28,

                "total_fat":
                    23,

                "saturated_fat":
                    12,

                "protein":
                    5,

                "fibre":
                    2,

                "salt":
                    0.9,

                "source":
                    "Demo"
            },


            {
                "product_name":
                    "Roasted Chickpea Snack",

                "brand":
                    "Demo Foods",

                "category":
                    "Snacks",

                "ingredients":
                    "Roasted chickpeas, spices, salt",

                "claims":
                    "High Protein, High Fibre",

                "calories":
                    330,

                "sugar":
                    3,

                "total_fat":
                    7,

                "saturated_fat":
                    1,

                "protein":
                    18,

                "fibre":
                    10,

                "salt":
                    0.7,

                "source":
                    "Demo"
            }
        ]


        for product in samples:

            result = calculate_foodgrade(
                product
            )

            conn.execute(
                """
                INSERT INTO products (

                    product_name,
                    brand,
                    category,
                    ingredients,
                    claims,

                    calories,
                    sugar,
                    total_fat,
                    saturated_fat,
                    protein,
                    fibre,
                    salt,

                    health_score,
                    health_grade,

                    claim_score,
                    claim_grade,

                    source,
                    data_quality_score
                )

                VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?
                )
                """,

                (
                    product["product_name"],
                    product["brand"],
                    product["category"],
                    product["ingredients"],
                    product["claims"],

                    product["calories"],
                    product["sugar"],
                    product["total_fat"],
                    product["saturated_fat"],
                    product["protein"],
                    product["fibre"],
                    product["salt"],

                    result["health_score"],
                    result["health_grade"],

                    result["claim_score"],
                    result["claim_grade"],

                    product["source"],
                    100
                )
            )


        conn.commit()


    conn.close()


# ============================================================
# UPDATE / INSERT IMPORTED PRODUCTS
# ============================================================

def upsert_imported_products(rows):

    conn = get_db()

    inserted = 0
    updated = 0


    for product in rows:

        result = calculate_foodgrade(
            product
        )

        barcode = (
            product.get("barcode")
            or None
        )


        existing = None

        if barcode:

            existing = conn.execute(
                """
                SELECT id
                FROM products
                WHERE barcode = ?
                """,
                (barcode,)
            ).fetchone()


        values = (

            product.get("product_name", ""),

            product.get("brand", ""),

            product.get("category", ""),

            product.get("ingredients", ""),

            product.get("claims", ""),

            product.get("calories", 0),

            product.get("sugar", 0),

            product.get("total_fat", 0),

            product.get("saturated_fat", 0),

            product.get("protein", 0),

            product.get("fibre", 0),

            product.get("salt", 0),

            result["health_score"],

            result["health_grade"],

            result["claim_score"],

            result["claim_grade"],

            product.get(
                "source",
                "Open Food Facts"
            ),

            product.get(
                "source_url",
                ""
            ),

            product.get(
                "image_url",
                ""
            ),

            product.get(
                "data_quality_score",
                0
            )
        )


        if existing:

            conn.execute(
                """
                UPDATE products SET

                    product_name = ?,
                    brand = ?,
                    category = ?,
                    ingredients = ?,
                    claims = ?,

                    calories = ?,
                    sugar = ?,
                    total_fat = ?,
                    saturated_fat = ?,
                    protein = ?,
                    fibre = ?,
                    salt = ?,

                    health_score = ?,
                    health_grade = ?,

                    claim_score = ?,
                    claim_grade = ?,

                    source = ?,
                    source_url = ?,
                    image_url = ?,
                    data_quality_score = ?

                WHERE id = ?
                """,

                values + (
                    existing["id"],
                )
            )

            updated += 1


        else:

            conn.execute(
                """
                INSERT INTO products (

                    product_name,
                    brand,
                    category,
                    ingredients,
                    claims,

                    calories,
                    sugar,
                    total_fat,
                    saturated_fat,
                    protein,
                    fibre,
                    salt,

                    health_score,
                    health_grade,

                    claim_score,
                    claim_grade,

                    source,
                    source_url,
                    image_url,
                    data_quality_score,

                    barcode
                )

                VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?, ?,
                    ?
                )
                """,

                values + (
                    barcode,
                )
            )

            inserted += 1


    conn.commit()

    conn.close()

    return inserted, updated


# ============================================================
# HOME / DASHBOARD
# ============================================================

@app.route("/")
def index():

    conn = get_db()

    # Select recent products with verified human-readable names
    # Filter out empty names, nutrition keys, table noise, and generic fallbacks
    all_rows = conn.execute(
        """
        SELECT *
        FROM products
        WHERE product_name IS NOT NULL
          AND product_name != ''
          AND product_name != 'Food Product'
          AND product_name != 'OCR Product'
          AND product_name NOT LIKE '%_g:%'
          AND product_name NOT LIKE '%_mg:%'
          AND product_name NOT LIKE '%_kcal:%'
        ORDER BY id DESC
        """
    ).fetchall()

    valid_products = [dict(r) for r in all_rows if is_valid_product_name(r["product_name"])][:6]

    # Fetch a featured product for the Nutrition Snapshot card (preferring a cereal or well-documented product)
    featured_product = None
    featured_row = conn.execute(
        """
        SELECT *
        FROM products
        WHERE product_name IS NOT NULL
          AND calories > 0
          AND (sugar > 0 OR protein > 0)
        ORDER BY CASE WHEN category = 'Cereal' THEN 0 ELSE 1 END, id ASC
        LIMIT 1
        """
    ).fetchone()
    if featured_row:
        featured_product = dict(featured_row)

    total = conn.execute(
        "SELECT COUNT(*) FROM products"
    ).fetchone()[0]

    imported = conn.execute(
        """
        SELECT COUNT(*)
        FROM products
        WHERE source = 'Open Food Facts'
        """
    ).fetchone()[0]

    conn.close()

    return render_template(
        "index.html",
        products=valid_products,
        featured_product=featured_product,
        total=total,
        imported=imported
    )


# ============================================================
# PRODUCTS
# ============================================================

@app.route("/products")
def products():

    query = request.args.get(
        "q",
        ""
    ).strip()

    category_filter = request.args.get(
        "category",
        ""
    ).strip()

    view_mode = request.args.get(
        "view",
        "grid"
    ).strip()

    conn = get_db()

    # Get distinct non-empty categories for filter pills
    cat_rows = conn.execute(
        """
        SELECT DISTINCT category
        FROM products
        WHERE category IS NOT NULL AND category != ''
        ORDER BY category ASC
        """
    ).fetchall()
    all_categories = [r["category"] for r in cat_rows]

    sql = "SELECT * FROM products WHERE 1=1"
    params = []

    if query:
        sql += " AND (product_name LIKE ? OR brand LIKE ? OR category LIKE ?)"
        pattern = f"%{query}%"
        params.extend([pattern, pattern, pattern])

    if category_filter:
        sql += " AND category = ?"
        params.append(category_filter)

    sql += " ORDER BY id DESC"

    rows = conn.execute(sql, params).fetchall()

    conn.close()

    return render_template(
        "products.html",
        products=rows,
        query=query,
        category_filter=category_filter,
        view_mode=view_mode,
        all_categories=all_categories
    )


# ============================================================
# ADD PRODUCT
# ============================================================

@app.route(
    "/product/new",
    methods=["GET", "POST"]
)
def new_product():

    if request.method == "POST":

        form = request.form


        try:

            product = {

                "product_name":
                    form["product_name"].strip(),

                "brand":
                    form.get(
                        "brand",
                        ""
                    ).strip(),

                "category":
                    form.get(
                        "category",
                        ""
                    ).strip(),

                "ingredients":
                    form.get(
                        "ingredients",
                        ""
                    ).strip(),

                "claims":
                    form.get(
                        "claims",
                        ""
                    ).strip(),

                "calories":
                    float(
                        form.get(
                            "calories"
                        ) or 0
                    ),

                "sugar":
                    float(
                        form.get(
                            "sugar"
                        ) or 0
                    ),

                "total_fat":
                    float(
                        form.get(
                            "total_fat"
                        ) or 0
                    ),

                "saturated_fat":
                    float(
                        form.get(
                            "saturated_fat"
                        ) or 0
                    ),

                "protein":
                    float(
                        form.get(
                            "protein"
                        ) or 0
                    ),

                "fibre":
                    float(
                        form.get(
                            "fibre"
                        ) or 0
                    ),

                "salt":
                    float(
                        form.get(
                            "salt"
                        ) or 0
                    ),

                "source":
                    "Manual"
            }


            if not product["product_name"]:

                raise ValueError(
                    "Product name is required."
                )


            result = calculate_foodgrade(
                product
            )


            conn = get_db()


            cursor = conn.execute(
                """
                INSERT INTO products (

                    product_name,
                    brand,
                    category,
                    ingredients,
                    claims,

                    calories,
                    sugar,
                    total_fat,
                    saturated_fat,
                    protein,
                    fibre,
                    salt,

                    health_score,
                    health_grade,

                    claim_score,
                    claim_grade,

                    source,
                    data_quality_score
                )

                VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?
                )
                """,

                (
                    product["product_name"],
                    product["brand"],
                    product["category"],
                    product["ingredients"],
                    product["claims"],

                    product["calories"],
                    product["sugar"],
                    product["total_fat"],
                    product["saturated_fat"],
                    product["protein"],
                    product["fibre"],
                    product["salt"],

                    result["health_score"],
                    result["health_grade"],

                    result["claim_score"],
                    result["claim_grade"],

                    "Manual",
                    result.get("data_quality_score", 100)
                )
            )


            product_id = cursor.lastrowid

            conn.commit()

            conn.close()


            flash(
                "Product added and analyzed successfully.",
                "success"
            )


            return redirect(
                url_for(
                    "product_detail",
                    product_id=product_id
                )
            )


        except ValueError as exc:

            flash(
                str(exc),
                "danger"
            )


    return render_template(
        "new_product.html"
    )


# ============================================================
# PRODUCT DETAIL
# ============================================================

@app.route(
    "/product/<int:product_id>"
)
def product_detail(product_id):

    conn = get_db()

    product = conn.execute(
        """
        SELECT *
        FROM products
        WHERE id = ?
        """,
        (product_id,)
    ).fetchone()

    if product is None:
        conn.close()
        flash(
            "Product not found.",
            "danger"
        )
        return redirect(
            url_for("products")
        )

    # Load all products for recommendation engine
    all_rows = conn.execute(
        "SELECT * FROM products ORDER BY id DESC"
    ).fetchall()
    conn.close()

    product_dict = dict(product)
    all_products = [dict(r) for r in all_rows]

    similar = recommend_similar_products(
        product_dict, all_products, top_n=5
    )
    healthier = recommend_healthier_alternatives(
        product_dict, all_products, top_n=3
    )

    foodgrade_result = calculate_foodgrade(product_dict)
    ml_prediction = predict_health_grade(
        product_dict,
        rule_grade=product_dict.get("health_grade")
    )

    return render_template(
        "product_detail.html",
        product=product,
        foodgrade=foodgrade_result,
        ml_prediction=ml_prediction,
        similar_products=similar,
        healthier_alternatives=healthier
    )


# ============================================================
# DELETE PRODUCT
# ============================================================

@app.route(
    "/product/<int:product_id>/delete",
    methods=["POST"]
)
def delete_product(product_id):

    conn = get_db()

    conn.execute(
        """
        DELETE FROM products
        WHERE id = ?
        """,
        (product_id,)
    )

    conn.commit()

    conn.close()


    flash(
        "Product deleted.",
        "success"
    )


    return redirect(
        url_for("products")
    )


# ============================================================
# METHODOLOGY
# ============================================================

@app.route("/methodology")
def methodology():

    return render_template(
        "methodology.html"
    )


# ============================================================
# DATA CENTER
# ============================================================

@app.route("/data-center")
def data_center():

    conn = get_db()


    total = conn.execute(
        "SELECT COUNT(*) FROM products"
    ).fetchone()[0]


    off_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM products
        WHERE source = 'Open Food Facts'
        """
    ).fetchone()[0]


    manual_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM products
        WHERE source = 'Manual'
        """
    ).fetchone()[0]


    conn.close()


    latest = (
        BASE_DIR
        / "data"
        / "processed"
        / "foodgrade_india_latest.csv"
    )


    return render_template(
        "data_center.html",

        total=total,

        off_count=off_count,

        manual_count=manual_count,

        latest_exists=latest.exists()
    )


# ============================================================
# IMPORT INDIAN PRODUCTS
# ============================================================

@app.route(
    "/data-center/import",
    methods=["POST"]
)
def import_indian_products():

    try:

        limit = int(
            request.form.get(
                "limit",
                "25"
            )
        )


        rows, payload = (
            fetch_indian_products(
                limit=limit
            )
        )


        if not rows:

            flash(
                "The source returned no usable Indian products. Try again later.",
                "warning"
            )

            return redirect(
                url_for("data_center")
            )


        save_dataset(
            rows,
            payload
        )


        inserted, updated = (
            upsert_imported_products(
                rows
            )
        )


        flash(
            f"Imported {len(rows)} products: "
            f"{inserted} new, "
            f"{updated} updated. "
            f"Cleaned CSV saved locally.",
            "success"
        )


    except Exception as exc:

        flash(
            f"Open Food Facts import failed: {exc}",
            "danger"
        )


    return redirect(
        url_for("data_center")
    )


# ============================================================
# V3 FOUNDATION
# ============================================================

@app.route("/v3")
def v3_foundation():

    return render_template(
        "v3_foundation.html"
    )


# ============================================================
# V3 PRODUCT ANALYSIS
# ============================================================

@app.route(
    "/v3/analyze/<int:product_id>"
)
def v3_analyze(product_id):

    conn = get_db()


    product = conn.execute(
        """
        SELECT *
        FROM products
        WHERE id = ?
        """,
        (product_id,)
    ).fetchone()


    conn.close()


    if product is None:

        flash(
            "Product not found.",
            "danger"
        )

        return redirect(
            url_for("products")
        )


    result = run_v3_pipeline(
        product
    )


    return render_template(
        "v3_analysis.html",
        product=product,
        result=result
    )


# ============================================================
# V3 EVALUATION
# ============================================================

@app.route("/v3/evaluation")
def v3_evaluation():

    conn = get_db()


    products = conn.execute(
        """
        SELECT *
        FROM products
        ORDER BY id ASC
        """
    ).fetchall()


    conn.close()


    evaluations = []

    passed = 0
    failed = 0


    for product in products:

        try:

            result = run_v3_pipeline(
                product
            )


            evaluations.append(
                {
                    "product": product,
                    "result": result,
                    "status": "PASS"
                }
            )


            passed += 1


        except Exception as exc:

            evaluations.append(
                {
                    "product": product,
                    "result": {
                        "error": str(exc)
                    },
                    "status": "FAIL"
                }
            )


            failed += 1


    return render_template(
        "v3_evaluation.html",

        evaluations=evaluations,

        products_tested=len(products),

        passed=passed,

        failed=failed
    )


# ============================================================
# V4 — WEB RETRIEVAL
# ============================================================

@app.route("/web-retrieval")
def web_retrieval():
    crawl_data = session.get("web_crawl")
    if not crawl_data and session.get("has_crawler_run"):
        crawl_file = BASE_DIR / "data" / "web" / "crawler_latest.json"
        if crawl_file.exists():
            try:
                raw_items = json.loads(crawl_file.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(raw_items, dict):
                    raw_items = [raw_items]
                crawl_data = {
                    "count": len(raw_items),
                    "items": raw_items,
                    "output": "data/web/crawler_latest.json",
                    "log": ""
                }
            except Exception:
                crawl_data = None

    return render_template(
        "web_retrieval.html",
        product=session.get("web_product"),
        analysis=session.get("web_analysis"),
        crawl=crawl_data,
        crawl_analysis=session.get("web_crawl_analysis"),
        crawl_selected=session.get("web_crawl_selected")
    )


# ============================================================
# V4 — SINGLE URL FETCH
# ============================================================

@app.route(
    "/web-retrieval/fetch",
    methods=["POST"]
)
def web_fetch():

    url = request.form.get(
        "url",
        ""
    ).strip()


    if not url:

        flash(
            "Please enter a URL.",
            "warning"
        )

        return redirect(
            url_for("web_retrieval")
        )


    try:

        product = fetch_product_url(
            url
        )


        session["web_product"] = product


        session.pop(
            "web_analysis",
            None
        )


        flash(
            "Product page retrieved successfully.",
            "success"
        )


    except Exception as exc:

        flash(
            f"Web retrieval failed: {exc}",
            "danger"
        )


    return redirect(
        url_for("web_retrieval")
    )


# ============================================================
# V4 — SINGLE URL → V3
# ============================================================

@app.route(
    "/web-retrieval/analyze",
    methods=["POST"]
)
def web_analyze():

    product = session.get(
        "web_product"
    )


    if not product:

        flash(
            "Fetch a product page first.",
            "warning"
        )

        return redirect(
            url_for("web_retrieval")
        )


    try:

        analysis = run_v3_pipeline(
            product
        )


        session["web_analysis"] = (
            analysis
        )


        flash(
            "Product successfully analyzed using V3 + FoodGrade.",
            "success"
        )


    except Exception as exc:

        flash(
            f"Product analysis failed: {exc}",
            "danger"
        )


    return redirect(
        url_for("web_retrieval")
    )


# ============================================================
# V4 — SCRAPY CRAWLER
# ============================================================

@app.route(
    "/web-retrieval/crawl",
    methods=["POST"]
)
def web_crawl():

    start_url = request.form.get(
        "start_url",
        ""
    ).strip()


    max_pages = request.form.get(
        "max_pages",
        5
    )


    delay = request.form.get(
        "delay",
        1
    )


    try:

        result = run_crawler(
            start_url,
            max_pages,
            delay
        )


        session["has_crawler_run"] = True
        session.pop("web_crawl", None)
        session.pop("web_crawl_analysis", None)
        session.pop("web_crawl_selected", None)

        flash(
            f"Crawler completed: {result['count']} page(s) exported.",
            "success"
        )

    except Exception as exc:
        flash(
            f"Crawler failed: {exc}",
            "danger"
        )

    return redirect(
        url_for("web_retrieval")
    )


# ============================================================
# V4 — CRAWLER RESULT → V3 → FOODGRADE
# ============================================================

@app.route(
    "/web-retrieval/crawl/analyze/<int:item_index>",
    methods=["POST"]
)
def web_crawl_analyze(item_index):

    crawl_file = BASE_DIR / "data" / "web" / "crawler_latest.json"
    items = []
    if crawl_file.exists():
        try:
            raw = json.loads(crawl_file.read_text(encoding="utf-8", errors="ignore"))
            items = raw if isinstance(raw, list) else [raw]
        except Exception:
            items = []

    if not items:
        crawl = session.get("web_crawl") or {}
        items = crawl.get("items", [])

    if not items:
        flash(
            "Run the crawler first.",
            "warning"
        )
        return redirect(
            url_for("web_retrieval")
        )

    if (
        item_index < 0
        or item_index >= len(items)
    ):
        flash(
            "Crawler result not found.",
            "danger"
        )
        return redirect(
            url_for("web_retrieval")
        )


    try:

        # ----------------------------------------------------
        # COPY CRAWLER ITEM
        # ----------------------------------------------------

        original = items[
            item_index
        ]


        product = dict(
            original
        )


        # ----------------------------------------------------
        # NORMALIZE TEXT FIELDS
        # ----------------------------------------------------

        text_fields = [

            "product_name",

            "brand",

            "category",

            "ingredients",

            "claims",

            "source",

            "url"
        ]


        for field in text_fields:

            product[field] = (
                product.get(field)
                or ""
            )


        # V3 / FoodGrade expects source_url
        product["source_url"] = (
            product.get("url")
            or ""
        )


        # ----------------------------------------------------
        # NORMALIZE NUMERIC FIELDS
        # ----------------------------------------------------

        numeric_fields = [

            "calories",

            "sugar",

            "total_fat",

            "saturated_fat",

            "protein",

            "fibre",

            "salt"
        ]


        for field in numeric_fields:

            try:

                value = product.get(
                    field
                )


                if (
                    value is None
                    or value == ""
                ):

                    product[field] = 0.0


                else:

                    product[field] = float(
                        value
                    )


            except (
                TypeError,
                ValueError
            ):

                product[field] = 0.0


        # ----------------------------------------------------
        # SEND INTO V3
        # ----------------------------------------------------

        analysis = run_v3_pipeline(
            product
        )


        session[
            "web_crawl_analysis"
        ] = analysis


        session[
            "web_crawl_selected"
        ] = item_index


        flash(
            "Crawler product successfully passed to the V3 + FoodGrade pipeline.",
            "success"
        )


    except Exception as exc:

        flash(
            f"Crawler product analysis failed: {exc}",
            "danger"
        )


    return redirect(
        url_for("web_retrieval")
    )

# ============================================================
# V5 — OCR + CONTENT-BASED RECOMMENDATION
# ============================================================

@app.route("/v5")
def v5_page():

    return render_template(
        "v5_ocr.html",

        ocr_text=session.get(
            "v5_ocr_text"
        ),

        ocr_product=session.get(
            "v5_ocr_product"
        ),

        ocr_analysis=session.get(
            "v5_ocr_analysis"
        ),

        recommendations=session.get(
            "v5_recommendations"
        )
    )


# ============================================================
# V5 — OCR
# ============================================================

@app.route(
    "/v5/ocr",
    methods=["POST"]
)
def v5_ocr():

    uploaded_file = request.files.get(
        "image"
    )


    if not uploaded_file:

        flash(
            "Please select a food product image.",
            "warning"
        )

        return redirect(
            url_for("v5_page")
        )


    filename = secure_filename(
        uploaded_file.filename
    )


    if not filename:

        flash(
            "Invalid image filename.",
            "danger"
        )

        return redirect(
            url_for("v5_page")
        )


    allowed_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp"
    }


    extension = (
        Path(filename)
        .suffix
        .lower()
    )


    if extension not in allowed_extensions:

        flash(
            "Please upload PNG, JPG, JPEG or WEBP.",
            "danger"
        )

        return redirect(
            url_for("v5_page")
        )

    # Validate file size (max 10 MB)
    uploaded_file.seek(0, 2)
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)
    if file_size > 10 * 1024 * 1024:
        flash(
            "Image file too large. Maximum allowed size is 10 MB.",
            "danger"
        )
        return redirect(
            url_for("v5_page")
        )


    try:

        ocr_dir = (
            BASE_DIR
            / "data"
            / "ocr"
        )


        ocr_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        image_path = (
            ocr_dir
            / filename
        )


        uploaded_file.save(
            image_path
        )


        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        text = perform_ocr(
            image_path
        )


        if not text:

            flash(
                "OCR could not detect readable text in the image.",
                "warning"
            )

            return redirect(
                url_for("v5_page")
            )


        # ----------------------------------------------------
        # EXTRACT PRODUCT DATA
        # ----------------------------------------------------

        product = (
            extract_product_from_text(
                text
            )
        )


        # ----------------------------------------------------
        # V3 ANALYSIS
        # ----------------------------------------------------

        analysis = run_v3_pipeline(
            product
        )


        # ----------------------------------------------------
        # SAVE OCR PRODUCT IN DATABASE
        # ----------------------------------------------------

        result = calculate_foodgrade(
            product
        )

        # --------------------------------------------------
        # PRODUCT-NAME VALIDATION GUARD
        # Block save if the extracted name is malformed, junk,
        # or a placeholder — show the user a correction form.
        # --------------------------------------------------
        extracted_name = product.get("product_name", "")
        if not is_valid_product_name(extracted_name):
            flash(
                "Product name could not be reliably extracted from the label. "
                "Please review and correct the name below before saving.",
                "warning"
            )
            return render_template(
                "v5_ocr.html",
                ocr_text=text,
                ocr_product=product,
                ocr_analysis=analysis,
                recommendations=[],
                name_needs_review=True,
            )

        conn = get_db()


        cursor = conn.execute(
            """
            INSERT INTO products (

                product_name,
                brand,
                category,
                ingredients,
                claims,

                calories,
                sugar,
                total_fat,
                saturated_fat,
                protein,
                fibre,
                salt,

                health_score,
                health_grade,

                claim_score,
                claim_grade,

                source,
                source_url,
                image_url,
                data_quality_score
            )

            VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?, ?, ?
            )
            """,

            (
                product.get(
                    "product_name",
                    "OCR Product"
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

                product.get(
                    "calories",
                    0
                ),

                product.get(
                    "sugar",
                    0
                ),

                product.get(
                    "total_fat",
                    0
                ),

                product.get(
                    "saturated_fat",
                    0
                ),

                product.get(
                    "protein",
                    0
                ),

                product.get(
                    "fibre",
                    0
                ),

                product.get(
                    "salt",
                    0
                ),

                result.get(
                    "health_score",
                    0
                ),

                result.get(
                    "health_grade",
                    "E"
                ),

                result.get(
                    "claim_score",
                    0
                ),

                result.get(
                    "claim_grade",
                    "E"
                ),

                "OCR",

                "",

                "",

                result.get("data_quality_score", 0)
            )
        )


        conn.commit()


        # ----------------------------------------------------
        # RECOMMENDATIONS
        # ----------------------------------------------------

        rows = conn.execute(
            """
            SELECT *
            FROM products
            ORDER BY id DESC
            """
        ).fetchall()


        conn.close()


        products = []

        for row in rows:

            products.append(
                dict(row)
            )


        recommendations = (
            recommend_similar_products(
                product,
                products,
                top_n=5
            )
        )


        # ----------------------------------------------------
        # DISPLAY V5 RESULTS DIRECTLY
        # ----------------------------------------------------
        # Do not store large OCR results in Flask's cookie session.
        # Render the result page directly instead.

        flash(
            "Food label OCR and product analysis completed.",
            "success"
        )

        return render_template(
            "v5_ocr.html",
            ocr_text=text,
            ocr_product=product,
            ocr_analysis=analysis,
            recommendations=recommendations
        )

    except OCRConnectionError as conn_err:
        flash(
            f"Standalone OCR Service Unavailable: {conn_err}",
            "warning"
        )
        return redirect(
            url_for("v5_page")
        )

    except OCRResponseError as resp_err:
        flash(
            f"OCR Service Error: {resp_err}",
            "danger"
        )
        return redirect(
            url_for("v5_page")
        )

    except Exception as exc:

        flash(
            f"V5 OCR failed: {exc}",
            "danger"
        )

        return redirect(
            url_for("v5_page")
        )

# ============================================================
# V5 — SAVE CORRECTED OCR PRODUCT NAME
# ============================================================

@app.route("/v5/ocr/save-corrected", endpoint="v5_ocr_save_corrected", methods=["POST"])
def v5_ocr_save_corrected():
    """Accept a user-corrected product name and save the OCR product to the DB."""
    corrected_name = request.form.get("corrected_name", "").strip()
    ocr_product_json = request.form.get("ocr_product_json", "{}")

    if not corrected_name or not is_valid_product_name(corrected_name):
        flash("Please enter a valid product name (at least 3 letters).", "warning")
        return redirect(url_for("v5_page"))

    try:
        product = json.loads(ocr_product_json)
    except (ValueError, TypeError):
        flash("Could not parse OCR product data. Please re-upload the image.", "danger")
        return redirect(url_for("v5_page"))

    product["product_name"] = corrected_name
    result = calculate_foodgrade(product)

    conn = get_db()
    conn.execute(
        """
        INSERT INTO products (
            product_name, brand, category, ingredients, claims,
            calories, sugar, total_fat, saturated_fat, protein, fibre, salt,
            health_score, health_grade, claim_score, claim_grade,
            source, source_url, image_url, data_quality_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            corrected_name,
            product.get("brand", ""),
            product.get("category", ""),
            product.get("ingredients", ""),
            product.get("claims", ""),
            product.get("calories", 0),
            product.get("sugar", 0),
            product.get("total_fat", 0),
            product.get("saturated_fat", 0),
            product.get("protein", 0),
            product.get("fibre", 0),
            product.get("salt", 0),
            result.get("health_score", 0),
            result.get("health_grade", "E"),
            result.get("claim_score", 0),
            result.get("claim_grade", "E"),
            "OCR",
            "",
            "",
            result.get("data_quality_score", 0),
        )
    )
    conn.commit()
    conn.close()

    flash(f"Product '{corrected_name}' saved successfully.", "success")
    return redirect(url_for("products"))


# ============================================================
# V6 — MACHINE LEARNING
# ============================================================

@app.route("/v6")
def v6_page():

    try:

        # Load cached metrics — no training on page load.
        # User must click "Train Model" to retrain.
        ml_result = load_model_metrics()

        return render_template(
            "v6_ml.html",
            ml_result=ml_result
        )

    except Exception as exc:

        return f"""
        <div style="font-family:Arial;padding:30px;">
            <h2>V6 Machine Learning Error</h2>
            <pre>{exc}</pre>
        </div>
        """

@app.route(
    "/v6/train",
    methods=["POST"]
)
def v6_train():

    try:

        result = train_health_model()

        session[
            "v6_ml_result"
        ] = result

        flash(
            "V6 Machine Learning model trained successfully.",
            "success"
        )

    except Exception as exc:

        flash(
            f"V6 ML training failed: {exc}",
            "danger"
        )

    return redirect(
        url_for("v6_page")
    )
# ============================================================
# V7 — TESTING & VALIDATION
# ============================================================

@app.route("/v7")
def v7_page():

    results = run_tests()

    return render_template(
        "v7_testing.html",
        results=results
    )
# ============================================================
# V8 — PROJECT OVERVIEW
# ============================================================

@app.route("/v8")
def v8_page():

    return render_template(
        "v8_overview.html"
    )
# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    init_db()

    seed_sample_products()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )