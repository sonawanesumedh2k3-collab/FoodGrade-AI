import json
import re
from urllib.parse import urlparse

import scrapy
from bs4 import BeautifulSoup


class ProductSpider(scrapy.Spider):

    name = "foodgrade_product"

    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_TIMEOUT": 15,
        "CONCURRENT_REQUESTS": 4,
        "LOG_LEVEL": "INFO",
    }

    BLOCKED_PATH_WORDS = {
        "terms",
        "terms-and-conditions",
        "privacy",
        "privacy-policy",
        "contact",
        "about",
        "login",
        "register",
        "account",
        "cart",
        "checkout",
        "shipping",
        "returns",
        "refund",
        "faq",
        "blog",
        "news",
    }

    def __init__(
        self,
        start_url=None,
        max_pages=10,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.start_url = start_url

        try:
            self.max_pages = max(
                1,
                int(max_pages or 10)
            )
        except ValueError:
            self.max_pages = 10

        self.count = 0

        self.seen = set()

        self.allowed_domains = []

        if start_url:

            hostname = urlparse(
                start_url
            ).hostname

            if hostname:
                self.allowed_domains = [
                    hostname
                ]

    # ========================================================
    # SCRAPY 2.19 START
    # ========================================================

    async def start(self):

        if not self.start_url:
            self.logger.error(
                "No start_url was provided."
            )
            return

        self.logger.info(
            "Starting FoodGrade crawler: %s",
            self.start_url
        )

        yield scrapy.Request(
            self.start_url,
            callback=self.parse,
            errback=self.handle_error,
            dont_filter=True
        )

    # ========================================================
    # ERROR HANDLER
    # ========================================================

    def handle_error(self, failure):

        self.logger.error(
            "Request failed: %s",
            failure
        )

    # ========================================================
    # MAIN PARSER
    # ========================================================

    def parse(self, response):

        if self.count >= self.max_pages:
            return

        if response.url in self.seen:
            return

        self.seen.add(
            response.url
        )

        soup = BeautifulSoup(
            response.text,
            "lxml"
        )

        page_title = ""

        if soup.title:

            page_title = soup.title.get_text(
                " ",
                strip=True
            )

        text = re.sub(
            r"\s+",
            " ",
            soup.get_text(" ")
        ).strip()

        # ----------------------------------------------------
        # JSON-LD
        # ----------------------------------------------------

        json_products = (
            self._extract_jsonld_products(
                soup
            )
        )

        # ----------------------------------------------------
        # PRODUCT INFORMATION
        # ----------------------------------------------------

        product_name = (
            self._extract_product_name(
                soup,
                json_products,
                page_title
            )
        )

        brand = (
            self._extract_brand(
                soup,
                json_products
            )
        )

        category = (
            self._extract_category(
                soup,
                json_products
            )
        )

        ingredients = (
            self._extract_ingredients(
                soup,
                json_products,
                text
            )
        )

        claims = (
            self._extract_claims(
                soup,
                json_products,
                text
            )
        )

        # ----------------------------------------------------
        # NUTRITION
        # ----------------------------------------------------

        calories = self._extract_nutrition(
            soup,
            json_products,
            text,
            ["calories", "energy"]
        )

        sugar = self._extract_nutrition(
            soup,
            json_products,
            text,
            ["sugar", "sugars"]
        )

        total_fat = self._extract_nutrition(
            soup,
            json_products,
            text,
            ["total fat", "fat"]
        )

        saturated_fat = self._extract_nutrition(
            soup,
            json_products,
            text,
            ["saturated fat", "saturates"]
        )

        protein = self._extract_nutrition(
            soup,
            json_products,
            text,
            ["protein"]
        )

        fibre = self._extract_nutrition(
            soup,
            json_products,
            text,
            ["fibre", "fiber"]
        )

        salt = self._extract_nutrition(
            soup,
            json_products,
            text,
            ["salt"]
        )

        image_url = self._extract_image(
            soup,
            json_products,
            response
        )

        # ----------------------------------------------------
        # PRODUCT PAGE FILTER
        # ----------------------------------------------------

        is_product = self._is_product_page(
            response.url,
            page_title,
            product_name,
            brand,
            ingredients,
            json_products,
            text
        )

        if is_product:

            self.count += 1

            self.logger.info(
                "Food product %s/%s: %s",
                self.count,
                self.max_pages,
                response.url
            )

            yield {
                "url":
                    response.url,

                "page_title":
                    page_title,

                "product_name":
                    product_name,

                "brand":
                    brand,

                "category":
                    category,

                "ingredients":
                    ingredients,

                "claims":
                    claims,

                "calories":
                    calories,

                "sugar":
                    sugar,

                "total_fat":
                    total_fat,

                "saturated_fat":
                    saturated_fat,

                "protein":
                    protein,

                "fibre":
                    fibre,

                "salt":
                    salt,

                "image_url":
                    image_url,

                "source":
                    "Scrapy Web Crawler"
            }

        else:

            self.logger.info(
                "Skipped non-product page: %s",
                response.url
            )

        # ----------------------------------------------------
        # STOP AFTER REQUIRED PRODUCT COUNT
        # ----------------------------------------------------

        if self.count >= self.max_pages:
            return

        # ----------------------------------------------------
        # FOLLOW SAME-DOMAIN LINKS
        # ----------------------------------------------------

        for href in soup.select(
            "a[href]"
        ):

            next_url = response.urljoin(
                href.get("href")
            )

            if not next_url.startswith(
                (
                    "http://",
                    "https://"
                )
            ):
                continue

            if next_url in self.seen:
                continue

            if not self._same_domain(
                next_url
            ):
                continue

            if self._blocked_url(
                next_url
            ):
                continue

            yield scrapy.Request(
                next_url,
                callback=self.parse,
                errback=self.handle_error
            )

    # ========================================================
    # PRODUCT PAGE DETECTION
    # ========================================================

    def _is_product_page(
        self,
        url,
        title,
        product_name,
        brand,
        ingredients,
        json_products,
        text
    ):

        if self._blocked_url(url):
            return False

        # Strongest signal:
        # Schema.org Product
        if json_products:
            return True

        lower_url = url.lower()

        product_url_words = [
            "/product/",
            "/products/",
            "/shop/",
            "/p/",
            "/item/"
        ]

        if any(
            word in lower_url
            for word in product_url_words
        ):
            return True

        lower_text = text.lower()

        food_words = [
            "ingredients",
            "nutrition facts",
            "nutrition information",
            "protein",
            "calories",
            "sugar",
            "fat",
            "fibre",
            "fiber",
            "salt"
        ]

        food_signal_count = sum(
            1
            for word in food_words
            if word in lower_text
        )

        if (
            product_name
            and (
                brand
                or ingredients
                or food_signal_count >= 2
            )
        ):
            return True

        return False

    # ========================================================
    # BLOCKED URL CHECK
    # ========================================================

    def _blocked_url(
        self,
        url
    ):

        path = (
            urlparse(
                url
            ).path
            .lower()
        )

        path_parts = set(
            part
            for part in path.split("/")
            if part
        )

        for word in self.BLOCKED_PATH_WORDS:

            if word in path_parts:
                return True

        return False

    # ========================================================
    # JSON-LD
    # ========================================================

    def _extract_jsonld_products(
        self,
        soup
    ):

        products = []

        scripts = soup.find_all(
            "script",
            {
                "type":
                    "application/ld+json"
            }
        )

        for script in scripts:

            raw = script.string

            if not raw:
                raw = script.get_text(
                    strip=True
                )

            if not raw:
                continue

            try:

                data = json.loads(
                    raw
                )

            except Exception:

                continue

            self._collect_json_products(
                data,
                products
            )

        return products

    def _collect_json_products(
        self,
        data,
        products
    ):

        if isinstance(
            data,
            list
        ):

            for item in data:

                self._collect_json_products(
                    item,
                    products
                )

            return

        if not isinstance(
            data,
            dict
        ):
            return

        graph = data.get(
            "@graph"
        )

        if graph:

            self._collect_json_products(
                graph,
                products
            )

        item_type = data.get(
            "@type",
            ""
        )

        if isinstance(
            item_type,
            list
        ):

            item_type = " ".join(
                str(x)
                for x in item_type
            )

        if "Product" in str(
            item_type
        ):

            products.append(
                data
            )

    # ========================================================
    # PRODUCT NAME
    # ========================================================

    def _extract_product_name(
        self,
        soup,
        products,
        title
    ):

        for product in products:

            name = product.get(
                "name"
            )

            if name:
                return str(
                    name
                ).strip()

        tag = soup.find(
            "meta",
            {
                "property":
                    "og:title"
            }
        )

        if tag and tag.get(
            "content"
        ):

            return tag.get(
                "content"
            ).strip()

        selectors = [
            "h1",
            ".product-title",
            ".product_name",
            ".product-name",
            "[itemprop='name']"
        ]

        for selector in selectors:

            tag = soup.select_one(
                selector
            )

            if tag:

                value = tag.get_text(
                    " ",
                    strip=True
                )

                if value:
                    return value

        return title.strip()

    # ========================================================
    # BRAND
    # ========================================================

    def _extract_brand(
        self,
        soup,
        products
    ):

        for product in products:

            brand = product.get(
                "brand"
            )

            if isinstance(
                brand,
                dict
            ):

                brand = brand.get(
                    "name"
                )

            if brand:

                return str(
                    brand
                ).strip()

        selectors = [
            "[itemprop='brand']",
            ".brand",
            ".product-brand",
            ".brand-name",
            "[class*='brand']"
        ]

        for selector in selectors:

            tag = soup.select_one(
                selector
            )

            if tag:

                value = tag.get_text(
                    " ",
                    strip=True
                )

                if value:
                    return value

        return ""

    # ========================================================
    # CATEGORY
    # ========================================================

    def _extract_category(
        self,
        soup,
        products
    ):

        for product in products:

            category = product.get(
                "category"
            )

            if category:

                return str(
                    category
                ).strip()

        tag = soup.select_one(
            "[itemprop='category']"
        )

        if tag:

            value = tag.get_text(
                " ",
                strip=True
            )

            if value:
                return value

        return ""

    # ========================================================
    # INGREDIENTS
    # ========================================================

    def _extract_ingredients(
        self,
        soup,
        products,
        text
    ):

        for product in products:

            ingredients = product.get(
                "ingredients"
            )

            if isinstance(
                ingredients,
                list
            ):

                values = []

                for ingredient in ingredients:

                    if isinstance(
                        ingredient,
                        dict
                    ):

                        value = (
                            ingredient.get(
                                "name"
                            )
                            or
                            ingredient.get(
                                "text"
                            )
                        )

                    else:

                        value = str(
                            ingredient
                        )

                    if value:
                        values.append(
                            value
                        )

                if values:

                    return ", ".join(
                        values
                    )

            elif ingredients:

                return str(
                    ingredients
                ).strip()

        selectors = [
            "[itemprop='ingredients']",
            ".ingredients",
            ".ingredient",
            ".ingredients-list",
            "[class*='ingredient']",
            "[id*='ingredient']"
        ]

        for selector in selectors:

            tags = soup.select(
                selector
            )

            for tag in tags:

                value = tag.get_text(
                    " ",
                    strip=True
                )

                if len(value) >= 5:

                    return value[:3000]

        match = re.search(
            r"ingredients?\s*[:\-]\s*(.+?)(?:nutrition|allergen|contains|storage|may contain|$)",
            text,
            flags=re.I
        )

        if match:

            value = match.group(
                1
            ).strip()

            if len(value) >= 5:

                return value[:3000]

        return ""

    # ========================================================
    # CLAIMS
    # ========================================================

    def _extract_claims(
        self,
        soup,
        products,
        text
    ):

        claims = []

        for product in products:

            description = product.get(
                "description"
            )

            if description:

                claims.extend(
                    self._find_claim_terms(
                        str(description)
                    )
                )

        claims.extend(
            self._find_claim_terms(
                text
            )
        )

        return ", ".join(
            list(
                dict.fromkeys(
                    claims
                )
            )
        )

    def _find_claim_terms(
        self,
        text
    ):

        terms = [

            "high protein",

            "high fibre",

            "high fiber",

            "low sugar",

            "no added sugar",

            "sugar free",

            "low fat",

            "fat free",

            "low sodium",

            "organic",

            "natural",

            "whole grain",

            "whole wheat",

            "rich in protein",

            "rich in fibre",

            "rich in fiber"
        ]

        lower = text.lower()

        found = []

        for term in terms:

            if term in lower:

                found.append(
                    term.title()
                )

        return found

    # ========================================================
    # NUTRITION
    # ========================================================

    def _extract_nutrition(
        self,
        soup,
        products,
        text,
        names
    ):

        # JSON-LD
        for product in products:

            nutrition = (
                product.get(
                    "nutrition"
                )
                or
                product.get(
                    "nutritionInformation"
                )
            )

            if isinstance(
                nutrition,
                dict
            ):

                for name in names:

                    value = (
                        self._json_nutrition_value(
                            nutrition,
                            name
                        )
                    )

                    if value is not None:

                        return value

        # HTML tables
        for row in soup.find_all(
            "tr"
        ):

            cells = row.find_all(
                ["th", "td"]
            )

            if len(cells) < 2:
                continue

            label = cells[0].get_text(
                " ",
                strip=True
            ).lower()

            value = cells[1].get_text(
                " ",
                strip=True
            )

            if not any(
                name.lower()
                in label
                for name in names
            ):
                continue

            number = re.search(
                r"([0-9]+(?:\.[0-9]+)?)",
                value
            )

            if number:

                return float(
                    number.group(1)
                )

        # Visible text
        for name in names:

            pattern = (
                rf"{re.escape(name)}"
                rf"\s*[:\-]?\s*"
                rf"([0-9]+(?:\.[0-9]+)?)"
                rf"\s*(kcal|g|mg)?"
            )

            match = re.search(
                pattern,
                text,
                flags=re.I
            )

            if match:

                try:

                    return float(
                        match.group(1)
                    )

                except ValueError:

                    pass

        return 0.0

    def _json_nutrition_value(
        self,
        nutrition,
        name
    ):

        mapping = {

            "calories": [
                "calories",
                "energy"
            ],

            "sugar": [
                "sugarContent",
                "sugar"
            ],

            "sugars": [
                "sugarContent",
                "sugar"
            ],

            "total fat": [
                "fatContent",
                "fat"
            ],

            "fat": [
                "fatContent",
                "fat"
            ],

            "saturated fat": [
                "saturatedFatContent",
                "saturatedFat"
            ],

            "saturates": [
                "saturatedFatContent",
                "saturatedFat"
            ],

            "protein": [
                "proteinContent",
                "protein"
            ],

            "fibre": [
                "fiberContent",
                "fibreContent",
                "fibre",
                "fiber"
            ],

            "fiber": [
                "fiberContent",
                "fibreContent",
                "fibre",
                "fiber"
            ],

            "salt": [
                "saltContent",
                "salt"
            ]
        }

        keys = mapping.get(
            name.lower(),
            []
        )

        for key in keys:

            value = nutrition.get(
                key
            )

            if value is None:
                continue

            if isinstance(
                value,
                (int, float)
            ):

                return float(
                    value
                )

            match = re.search(
                r"[0-9]+(?:\.[0-9]+)?",
                str(value)
            )

            if match:

                return float(
                    match.group()
                )

        return None

    # ========================================================
    # IMAGE
    # ========================================================

    def _extract_image(
        self,
        soup,
        products,
        response
    ):

        for product in products:

            image = product.get(
                "image"
            )

            if isinstance(
                image,
                list
            ):

                if image:
                    image = image[0]

            if isinstance(
                image,
                dict
            ):

                image = image.get(
                    "url"
                )

            if image:

                return response.urljoin(
                    str(image)
                )

        meta = soup.find(
            "meta",
            {
                "property":
                    "og:image"
            }
        )

        if meta and meta.get(
            "content"
        ):

            return response.urljoin(
                meta.get(
                    "content"
                )
            )

        return ""

    # ========================================================
    # DOMAIN
    # ========================================================

    def _same_domain(
        self,
        url
    ):

        hostname = urlparse(
            url
        ).hostname

        if not hostname:
            return False

        if not self.allowed_domains:
            return True

        allowed = (
            self.allowed_domains[0]
        )

        return (
            hostname == allowed
            or
            hostname.endswith(
                "." + allowed
            )
        )