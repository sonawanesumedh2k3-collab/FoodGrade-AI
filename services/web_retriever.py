import json
from pathlib import Path
from urllib.parse import urlparse
import requests
from services.web_product_extractor import extract_product_from_html

BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "data" / "web"
USER_AGENT = "FoodGradeAI/4.0 (educational project; contact: project-local)"


def validate_url(url):
    parsed = urlparse((url or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def fetch_product_url(url, timeout=15):
    url = (url or "").strip()
    if not validate_url(url):
        raise ValueError("Please enter a valid http:// or https:// URL.")
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "html" not in content_type.lower() and not response.text.lstrip().startswith("<"):
        raise ValueError("The URL did not return an HTML page.")
    result = extract_product_from_html(response.url, response.text, response.headers)
    result["http_status"] = response.status_code
    result["content_type"] = content_type
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    stamp = __import__("datetime").datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe = __import__("re").sub(r"[^a-zA-Z0-9_-]", "_", urlparse(response.url).netloc)[:60]
    html_path = WEB_DIR / f"retrieved_{stamp}_{safe}.html"
    json_path = WEB_DIR / f"retrieved_{stamp}_{safe}.json"
    html_path.write_text(response.text, encoding="utf-8", errors="ignore")
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["saved_html"] = str(html_path.relative_to(BASE_DIR))
    result["saved_json"] = str(json_path.relative_to(BASE_DIR))
    return result
