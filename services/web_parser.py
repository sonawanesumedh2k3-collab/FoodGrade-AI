import json
import re
from bs4 import BeautifulSoup


def soup_from_html(html):
    return BeautifulSoup(html or "", "lxml")


def clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def extract_jsonld(soup):
    records = []
    for tag in soup.find_all("script", attrs={"type": re.compile(r"application/ld\+json", re.I)}):
        raw = tag.string or tag.get_text()
        if not raw:
            continue
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict) and isinstance(data.get("@graph"), list):
                records.extend(data["@graph"])
            else:
                records.append(data)
        except (json.JSONDecodeError, TypeError):
            continue
    return records


def find_product_jsonld(records):
    for record in records:
        if not isinstance(record, dict):
            continue
        typ = record.get("@type", "")
        types = typ if isinstance(typ, list) else [typ]
        if any(str(x).lower() == "product" for x in types):
            return record
    return None


def page_title(soup):
    return clean_text(soup.title.get_text()) if soup.title else ""


def visible_text(soup):
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return clean_text(soup.get_text(" "))
