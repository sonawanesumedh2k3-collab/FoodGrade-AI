import datetime
import ipaddress
import json
from pathlib import Path
import re
import socket
from urllib.parse import urlparse, urljoin
import requests
from services.web_product_extractor import extract_product_from_html

BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "data" / "web"
USER_AGENT = "FoodGradeAI/4.0 (educational research prototype; contact: project-local)"
MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB maximum response size
MAX_REDIRECTS = 5


def validate_url(url: str) -> bool:
    """Basic syntactic validation for http/https URLs."""
    parsed = urlparse((url or "").strip())
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def is_safe_destination(url: str) -> tuple[bool, str]:
    """
    Security check against Server-Side Request Forgery (SSRF):
    - Must be http or https scheme.
    - Hostname must resolve to valid public IP addresses.
    - Prohibits loopback (127.0.0.0/8, ::1), private RFC1918 (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16),
      link-local/cloud metadata (169.254.0.0/16, fe80::/10), broadcast, and multicast addresses.
    """
    if not validate_url(url):
        return False, "Please enter a valid http:// or https:// URL."

    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").strip().lower()

    if not hostname:
        return False, "Invalid URL: missing hostname."

    # Direct name checks
    if hostname in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}:
        return False, "Access to localhost or loopback addresses is prohibited."

    if "metadata" in hostname or hostname.endswith(".internal") or hostname.endswith(".local"):
        return False, "Access to internal or cloud metadata endpoints is prohibited."

    # Resolve IP addresses
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False, f"Cannot resolve hostname '{hostname}'."
    except Exception as exc:
        return False, f"DNS resolution failed: {exc}"

    if not addr_info:
        return False, f"No DNS records found for '{hostname}'."

    for family, _, _, _, sockaddr in addr_info:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
            if ip.is_loopback:
                return False, f"Destination resolves to loopback address ({ip_str})."
            if ip.is_private:
                return False, f"Destination resolves to private network address ({ip_str})."
            if ip.is_link_local:
                return False, f"Destination resolves to link-local address ({ip_str})."
            if ip.is_multicast:
                return False, f"Destination resolves to multicast address ({ip_str})."
            if ip.is_reserved:
                return False, f"Destination resolves to reserved address ({ip_str})."
            if ip_str == "0.0.0.0":
                return False, "Destination resolves to unspecified address."
        except ValueError:
            return False, f"Invalid resolved IP address: {ip_str}."

    return True, ""


def fetch_product_url(url: str, timeout: int = 15) -> dict:
    """
    Fetch a product page safely with:
    - SSRF prevention at start and each redirect hop
    - Timeout protection
    - Response size cap (5 MB)
    - Content-type validation
    """
    current_url = (url or "").strip()
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
        "Accept-Language": "en-US,en;q=0.8",
    }

    session = requests.Session()
    redirect_count = 0

    while True:
        safe, reason = is_safe_destination(current_url)
        if not safe:
            raise ValueError(f"SSRF Protection blocked request: {reason}")

        try:
            response = session.get(
                current_url,
                headers=headers,
                timeout=timeout,
                allow_redirects=False,
                stream=True,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"HTTP request failed: {exc}")

        # Check for redirects manually to inspect destination safely
        if response.is_redirect or response.status_code in {301, 302, 303, 307, 308}:
            redirect_count += 1
            if redirect_count > MAX_REDIRECTS:
                raise ValueError("Exceeded maximum allowed redirects.")
            location = response.headers.get("Location")
            if not location:
                raise ValueError("Redirect response missing Location header.")
            current_url = urljoin(current_url, location)
            continue

        break

    response.raise_for_status()

    # Enforce response size limits
    content_bytes = bytearray()
    for chunk in response.iter_content(chunk_size=32768):
        content_bytes.extend(chunk)
        if len(content_bytes) > MAX_RESPONSE_BYTES:
            raise ValueError("The requested page exceeded the maximum response size limit (5 MB).")

    encoding = response.encoding or "utf-8"
    try:
        html_text = content_bytes.decode(encoding, errors="replace")
    except Exception:
        html_text = content_bytes.decode("utf-8", errors="replace")

    content_type = response.headers.get("Content-Type", "")
    if "html" not in content_type.lower() and not html_text.lstrip().startswith("<"):
        raise ValueError("The URL did not return an HTML document.")

    result = extract_product_from_html(current_url, html_text, response.headers)
    result["http_status"] = response.status_code
    result["content_type"] = content_type

    WEB_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_host = re.sub(r"[^a-zA-Z0-9_-]", "_", urlparse(current_url).netloc)[:60]
    html_path = WEB_DIR / f"retrieved_{stamp}_{safe_host}.html"
    json_path = WEB_DIR / f"retrieved_{stamp}_{safe_host}.json"

    html_path.write_text(html_text, encoding="utf-8", errors="ignore")
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["saved_html"] = str(html_path.relative_to(BASE_DIR))
    result["saved_json"] = str(json_path.relative_to(BASE_DIR))

    return result
