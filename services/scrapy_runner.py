import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


BASE_DIR = Path(__file__).resolve().parents[1]

SPIDER = (
    BASE_DIR
    / "crawler"
    / "foodgrade_crawler"
    / "spiders"
    / "product_spider.py"
)

OUTPUT_DIR = BASE_DIR / "data" / "web"


def run_crawler(start_url, max_pages=10, delay=1.0):

    parsed = urlparse(start_url)

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Please enter a valid start URL.")

    hostname = (parsed.hostname or "").lower()
    if "openfoodfacts.org" in hostname:
        raise ValueError(
            "Crawling Open Food Facts with Scrapy is disabled to respect site terms and rate limits. "
            "Please use the permitted Data Center / Open Food Facts API import feature instead."
        )

    # Basic SSRF prevention on crawler start URL
    if hostname in {"localhost", "127.0.0.1", "0.0.0.0", "::1"} or "metadata" in hostname:
        raise ValueError("Crawling localhost or internal metadata targets is prohibited (SSRF protection).")

    max_pages = max(1, min(int(max_pages), 50))
    delay = max(0.0, min(float(delay), 10.0))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = OUTPUT_DIR / "crawler_latest.json"

    if output.exists():
        output.unlink()

    cmd = [
        sys.executable,
        "-m",
        "scrapy",
        "runspider",
        str(SPIDER),

        "-a",
        f"start_url={start_url}",

        "-a",
        f"max_pages={max_pages}",

        # Request delay
        "-s",
        f"DOWNLOAD_DELAY={delay}",

        # Respect robots.txt
        "-s",
        "ROBOTSTXT_OBEY=True",

        # Safety limits
        "-s",
        f"CLOSESPIDER_ITEMCOUNT={max_pages}",

        "-s",
        f"CLOSESPIDER_PAGECOUNT={max_pages}",

        "-s",
        "CLOSESPIDER_TIMEOUT=30",

        "-s",
        "DOWNLOAD_TIMEOUT=15",

        # Keep the crawler controlled
        "-s",
        "CONCURRENT_REQUESTS=4",

        "-s",
        "LOG_LEVEL=INFO",

        # Export results
        "-o",
        str(output),
    ]

    proc = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        timeout=45
    )

    log = (
        (proc.stdout or "")
        + "\n"
        + (proc.stderr or "")
    )

    if proc.returncode != 0:
        raise RuntimeError(
            (log or "Scrapy crawler failed.")[-5000:]
        )

    rows = []

    if output.exists():

        content = output.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()

        if content:

            # Normal Scrapy JSON export
            try:
                data = json.loads(content)

                if isinstance(data, list):
                    rows = data

                elif isinstance(data, dict):
                    rows = [data]

            except json.JSONDecodeError:

                # Fallback for JSON Lines
                for line in content.splitlines():

                    if not line.strip():
                        continue

                    try:
                        rows.append(
                            json.loads(line)
                        )
                    except json.JSONDecodeError:
                        continue

    return {
        "count": len(rows),
        "items": rows,
        "output": str(
            output.relative_to(BASE_DIR)
        ),
        "log": log[-5000:]
    }