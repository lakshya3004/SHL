#!/usr/bin/env python3
"""
SHL Catalog Scraper - Fetches the live catalog using the SHL API/search endpoint.
Run this once to populate data/catalog_seed.json
"""
import sys
import os
import json
import time
import re
import requests
from bs4 import BeautifulSoup
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

TEST_TYPE_MAP = {
    "A": "Ability",
    "B": "Behavioral",
    "C": "Competency",
    "D": "Development",
    "E": "Exercise",
    "K": "Knowledge",
    "M": "Motivation",
    "P": "Personality",
    "S": "Simulation",
}


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def get_page(url, session, retries=3):
    for attempt in range(retries):
        try:
            time.sleep(1.5 + attempt)
            r = session.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.text
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {url}: {e}")
    return None


def parse_catalog_page(html):
    """Parse one page of the SHL catalog table."""
    soup = BeautifulSoup(html, "html.parser")
    assessments = []

    # The SHL catalog renders as a table with specific structure
    # Try multiple selectors
    rows = soup.select("table tbody tr")
    if not rows:
        rows = soup.select(".product-catalogue__row")
    if not rows:
        # Try the JS-rendered data in script tags
        scripts = soup.find_all("script", type="application/json")
        for s in scripts:
            try:
                data = json.loads(s.string)
                # Check if it looks like catalog data
                if isinstance(data, list) and data and "name" in str(data[0]):
                    return data
            except Exception:
                pass

    for row in rows:
        try:
            link = row.find("a", href=True)
            if not link:
                continue
            name = link.get_text(strip=True)
            href = link["href"]
            url = href if href.startswith("http") else BASE_URL + href

            cols = row.find_all("td")
            remote = False
            adaptive = False
            test_types_raw = ""

            if len(cols) >= 4:
                col_texts = [c.get_text(strip=True) for c in cols]
                remote = any("yes" in t.lower() for t in col_texts[1:3])
                adaptive = any("yes" in t.lower() for t in col_texts[2:4])
                test_types_raw = col_texts[-1] if col_texts else ""

            # Map type codes
            type_codes = [c for c in test_types_raw.split() if c in TEST_TYPE_MAP]
            test_type = TEST_TYPE_MAP.get(type_codes[0], "General Assessment") if type_codes else "General Assessment"

            assessments.append({
                "id": slugify(name),
                "name": name,
                "url": url,
                "test_type": test_type,
                "test_type_codes": type_codes,
                "remote_support": remote,
                "adaptive": adaptive,
                "description": f"SHL {name} assessment.",
                "keywords": [],
            })
        except Exception as e:
            logger.warning(f"Row parse error: {e}")

    return assessments


def scrape_all_pages(max_pages=20):
    session = requests.Session()
    all_items = []
    
    # Try with filter for Individual Test Solutions
    # SHL uses URL params for filtering
    filter_urls = [
        CATALOG_URL + "?type=1",   # Individual Test Solutions filter
        CATALOG_URL,
    ]

    for base in filter_urls:
        for page in range(1, max_pages + 1):
            if page == 1:
                url = base
            else:
                sep = "&" if "?" in base else "?"
                url = f"{base}{sep}start={(page-1)*12}"  # SHL uses 12 per page

            logger.info(f"Fetching page {page}: {url}")
            html = get_page(url, session)
            if not html:
                break

            items = parse_catalog_page(html)
            if not items:
                logger.info(f"No items on page {page}, stopping.")
                break
            
            all_items.extend(items)
            logger.info(f"Found {len(items)} items on page {page}. Total: {len(all_items)}")

        if all_items:
            break

    return all_items


def deduplicate(items):
    seen_ids = set()
    result = []
    for item in items:
        if item["id"] not in seen_ids:
            seen_ids.add(item["id"])
            result.append(item)
    return result


def main():
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "catalog_seed.json"
    )

    logger.info("Starting SHL catalog scrape...")
    items = scrape_all_pages(max_pages=15)
    items = deduplicate(items)

    if not items:
        logger.error("No items scraped! Check the catalog URL and HTML structure.")
        logger.info("Using embedded fallback catalog instead...")
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    logger.success(f"Saved {len(items)} assessments to {output_path}")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
