"""
pdf_fetcher.py
--------------
Downloads academic paper PDFs from arXiv and extracts clean text.
Responsibilities:
  - Download PDFs from arXiv PDF URLs
  - Extract and clean text using PyMuPDF
  - Cache results to disk to avoid redundant downloads
  - Truncate text to a configurable token limit for the LLM context window
"""

import os
import re
import time
import logging
import hashlib
import requests
import fitz  # PyMuPDF

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pdf_fetcher")

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_DIR        = "cache/pdfs"
MAX_CHARS        = 12000   # ~3000 tokens — covers abstract + intro + conclusion
REQUEST_TIMEOUT  = 30      # seconds
RETRY_ATTEMPTS   = 3
RETRY_DELAY      = 5       # seconds between retries


# ── Cache Helpers ──────────────────────────────────────────────────────────────

def _get_cache_path(pdf_url: str) -> str:
    """Generate a unique cache file path from the PDF URL."""
    url_hash = hashlib.md5(pdf_url.encode()).hexdigest()
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{url_hash}.txt")


def _is_cached(pdf_url: str) -> bool:
    """Check if text for this URL is already cached on disk."""
    return os.path.exists(_get_cache_path(pdf_url))


def _load_cache(pdf_url: str) -> str:
    """Load cached text from disk."""
    with open(_get_cache_path(pdf_url), "r", encoding="utf-8") as f:
        return f.read()


def _save_cache(pdf_url: str, text: str) -> None:
    """Save extracted text to disk cache."""
    with open(_get_cache_path(pdf_url), "w", encoding="utf-8") as f:
        f.write(text)


# ── Text Cleaning ──────────────────────────────────────────────────────────────

def _clean_text(raw: str) -> str:
    """
    Clean raw PDF text for LLM consumption.
    - Remove excessive whitespace and newlines
    - Remove page headers/footers (short lines of all caps)
    - Collapse multiple spaces
    """
    lines = raw.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines, very short lines (page numbers, headers)
        if not stripped or len(stripped) < 4:
            continue
        # Skip lines that are purely numeric (page numbers)
        if stripped.isdigit():
            continue
        cleaned.append(stripped)

    text = " ".join(cleaned)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ── Core Fetch Function ────────────────────────────────────────────────────────

def fetch_pdf_text(pdf_url: str, max_chars: int = MAX_CHARS) -> str | None:
    """
    Download a PDF from the given URL and extract clean text.
    Returns cached result if available.

    Args:
        pdf_url:   Direct URL to the PDF (e.g. from arXiv search results).
        max_chars: Maximum characters to return (truncates from end).

    Returns:
        Extracted text string, or None if download/extraction fails.

    Example:
        >>> text = fetch_pdf_text("https://arxiv.org/pdf/2401.12345")
        >>> print(text[:200])
    """
    if not pdf_url:
        logger.warning("Empty PDF URL provided.")
        return None

    # Return cached result if available
    if _is_cached(pdf_url):
        logger.info(f"Cache hit: {pdf_url}")
        text = _load_cache(pdf_url)
        return text[:max_chars]

    logger.info(f"Fetching PDF: {pdf_url}")

    # Download PDF with retries
    pdf_bytes = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = requests.get(
                pdf_url,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "ARA-ResearchAssistant/1.0"},
            )
            response.raise_for_status()
            pdf_bytes = response.content
            logger.info(f"Downloaded PDF ({len(pdf_bytes) / 1024:.1f} KB): {pdf_url}")
            break
        except requests.exceptions.Timeout:
            logger.warning(f"Attempt {attempt}/{RETRY_ATTEMPTS} timed out.")
        except requests.exceptions.HTTPError as e:
            logger.warning(f"Attempt {attempt}/{RETRY_ATTEMPTS} HTTP error: {e}")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Attempt {attempt}/{RETRY_ATTEMPTS} connection error: {e}")

        if attempt < RETRY_ATTEMPTS:
            logger.info(f"Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
        else:
            logger.error(f"All {RETRY_ATTEMPTS} attempts failed for: {pdf_url}")
            return None

    if not pdf_bytes:
        return None

    # Extract text with PyMuPDF
    try:
        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        raw  = ""
        for page in doc:
            raw += page.get_text()
        doc.close()
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed for {pdf_url}: {e}")
        return None

    if not raw.strip():
        logger.warning(f"Extracted empty text from: {pdf_url}")
        return None

    # Clean and truncate
    text = _clean_text(raw)
    text = text[:max_chars]

    # Save to cache
    _save_cache(pdf_url, text)
    logger.info(f"Cached extracted text ({len(text)} chars): {pdf_url}")

    return text


def fetch_papers_text(papers: list[dict], max_chars: int = MAX_CHARS) -> list[dict]:
    """
    Fetch and extract text for a list of paper dicts (from arxiv_search).
    Adds a 'text' key to each paper dict in place.
    Skips papers where extraction fails and logs a warning.

    Args:
        papers:    List of paper dicts from arxiv_search.search_papers()
        max_chars: Max characters per paper.

    Returns:
        List of paper dicts with 'text' key added.
        Papers where extraction failed will have 'text': None.
    """
    logger.info(f"Fetching text for {len(papers)} papers...")
    success = 0

    for i, paper in enumerate(papers, 1):
        pdf_url = paper.get("pdf_url")
        logger.info(f"[{i}/{len(papers)}] {paper.get('title', 'Unknown')[:60]}...")

        text = fetch_pdf_text(pdf_url, max_chars=max_chars)
        paper["text"] = text

        if text:
            success += 1
        else:
            logger.warning(f"Failed to extract text for: {paper.get('title')}")

        # Polite delay between requests to avoid hammering arXiv
        if i < len(papers):
            time.sleep(1)

    logger.info(f"Successfully extracted text for {success}/{len(papers)} papers.")
    return papers


# ── Quick test harness ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from tools.arxiv_search import search_papers

    print("=" * 60)
    print("ARA — PDF Fetcher Test Harness")
    print("=" * 60)

    # Search for 3 papers
    papers = search_papers("transformer pruning efficient inference", max_results=3)

    if not papers:
        print("No papers found. Check your internet connection.")
        exit(1)

    # Fetch text for all papers
    papers = fetch_papers_text(papers)

    # Print results
    for i, paper in enumerate(papers, 1):
        print(f"\n[{i}] {paper['title']}")
        print(f"    PDF URL : {paper['pdf_url']}")
        if paper["text"]:
            print(f"    Chars   : {len(paper['text'])}")
            print(f"    Preview : {paper['text'][:200]}...")
        else:
            print("    Status  : FAILED — text extraction unsuccessful")

    print("\n" + "=" * 60)
    print(f"Cache saved to: {os.path.abspath(CACHE_DIR)}")