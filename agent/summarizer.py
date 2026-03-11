"""
summarizer.py
-------------
Summarizes academic papers using a locally hosted Ollama LLM.
Responsibilities:
  - Load the summarization prompt from disk
  - Send paper text to Ollama and parse structured JSON response
  - Validate the returned JSON schema
  - Cache summaries to disk to avoid redundant LLM calls
  - Handle malformed output with up to 2 retries
"""

import os
import json
import time
import logging
import hashlib
import requests
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("summarizer")

# ── Configuration ──────────────────────────────────────────────────────────────
OLLAMA_URL     = "http://localhost:11434/api/chat"
MODEL_NAME     = "llama3:8b"
REQUEST_TIMEOUT = 240
MAX_RETRIES    = 2

PROMPT_PATH    = Path(__file__).parent.parent / "prompts" / "summarize.txt"
CACHE_DIR      = "cache/summaries"

REQUIRED_KEYS  = {"title", "authors", "year", "problem", "method", "results", "limitations", "keywords"}

# ── Fallback summary returned when all retries fail ───────────────────────────
def _fallback_summary(paper: dict) -> dict:
    return {
        "title":       paper.get("title", "Unknown"),
        "authors":     paper.get("authors", "Unknown"),
        "year":        paper.get("published", "Unknown")[:4],
        "problem":     "Could not extract — LLM summarization failed.",
        "method":      "Not available.",
        "results":     "Not available.",
        "limitations": "Not available.",
        "keywords":    [],
        "_failed":     True,
    }


# ── Cache Helpers ──────────────────────────────────────────────────────────────

def _cache_path(paper_id: str) -> str:
    safe_id = hashlib.md5(paper_id.encode()).hexdigest()
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{safe_id}.json")


def _is_cached(paper_id: str) -> bool:
    return os.path.exists(_cache_path(paper_id))


def _load_cache(paper_id: str) -> dict:
    with open(_cache_path(paper_id), "r", encoding="utf-8") as f:
        return json.load(f)


def _save_cache(paper_id: str, summary: dict) -> None:
    with open(_cache_path(paper_id), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# ── Prompt Loader ──────────────────────────────────────────────────────────────

def _load_prompt() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


# ── JSON Validation ────────────────────────────────────────────────────────────

def _validate(data: dict) -> dict:
    """
    Validate and coerce the LLM JSON output to the expected schema.
    Raises ValueError if required keys are missing or types are wrong.
    """
    missing = REQUIRED_KEYS - data.keys()
    if missing:
        raise ValueError(f"Missing keys: {missing}")

    # Coerce keywords to list if LLM returned a string
    if isinstance(data["keywords"], str):
        data["keywords"] = [k.strip() for k in data["keywords"].split(",")]

    # Ensure year is a string
    data["year"] = str(data["year"])

    return data


# ── Core Summarize Function ────────────────────────────────────────────────────

def summarize_paper(paper: dict) -> dict:
    """
    Summarize a single paper using the local Ollama LLM.
    Returns a structured summary dict. Falls back gracefully on failure.

    Args:
        paper: A paper dict from arxiv_search / pdf_fetcher with a 'text' key.

    Returns:
        Summary dict with keys: title, authors, year, problem, method,
        results, limitations, keywords. Plus '_failed': True on failure.

    Example:
        >>> summary = summarize_paper(paper)
        >>> print(summary["problem"])
    """
    paper_id = paper.get("id", paper.get("title", "unknown"))

    # Return cached summary if available
    if _is_cached(paper_id):
        logger.info(f"Cache hit (summary): {paper_id}")
        return _load_cache(paper_id)

    # Skip papers where text extraction failed
    text = paper.get("text")
    if not text:
        logger.warning(f"No text available for: {paper.get('title')} — using fallback.")
        return _fallback_summary(paper)

    # Load system prompt
    try:
        system_prompt = _load_prompt()
    except FileNotFoundError as e:
        logger.error(e)
        return _fallback_summary(paper)

    # Build user message — include title for context
    user_message = (
        f"Paper title: {paper.get('title', 'Unknown')}\n\n"
        f"Paper text:\n{text[:6000]}"
    )

    logger.info(f"Summarizing: '{paper.get('title', paper_id)[:60]}...'")

    # Query Ollama with retries
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model":   MODEL_NAME,
                    "stream":  False,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_message},
                    ],
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            raw_text = response.json()["message"]["content"].strip()
            logger.info(f"Raw LLM output (attempt {attempt}):\n{raw_text}")

        except requests.exceptions.Timeout:
            logger.warning(f"Attempt {attempt} timed out.")
            if attempt <= MAX_RETRIES:
                time.sleep(3)
                continue
            return _fallback_summary(paper)

        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Is it running? Try: ollama serve")
            return _fallback_summary(paper)

        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt}: {e}")
            return _fallback_summary(paper)

        # Strip markdown fences if present
        if "```" in raw_text:
            lines = raw_text.splitlines()
            raw_text = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            ).strip()

        # Parse JSON
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSONDecodeError on attempt {attempt}: {e}")
            if attempt <= MAX_RETRIES:
                time.sleep(2)
                continue
            return _fallback_summary(paper)

        # Validate schema
        try:
            summary = _validate(parsed)
            _save_cache(paper_id, summary)
            logger.info(f"Summary complete: '{summary['title'][:60]}'")
            return summary
        except ValueError as e:
            logger.warning(f"Schema validation failed on attempt {attempt}: {e}")
            if attempt <= MAX_RETRIES:
                time.sleep(2)
                continue
            return _fallback_summary(paper)

    return _fallback_summary(paper)


def summarize_papers(papers: list[dict]) -> list[dict]:
    """
    Summarize a list of papers. Adds a 'summary' key to each paper dict.

    Args:
        papers: List of paper dicts with 'text' key from pdf_fetcher.

    Returns:
        Same list with 'summary' key added to each paper.
    """
    logger.info(f"Summarizing {len(papers)} papers...")
    success = 0

    for i, paper in enumerate(papers, 1):
        logger.info(f"[{i}/{len(papers)}] Processing...")
        summary = summarize_paper(paper)
        paper["summary"] = summary

        if not summary.get("_failed"):
            success += 1

    logger.info(f"Summarization complete: {success}/{len(papers)} successful.")
    return papers


# ── Quick test harness ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from tools.arxiv_search import search_papers
    from tools.pdf_fetcher  import fetch_papers_text

    print("=" * 60)
    print("ARA — LLM Summarizer Test Harness")
    print("=" * 60)

    # Step 1: Search
    papers = search_papers("transformer pruning efficient inference", max_results=2)
    if not papers:
        print("No papers found.")
        exit(1)

    # Step 2: Fetch PDFs
    papers = fetch_papers_text(papers)

    # Step 3: Summarize
    papers = summarize_papers(papers)

    # Step 4: Print results
    for i, paper in enumerate(papers, 1):
        s = paper["summary"]
        print(f"\n{'='*60}")
        print(f"[{i}] {s['title']}")
        print(f"     Authors    : {s['authors']}")
        print(f"     Year       : {s['year']}")
        print(f"     Problem    : {s['problem']}")
        print(f"     Method     : {s['method']}")
        print(f"     Results    : {s['results']}")
        print(f"     Limitations: {s['limitations']}")
        print(f"     Keywords   : {', '.join(s['keywords'])}")
        if s.get("_failed"):
            print("     *** SUMMARIZATION FAILED — fallback used ***")

    print(f"\nSummaries cached to: {os.path.abspath(CACHE_DIR)}")