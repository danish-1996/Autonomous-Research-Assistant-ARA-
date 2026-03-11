"""
gap_analyzer.py
---------------
Identifies open research questions across a set of paper summaries
using the locally hosted Ollama LLM.
Responsibilities:
  - Build a combined prompt from all paper summaries
  - Query Ollama for research gaps in JSON format
  - Validate and return structured gap list
  - Fallback gracefully if LLM fails
"""

import json
import time
import logging
import requests
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gap_analyzer")

# ── Configuration ──────────────────────────────────────────────────────────────
OLLAMA_URL      = "http://localhost:11434/api/chat"
MODEL_NAME      = "llama3:8b"
REQUEST_TIMEOUT = 240
MAX_RETRIES     = 2

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "gap_analysis.txt"

# ── Fallback ───────────────────────────────────────────────────────────────────
FALLBACK_GAPS = {
    "gaps": [
        {
            "title": "Gap analysis unavailable",
            "description": "The LLM could not complete gap analysis. Try running again.",
            "relevant_papers": [],
        }
    ]
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_prompt() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt not found: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def _build_summaries_text(papers: list[dict]) -> str:
    """
    Concatenate all paper summaries into a single text block
    for the LLM to reason across.
    """
    lines = []
    for i, paper in enumerate(papers, 1):
        s = paper.get("summary", {})
        if s.get("_failed"):
            continue
        lines.append(
            f"Paper {i}: {s.get('title', 'Unknown')}\n"
            f"  Problem    : {s.get('problem', 'N/A')}\n"
            f"  Method     : {s.get('method', 'N/A')}\n"
            f"  Results    : {s.get('results', 'N/A')}\n"
            f"  Limitations: {s.get('limitations', 'N/A')}\n"
            f"  Keywords   : {', '.join(s.get('keywords', []))}\n"
        )
    return "\n".join(lines)


def _validate_gaps(data: dict) -> dict:
    """Validate the gaps JSON structure."""
    if "gaps" not in data:
        raise ValueError("Missing 'gaps' key in response")
    if not isinstance(data["gaps"], list):
        raise ValueError("'gaps' must be a list")
    for gap in data["gaps"]:
        for key in ("title", "description", "relevant_papers"):
            if key not in gap:
                raise ValueError(f"Gap missing key: '{key}'")
        if isinstance(gap["relevant_papers"], str):
            gap["relevant_papers"] = [gap["relevant_papers"]]
    return data


# ── Core Function ──────────────────────────────────────────────────────────────

def analyze_gaps(papers: list[dict]) -> dict:
    """
    Identify research gaps across all paper summaries.

    Args:
        papers: List of paper dicts with 'summary' key from summarizer.

    Returns:
        Dict with 'gaps' key containing a list of gap dicts, each with:
          - title            : Short gap name
          - description      : What is unexplored
          - relevant_papers  : List of paper titles that hint at this gap

    Example:
        >>> gaps = analyze_gaps(papers)
        >>> for gap in gaps["gaps"]:
        ...     print(gap["title"])
    """
    valid = [p for p in papers if p.get("summary") and not p["summary"].get("_failed")]
    if not valid:
        logger.warning("No valid summaries for gap analysis.")
        return FALLBACK_GAPS

    # Load prompt
    try:
        system_prompt = _load_prompt()
    except FileNotFoundError as e:
        logger.error(e)
        return FALLBACK_GAPS

    # Build user message
    summaries_text = _build_summaries_text(valid)
    user_message = (
        f"Here are {len(valid)} paper summaries from a literature search.\n\n"
        f"{summaries_text}\n\n"
        f"Identify the key research gaps across these papers."
    )

    logger.info(f"Analyzing gaps across {len(valid)} papers...")

    # Query Ollama with retries
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model":  MODEL_NAME,
                    "stream": False,
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
            return FALLBACK_GAPS
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Is it running?")
            return FALLBACK_GAPS
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return FALLBACK_GAPS

        # Strip markdown fences
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
            return FALLBACK_GAPS

        # Validate
        try:
            gaps = _validate_gaps(parsed)
            logger.info(f"Found {len(gaps['gaps'])} research gaps.")
            return gaps
        except ValueError as e:
            logger.warning(f"Validation failed on attempt {attempt}: {e}")
            if attempt <= MAX_RETRIES:
                time.sleep(2)
                continue
            return FALLBACK_GAPS

    return FALLBACK_GAPS


def format_gaps(gaps: dict) -> str:
    """
    Format gaps dict into a readable string for logging or display.
    """
    lines = []
    for i, gap in enumerate(gaps.get("gaps", []), 1):
        lines.append(f"\n[Gap {i}] {gap['title']}")
        lines.append(f"  {gap['description']}")
        if gap.get("relevant_papers"):
            lines.append(f"  Related: {', '.join(gap['relevant_papers'])}")
    return "\n".join(lines)


# ── Quick test harness ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from tools.arxiv_search import search_papers
    from tools.pdf_fetcher  import fetch_papers_text
    from agent.summarizer   import summarize_papers

    print("=" * 60)
    print("ARA — Gap Analyzer Test Harness")
    print("=" * 60)

    papers = search_papers("transformer pruning efficient inference", max_results=6)
    papers = fetch_papers_text(papers)
    papers = summarize_papers(papers)

    gaps = analyze_gaps(papers)
    print(format_gaps(gaps))