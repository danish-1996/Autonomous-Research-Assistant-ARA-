"""
arxiv_search.py
---------------
Queries the arXiv API for academic papers matching a search topic.
Responsibilities:
  - Accept a search query string and max results count
  - Fetch paper metadata from the arXiv API
  - Return a clean list of structured paper dicts
  - Handle rate limiting and connection errors gracefully
"""

import time
import logging
import arxiv

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("arxiv_search")

# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_MAX_RESULTS = 10
RETRY_ATTEMPTS      = 3
RETRY_DELAY         = 5  # seconds between retries


# ── Search Function ────────────────────────────────────────────────────────────

def search_papers(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict]:
    """
    Search arXiv for papers matching the query string.

    Args:
        query:       Natural language or keyword search query.
        max_results: Maximum number of papers to return (default 10).

    Returns:
        List of paper dicts, each containing:
          - id          : arXiv paper ID (e.g. "2301.07041")
          - title       : Full paper title
          - authors     : Comma-separated author names
          - abstract    : Full abstract text
          - published   : Publication date (YYYY-MM-DD)
          - pdf_url     : Direct URL to the PDF
          - arxiv_url   : URL to the arXiv abstract page
          - categories  : List of arXiv category tags

    Example:
        >>> papers = search_papers("transformer pruning efficient inference", max_results=5)
        >>> print(papers[0]["title"])
    """
    if not query or not query.strip():
        logger.warning("Empty query provided — returning empty list.")
        return []

    logger.info(f"Searching arXiv for: '{query}' (max {max_results} results)")

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            papers = []
            for result in client.results(search):
                paper = {
                    "id":         result.entry_id.split("/")[-1],
                    "title":      result.title.strip(),
                    "authors":    ", ".join(a.name for a in result.authors),
                    "abstract":   result.summary.strip(),
                    "published":  result.published.strftime("%Y-%m-%d"),
                    "pdf_url":    result.pdf_url,
                    "arxiv_url":  result.entry_id,
                    "categories": result.categories,
                }
                papers.append(paper)

            logger.info(f"Found {len(papers)} papers for query: '{query}'")
            return papers

        except Exception as e:
            logger.warning(f"Attempt {attempt}/{RETRY_ATTEMPTS} failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                logger.info(f"Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"All {RETRY_ATTEMPTS} attempts failed for query: '{query}'")
                return []


def format_paper_summary(paper: dict) -> str:
    """
    Format a paper dict into a human-readable summary string.
    Useful for logging and UI display.

    Args:
        paper: A paper dict returned by search_papers()

    Returns:
        Formatted multi-line string summary.
    """
    return (
        f"Title    : {paper['title']}\n"
        f"Authors  : {paper['authors']}\n"
        f"Published: {paper['published']}\n"
        f"PDF      : {paper['pdf_url']}\n"
        f"Abstract : {paper['abstract'][:200]}...\n"
    )


# ── Quick test harness ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "energy efficient LLM inference edge devices",
        "transformer pruning quantization",
        "this query should return results even if vague",
    ]

    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"Query: '{query}'")
        print("=" * 60)

        papers = search_papers(query, max_results=3)

        if not papers:
            print("No results found.")
            continue

        for i, paper in enumerate(papers, 1):
            print(f"\n[{i}] {format_paper_summary(paper)}")