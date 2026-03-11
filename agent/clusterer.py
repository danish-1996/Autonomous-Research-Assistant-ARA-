"""
clusterer.py
------------
Groups papers into thematic clusters using TF-IDF and K-Means.
Responsibilities:
  - Build a TF-IDF matrix from paper keywords, titles, and abstracts
  - Cluster papers into N themes using K-Means
  - Assign a human-readable label to each cluster
  - Return papers annotated with their cluster assignment
"""

import logging
import math

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("clusterer")


# ── Core Clustering Function ───────────────────────────────────────────────────

def cluster_papers(papers: list[dict], n_clusters: int = None) -> dict:
    """
    Group papers into thematic clusters based on their summaries.

    Args:
        papers:     List of paper dicts with 'summary' key from summarizer.
        n_clusters: Number of clusters. Auto-determined if None.

    Returns:
        Dict mapping cluster_label → list of paper dicts in that cluster.

    Example:
        >>> clusters = cluster_papers(papers)
        >>> for label, group in clusters.items():
        ...     print(label, len(group))
    """
    # Filter to papers with valid summaries
    valid = [p for p in papers if p.get("summary") and not p["summary"].get("_failed")]

    if not valid:
        logger.warning("No valid summaries found for clustering.")
        return {"Uncategorized": papers}

    if len(valid) == 1:
        label = _make_label(valid[0]["summary"]["keywords"])
        return {label: valid}

    # Auto-determine cluster count: sqrt(n/2) capped between 2 and 5
    if n_clusters is None:
        n_clusters = min(max(2, math.ceil(math.sqrt(len(valid) / 2))), 5)
        # Can't have more clusters than papers
        n_clusters = min(n_clusters, len(valid))

    logger.info(f"Clustering {len(valid)} papers into {n_clusters} themes...")

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans

        # Build document strings from summary fields
        docs = []
        for p in valid:
            s = p["summary"]
            doc = " ".join([
                s.get("title", ""),
                s.get("problem", ""),
                s.get("method", ""),
                " ".join(s.get("keywords", [])),
            ])
            docs.append(doc)

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        X = vectorizer.fit_transform(docs)

        # K-Means clustering
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        # Get top terms per cluster for labeling
        terms = vectorizer.get_feature_names_out()
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

        clusters = {}
        for cluster_id in range(n_clusters):
            # Top 3 terms from centroid as cluster label
            top_terms = [terms[i] for i in order_centroids[cluster_id, :3]]
            label = " / ".join(t.title() for t in top_terms)

            cluster_papers = [valid[i] for i, l in enumerate(labels) if l == cluster_id]
            clusters[label] = cluster_papers
            logger.info(f"Cluster '{label}': {len(cluster_papers)} papers")

        return clusters

    except ImportError:
        logger.warning("scikit-learn not installed — falling back to keyword clustering.")
        return _keyword_fallback(valid)


def _keyword_fallback(papers: list[dict]) -> dict:
    """
    Simple fallback clustering when scikit-learn is unavailable.
    Groups papers by their first keyword.
    """
    clusters = {}
    for paper in papers:
        keywords = paper["summary"].get("keywords", [])
        label = keywords[0].title() if keywords else "General"
        clusters.setdefault(label, []).append(paper)
    return clusters


def _make_label(keywords: list) -> str:
    """Generate a cluster label from a list of keywords."""
    return " / ".join(k.title() for k in keywords[:3]) if keywords else "General"


def format_clusters(clusters: dict) -> str:
    """
    Format clusters into a readable string for logging or display.

    Args:
        clusters: Dict from cluster_papers()

    Returns:
        Multi-line formatted string.
    """
    lines = []
    for label, group in clusters.items():
        lines.append(f"\nTheme: {label} ({len(group)} papers)")
        for p in group:
            title = p["summary"].get("title", p.get("title", "Unknown"))
            lines.append(f"  - {title[:70]}")
    return "\n".join(lines)


# ── Quick test harness ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from tools.arxiv_search import search_papers
    from tools.pdf_fetcher  import fetch_papers_text
    from agent.summarizer   import summarize_papers

    print("=" * 60)
    print("ARA — Clusterer Test Harness")
    print("=" * 60)

    papers = search_papers("transformer pruning efficient inference", max_results=6)
    papers = fetch_papers_text(papers)
    papers = summarize_papers(papers)

    clusters = cluster_papers(papers)
    print(format_clusters(clusters))