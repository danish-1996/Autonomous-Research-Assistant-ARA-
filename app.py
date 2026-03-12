"""
app.py
------
Gradio entry point for the Autonomous Research Assistant (ARA).
Provides a clean interface with:
  - Query input with paper count selector
  - Step-by-step progress updates
  - Full report display with download button
"""

import os
import gradio as gr

from tools.arxiv_search  import search_papers
from tools.pdf_fetcher   import fetch_papers_text
from agent.summarizer    import summarize_paper
from agent.clusterer     import cluster_papers
from agent.gap_analyzer  import analyze_gaps
from agent.report_gen    import generate_report

# ── CSS ────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0f1117;
    --bg-panel:  #161b27;
    --bg-card:   #1c2333;
    --border:    #2a3244;
    --accent:    #4f8ef7;
    --green:     #3ecf8e;
    --text:      #e2e8f0;
    --text-dim:  #64748b;
    --mono:      'JetBrains Mono', monospace;
    --sans:      'Inter', sans-serif;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: var(--sans) !important;
    color: var(--text) !important;
}

#ara-header {
    padding: 28px 0 16px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
#ara-header h1 {
    font-family: var(--mono) !important;
    font-size: 1.5rem !important;
    color: var(--accent) !important;
    letter-spacing: 0.1em;
    margin: 0 0 6px 0;
}
#ara-header p {
    color: var(--text-dim);
    font-size: 0.85rem;
    margin: 0;
}

#query-input textarea {
    font-family: var(--sans) !important;
    font-size: 0.95rem !important;
    background: var(--bg-card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
#query-input textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(79,142,247,0.15) !important;
}

#run-btn {
    font-family: var(--mono) !important;
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.05em !important;
}
#run-btn:hover { background: #3b7de8 !important; }

#clear-btn {
    font-family: var(--mono) !important;
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-dim) !important;
    border-radius: 6px !important;
    font-size: 0.88rem !important;
}

#progress-box textarea {
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    background: var(--bg-card) !important;
    color: var(--green) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    line-height: 1.7 !important;
}

#report-out {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 16px !important;
    font-size: 0.88rem !important;
    color: var(--text) !important;
}
#report-out h1, #report-out h2, #report-out h3,
#report-out h4, #report-out p, #report-out li,
#report-out strong, #report-out em, #report-out blockquote {
    color: var(--text) !important;
}
#report-out h1 { color: var(--accent) !important; }
#report-out h2 { color: var(--accent) !important; }
#report-out h3 { color: #93c5fd !important; }
#report-out strong { color: #fff !important; }
#report-out blockquote { color: var(--text-dim) !important; }
#report-out code {
    background: #0f1117 !important;
    color: var(--green) !important;
    padding: 1px 5px !important;
    border-radius: 3px !important;
}
"""

# ── Pipeline Runner ────────────────────────────────────────────────────────────

def run_pipeline(query: str, max_results: int):
    """
    Run the full ARA pipeline yielding progress updates at each step.
    """
    if not query or not query.strip():
        yield "⚠️  Please enter a research topic.", "*No report generated.*", None
        return

    log = ""

    def step(msg):
        nonlocal log
        log += f"{msg}\n"
        return log

    # Step 1 — Search
    yield step(f"🔍  Searching arXiv for: '{query}'..."), "*Working...*", None
    papers = search_papers(query, max_results=int(max_results))
    if not papers:
        yield step("❌  No papers found. Try a different query."), "*No results.*", None
        return
    yield step(f"✅  Found {len(papers)} papers.\n"), "*Working...*", None

    # Step 2 — Fetch PDFs
    yield step("📄  Fetching and extracting PDF text..."), "*Working...*", None
    papers = fetch_papers_text(papers)
    fetched = sum(1 for p in papers if p.get("text"))
    yield step(f"✅  Extracted text from {fetched}/{len(papers)} papers.\n"), "*Working...*", None

    # Step 3 — Summarize paper by paper
    yield step("🧠  Summarizing papers with LLM..."), "*Working...*", None
    for i, paper in enumerate(papers, 1):
        yield step(f"     [{i}/{len(papers)}] {paper.get('title', '')[:55]}..."), "*Working...*", None
        paper["summary"] = summarize_paper(paper)

    success = sum(1 for p in papers if not p.get("summary", {}).get("_failed"))
    yield step(f"✅  Summarized {success}/{len(papers)} papers.\n"), "*Working...*", None

    # Step 4 — Cluster
    yield step("🗂️   Clustering papers by theme..."), "*Working...*", None
    clusters = cluster_papers(papers)
    yield step(f"✅  Identified {len(clusters)} thematic clusters.\n"), "*Working...*", None

    # Step 5 — Gap Analysis
    yield step("🔬  Analyzing research gaps..."), "*Working...*", None
    gaps = analyze_gaps(papers)
    n_gaps = len(gaps.get("gaps", []))
    yield step(f"✅  Found {n_gaps} research gaps.\n"), "*Working...*", None

    # Step 6 — Generate Report
    yield step("📝  Generating literature review report..."), "*Working...*", None
    report = generate_report(query, papers, clusters, gaps)

    # Find saved report file for download
    safe_query  = "_".join(query.lower().split())[:50]
    outputs_dir = "outputs"
    report_path = None
    try:
        saved = sorted(
            [f for f in os.listdir(outputs_dir) if f.startswith(f"report_{safe_query}")],
            reverse=True,
        )
        if saved:
            report_path = os.path.join(outputs_dir, saved[0])
    except Exception:
        pass

    yield step("✅  Done! Report saved to outputs/\n"), report, report_path


def clear_all():
    return "", "", "*Run the pipeline to generate a literature review...*", None


# ── Build UI ───────────────────────────────────────────────────────────────────

with gr.Blocks(title="ARA — Autonomous Research Assistant") as demo:

    gr.HTML("""
        <div id="ara-header">
            <h1>◈ AUTONOMOUS RESEARCH ASSISTANT</h1>
            <p>Search arXiv → Extract PDFs → Summarize → Cluster → Identify Gaps → Generate Report</p>
        </div>
    """)

    with gr.Row():

        # ── Left: Input + Progress ─────────────────────────────────────────────
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Research Topic",
                placeholder='e.g. "energy-efficient LLM inference on edge devices"',
                lines=2,
                elem_id="query-input",
            )
            max_results = gr.Slider(
                minimum=3,
                maximum=15,
                value=6,
                step=1,
                label="Number of papers to analyze",
            )
            with gr.Row():
                run_btn   = gr.Button("[ RUN PIPELINE ]", elem_id="run-btn",   scale=3)
                clear_btn = gr.Button("[ CLEAR ]",        elem_id="clear-btn", scale=1)

            progress = gr.Textbox(
                label="Pipeline Progress",
                lines=14,
                interactive=False,
                elem_id="progress-box",
            )

        # ── Right: Report Output ───────────────────────────────────────────────
        with gr.Column(scale=3):
            report_out = gr.Markdown(
                value="*Run the pipeline to generate a literature review...*",
                elem_id="report-out",
            )
            download_btn = gr.File(
                label="Download Report (.md)",
                interactive=False,
            )

    # ── Event Wiring ───────────────────────────────────────────────────────────

    run_btn.click(
        fn=run_pipeline,
        inputs=[query_input, max_results],
        outputs=[progress, report_out, download_btn],
    )

    clear_btn.click(
        fn=clear_all,
        outputs=[query_input, progress, report_out, download_btn],
    )


# ── Launch ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        css=CSS,
    )