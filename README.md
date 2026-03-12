# Autonomous Research Assistant (ARA)

Give it a research topic. Get back a full literature review — powered by a locally hosted LLM.

---

## Overview

ARA is an AI-powered agent that automates academic literature review. Type a research topic — ARA autonomously searches arXiv, fetches and parses relevant papers, summarizes each one using a locally-hosted LLM, clusters them by theme, identifies research gaps, and produces a structured Markdown report.

Fully local. No API keys. No data leaves your machine.

---

## How It Works

```
User: "energy-efficient LLM inference on edge devices"

  Step 1 → arXiv Search    : Finds 10 relevant papers
  Step 2 → PDF Fetcher     : Downloads and extracts text from each paper
  Step 3 → LLM Summarizer  : Generates structured JSON summary per paper
  Step 4 → Clusterer       : Groups papers into themes
  Step 5 → Gap Analyzer    : Identifies open research questions
  Step 6 → Report Output   : Saves full literature review to outputs/
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| LLM Backend | Ollama + `llama3:8b` | Local summarization, gap analysis, report writing |
| Paper Source | arXiv API (`arxiv.py`) | Search and retrieve academic papers |
| PDF Parsing | PyMuPDF (`fitz`) | Extract clean text from paper PDFs |
| Clustering | scikit-learn TF-IDF | Group papers by topic similarity |
| UI | Gradio `gr.Blocks` | Web interface for input and report display |
| Caching | `diskcache` | Avoid re-fetching already processed papers |
| Output | Markdown + JSON | Human-readable report and structured summaries |

---

## Project Structure

```
autonomous-research-assistant/
├── app.py                    # Gradio entry point
├── agent/
│   ├── __init__.py
│   ├── orchestrator.py       # Pipeline manager
│   ├── summarizer.py         # Per-paper LLM summarization
│   ├── clusterer.py          # Topic clustering
│   ├── gap_analyzer.py       # Research gap detection
│   └── report_gen.py         # Final report assembly
├── tools/
│   ├── __init__.py
│   ├── arxiv_search.py       # arXiv API wrapper
│   └── pdf_fetcher.py        # PDF download + text extraction
├── prompts/
│   ├── summarize.txt         # Paper summarization prompt
│   ├── gap_analysis.txt      # Research gap detection prompt
│   └── report.txt            # Report generation prompt
├── outputs/                  # Generated reports saved here
├── cache/                    # Cached PDFs and summaries
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Prerequisites

1. Python 3.12+
2. Ollama with `llama3:8b` pulled:
   ```bash
   ollama pull llama3:8b
   ```

---

## Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/autonomous-research-assistant.git
cd autonomous-research-assistant

# 2. Create and activate a virtual environment
python -m venv venv
source venv/Scripts/activate    # Windows (Git Bash)
source venv/bin/activate        # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make sure Ollama is running
ollama serve

# 5. Launch the app
python app.py
```

Open your browser at `http://localhost:7860`

---

## Error Handling

| Scenario | Behavior |
|---|---|
| PDF download fails | Skip paper, log warning, continue with rest |
| LLM returns malformed JSON | Retry up to 2 times with stricter prompt |
| arXiv rate limit hit | Exponential backoff, max 3 retries |
| Paper exceeds context window | Truncate to first 3000 tokens |
| Query returns no results | Friendly UI error with suggestions |
| Repeated query | Served instantly from disk cache |

---

## License

[MIT](LICENSE)
