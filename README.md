

https://github.com/user-attachments/assets/09017a0c-095c-464f-a88d-b3e250c8f8fd



# DataScraper Agent

**🔴 Live dashboard:** https://datascraperagent-hwujjbr5i62ysmkviv7ebb.streamlit.app/

A small AI agent that finds remote AI/agent-developer job listings, scores
each one for relevance using Claude, and — if the first pass comes up short —
autonomously decides to broaden its own search and try again.

## Why this is an agent, not a script with a label on it

A plain scraper follows one fixed path: fetch -> filter by keyword -> save.
Every decision is hard-coded by the programmer ahead of time.

This project is structured around the classic agent loop instead:

```
PERCEIVE  ->  REASON  ->  DECIDE  ->  ACT
(scraper.py)  (agent.py: Claude scores)  (agent.py: branch on results)  (agent.py: retry or output)
```

1. **Perceive** (`scraper.py`) — pull raw, unopinionated data from the
   environment (RemoteOK's public API). This layer has zero judgement in it;
   it doesn't know what "relevant" means.
2. **Reason** (`agent.py: score_jobs`) — a cheap keyword net first drops
   obviously-unrelated noise from the raw feed (RemoteOK returns bartenders,
   gardeners, sales reps alongside tech roles — no LLM call is needed to rule
   those out). Whatever survives that recall pass is handed to Claude, which
   *judges* it against a candidate profile: not `if "python" in title`, but
   Claude reading each job's title, tags, and description and producing a
   relevance score (0-10) plus a one-sentence rationale. The actual relevance
   **judgement** is delegated to the model, not encoded in `if/else` branches
   — the keyword net only decides what's worth judging at all.
3. **Decide** (`agent.py: run_agent`) — the agent inspects the *outcome* of
   its own reasoning step and branches on it. If fewer than 3 jobs cleared
   the relevance bar, the agent concludes its search was too narrow and
   decides, on its own, to act again — it doesn't just report "0 results
   found" and stop.
4. **Act** (`agent.py: suggest_broader_keywords` + a second scoring pass) —
   the agent asks Claude to *generate* new search keywords (Claude decides
   what those keywords are; they aren't a hard-coded synonym list), re-filters
   the same raw job pool with them, and re-scores the new candidates. This is
   capped at one retry so the loop can't run away and rack up API cost.

The key difference from a static script: **the control flow depends on what
the LLM concludes**, not just what the LLM outputs. A prompt-wrapper script
would call the model once and print the answer. This agent calls the model,
evaluates whether its own goal ("find >=3 relevant jobs") was met, and takes
a different action depending on the answer — including a decision (which
keywords to search next) that only the LLM makes, not the developer.

## Cost control

- Jobs are **batched** into groups of 20 per Claude call (see `BATCH_SIZE` in
  `agent.py`), instead of one API call per job. A typical RemoteOK pull
  (~100 jobs) costs ~5 calls for the first scoring pass, not ~100.
- Job descriptions are truncated to 600 characters before being sent to
  Claude — enough context to judge relevance, not enough to burn tokens on
  boilerplate HTML.
- The broaden-and-retry loop only re-scores jobs that weren't already
  scored in the first pass, and is hard-capped at **one** retry.
- Structured JSON output (`output_config.format`) is used instead of
  free-form text, so responses are compact and require no retry-on-parse-
  failure logic.

## Project structure

```
datascraper_agent/
  scraper.py                       # Perception — RemoteOK API fetch only, no opinions
  agent.py                          # Reasoning + autonomous decision loop + main entrypoint
  streamlit_app.py                  # Dashboard — visualizes the agent's saved results
  requirements.txt
  .env.example
  .streamlit/secrets.toml.example   # Template for Streamlit Community Cloud secrets
  .github/workflows/daily-scrape.yml  # Daily automated run (see below)
  output/                           # timestamped CSV + JSON results (+ run metadata) land here
```

## Setup

```bash
cd datascraper_agent
pip install -r requirements.txt
cp .env.example .env   # then edit .env and add your real ANTHROPIC_API_KEY
```

## Run

```bash
python agent.py
```

This will:

1. Fetch live remote job listings from RemoteOK.
2. Score every job for relevance to a "freelance AI agent developer"
   profile (Python, LLM agents, automation, RAG, MCP).
3. Keep only jobs scoring >= 6/10, sorted highest-first.
4. If fewer than 3 jobs qualify, autonomously broaden the search once and
   re-score the newly surfaced candidates.
5. Save the final ranked list to `output/relevant_jobs_<timestamp>.csv` and
   `.json`, and print a summary with the top 5 matches.

## Dashboard

A Streamlit dashboard visualizes whatever the agent last saved to `output/`:

```bash
streamlit run streamlit_app.py
```

It auto-detects the most recent `relevant_jobs_*.json` file (falling back to
a manual file upload if none exists), and shows:

- Summary metrics — jobs fetched, relevant matches, relevance threshold, and
  whether the autonomous keyword-broadening step fired (with the keywords it
  chose, read from the matching `run_meta_*.json` sidecar file).
- A highlighted card for the #1 scored job.
- A sortable, filterable results table (minimum-score slider + text search)
  with a clickable "Apply" link per row.
- A **Run Agent Now** button that re-executes `agent.py` live (using
  `ANTHROPIC_API_KEY` from your environment/`.env`), shows a spinner while it
  runs, and refreshes the dashboard with the new results afterward.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `CLAUDE_MODEL` | `claude-opus-5` | Model used for scoring/keyword suggestion |

`agent.py` also reads a local `.env` file directly (no extra dependency
needed) if `ANTHROPIC_API_KEY` isn't already set in your shell.

## How this stays fresh

A GitHub Actions workflow ([.github/workflows/daily-scrape.yml](.github/workflows/daily-scrape.yml))
runs the agent automatically once a day (04:00 UTC / 09:00 Pakistan time), and
can also be triggered manually from the repo's **Actions** tab
(`workflow_dispatch`). Each run:

1. Checks out the repo and installs `requirements.txt`.
2. Runs `python agent.py`, using `ANTHROPIC_API_KEY` from the repo's
   **GitHub Actions secret** of the same name (see setup step below).
3. Commits the newly generated `output/relevant_jobs_*.json`,
   `relevant_jobs_*.csv`, and `run_meta_*.json` files back to the repo as
   `github-actions[bot]`, and pushes — so the results in the repo (and
   whatever the deployed Streamlit dashboard reads) are never more than a
   day stale, with zero manual intervention.

If a run produces no new result files (e.g. the agent failed), the workflow
skips the commit step instead of pushing an empty change.

### Manual setup — you still need to do these yourself

These require your own GitHub/Streamlit login, so they can't be automated:

1. **Add the GitHub Actions secret** — in this repo, go to
   **Settings → Secrets and variables → Actions → New repository secret**,
   name it exactly **`ANTHROPIC_API_KEY`**, and paste your real Anthropic key.
   Without this, the daily workflow run will fail.
2. **Deploy the dashboard on Streamlit Community Cloud** — go to
   [share.streamlit.io](https://share.streamlit.io), connect this GitHub
   repo, set the main file path to `streamlit_app.py`, then under the app's
   **Settings → Secrets**, paste:
   ```toml
   ANTHROPIC_API_KEY = "your_key_here"
   ```
   (see `.streamlit/secrets.toml.example` for the exact format).
