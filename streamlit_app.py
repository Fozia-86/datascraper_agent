"""Streamlit dashboard for the DataScraper Agent.

Read-only view over whatever the agent last produced in output/, plus a
button that re-runs agent.py live and refreshes the page with new results.
This file does no scoring or reasoning itself — it only visualizes what
scraper.py + agent.py already decided.
"""

import glob
import html
import json
import os
import subprocess
import sys

import pandas as pd
import streamlit as st

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
AGENT_SCRIPT = os.path.join(PROJECT_DIR, "agent.py")
AGENT_TIMEOUT_SECONDS = 600

st.set_page_config(
    page_title="DataScraper Agent Dashboard",
    page_icon="🛰️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .top-match-card {
        border-left: 6px solid #22c55e;
        background-color: rgba(34, 197, 94, 0.08);
        border-radius: 0.5rem;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1.2rem;
    }
    .top-match-card h4 {
        margin: 0 0 0.35rem 0;
    }
    .top-match-card .reasoning {
        color: var(--text-color, #4b5563);
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def find_latest_results() -> tuple[str | None, str | None]:
    """Return (jobs_json_path, matching_meta_json_path_or_None), most recent first.

    Filenames are timestamped relevant_jobs_YYYYMMDD_HHMMSS.json, so a plain
    lexicographic sort on the basename orders them chronologically.
    """
    pattern = os.path.join(PROJECT_DIR, "**", "relevant_jobs_*.json")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None, None

    candidates.sort(key=os.path.basename)
    latest_jobs_path = candidates[-1]

    timestamp = os.path.basename(latest_jobs_path).removeprefix("relevant_jobs_").removesuffix(".json")
    meta_path = os.path.join(os.path.dirname(latest_jobs_path), f"run_meta_{timestamp}.json")
    return latest_jobs_path, (meta_path if os.path.exists(meta_path) else None)


def load_jobs(jobs_path: str) -> pd.DataFrame:
    with open(jobs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if df.empty:
        return df
    for col in ["title", "company", "location", "url", "reasoning", "date"]:
        if col not in df.columns:
            df[col] = ""
    if "score" not in df.columns:
        df["score"] = 0
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def load_meta(meta_path: str | None) -> dict:
    if not meta_path:
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_anthropic_api_key() -> str | None:
    """os.environ first (local dev, GitHub Actions); st.secrets as a fallback
    for Streamlit Community Cloud, which injects secrets into st.secrets but
    not into the process environment a subprocess would inherit."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        # No secrets.toml locally (StreamlitSecretNotFoundError) or key not
        # defined in it (KeyError) — either way, no key available this route.
        return None


def run_agent_now() -> tuple[bool, str]:
    env = os.environ.copy()
    api_key = resolve_anthropic_api_key()
    if not api_key:
        return False, (
            "ANTHROPIC_API_KEY is not set. Set it in your environment/.env locally, "
            "or under this app's Secrets settings on Streamlit Community Cloud."
        )
    env["ANTHROPIC_API_KEY"] = api_key

    result = subprocess.run(
        [sys.executable, AGENT_SCRIPT],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=AGENT_TIMEOUT_SECONDS,
        env=env,
    )
    ok = result.returncode == 0
    log = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
    return ok, log


# --- Header + "Run Agent Now" -------------------------------------------

header_col, button_col = st.columns([4, 1])
with header_col:
    st.title("🛰️ DataScraper Agent")
    st.caption("Remote AI / agent-developer job listings, scored for relevance by Claude.")
with button_col:
    st.write("")
    run_clicked = st.button("🚀 Run Agent Now", width="stretch")

if run_clicked:
    with st.spinner("Running agent.py — fetching jobs and scoring with Claude..."):
        success, log = run_agent_now()
    if success:
        st.success("Agent run complete. Refreshing dashboard with new results...")
        with st.expander("Agent run log"):
            st.code(log or "(no output)", language="text")
        st.rerun()
    else:
        st.error("Agent run failed. See the log below.")
        with st.expander("Agent run log", expanded=True):
            st.code(log or "(no output)", language="text")

st.divider()

# --- Load results ---------------------------------------------------------

jobs_path, meta_path = find_latest_results()

if jobs_path is None:
    st.warning(
        "No `relevant_jobs_*.json` results found yet under this project. "
        "Run the agent above, or upload a results file manually."
    )
    uploaded = st.file_uploader("Upload a relevant_jobs_*.json file", type="json")
    if uploaded is None:
        st.stop()
    df = pd.DataFrame(json.load(uploaded))
    if not df.empty:
        df["score"] = pd.to_numeric(df.get("score", 0), errors="coerce").fillna(0).astype(int)
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    meta = {}
    source_label = uploaded.name
else:
    df = load_jobs(jobs_path)
    meta = load_meta(meta_path)
    source_label = os.path.basename(jobs_path)

st.caption(f"Showing results from `{source_label}`" + (" (no run metadata found for this file)" if jobs_path and not meta_path else ""))

# --- Summary metrics --------------------------------------------------------

m1, m2, m3, m4 = st.columns(4)
m1.metric("Jobs fetched", meta.get("total_fetched", "N/A"))
m2.metric("Relevant matches", len(df))
m3.metric("Relevance threshold", f"≥ {meta.get('relevance_threshold', 6)}/10")
broaden_attempted = meta.get("broaden_attempted", meta.get("broadened", False))
broaden_found_new = meta.get("broaden_found_new_relevant", False)
m4.metric("Search broadened?", "Attempted" if broaden_attempted else "Not needed")

if broaden_attempted and meta.get("broaden_keywords"):
    keyword_list = ", ".join(f"**{k}**" for k in meta["broaden_keywords"])
    if broaden_found_new:
        st.info(f"🔎 Fewer than 3 relevant jobs were found, so the agent autonomously "
                f"broadened its search with keywords: {keyword_list} — and it worked, "
                f"turning up additional relevant job(s).")
    else:
        scored_count = meta.get("broaden_new_candidates_scored", 0)
        st.info(f"🔎 Fewer than 3 relevant jobs were found, so the agent autonomously "
                f"broadened its search with keywords: {keyword_list}. It re-scored "
                f"{scored_count} new candidate job(s) matching those keywords, but none "
                f"cleared the relevance bar — the current live RemoteOK feed simply doesn't "
                f"have more matches right now. This is a real outcome of the retry, not a bug.")

st.divider()

# --- Top match card ----------------------------------------------------

if not df.empty:
    top = df.iloc[0]
    st.markdown(
        f"""
        <div class="top-match-card">
            <h4>🏆 Top match — {html.escape(str(top['title']))} @ {html.escape(str(top['company']))}</h4>
            <p>Score: <strong>{int(top['score'])}/10</strong> &nbsp;|&nbsp; {html.escape(str(top['location']))} &nbsp;|&nbsp; {html.escape(str(top['date']))}</p>
            <p class="reasoning">"{html.escape(str(top['reasoning']))}"</p>
            <p><a href="{html.escape(str(top['url']), quote=True)}" target="_blank">Apply ↗</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("No relevant jobs in this results file.")
    st.stop()

# --- Filter + table ----------------------------------------------------

filter_col, search_col = st.columns([2, 3])
with filter_col:
    min_score = st.slider("Minimum relevance score", min_value=0, max_value=10, value=6)
with search_col:
    search = st.text_input("Search title / company", placeholder="e.g. LangChain, backend, agent...")

filtered = df[df["score"] >= min_score]
if search:
    needle = search.lower()
    filtered = filtered[
        filtered["title"].str.lower().str.contains(needle, na=False)
        | filtered["company"].str.lower().str.contains(needle, na=False)
    ]

st.caption(f"{len(filtered)} of {len(df)} jobs shown")

st.dataframe(
    filtered[["title", "company", "location", "score", "reasoning", "url", "date"]],
    column_config={
        "title": st.column_config.TextColumn("Title", width="medium"),
        "company": st.column_config.TextColumn("Company"),
        "location": st.column_config.TextColumn("Location"),
        "score": st.column_config.NumberColumn("Score", format="%d/10"),
        "reasoning": st.column_config.TextColumn("Why it matched", width="large"),
        "url": st.column_config.LinkColumn("Apply", display_text="Apply ↗"),
        "date": st.column_config.TextColumn("Date posted"),
    },
    hide_index=True,
    width="stretch",
)
