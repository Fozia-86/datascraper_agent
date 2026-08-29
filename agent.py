"""Reasoning + decision layer, and the agent's main entrypoint.

This is what makes the project an *agent* rather than a scraper with a label
slapped on it: the LLM does the relevance judgement (not keyword matching),
and the control flow branches on what the LLM concludes — including an
autonomous decision to broaden its own search when the first pass comes up
short. See README.md for the full perceive -> reason -> decide -> act
breakdown.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from datetime import datetime, timezone

import anthropic

from scraper import fetch_jobs

MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5")
BATCH_SIZE = 20  # jobs per Claude call, to keep API calls (and cost) low
RELEVANCE_THRESHOLD = 6
MIN_RELEVANT_JOBS = 3
MAX_DESCRIPTION_CHARS = 600  # trims noisy HTML descriptions before they hit the prompt

BASE_KEYWORDS = [
    "ai", "artificial intelligence", "agent", "agentic", "llm", "gpt",
    "machine learning", "ml", "nlp", "generative ai", "genai", "rag",
    "mcp", "prompt engineer", "chatbot", "automation", "python",
]

CANDIDATE_PROFILE = (
    "A freelance AI agent developer with strengths in: Python, building "
    "LLM-powered agents (tool use / function calling / agentic loops), "
    "workflow automation, Retrieval-Augmented Generation (RAG), and the "
    "Model Context Protocol (MCP)."
)

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "score": {"type": "integer"},
                    "reasoning": {"type": "string"},
                },
                "required": ["index", "score", "reasoning"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}

KEYWORDS_SCHEMA = {
    "type": "object",
    "properties": {
        "keyword_1": {"type": "string"},
        "keyword_2": {"type": "string"},
        "keyword_3": {"type": "string"},
    },
    "required": ["keyword_1", "keyword_2", "keyword_3"],
    "additionalProperties": False,
}


def load_env_file(path: str = ".env") -> None:
    """Minimal dependency-free .env loader (never overrides an already-set var)."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_client() -> anthropic.Anthropic:
    load_env_file()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY is not set.\n"
            "Copy .env.example to .env and add your key, or export it in your shell."
        )
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def truncate(text: str, n: int = MAX_DESCRIPTION_CHARS) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + "..."


def score_job_batch(client: anthropic.Anthropic, jobs: list[dict]) -> dict[int, dict]:
    """Send one batch of jobs to Claude in a single call, get back {index: {score, reasoning}}."""
    listing_lines = []
    for i, job in enumerate(jobs):
        listing_lines.append(
            f"[{i}] Title: {job['title']}\n"
            f"    Company: {job['company']}\n"
            f"    Tags: {', '.join(job['tags'])}\n"
            f"    Description: {truncate(job['description'])}"
        )
    listings_text = "\n\n".join(listing_lines)

    prompt = (
        f"Candidate profile:\n{CANDIDATE_PROFILE}\n\n"
        f"Below are {len(jobs)} remote job listings, each tagged with an [index].\n"
        f"For EVERY job, score how relevant it is for this candidate on a 0-10 "
        f"scale (0 = totally unrelated, 10 = perfect match), and give exactly "
        f"one sentence of reasoning for the score.\n\n"
        f"{listings_text}\n\n"
        f"Return one score and one reasoning sentence for every [index] listed above."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        output_config={"format": {"type": "json_schema", "schema": SCORE_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    return {item["index"]: item for item in data["scores"]}


def score_jobs(client: anthropic.Anthropic, jobs: list[dict]) -> list[dict]:
    """Score all jobs in batches and attach score/reasoning to each job dict."""
    scored = []
    for start in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[start : start + BATCH_SIZE]
        results = score_job_batch(client, batch)
        for i, job in enumerate(batch):
            result = results.get(i, {"score": 0, "reasoning": "No score returned by model."})
            scored_job = dict(job)
            scored_job["score"] = result["score"]
            scored_job["reasoning"] = result["reasoning"]
            scored.append(scored_job)
    return scored


def suggest_broader_keywords(client: anthropic.Anthropic, current_relevant_count: int) -> list[str]:
    """Ask Claude to autonomously propose 3 keywords to widen the search."""
    prompt = (
        f"A job search for this candidate profile only turned up "
        f"{current_relevant_count} relevant remote job listing(s) scoring "
        f">= {RELEVANCE_THRESHOLD}/10:\n\n{CANDIDATE_PROFILE}\n\n"
        f"Suggest exactly 3 additional keywords or short job-title fragments "
        f"(e.g. related roles, adjacent skills, or synonyms) that could be used "
        f"to re-filter the same pool of job listings and surface more relevant "
        f"matches for this candidate."
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        output_config={"format": {"type": "json_schema", "schema": KEYWORDS_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    return [data["keyword_1"], data["keyword_2"], data["keyword_3"]]


def _keyword_patterns(keywords: list[str]) -> list[re.Pattern]:
    """Turn keyword strings into word-boundary regexes.

    Plain substring matching would let bare "ai" match inside "email" or
    "domain" — word boundaries avoid that. Claude-suggested keywords also
    sometimes arrive as slash/comma-joined phrases (e.g. "LangChain / vector
    database"), which would never literally appear in a job description as
    one string, so each is split into its component terms first.
    """
    terms = []
    for kw in keywords:
        for part in re.split(r"[\/,]", kw):
            part = part.strip().lower()
            if part:
                terms.append(part)
    return [re.compile(r"\b" + re.escape(term) + r"\b") for term in terms]


def keyword_filter(jobs: list[dict], keywords: list[str]) -> list[dict]:
    """Return jobs whose title/company/tags/description mention any keyword."""
    patterns = _keyword_patterns(keywords)
    matched = []
    for job in jobs:
        haystack = " ".join(
            [job["title"], job["company"], " ".join(job["tags"]), job["description"]]
        ).lower()
        if any(p.search(haystack) for p in patterns):
            matched.append(job)
    return matched


def save_results(
    relevant: list[dict],
    total_fetched: int,
    broaden_attempted: bool,
    broaden_keywords: list[str],
    broaden_found_new_relevant: bool,
    broaden_new_candidates_scored: int,
) -> tuple[str, str, str]:
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"relevant_jobs_{timestamp}.csv")
    json_path = os.path.join(output_dir, f"relevant_jobs_{timestamp}.json")
    meta_path = os.path.join(output_dir, f"run_meta_{timestamp}.json")

    fields = ["title", "company", "location", "url", "score", "reasoning", "date"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for job in relevant:
            writer.writerow({k: job.get(k, "") for k in fields})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([{k: job.get(k, "") for k in fields} for job in relevant], f, indent=2)

    # Sidecar with run-level facts the results file itself doesn't carry —
    # the dashboard reads this for the summary metrics.
    meta = {
        "timestamp": timestamp,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "total_fetched": total_fetched,
        "total_relevant": len(relevant),
        "broaden_attempted": broaden_attempted,
        "broaden_keywords": broaden_keywords,
        "broaden_found_new_relevant": broaden_found_new_relevant,
        "broaden_new_candidates_scored": broaden_new_candidates_scored,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return csv_path, json_path, meta_path


def print_summary(
    total_fetched: int,
    relevant: list[dict],
    broaden_attempted: bool,
    broaden_keywords: list[str],
    output_paths: tuple[str, str, str],
) -> None:
    csv_path, json_path, meta_path = output_paths
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total jobs fetched:        {total_fetched}")
    print(f"Total relevant (score>=6): {len(relevant)}")
    if broaden_attempted:
        print(f"Search broadened:          attempted -> {', '.join(broaden_keywords)}")
    else:
        print("Search broadened:          not needed (>=3 relevant jobs found on first pass)")
    print(f"Saved CSV:                 {csv_path}")
    print(f"Saved JSON:                {json_path}")
    print(f"Saved run metadata:        {meta_path}")
    print("\nTop 5 matches:")
    for job in relevant[:5]:
        print(f"  [{job['score']}/10] {job['title']} @ {job['company']} - {job['url']}")
        print(f"        {job['reasoning']}")
    print("=" * 60)


def run_agent() -> None:
    client = get_client()

    print("STEP 1 - Perception: fetching remote job listings from RemoteOK...")
    raw_jobs = fetch_jobs()
    print(f"  Fetched {len(raw_jobs)} raw job listings.\n")

    # Cheap recall net over the whole RemoteOK feed (bartenders, gardeners, sales
    # reps, etc. don't need an LLM call to rule out). The actual relevance
    # *judgement* on whatever passes this net is still 100% Claude's call.
    candidate_jobs = keyword_filter(raw_jobs, BASE_KEYWORDS)
    print(
        f"  Pre-filtered to {len(candidate_jobs)} AI/tech-adjacent candidate(s) "
        f"before sending anything to Claude.\n"
    )

    print("STEP 2 - Reasoning: scoring jobs for relevance with Claude...")
    scored = score_jobs(client, candidate_jobs)
    relevant = sorted(
        (j for j in scored if j["score"] >= RELEVANCE_THRESHOLD),
        key=lambda j: j["score"],
        reverse=True,
    )
    print(f"  {len(relevant)} job(s) scored >= {RELEVANCE_THRESHOLD}/10.\n")

    broaden_attempted = False
    broaden_found_new_relevant = False
    broaden_new_candidates_scored = 0
    broaden_keywords: list[str] = []

    if len(relevant) < MIN_RELEVANT_JOBS:
        broaden_attempted = True
        print("STEP 3 - Autonomous action: too few relevant jobs found, deciding to broaden search.")
        broaden_keywords = suggest_broader_keywords(client, len(relevant))
        print(
            f"  Only {len(relevant)} relevant job(s) found, broadening search "
            f"with keywords: {', '.join(broaden_keywords)}"
        )

        broadened_pool = keyword_filter(raw_jobs, broaden_keywords)
        already_scored_urls = {j["url"] for j in scored}
        new_jobs = [j for j in broadened_pool if j["url"] not in already_scored_urls]
        broaden_new_candidates_scored = len(new_jobs)
        print(f"  Re-filtered pool contains {len(new_jobs)} new candidate job(s) to score.\n")

        if new_jobs:
            newly_scored = score_jobs(client, new_jobs)
            scored.extend(newly_scored)
            relevant = sorted(
                (j for j in scored if j["score"] >= RELEVANCE_THRESHOLD),
                key=lambda j: j["score"],
                reverse=True,
            )
            broaden_found_new_relevant = any(j["score"] >= RELEVANCE_THRESHOLD for j in newly_scored)
            print(f"  After broadening: {len(relevant)} job(s) now score >= {RELEVANCE_THRESHOLD}/10.\n")
        else:
            print("  No new candidate jobs matched the broadened keywords; keeping original results.\n")

    print("STEP 4 - Output: saving ranked results...")
    output_paths = save_results(
        relevant,
        len(raw_jobs),
        broaden_attempted,
        broaden_keywords,
        broaden_found_new_relevant,
        broaden_new_candidates_scored,
    )
    print_summary(len(raw_jobs), relevant, broaden_attempted, broaden_keywords, output_paths)


if __name__ == "__main__":
    run_agent()
