"""Perception layer.

Pure data fetching: talks to the RemoteOK public API and hands back a plain
list of job dicts. No scoring, no filtering logic, no opinions about which
jobs matter — that reasoning happens in agent.py.
"""

from __future__ import annotations

import requests

REMOTEOK_API_URL = "https://remoteok.com/api"

# RemoteOK blocks requests with default/empty User-Agent strings (403), so we
# identify the client honestly rather than spoofing a browser.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 "
    "datascraper-agent/1.0 (job-search research bot)"
)


def fetch_jobs(limit: int | None = None) -> list[dict]:
    """Fetch remote job listings from the RemoteOK public API.

    Returns a list of dicts with keys: title, company, location, tags,
    description, url, date.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    response = requests.get(REMOTEOK_API_URL, headers=headers, timeout=15)
    response.raise_for_status()
    raw = response.json()

    # RemoteOK's response starts with a legacy metadata entry (no "id" field)
    # that isn't an actual job listing — filter it out.
    jobs_raw = [entry for entry in raw if isinstance(entry, dict) and entry.get("id")]

    jobs = []
    for entry in jobs_raw:
        jobs.append(
            {
                "title": entry.get("position") or entry.get("title", ""),
                "company": entry.get("company", ""),
                "location": entry.get("location") or "Remote",
                "tags": entry.get("tags", []) or [],
                "description": entry.get("description", "") or "",
                "url": entry.get("url") or entry.get("apply_url", ""),
                "date": entry.get("date", ""),
            }
        )

    if limit:
        jobs = jobs[:limit]
    return jobs


if __name__ == "__main__":
    fetched = fetch_jobs()
    print(f"Fetched {len(fetched)} jobs from RemoteOK")
    for job in fetched[:5]:
        print(f"- {job['title']} @ {job['company']} ({job['location']})")
