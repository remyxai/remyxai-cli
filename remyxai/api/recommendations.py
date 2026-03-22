"""
remyxai/api/recommendations.py

Client calls for the Remyx recommendations API.
Wraps /api/v1.0/papers/* endpoints in engine/app/api/papers.py.

─── Response envelope ────────────────────────────────────────────────────────

Every recommendation returned by the API follows this source-agnostic shape:

  {
    "recommendation_id": str,
    "source_type":        str,    # "arxiv_paper" | "github_repo" | ...
    "resource_id":        str,    # stable ID (arxiv_id, github slug, etc.)
    "title":              str,    # always safe to display
    "url":                str,    # always a valid link
    "relevance_score":    float,  # 0.0 – 1.0
    "reasoning":          str,
    "suggested_experiment": str,
    "interest_id":        str,
    "interest_name":      str,
    "alerted_at":         str,    # ISO datetime
    "resource":           dict,   # source-specific payload — key off source_type
  }

resource payload by source_type:

  "arxiv_paper":
    { arxiv_id, authors, abstract, abstract_summary, published_at,
      categories, has_docker, docker_image, github_url }

  "github_repo":  (future — GitHubResource)
    { owner, repo, description, stars, language, topics,
      last_pushed_at, has_docker, docker_image }

Always use top-level title / url / resource_id for generic display.
Inspect source_type only when you need source-specific fields.
"""
from __future__ import annotations

import logging
import requests
from typing import Any, Dict, List, Optional

from . import BASE_URL, HEADERS, get_headers, log_api_response

logger = logging.getLogger(__name__)


def _h(api_key: Optional[str] = None) -> dict:
    return get_headers(api_key) if api_key else HEADERS


# ─── digest ──────────────────────────────────────────────────────────────────

def get_recommendations_digest(
    limit: int = 5,
    period: str = "today",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch recommendations grouped by Research Interest.

    Calls GET /api/v1.0/papers/recommended/digest

    Args:
        limit:   Items per interest (1–10, default 5).
        period:  "today" | "week" | "all"  (default "today").
        api_key: Optional explicit API key.

    Returns:
        {
          "date":         str,          # "YYYY-MM-DD"
          "period":       str,
          "source_types": [str],        # e.g. ["arxiv_paper"]
          "interests": [
            {
              "id":                 str,
              "name":               str,
              "daily_papers_count": int,
              "recommendations":    [ <envelope>, ... ],
              "count":              int,
            }
          ],
          "total_papers": int,          # sum across all interests
        }
    """
    url = f"{BASE_URL}/papers/recommended/digest"
    params = {
        "limit": max(1, min(limit, 10)),
        "period": period,
    }
    r = requests.get(url, headers=_h(api_key), params=params, timeout=30)
    log_api_response(r)
    r.raise_for_status()
    return r.json()


# ─── flat list ───────────────────────────────────────────────────────────────

def list_recommended(
    interest_id: Optional[str] = None,
    limit: int = 20,
    period: str = "all",
    source_type: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List recommendations (flat, optionally filtered).

    Calls GET /api/v1.0/papers/recommended

    Args:
        interest_id: Filter to one Research Interest UUID.
        limit:       Max results (1–50, default 20).
        period:      "today" | "week" | "all"  (default "all").
        source_type: Optional filter e.g. "arxiv_paper" or "github_repo".
        api_key:     Optional explicit API key.

    Returns:
        {
          "papers":      [ <envelope>, ... ],   # field name from engine
          "count":       int,
          "period":      str,
          "interest_id": str | None,
        }
    """
    url = f"{BASE_URL}/papers/recommended"
    params: Dict[str, Any] = {
        "limit": max(1, min(limit, 50)),
        "period": period,
    }
    if interest_id:
        params["interest_id"] = interest_id
    if source_type:
        params["source_type"] = source_type

    r = requests.get(url, headers=_h(api_key), params=params, timeout=30)
    log_api_response(r)
    r.raise_for_status()
    return r.json()


# ─── trigger refresh ─────────────────────────────────────────────────────────

def trigger_recommendations_refresh(
    interest_id: Optional[str] = None,
    num_results: Optional[int] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Kick off async recommendation generation for one or all interests.

    Calls POST /api/v1.0/papers/recommended/refresh

    Args:
        interest_id: Specific interest UUID; omit to refresh all active.
        num_results: Items per interest; omit to use interest.daily_papers_count.
        api_key:     Optional explicit API key.

    Returns:
        { "tasks": [ { "task_id", "interest_id", "interest_name", "status" } ] }
    """
    url = f"{BASE_URL}/papers/recommended/refresh"
    payload: Dict[str, Any] = {}
    if interest_id:
        payload["interest_id"] = interest_id
    if num_results is not None:
        payload["num_results"] = num_results

    r = requests.post(url, json=payload, headers=_h(api_key), timeout=30)
    log_api_response(r)
    r.raise_for_status()
    return r.json()


# ─── poll refresh task ───────────────────────────────────────────────────────

def poll_refresh_task(
    task_id: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Poll a refresh task started by trigger_recommendations_refresh().

    Calls GET /api/v1.0/papers/recommended/refresh/<task_id>

    Returns:
        {
          "task_id":   str,
          "status":    "pending" | "running" | "completed" | "failed",
          "progress":  int,       # 0–100
          "message":   str,
          "result":    dict | None,   # present when completed
          "error":     str | None,    # present when failed
        }
    """
    url = f"{BASE_URL}/papers/recommended/refresh/{task_id}"
    r = requests.get(url, headers=_h(api_key), timeout=30)
    log_api_response(r)
    r.raise_for_status()
    return r.json()
