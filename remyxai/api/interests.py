"""
remyxai/api/interests.py

Client calls for the Research Interests management API.
Wraps /api/v1.0/interests/* endpoints in engine/app/api/interests.py.

A ResearchInterest is a named, natural-language description of what to
track. The recommendation pipeline uses it to match papers, GitHub repos,
and future sources — no changes needed here when new sources ship.
"""
from __future__ import annotations

import logging
import requests
from typing import Any, Dict, List, Optional

from . import BASE_URL, HEADERS, get_headers, log_api_response

logger = logging.getLogger(__name__)


def _h(api_key: Optional[str] = None) -> dict:
    return get_headers(api_key) if api_key else HEADERS


def list_interests(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all Research Interests for the current user.

    Calls GET /api/v1.0/interests

    Returns list of:
      { id, name, context, daily_count, is_active,
        created_at, last_recommendation_at }
    """
    r = requests.get(f"{BASE_URL}/interests", headers=_h(api_key), timeout=30)
    log_api_response(r)
    r.raise_for_status()
    return r.json().get("interests", [])


def get_interest(
    interest_id: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get a single Research Interest by UUID.

    Calls GET /api/v1.0/interests/<id>
    """
    r = requests.get(
        f"{BASE_URL}/interests/{interest_id}", headers=_h(api_key), timeout=30
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()


def create_interest(
    name: str,
    context: str,
    daily_count: int = 2,
    is_active: bool = True,
    source_repo_url: Optional[str] = None,
    source_repo_metadata: Optional[Dict[str, Any]] = None,
    generated_report: Optional[str] = None,
    repo_analysis: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new Research Interest.

    Calls POST /api/v1.0/interests

    Args:
        name:        Short label, e.g. "RAG & Retrieval".
        context:     Natural language description of what to track.
                     Can also be a HuggingFace or GitHub URL — the server
                     will scrape it and generate an embedding.
        daily_count: Items to surface per day (1–10, default 2).
        is_active:   Include in daily digest immediately (default True).
        source_repo_url, source_repo_metadata, generated_report,
        repo_analysis: Repo-sourced fields; include after an
                       analyze-repo flow to persist the analysis payload.
    """
    body: Dict[str, Any] = {
        "name": name,
        "context": context,
        "daily_count": daily_count,
        "is_active": is_active,
    }
    if source_repo_url is not None:
        body["source_repo_url"] = source_repo_url
    if source_repo_metadata is not None:
        body["source_repo_metadata"] = source_repo_metadata
    if generated_report is not None:
        body["generated_report"] = generated_report
    if repo_analysis is not None:
        body["repo_analysis"] = repo_analysis

    r = requests.post(
        f"{BASE_URL}/interests",
        json=body,
        headers=_h(api_key),
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()


def update_interest(
    interest_id: str,
    name: Optional[str] = None,
    context: Optional[str] = None,
    daily_count: Optional[int] = None,
    is_active: Optional[bool] = None,
    source_repo_url: Optional[str] = None,
    source_repo_metadata: Optional[Dict[str, Any]] = None,
    generated_report: Optional[str] = None,
    repo_analysis: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update a Research Interest. Only supplied fields are changed.

    Calls PUT /api/v1.0/interests/<id>

    Changing name or context invalidates the pre-ranked recommendation
    pool for this interest. The server handles invalidation automatically
    and returns pool_invalidated: N in the response when this occurs.
    """
    payload: Dict[str, Any] = {}
    if name is not None:
        payload["name"] = name
    if context is not None:
        payload["context"] = context
    if daily_count is not None:
        payload["daily_count"] = daily_count
    if is_active is not None:
        payload["is_active"] = is_active
    if source_repo_url is not None:
        payload["source_repo_url"] = source_repo_url
    if source_repo_metadata is not None:
        payload["source_repo_metadata"] = source_repo_metadata
    if generated_report is not None:
        payload["generated_report"] = generated_report
    if repo_analysis is not None:
        payload["repo_analysis"] = repo_analysis

    r = requests.put(
        f"{BASE_URL}/interests/{interest_id}",
        json=payload,
        headers=_h(api_key),
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()


# ─── repo-sourced interest helpers ───────────────────────────────────────────


def analyze_repo(
    repo_url: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Kick off async repo analysis to seed a Research Interest.

    Calls POST /api/v1.0/interests/analyze-repo

    Returns: {"task_id": "...", "status_url": "..."}  (HTTP 202)
    """
    r = requests.post(
        f"{BASE_URL}/interests/analyze-repo",
        json={"repo_url": repo_url},
        headers=_h(api_key),
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()


def poll_repo_analysis(
    task_id: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Poll a repo-analysis task.

    Calls GET /api/v1.0/interests/analyze-repo/<task_id>
    """
    r = requests.get(
        f"{BASE_URL}/interests/analyze-repo/{task_id}",
        headers=_h(api_key),
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()


def regenerate_interest(
    interest_id: str,
    repo_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Re-run repo analysis for an existing Research Interest.

    Calls POST /api/v1.0/interests/<id>/regenerate

    Returns the same shape as analyze_repo (task_id + status_url).
    Apply the returned payload via update_interest once polling completes.
    """
    body: Dict[str, Any] = {}
    if repo_url is not None:
        body["repo_url"] = repo_url

    r = requests.post(
        f"{BASE_URL}/interests/{interest_id}/regenerate",
        json=body,
        headers=_h(api_key),
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()


def list_github_repos(
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List repos the caller can pick from via their GitHub integration.

    Calls GET /api/v1.0/interests/github/repos

    Returns {"connected": bool, "repos": [...]}
    """
    r = requests.get(
        f"{BASE_URL}/interests/github/repos",
        headers=_h(api_key),
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()


def delete_interest(
    interest_id: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Delete a Research Interest and its associated recommendations.

    Calls DELETE /api/v1.0/interests/<id>
    """
    r = requests.delete(
        f"{BASE_URL}/interests/{interest_id}", headers=_h(api_key), timeout=30
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()


def toggle_interest(
    interest_id: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Toggle is_active on a Research Interest without deleting it.

    Calls POST /api/v1.0/interests/<id>/toggle
    """
    r = requests.post(
        f"{BASE_URL}/interests/{interest_id}/toggle",
        headers=_h(api_key),
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()
