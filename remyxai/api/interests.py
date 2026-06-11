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
    name: Optional[str] = None,
    context: Optional[str] = None,
    daily_count: int = 2,
    is_active: bool = True,
    *,
    # ── repo-sourced fields (from analyze_repo) ──────────────────────────
    source_repo_url: Optional[str] = None,
    source_repo_metadata: Optional[Any] = None,
    repo_analysis: Optional[Any] = None,
    generated_report: Optional[str] = None,
    report_generated_at: Optional[str] = None,
    # ── provisioning ("Automate paper PRs on this repo") ─────────────────
    provision_action: Optional[bool] = None,
    provision_auto_merge: Optional[bool] = None,
    provision_repo_url: Optional[str] = None,
    # ── project-sourced fields ───────────────────────────────────────────
    project_id: Optional[str] = None,
    mode: Optional[str] = None,
    included_experiment_ids: Optional[List[str]] = None,
    auto_update_from_experiments: Optional[bool] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new Research Interest.

    Calls POST /api/v1.0/interests

    Args:
        name:        Short label, e.g. "RAG & Retrieval".
        context:     Natural language description of what to track.
                     Can also be a HuggingFace or GitHub URL — the server
                     will scrape it and generate an embedding. Optional for
                     project-mode interests (the server fills a placeholder).
        daily_count: Items to surface per day (1–10, default 2).
        is_active:   Include in daily digest immediately (default True).

    Repo-sourced kwargs (from a prior ``analyze_repo`` run): when
    ``source_repo_url`` is set the server links the interest to its
    canonical ExperimentHistory and dispatches experiment-history
    extraction (response carries ``history_extraction_task_id``).

    Provisioning kwargs map to the "Automate paper PRs on this repo"
    choice. Leave ``provision_action`` unset/False for "Not now". Set it
    True with ``provision_auto_merge=False`` to open a setup PR for review,
    or ``True`` to set it up automatically (response carries
    ``provision_task_id``).

    Project-sourced kwargs: pass ``project_id`` (and optionally
    ``mode='project_structured'``, ``included_experiment_ids``,
    ``auto_update_from_experiments``) to build context from a project's
    experiments. Response carries ``build_task_id``.
    """
    payload: Dict[str, Any] = {
        "daily_count": daily_count,
        "is_active": is_active,
    }
    if name is not None:
        payload["name"] = name
    if context is not None:
        payload["context"] = context

    # Repo-sourced fields — only forwarded when present.
    if source_repo_url is not None:
        payload["source_repo_url"] = source_repo_url
    if source_repo_metadata is not None:
        payload["source_repo_metadata"] = source_repo_metadata
    if repo_analysis is not None:
        payload["repo_analysis"] = repo_analysis
    if generated_report is not None:
        payload["generated_report"] = generated_report
    if report_generated_at is not None:
        payload["report_generated_at"] = report_generated_at

    # Provisioning — omitting provision_action entirely means "Not now".
    if provision_action is not None:
        payload["provision_action"] = provision_action
    if provision_auto_merge is not None:
        payload["provision_auto_merge"] = provision_auto_merge
    if provision_repo_url is not None:
        payload["provision_repo_url"] = provision_repo_url

    # Project-sourced fields.
    if project_id is not None:
        payload["project_id"] = project_id
    if mode is not None:
        payload["mode"] = mode
    if included_experiment_ids is not None:
        payload["included_experiment_ids"] = included_experiment_ids
    if auto_update_from_experiments is not None:
        payload["auto_update_from_experiments"] = auto_update_from_experiments

    r = requests.post(
        f"{BASE_URL}/interests",
        json=payload,
        headers=_h(api_key),
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()


# ─── Repo analysis (create-from-repo flow) ─────────────────────────────────

def analyze_repo(
    repo_url: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Kick off async analysis of a GitHub repo for a potential Research
    Interest. The server clones/reads the repo and generates a profile
    report (used as the interest context) plus structured repo_analysis.

    Calls POST /api/v1.0/interests/analyze-repo

    Returns 202 { task_id, status_url }. Poll with poll_analyze_repo.
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


def poll_analyze_repo(
    task_id: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Poll a repo-analysis task.

    Calls GET /api/v1.0/interests/analyze-repo/<task_id>

    Returns the task dict { status, progress, message, result }. On
    completion `result` carries:
      { full_name, source_repo_url, source_repo_metadata,
        report_markdown, repo_analysis, report_generated_at }
    """
    r = requests.get(
        f"{BASE_URL}/interests/analyze-repo/{task_id}",
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

    r = requests.put(
        f"{BASE_URL}/interests/{interest_id}",
        json=payload,
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


# ─── Outrider provisioning  ──────────────────────────────────────

def provision_action(
    interest_id: str,
    repo_url: Optional[str] = None,
    auto_merge: bool = True,
    branch: Optional[str] = None,
    workflow_filename: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Provision the Outrider recommendation Action on a repo, server-side
    (the Remyx GitHub App sets secrets, writes the workflow, opens a
    bot-authored setup PR, and — when auto_merge — merges it and fires
    the first run).

    Calls POST /api/v1.0/interests/<id>/provision-action

    Args:
        repo_url:   GitHub URL; defaults to the interest's source repo.
        auto_merge: merge the setup PR + dispatch the first run ("set it
                    up for me"). False opens the PR for the user to review.

    Returns 202 { task_id, status_url }. Poll with poll_provision_action.
    """
    payload: Dict[str, Any] = {"auto_merge": auto_merge}
    if repo_url:
        payload["repo_url"] = repo_url
    if branch:
        payload["branch"] = branch
    if workflow_filename:
        payload["workflow_filename"] = workflow_filename

    r = requests.post(
        f"{BASE_URL}/interests/{interest_id}/provision-action",
        json=payload,
        headers=_h(api_key),
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()


def poll_provision_action(
    interest_id: str,
    task_id: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Poll a provision-action task.

    Calls GET /api/v1.0/interests/<id>/provision-action/<task_id>

    Returns the task dict: { status, progress, message, result, error }.
    On completion, `result` carries pr_url, secret_set, merged, dispatched,
    model_key_missing, key_label, repo.
    """
    r = requests.get(
        f"{BASE_URL}/interests/{interest_id}/provision-action/{task_id}",
        headers=_h(api_key),
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()


def get_provision_status(
    interest_id: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Latest provisioning state for an interest.

    Calls GET /api/v1.0/interests/<id>/provision-action

    Returns { provisioned: False } if none, else the provisioned-action
    record with provisioned: True.
    """
    r = requests.get(
        f"{BASE_URL}/interests/{interest_id}/provision-action",
        headers=_h(api_key),
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()
