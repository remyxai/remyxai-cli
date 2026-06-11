"""
remyxai/api/projects.py

Client calls for the Projects API.
Wraps /api/v1.0/projects/* and /api/v1.0/experiments in
engine/app/api/projects.py and engine/app/api/experiments.py.

Used by the create-research-interest-from-a-project flow: list the
caller's projects to pick a project_id, and optionally list a project's
experiments to curate `included_experiment_ids`.
"""
from __future__ import annotations

import logging
import requests
from typing import Any, Dict, List, Optional

from . import BASE_URL, HEADERS, get_headers, log_api_response

logger = logging.getLogger(__name__)


def _h(api_key: Optional[str] = None) -> dict:
    return get_headers(api_key) if api_key else HEADERS


def list_projects(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List non-archived projects for the caller's team.

    Calls GET /api/v1.0/projects

    Returns list of:
      { id, name, description, team_id, is_archived, config,
        scope_context, created_at, updated_at }
    """
    r = requests.get(f"{BASE_URL}/projects", headers=_h(api_key), timeout=30)
    log_api_response(r)
    r.raise_for_status()
    return r.json().get("projects", [])


def list_experiments(
    project_id: Optional[str] = None,
    limit: int = 100,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List experiments, optionally scoped to a project.

    Calls GET /api/v1.0/experiments[?project_id=<id>&limit=<n>]

    Returns a list of experiment dicts (id, name, project_id, ...).
    The server caps `limit` at 100.
    """
    params: Dict[str, Any] = {"limit": limit}
    if project_id:
        params["project_id"] = project_id
    r = requests.get(
        f"{BASE_URL}/experiments",
        headers=_h(api_key),
        params=params,
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    body = r.json()
    # The experiments endpoint returns either a bare list or an envelope.
    if isinstance(body, dict):
        return body.get("experiments", body.get("results", []))
    return body
