"""
remyxai/api/projects.py

Client calls for the Projects management API.
Wraps /api/v1.0/projects/* endpoints.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from . import BASE_URL, HEADERS, get_headers, log_api_response

logger = logging.getLogger(__name__)


def _h(api_key: Optional[str] = None) -> dict:
    return get_headers(api_key) if api_key else HEADERS


def list_projects(
    team_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List non-archived projects for the caller's team.

    Calls GET /api/v1.0/projects
    """
    params: Dict[str, Any] = {}
    if team_id:
        params["team_id"] = team_id

    r = requests.get(
        f"{BASE_URL}/projects",
        headers=_h(api_key),
        params=params,
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json().get("projects", [])


def get_project(
    project_id: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get a project with full config.

    Calls GET /api/v1.0/projects/<id>
    """
    r = requests.get(
        f"{BASE_URL}/projects/{project_id}",
        headers=_h(api_key),
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()


def configure_eval_template(
    project_id: str,
    template_name: str,
    template: Dict[str, Any],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create or replace an eval template on a project's config.

    Calls POST /api/v1.0/projects/<id>/eval-templates/<name>

    Args:
        project_id:    UUID of the target project.
        template_name: Name for this eval template (e.g. "default").
        template:      Template definition — passed as the request body.
    """
    r = requests.post(
        f"{BASE_URL}/projects/{project_id}/eval-templates/{template_name}",
        headers=_h(api_key),
        json=template,
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()


def set_decision_policy(
    project_id: str,
    policy_name: str,
    policy: Dict[str, Any],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create or replace a decision policy on a project's config.

    Calls POST /api/v1.0/projects/<id>/decision-policies/<name>

    Args:
        project_id:  UUID of the target project.
        policy_name: Name for this policy (e.g. "default", "strict").
        policy:      Policy definition — passed as the request body.
                     Rules keyed by disposition (ship/reject/iterate) with
                     combinators + predicates over metric deltas,
                     confidence bands, and sample sizes.
    """
    r = requests.post(
        f"{BASE_URL}/projects/{project_id}/decision-policies/{policy_name}",
        headers=_h(api_key),
        json=policy,
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()
