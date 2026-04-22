"""
remyxai/api/experiments.py

Client calls for the Experiments management + validation API.
Wraps /api/v1.0/experiments/* and /api/v1.0/eval-env/runs endpoints.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from . import BASE_URL, HEADERS, get_headers, log_api_response

logger = logging.getLogger(__name__)


def _h(api_key: Optional[str] = None) -> dict:
    return get_headers(api_key) if api_key else HEADERS


def list_experiments(
    project_id: Optional[str] = None,
    status: Optional[str] = None,
    initiative: Optional[str] = None,
    limit: int = 20,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List experiments with optional filters.

    Calls GET /api/v1.0/experiments
    """
    params: Dict[str, Any] = {"limit": limit}
    if project_id:
        params["project_id"] = project_id
    if status:
        params["status"] = status
    if initiative:
        params["initiative"] = initiative

    r = requests.get(
        f"{BASE_URL}/experiments",
        headers=_h(api_key),
        params=params,
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json().get("experiments", [])


def get_experiment(
    experiment_id: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get a single experiment by UUID.

    Calls GET /api/v1.0/experiments/<id>
    """
    r = requests.get(
        f"{BASE_URL}/experiments/{experiment_id}",
        headers=_h(api_key),
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()


def start_validation_run(
    experiment_id: str,
    template_id: str,
    github_url: str,
    variants: List[Dict[str, Any]],
    seeds: int = 1,
    pr_number: Optional[int] = None,
    pr_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Start a validation run for an experiment.

    Calls POST /api/v1.0/eval-env/runs with the parent experiment_id and
    a locked EvalTemplate. The engine builds per-variant Docker images,
    runs the evaluation inside them, collects results, and computes a
    Pass/Warn/Fail verdict against the template's decision criteria.

    Args:
        experiment_id:  Parent experiment UUID (required).
        template_id:    UUID of a locked EvalTemplate (required).
        github_url:     Target repo URL (required).
        variants:       [{name, ref?, commit_sha}, ...] — at least one;
                        typically baseline + feature.
        seeds:          Per-variant seed count.
        pr_number:      PR number for lineage (optional).
        pr_url:         PR URL for lineage (optional).

    Returns:
        {"run": {...}, "build_ids": {variant: uuid}, "status": "building_envs"}
    """
    body: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "template_id": template_id,
        "github_url": github_url,
        "variants": variants,
        "seeds": seeds,
    }
    if pr_number is not None:
        body["pr_number"] = pr_number
    if pr_url is not None:
        body["pr_url"] = pr_url

    r = requests.post(
        f"{BASE_URL}/eval-env/runs",
        headers=_h(api_key),
        json=body,
        timeout=60,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()


def get_validation_run(
    run_id: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Poll a validation run.

    Calls GET /api/v1.0/eval-env/runs/<id>
    """
    r = requests.get(
        f"{BASE_URL}/eval-env/runs/{run_id}",
        headers=_h(api_key),
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()
