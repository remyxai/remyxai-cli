"""
remyxai/api/outrider.py

Client calls for the Outrider installations API.
Wraps /api/v1.0/outrider/* endpoints in engine/app/api/outrider.py.

An *installation* is one provisioned agent: an (interest, repo) pair with
its own scoped ``REMYX_API_KEY``. The CLI needs these to re-provision a repo
whose install already exists — provisioning is idempotent server-side and
short-circuits with "Already enabled" once a repo is fully installed, so
changing a live install's configuration means revoking it first and letting
the provisioner re-drive.
"""
from __future__ import annotations

import logging
import requests
from typing import Any, Dict, List, Optional

from . import BASE_URL, HEADERS, get_headers, log_api_response

logger = logging.getLogger(__name__)


def _h(api_key: Optional[str] = None) -> dict:
    return get_headers(api_key) if api_key else HEADERS


def list_installations(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List the caller's provisioned Outrider installations, newest first.

    Calls GET /api/v1.0/outrider/installations

    Revoked (paused) installs are included — filter on ``revoked``.
    Each row carries id, interest_id, interest_name, repo_full_name,
    workflow_filename, pr_url, merged, dispatched, model_provider,
    model_key_set, revoked.
    """
    r = requests.get(
        f"{BASE_URL}/outrider/installations", headers=_h(api_key), timeout=30
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json().get("installations", [])


def revoke_installation(
    installation_id: str, api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Pause an installation: mark it revoked and kill its scoped API key.

    Calls POST /api/v1.0/outrider/installations/<id>/revoke

    The workflow files stay in the repo; its next run fails auth immediately.
    Re-provisioning the same interest onto the repo resumes it — and, because
    the provisioner skips its "already enabled" short-circuit on a revoked
    row, re-drives every step (fresh key, current workflow YAML).

    Returns { revoked: True, installation: {...} }.
    """
    r = requests.post(
        f"{BASE_URL}/outrider/installations/{installation_id}/revoke",
        headers=_h(api_key),
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()
