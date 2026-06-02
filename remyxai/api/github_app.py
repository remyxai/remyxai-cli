"""
remyxai/api/github_app.py

Client calls for the Remyx GitHub App (remyx-ai[bot]) endpoints.
Wraps /api/v1.0/github/app/* in engine/app/api/github_app_api.py.

The App is what lets the engine provision the Outrider recommendation
workflow, set repo secrets, and open bot-authored PRs without the user
granting a personal token. Installing the App is an interactive browser
step; these endpoints let the CLI surface the install link and poll for
completion.
"""
from __future__ import annotations

import logging
import requests
from typing import Any, Dict, Optional

from . import BASE_URL, HEADERS, get_headers, log_api_response

logger = logging.getLogger(__name__)


def _h(api_key: Optional[str] = None) -> dict:
    return get_headers(api_key) if api_key else HEADERS


def get_app_install_url(api_key: Optional[str] = None) -> Dict[str, Any]:
    """Return the Remyx GitHub App install URL for the current user.

    Calls GET /api/v1.0/github/app/install-url

    Returns { install_url, state, app_slug, configured }. When the App
    isn't configured on the server, returns { configured: False, error }.
    """
    r = requests.get(
        f"{BASE_URL}/github/app/install-url", headers=_h(api_key), timeout=30
    )
    log_api_response(r)
    if r.status_code == 503:
        return r.json()  # {configured: False, error}
    r.raise_for_status()
    return r.json()


def is_app_installed(repo: str, api_key: Optional[str] = None) -> bool:
    """Whether the Remyx App is installed on `repo` (owner/name).

    Calls GET /api/v1.0/github/app/installation?repo=owner/name
    """
    r = requests.get(
        f"{BASE_URL}/github/app/installation",
        params={"repo": repo},
        headers=_h(api_key),
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return bool(r.json().get("installed"))
