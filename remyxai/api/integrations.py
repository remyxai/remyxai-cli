"""
remyxai/api/integrations.py

Client calls for the JWT integrations API.
Wraps /api/v1.0/integrations/* in engine/app/api/integrations.py.

Used by the CLI to connect/check credential-based providers — notably
the model provider (`claude_code` → Anthropic key) that Outrider needs
the engine to push to the repo as a secret.
"""
from __future__ import annotations

import logging
import requests
from typing import Any, Dict, Optional

from . import BASE_URL, HEADERS, get_headers, log_api_response

logger = logging.getLogger(__name__)


def _h(api_key: Optional[str] = None) -> dict:
    return get_headers(api_key) if api_key else HEADERS


def get_integration_status(
    provider: str, api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Connection status for a provider.

    Calls GET /api/v1.0/integrations/<provider>/status
    Returns { connected: bool, connection?: {...} }.
    """
    r = requests.get(
        f"{BASE_URL}/integrations/{provider}/status",
        headers=_h(api_key),
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()


def connect_credential(
    provider: str,
    credentials: Dict[str, Any],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Connect a credential-based provider (e.g. claude_code).

    Calls POST /api/v1.0/integrations/<provider>/connect
    Body is the provider's credential dict, e.g. {"api_key": "sk-ant-..."}.
    The engine validates the key against the provider before storing it
    encrypted. Returns { connection, user_info } on success.
    """
    r = requests.post(
        f"{BASE_URL}/integrations/{provider}/connect",
        json=credentials,
        headers=_h(api_key),
        timeout=30,
    )
    log_api_response(r)
    r.raise_for_status()
    return r.json()
