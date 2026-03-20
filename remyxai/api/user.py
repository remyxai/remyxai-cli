import requests
from typing import Optional
from . import BASE_URL, HEADERS, get_headers, log_api_response


def _resolve_headers(api_key: Optional[str] = None) -> dict:
    """Return headers using explicit key if provided, else module-level default."""
    if api_key:
        return get_headers(api_key)
    return HEADERS


def get_user_profile(api_key: Optional[str] = None):
    url = f"{BASE_URL}user"
    response = requests.get(url, headers=_resolve_headers(api_key))
    return response.json()


def get_user_credits(api_key: Optional[str] = None):
    url = f"{BASE_URL}user/credits"
    response = requests.get(url, headers=_resolve_headers(api_key))
    return response.json()
