import os
import logging

BASE_URL = "https://engine.remyx.ai/api/v1.0"


def get_api_key(api_key=None):
    """
    Resolve the API key from (in priority order):
      1. Explicit api_key argument
      2. REMYXAI_API_KEY environment variable

    Raises ValueError if no key is found.
    """
    key = api_key or os.getenv("REMYXAI_API_KEY")
    if not key:
        raise ValueError(
            "REMYXAI_API_KEY not found. "
            "Pass api_key= or set the REMYXAI_API_KEY environment variable."
        )
    return key


def get_headers(api_key=None):
    """
    Build authorization headers, resolving the key lazily.

    Args:
        api_key: Optional explicit key. Falls back to env var.
    """
    return {
        "Authorization": f"Bearer {get_api_key(api_key)}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Backwards compatibility
#
# Existing code throughout the package does:
#   from . import HEADERS, REMYXAI_API_KEY
#
# We preserve these module-level names so nothing breaks.  When the env var
# is set (the normal CLI / AG2 path), they work exactly as before.  When it
# is NOT set (e.g. HF Space at import time), we set safe defaults and the
# actual key is resolved lazily via get_headers(api_key=...) at call time.
# ---------------------------------------------------------------------------

REMYXAI_API_KEY = os.getenv("REMYXAI_API_KEY", "")

if REMYXAI_API_KEY:
    logging.info(f"Using API Key: {REMYXAI_API_KEY[:8]}...")
    HEADERS = {
        "Authorization": f"Bearer {REMYXAI_API_KEY}",
        "Content-Type": "application/json",
    }
else:
    logging.debug(
        "REMYXAI_API_KEY not set at import time. "
        "Use get_headers(api_key=...) or set the env var before making API calls."
    )
    HEADERS = {
        "Authorization": "Bearer ",
        "Content-Type": "application/json",
    }


def log_api_response(response):
    """Log the response from the API based on the status code."""
    if response.status_code in [200, 201]:
        logging.debug(
            f"API call successful: {response.url}, Status: {response.status_code}"
        )
    else:
        logging.error(
            f"API call failed: {response.url}, "
            f"Status: {response.status_code}, Response: {response.text}"
        )
