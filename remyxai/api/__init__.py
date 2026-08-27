import hashlib
import logging
import os

DEFAULT_BASE_URL = "https://engine.remyx.ai/api/v1.0"
API_PATH = "/api/v1.0"


def resolve_base_url(value=None):
    """Engine API base URL, from ``REMYXAI_API_URL`` or the production default.

    Accepts either a full API base (``https://host/api/v1.0``) or just an
    origin (``http://localhost:5000``) — the version path is appended when it's
    missing, so pointing the CLI at a test server is one env var and no
    guessing about the prefix.
    """
    raw = (value if value is not None else os.getenv("REMYXAI_API_URL", "")).strip()
    if not raw:
        return DEFAULT_BASE_URL
    raw = raw.rstrip("/")
    return raw if API_PATH in raw else raw + API_PATH


# Read once at import: callers do `from . import BASE_URL`, so the env var has
# to be set before the CLI starts (the normal shell-export case).
BASE_URL = resolve_base_url()


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

# Fixed, non-secret inputs: the fingerprint has to be reproducible across
# machines and runs, otherwise it cannot be compared, which is its whole point.
_FINGERPRINT_SALT = b"remyxai-cli-key-fingerprint-v1"
_FINGERPRINT_ROUNDS = 1000


def key_fingerprint(key):
    """A short, stable, non-reversible identifier for an API key.

    Logging a slice of the key put real key material into log files, and logs
    travel a lot further than the shell that exported the key. A truncated
    digest answers the only question that line was there to answer — *which*
    key is this? — without carrying any of it: the same key always yields the
    same fingerprint, different keys effectively never collide, and it cannot
    be worked back to the secret.

    Uses PBKDF2 rather than a bare SHA-256. This is not password storage, so
    the slow-KDF argument does not really apply, but deriving through one costs
    well under a millisecond here and means a logged fingerprint cannot be used
    to cheaply confirm a guessed key. It also keeps static analysis quiet
    without a standing dismissal.
    """
    if not key:
        return "none"
    digest = hashlib.pbkdf2_hmac(
        "sha256", key.encode("utf-8"), _FINGERPRINT_SALT, _FINGERPRINT_ROUNDS
    )
    return "fp:" + digest.hex()[:8]


def key_hint(key, source="REMYXAI_API_KEY"):
    """Describe the key in use for logs and support output, without leaking it.

    Includes the length because a truncated or whitespace-mangled key is a
    common misconfiguration and the length is the tell; for a fixed-format key
    it reveals nothing the format does not already.
    """
    if not key:
        return f"no API key ({source} not set)"
    return f"API key from {source} ({key_fingerprint(key)}, {len(key)} chars)"


REMYXAI_API_KEY = os.getenv("REMYXAI_API_KEY", "")

if REMYXAI_API_KEY:
    # Identify the key, never quote it. Debug rather than info: this fires on
    # every import, and the matching not-set branch below is already debug.
    logging.debug("Using %s", key_hint(REMYXAI_API_KEY))
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
    if 200 <= response.status_code < 300:
        logging.debug(
            f"API call successful: {response.url}, Status: {response.status_code}"
        )
    else:
        logging.error(
            f"API call failed: {response.url}, "
            f"Status: {response.status_code}, Response: {response.text}"
        )
