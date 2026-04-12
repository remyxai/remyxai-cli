"""
DockerGen API client for generating validated Dockerfiles from GitHub repos.

Used by:
- remyxai CLI commands
- DockerGenClient (for HF Space and external integrations)
"""
import logging
import requests
from typing import Optional
from dataclasses import dataclass, asdict, field
from . import BASE_URL, HEADERS, get_headers, log_api_response

logger = logging.getLogger(__name__)


@dataclass
class DockerfileResult:
    """
    Represents the result of a Dockerfile generation task.
    """

    task_id: str
    github_url: str
    status: str  # pending|generating|building|validating|completed|failed
    dockerfile_text: Optional[str] = None
    branch: Optional[str] = None
    base_image: Optional[str] = None
    failure_reason: Optional[str] = None
    build_attempts: int = 0
    generation_log: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DockerfileResult":
        """Create DockerfileResult from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


def _resolve_headers(api_key: Optional[str] = None) -> dict:
    """Return headers using explicit key if provided, else module-level default."""
    if api_key:
        return get_headers(api_key)
    return HEADERS


def generate_dockerfile(
    github_url: str,
    branch: Optional[str] = None,
    gpu: bool = False,
    api_key: Optional[str] = None,
) -> DockerfileResult:
    """
    Request Dockerfile generation for a GitHub repo.

    Args:
        github_url: GitHub repository URL
        branch: Git branch (default: repo's default branch)
        gpu: Whether to use a GPU/CUDA base image
        api_key: Optional explicit API key. Falls back to REMYXAI_API_KEY env var.

    Returns:
        DockerfileResult with task_id and initial status.
        If a cached result exists, status will be "completed" with dockerfile_text.
    """
    url = f"{BASE_URL}/dockergen/generate"

    payload = {"github_url": github_url.strip()}
    if branch:
        payload["branch"] = branch
    if gpu:
        payload["gpu"] = True

    logging.info(f"POST request to {url}")
    logging.debug(f"Payload: {payload}")

    headers = _resolve_headers(api_key)
    response = requests.post(url, json=payload, headers=headers, timeout=30)

    log_api_response(response)

    if response.status_code in (200, 202):
        return DockerfileResult.from_dict(response.json())
    elif response.status_code == 429:
        raise RuntimeError(
            "Rate limit exceeded. Try again later or check your daily quota."
        )
    else:
        logging.error(f"Failed to generate dockerfile: {response.status_code}")
        response.raise_for_status()


def get_dockergen_status(
    task_id: str,
    api_key: Optional[str] = None,
) -> DockerfileResult:
    """
    Check the status of a Dockerfile generation task.

    Args:
        task_id: UUID of the generation task
        api_key: Optional explicit API key.

    Returns:
        DockerfileResult with current status and results if completed.
    """
    url = f"{BASE_URL}/dockergen/status/{task_id}"

    logging.info(f"GET request to {url}")

    headers = _resolve_headers(api_key)
    response = requests.get(url, headers=headers, timeout=30)

    log_api_response(response)

    if response.status_code == 200:
        return DockerfileResult.from_dict(response.json())
    elif response.status_code == 404:
        logging.warning(f"Task {task_id} not found")
        raise ValueError(f"Task {task_id} not found")
    else:
        logging.error(f"Failed to get status: {response.status_code}")
        response.raise_for_status()


def lookup_dockerfile(
    github_url: str,
    branch: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[DockerfileResult]:
    """
    Look up a cached Dockerfile for a GitHub repo.

    Args:
        github_url: GitHub repository URL
        branch: Optional branch filter
        api_key: Optional explicit API key.

    Returns:
        DockerfileResult if a cached Dockerfile exists, None otherwise.
    """
    url = f"{BASE_URL}/dockergen/lookup"

    params = {"github_url": github_url.strip()}
    if branch:
        params["branch"] = branch

    logging.info(f"GET request to {url}")
    logging.debug(f"Params: {params}")

    headers = _resolve_headers(api_key)
    response = requests.get(url, headers=headers, params=params, timeout=30)

    if response.status_code == 404:
        logging.info(f"No cached Dockerfile for {github_url}")
        return None

    log_api_response(response)

    if response.status_code == 200:
        return DockerfileResult.from_dict(response.json())
    else:
        logging.error(f"Failed to lookup dockerfile: {response.status_code}")
        response.raise_for_status()
