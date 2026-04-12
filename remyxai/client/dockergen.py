"""
Python client for DockerGen API

Thin wrapper around remyxai.api.dockergen for convenience.
Used by the HF Space (remyx-builder) and external integrations.
"""
import time
from typing import Optional

from remyxai.api.dockergen import (
    DockerfileResult,
    generate_dockerfile as api_generate_dockerfile,
    get_dockergen_status as api_get_dockergen_status,
    lookup_dockerfile as api_lookup_dockerfile,
)


class DockerGenClient:
    """
    Client for interacting with Remyx DockerGen API.

    Examples:

        # Using environment variable (default -- works with CLI)
        >>> from remyxai.client.dockergen import DockerGenClient
        >>>
        >>> client = DockerGenClient()
        >>> result = client.generate("https://github.com/org/repo")

        # Using explicit API key (e.g. in a web app or HF Space)
        >>> client = DockerGenClient(api_key="your_key_here")
        >>> result = client.generate_and_poll("https://github.com/org/repo")
        >>> print(result.dockerfile_text)
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize DockerGen client.

        Args:
            api_key: Optional explicit API key. If not provided, the
                     REMYXAI_API_KEY environment variable is used.
        """
        self.api_key = api_key

    def generate(
        self,
        github_url: str,
        branch: Optional[str] = None,
        gpu: bool = False,
    ) -> DockerfileResult:
        """Submit a Dockerfile generation request. Returns immediately."""
        return api_generate_dockerfile(
            github_url, branch=branch, gpu=gpu, api_key=self.api_key
        )

    def status(self, task_id: str) -> DockerfileResult:
        """Check the status of a generation task."""
        return api_get_dockergen_status(task_id, api_key=self.api_key)

    def lookup(
        self,
        github_url: str,
        branch: Optional[str] = None,
    ) -> Optional[DockerfileResult]:
        """Look up a cached Dockerfile for a repo. Returns None if not found."""
        return api_lookup_dockerfile(
            github_url, branch=branch, api_key=self.api_key
        )

    def generate_and_poll(
        self,
        github_url: str,
        branch: Optional[str] = None,
        gpu: bool = False,
        timeout: int = 300,
        poll_interval: int = 5,
        on_status: callable = None,
    ) -> DockerfileResult:
        """
        Generate a Dockerfile and poll until completion or timeout.

        Args:
            github_url: GitHub repository URL
            branch: Git branch (optional)
            gpu: Whether to use GPU/CUDA base image
            timeout: Max seconds to wait (default 300)
            poll_interval: Seconds between polls (default 5)
            on_status: Optional callback(DockerfileResult) called on each poll

        Returns:
            DockerfileResult with final status. If timed out, status will
            still be in-progress -- caller can use task_id to check later.
        """
        result = self.generate(github_url, branch=branch, gpu=gpu)

        if result.status in ("completed", "failed"):
            return result

        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(poll_interval)
            result = self.status(result.task_id)
            if on_status:
                on_status(result)
            if result.status in ("completed", "failed"):
                return result

        return result
