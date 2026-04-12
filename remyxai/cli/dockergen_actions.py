# remyxai/cli/dockergen_actions.py
"""
CLI actions for generating validated Dockerfiles from GitHub repos.
"""
import json
import logging
import sys
from typing import Optional

from remyxai.api.dockergen import (
    generate_dockerfile,
    get_dockergen_status,
    lookup_dockerfile,
)
from remyxai.client.dockergen import DockerGenClient

logger = logging.getLogger(__name__)

STATUS_ICONS = {
    "pending": "\u23f3",       # hourglass
    "generating": "\u2699\ufe0f",  # gear
    "building": "\U0001f528",  # hammer
    "validating": "\U0001f504",  # arrows
    "completed": "\u2705",     # check
    "failed": "\u274c",        # cross
}


def _print_result(result, output_format: str = "text"):
    """Print a DockerfileResult in the requested format."""
    if output_format == "json":
        print(json.dumps(result.to_dict(), indent=2))
        return

    icon = STATUS_ICONS.get(result.status, "?")
    print(f"\n{icon} Status: {result.status}")
    print(f"   Task ID: {result.task_id}")
    print(f"   Repo: {result.github_url}")
    if result.branch:
        print(f"   Branch: {result.branch}")

    if result.status == "completed":
        if result.build_attempts:
            print(f"   Build attempts: {result.build_attempts}")
        if result.base_image:
            print(f"   Base image: {result.base_image}")
        if result.completed_at:
            print(f"   Completed: {result.completed_at}")
        print(f"\n{'=' * 60}")
        print(result.dockerfile_text)
        print(f"{'=' * 60}")
    elif result.status == "failed":
        if result.failure_reason:
            print(f"   Error: {result.failure_reason}")


def handle_generate(
    github_url: str,
    branch: Optional[str] = None,
    gpu: bool = False,
    output_format: str = "text",
):
    """Handle fire-and-forget generation request."""
    try:
        result = generate_dockerfile(github_url, branch=branch, gpu=gpu)
        _print_result(result, output_format)

        if output_format == "text" and result.status not in ("completed", "failed"):
            print(f"\n\U0001f4a1 Check progress with:")
            print(f"   remyxai dockergen status {result.task_id}")
    except Exception as e:
        print(f"\u274c Generation failed: {e}")
        logger.error(f"Generate error: {e}", exc_info=True)
        sys.exit(1)


def handle_generate_wait(
    github_url: str,
    branch: Optional[str] = None,
    gpu: bool = False,
    timeout: int = 300,
    output_format: str = "text",
    output_file: Optional[str] = None,
):
    """Handle blocking generation request with polling."""
    try:
        client = DockerGenClient()
        if output_format == "text":
            print(f"\n\U0001f433 Generating Dockerfile for {github_url}...")
            print(f"   This may take a few minutes.\n")

        last_status = None

        def on_status(result):
            nonlocal last_status
            if result.status != last_status:
                last_status = result.status
                if output_format == "text":
                    icon = STATUS_ICONS.get(result.status, "?")
                    print(f"   {icon} {result.status}")

        result = client.generate_and_poll(
            github_url,
            branch=branch,
            gpu=gpu,
            timeout=timeout,
            poll_interval=5,
            on_status=on_status,
        )

        if output_file and result.status == "completed" and result.dockerfile_text:
            with open(output_file, "w") as f:
                f.write(result.dockerfile_text)
            if output_format == "text":
                print(f"\n\u2705 Dockerfile written to {output_file}")
            return

        _print_result(result, output_format)

        if result.status not in ("completed", "failed"):
            if output_format == "text":
                print(f"\n\u23f3 Still processing after {timeout}s timeout.")
                print(f"   Check later with: remyxai dockergen status {result.task_id}")
    except Exception as e:
        print(f"\u274c Generation failed: {e}")
        logger.error(f"Generate wait error: {e}", exc_info=True)
        sys.exit(1)


def handle_status(task_id: str, output_format: str = "text"):
    """Handle status check."""
    try:
        result = get_dockergen_status(task_id)
        _print_result(result, output_format)
    except ValueError as e:
        print(f"\u274c {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\u274c Status check failed: {e}")
        logger.error(f"Status error: {e}", exc_info=True)
        sys.exit(1)


def handle_lookup(
    github_url: str,
    branch: Optional[str] = None,
    output_format: str = "text",
):
    """Handle cached Dockerfile lookup."""
    try:
        result = lookup_dockerfile(github_url, branch=branch)

        if result is None:
            if output_format == "json":
                print(json.dumps({"found": False, "github_url": github_url}))
            else:
                print(f"\nNo cached Dockerfile found for {github_url}")
                print(f"\n\U0001f4a1 Generate one with:")
                print(f"   remyxai dockergen generate {github_url} --wait")
            return

        _print_result(result, output_format)
    except Exception as e:
        print(f"\u274c Lookup failed: {e}")
        logger.error(f"Lookup error: {e}", exc_info=True)
        sys.exit(1)
