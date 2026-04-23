"""
CLI action handlers for Research Interest management.
Called by the `remyxai interests` command group in commands.py.
"""
from __future__ import annotations

import json
import re
import sys
import textwrap
import time
from typing import Optional

import click

from remyxai.api.interests import (
    analyze_repo,
    create_interest,
    delete_interest,
    get_interest,
    list_github_repos,
    list_interests,
    poll_repo_analysis,
    regenerate_interest,
    toggle_interest,
    update_interest,
)


# ─── repo-analysis polling helper ────────────────────────────────────────────


def _wait_for_repo_analysis(
    task_id: str,
    timeout_s: int = 180,
    poll_interval_s: float = 2.0,
) -> dict:
    """Block until a repo-analysis task completes or fails.

    Returns the inner `result` payload (report_markdown, repo_analysis,
    source_repo_metadata, ...) — the Redis-backed task envelope wraps
    that payload, which callers shouldn't have to unwrap themselves.
    """
    click.echo(f"  task_id: {task_id}")
    click.echo("  Waiting for analysis to complete (up to {}s)...".format(timeout_s))

    deadline = time.monotonic() + timeout_s
    last_message = ""
    while time.monotonic() < deadline:
        task = poll_repo_analysis(task_id)
        status = task.get("status") or ""
        message = task.get("message") or ""

        if message and message != last_message:
            click.echo(f"    • {message}")
            last_message = message

        if status in ("complete", "completed", "done"):
            return task.get("result") or {}
        if status in ("failed", "error"):
            click.echo(
                f"❌ Analysis failed: {task.get('error') or message or 'unknown error'}",
                err=True,
            )
            sys.exit(1)

        time.sleep(poll_interval_s)

    click.echo(
        f"❌ Analysis did not complete within {timeout_s}s. "
        f"Poll manually with task_id={task_id}",
        err=True,
    )
    sys.exit(1)


# ─── formatting helpers ──────────────────────────────────────────────────────

def _print_interest(i: dict, verbose: bool = False) -> None:
    icon = "✅" if i.get("is_active") else "⏸ "
    last = i.get("last_recommendation_at")
    last_str = f"  last reco: {last[:10]}" if last else ""

    click.echo(f"\n  {icon}  {i['name']}")
    click.echo(f"       id:          {i['id']}")
    click.echo(f"       daily_count: {i.get('daily_count', 2)}")
    click.echo(f"       active:      {i.get('is_active', True)}{last_str}")

    if verbose:
        ctx = (i.get("context") or "").strip()
        if ctx:
            wrapped = textwrap.fill(
                ctx, width=72,
                initial_indent="       context:    ",
                subsequent_indent="                   ",
            )
            click.echo(wrapped)


# ─── name-or-id resolver ─────────────────────────────────────────────────────

def _resolve_interest_id(name_or_id: str) -> str:
    """Accept either a UUID or an interest name and return the UUID.

    If name_or_id looks like a UUID (36 chars, hyphen-separated),
    return it directly. Otherwise fetch all interests and match by name
    case-insensitively, returning the first match found.
    """
    if re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        name_or_id, re.IGNORECASE
    ):
        return name_or_id

    try:
        interests = list_interests()
    except Exception as e:
        click.echo("Failed to fetch interests: {}".format(e), err=True)
        sys.exit(1)

    needle = name_or_id.lower()
    for interest in interests:
        if interest.get("name", "").lower() == needle:
            return interest["id"]

    names = [i.get("name", "") for i in interests]
    available = ", ".join(names) if names else "none"
    click.echo(
        "No interest found with name {!r}. Available: {}".format(name_or_id, available),
        err=True,
    )
    sys.exit(1)


# ─── list ────────────────────────────────────────────────────────────────────

def handle_interests_list(output_format: str = "text") -> None:
    try:
        interests = list_interests()
    except Exception as e:
        click.echo(f"❌ Failed to fetch interests: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(interests, indent=2))
        return

    if not interests:
        click.echo(
            "\n  No Research Interests found.\n"
            "  Create one:  remyxai interests create\n"
        )
        return

    active = sum(1 for i in interests if i.get("is_active"))
    click.echo(
        f"\n🎯  Research Interests  "
        f"({len(interests)} total, {active} active)"
    )
    click.echo("━" * 60)
    for i in interests:
        _print_interest(i, verbose=True)
    click.echo()


# ─── get ─────────────────────────────────────────────────────────────────────

def handle_interests_get(interest_id: str, output_format: str = "text") -> None:
    interest_id = _resolve_interest_id(interest_id)
    try:
        interest = get_interest(interest_id)
    except Exception as e:
        click.echo(f"❌ Failed to fetch interest: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(interest, indent=2))
        return

    click.echo("\n🎯  Research Interest")
    click.echo("━" * 60)
    _print_interest(interest, verbose=True)
    click.echo()


# ─── create ──────────────────────────────────────────────────────────────────

def handle_interests_create(
    name: Optional[str],
    context: Optional[str],
    daily_count: int,
    inactive: bool,
    output_format: str,
    repo: Optional[str] = None,
) -> None:
    # Repo-sourced flow: kick off analysis, poll, use the returned
    # markdown as context, and persist the repo fields on save.
    repo_payload: Optional[dict] = None
    if repo:
        click.echo(f"\n🔍  Analyzing {repo} to seed a Research Interest...")
        try:
            kickoff = analyze_repo(repo)
        except Exception as e:
            click.echo(f"❌ Failed to start repo analysis: {e}", err=True)
            sys.exit(1)
        repo_payload = _wait_for_repo_analysis(kickoff["task_id"])
        click.echo("✅  Analysis complete.\n")

        # Prefer the server-generated markdown as context; fall back to the
        # user's --context if they supplied one.
        if not context:
            context = (
                repo_payload.get("report_markdown")
                or repo_payload.get("generated_report")
                or ""
            )
        # Auto-name from repo "owner/name" if not provided.
        if not name:
            meta = repo_payload.get("source_repo_metadata") or {}
            auto = (
                repo_payload.get("full_name")
                or meta.get("full_name")
                or meta.get("name")
            )
            if auto:
                name = auto
                click.echo(f"   Using auto-generated name: {name}\n")

    if not name:
        name = click.prompt("  Interest name (e.g. 'RAG & Retrieval')")
    if not context:
        click.echo(
            "  Context: describe what you want to track in natural language.\n"
            "  Can also be a HuggingFace or GitHub URL.\n"
            "  Examples:\n"
            "    'Retrieval-augmented generation, hybrid search, re-ranking'\n"
            "    'https://huggingface.co/ibm-granite/granitelib-rag-r1.0'\n"
        )
        context = click.prompt("  Context")

    try:
        create_kwargs = {
            "name": name,
            "context": context,
            "daily_count": daily_count,
            "is_active": not inactive,
        }
        if repo_payload:
            create_kwargs.update({
                "source_repo_url": repo,
                "source_repo_metadata": repo_payload.get("source_repo_metadata"),
                "generated_report": (
                    repo_payload.get("report_markdown")
                    or repo_payload.get("generated_report")
                ),
                "repo_analysis": repo_payload.get("repo_analysis"),
            })
        result = create_interest(**create_kwargs)
    except Exception as e:
        click.echo(f"❌ Failed to create interest: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
        return

    click.echo(f"\n✅  Created '{result['name']}'  (id: {result['id']})")
    click.echo(
        f"   daily_count: {result['daily_count']}  |  "
        f"active: {result['is_active']}\n"
    )
    click.echo(
        "  Trigger your first recommendations:\n"
        f"  remyxai papers refresh --interest {result['name']!r} --wait\n"
    )


# ─── update ──────────────────────────────────────────────────────────────────

def handle_interests_update(
    interest_id: str,
    name: Optional[str],
    context: Optional[str],
    daily_count: Optional[int],
    is_active: Optional[bool],
    output_format: str,
) -> None:
    interest_id = _resolve_interest_id(interest_id)
    if not any(v is not None for v in [name, context, daily_count, is_active]):
        click.echo(
            "❌ Provide at least one field to update:\n"
            "   --name, --context, --daily-count, --activate, --deactivate",
            err=True,
        )
        sys.exit(1)

    try:
        result = update_interest(
            interest_id=interest_id,
            name=name,
            context=context,
            daily_count=daily_count,
            is_active=is_active,
        )
    except Exception as e:
        click.echo(f"❌ Failed to update interest: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
        return

    click.echo(f"\n✅  Updated '{result['name']}'")
    if result.get("pool_invalidated"):
        click.echo(
            "   ℹ️  Context changed — recommendation pool cleared.\n"
            f"   Run:  remyxai papers refresh --interest {result['name']!r} --wait"
        )
    click.echo()


# ─── delete ──────────────────────────────────────────────────────────────────

def handle_interests_delete(
    interest_id: str,
    yes: bool,
    output_format: str,
) -> None:
    interest_id = _resolve_interest_id(interest_id)
    if not yes:
        try:
            i = get_interest(interest_id)
            label = i.get("name", interest_id)
        except Exception:
            label = interest_id

        click.confirm(
            f"  Delete Research Interest '{label}'? "
            "This removes all associated recommendations.",
            abort=True,
        )

    try:
        result = delete_interest(interest_id)
    except Exception as e:
        click.echo(f"❌ Failed to delete interest: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
        return

    click.echo(f"\n🗑️  Deleted interest {interest_id}\n")


# ─── toggle ──────────────────────────────────────────────────────────────────

def handle_interests_toggle(interest_id: str, output_format: str) -> None:
    interest_id = _resolve_interest_id(interest_id)
    try:
        result = toggle_interest(interest_id)
    except Exception as e:
        click.echo(f"❌ Failed to toggle interest: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
        return

    state = "active ✅" if result.get("is_active") else "paused ⏸"
    click.echo(f"\n  '{result['name']}' is now {state}\n")


# ─── regenerate ──────────────────────────────────────────────────────────────

def handle_interests_regenerate(
    interest_id: str,
    repo_url: Optional[str],
    output_format: str,
) -> None:
    """Re-run repo analysis against an existing interest and persist the
    refreshed payload via update_interest once the task completes."""
    interest_id = _resolve_interest_id(interest_id)

    try:
        kickoff = regenerate_interest(interest_id, repo_url=repo_url)
    except Exception as e:
        click.echo(f"❌ Failed to start regenerate: {e}", err=True)
        sys.exit(1)

    click.echo(
        f"\n🔁  Regenerating repo analysis for interest {interest_id}..."
    )
    payload = _wait_for_repo_analysis(kickoff["task_id"])

    try:
        result = update_interest(
            interest_id=interest_id,
            context=(
                payload.get("report_markdown") or payload.get("generated_report")
            ),
            source_repo_metadata=payload.get("source_repo_metadata"),
            generated_report=(
                payload.get("report_markdown") or payload.get("generated_report")
            ),
            repo_analysis=payload.get("repo_analysis"),
        )
    except Exception as e:
        click.echo(f"❌ Failed to apply regenerated payload: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
        return

    click.echo(f"\n✅  Regenerated '{result.get('name', interest_id)}'")
    if result.get("pool_invalidated"):
        click.echo(
            "   ℹ️  Recommendation pool cleared.\n"
            f"   Run:  remyxai papers refresh --interest "
            f"{result['name']!r} --wait"
        )
    click.echo()


# ─── list-repos ──────────────────────────────────────────────────────────────

def handle_interests_list_repos(output_format: str) -> None:
    """List GitHub repos the caller can source an interest from."""
    try:
        result = list_github_repos()
    except Exception as e:
        click.echo(f"❌ Failed to list GitHub repos: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
        return

    if not result.get("connected"):
        click.echo(
            "\n  GitHub is not connected for this account.\n"
            "  Connect at Settings → Integrations, then retry.\n"
        )
        return

    repos = result.get("repos") or []
    if not repos:
        click.echo("\n  No repos visible via the connected GitHub integration.\n")
        return

    click.echo(f"\n🐙  GitHub repos  ({len(repos)})")
    click.echo("━" * 60)
    for r in repos:
        name = r.get("full_name") or r.get("name") or "(unnamed)"
        url = r.get("html_url") or r.get("url") or ""
        private_tag = " [private]" if r.get("private") else ""
        click.echo(f"  {name}{private_tag}")
        if url:
            click.echo(f"       {url}")
    click.echo()
