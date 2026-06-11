"""
CLI action handlers for Outrider lifecycle management.

`remyxai outrider init` sets up Outrider on a GitHub repo by driving the
Remyx engine — the same server-side "set it up for me" flow as the web app.
The engine, via the Remyx GitHub App (remyx-ai[bot]), sets the repo secrets,
writes the workflow, opens a bot-authored setup PR, and (in `auto` mode)
merges it and fires the first run.

Nothing touches the user's local git or a personal `gh` token; the CLI only
needs the user's REMYX_API_KEY. Flow:

  1. Resolve the target repo (owner/name).
  2. Resolve the ResearchInterest (provided UUID, auto-created, or prompted).
  3. Ensure the Remyx GitHub App is installed on the repo (surface the install
     link + poll — installing is an interactive browser step).
  4. Ensure a model provider (Anthropic via `claude_code`) is connected.
  5. Kick off provisioning and report the bot-authored setup PR.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from typing import Optional

import click

from remyxai.api.interests import (
    get_interest,
    provision_action,
    poll_provision_action,
)
from remyxai.cli.interest_actions import (
    RepoAnalysisError,
    create_interest_from_repo,
    _kick_off_recommendations,
)
from remyxai.api.integrations import connect_credential, get_integration_status
from remyxai.api.github_app import get_app_install_url, is_app_installed

logger = logging.getLogger(__name__)

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# Anthropic via the `claude_code` integration — the only model provider wired
# today (the engine's MODEL_PROVIDERS registry is the source of truth).
MODEL_PROVIDER = "claude_code"

INSTALL_POLL_INTERVAL = 5     # seconds between App-install checks
INSTALL_POLL_TIMEOUT = 300    # stop waiting for the browser install after 5 min
PROVISION_POLL_INTERVAL = 3
PROVISION_POLL_TIMEOUT = 300


# ─── repo resolution (read-only; never mutates the working tree) ───────────

def _normalize_repo(value: str) -> Optional[str]:
    """Accept owner/name, an https URL, or an ssh URL → 'owner/name'."""
    value = (value or "").strip()
    if value.endswith(".git"):
        value = value[:-4]
    for pat in (
        r"^git@github\.com:([\w.-]+/[\w.-]+)$",
        r"^https?://github\.com/([\w.-]+/[\w.-]+)$",
        r"^([\w.-]+/[\w.-]+)$",
    ):
        m = re.match(pat, value)
        if m:
            return m.group(1)
    return None


def _detect_github_repo_from_cwd() -> Optional[str]:
    """Parse `origin` of the cwd's git repo → 'owner/name', or None."""
    try:
        out = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return _normalize_repo(out)


# ─── interest resolution ───────────────────────────────────────────────────

def _resolve_interest_id(interest_id, auto_interest, repo, repo_url, api_key):
    """Resolve to a validated interest UUID via flag / auto-create / prompt."""
    if interest_id:
        if not UUID_RE.match(interest_id):
            raise click.UsageError(
                f"--interest must be a UUID, got: {interest_id!r}"
            )
        try:
            get_interest(interest_id, api_key=api_key)
        except Exception as e:
            raise click.ClickException(
                f"interest {interest_id} could not be fetched from "
                f"engine.remyx.ai: {e}. Check the UUID and your REMYX_API_KEY."
            )
        return interest_id

    if auto_interest:
        click.echo(
            "Creating a Research Interest from this repo (may take 30-90s)…"
        )
        # Use the analyze-repo flow so the interest gets a rich,
        # ExperimentHistory-derived context (and the server dispatches
        # extraction) instead of a URL-only stub. Paper-PR
        # provisioning is handled separately below by `outrider init`,
        # so we don't provision here (automate="none").
        try:
            created = create_interest_from_repo(
                repo_url,
                name=repo.split("/")[-1],
                daily_count=3,
                is_active=True,
                automate="none",
                api_key=api_key,
                echo=click.echo,
            )
        except RepoAnalysisError as e:
            raise click.ClickException(
                f"interest creation failed during repo analysis: {e}\n"
                f"  Try again, or create one at engine.remyx.ai and re-run "
                f"with --interest <uuid>."
            )
        except Exception as e:
            raise click.ClickException(
                f"interest creation failed: {e}\n"
                f"  Create one at engine.remyx.ai and re-run with "
                f"--interest <uuid>."
            )
        new_id = created.get("id")
        if not new_id or not UUID_RE.match(new_id):
            raise click.ClickException(
                f"interest creation did not return a UUID: {created}"
            )
        click.echo(f"✓ Created interest: {new_id}")
        if created.get("history_extraction_task_id"):
            click.echo(
                "  🧪 experiment-history extraction dispatched; "
                "interest context will keep deepening as it completes."
            )
        return new_id

    typed = click.prompt("Remyx interest UUID (from engine.remyx.ai)").strip()
    if not UUID_RE.match(typed):
        raise click.UsageError(f"interest UUID is malformed: {typed!r}")
    return typed


# ─── preflight helpers ─────────────────────────────────────────────────────

def _ensure_app_installed(repo, api_key, no_wait, sleep=time.sleep):
    """Confirm the Remyx App is installed on `repo`; otherwise surface the
    install link and poll until it is (installing is a browser step)."""
    if is_app_installed(repo, api_key=api_key):
        click.echo(f"✓ Remyx GitHub App is installed on {repo}")
        return

    info = get_app_install_url(api_key=api_key)
    if not info.get("configured", True) or not info.get("install_url"):
        raise click.ClickException(
            "The Remyx GitHub App isn't configured on the server. "
            "Contact Remyx support."
        )
    click.echo("")
    click.secho(
        "Action needed — install the Remyx GitHub App on this repo:",
        fg="yellow", bold=True,
    )
    click.echo(f"  {info['install_url']}")
    click.echo("  (grant it access to the repo, then come back here)")
    if no_wait:
        raise click.ClickException(
            "App not installed yet. Install it via the link above, then re-run."
        )
    click.echo("\nWaiting for the install to complete…")
    waited = 0
    while waited < INSTALL_POLL_TIMEOUT:
        sleep(INSTALL_POLL_INTERVAL)
        waited += INSTALL_POLL_INTERVAL
        if is_app_installed(repo, api_key=api_key):
            click.echo(f"✓ Remyx GitHub App is now installed on {repo}")
            return
    raise click.ClickException(
        f"Timed out after {INSTALL_POLL_TIMEOUT}s waiting for the App install. "
        f"Install it via the link above and re-run."
    )


def _ensure_model_provider(anthropic_key, api_key):
    """Ensure a model provider (Anthropic) is connected. Returns True if so.

    Non-fatal when absent: provisioning still proceeds, but the first run
    can't complete until a key is connected — so we warn loudly.
    """
    try:
        status = get_integration_status(MODEL_PROVIDER, api_key=api_key)
    except Exception:
        status = {"connected": False}
    if status.get("connected"):
        click.echo("✓ Model provider (Claude Code) is connected")
        return True

    key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        click.secho(
            "⚠ No model provider connected. Provisioning will proceed, but the "
            "first run can't complete until you connect an Anthropic key "
            "(pass --anthropic-key, set ANTHROPIC_API_KEY, or connect Claude "
            "Code in Integrations).",
            fg="yellow",
        )
        return False
    try:
        connect_credential(MODEL_PROVIDER, {"api_key": key}, api_key=api_key)
    except Exception as e:
        raise click.ClickException(f"Failed to connect the Anthropic key: {e}")
    click.echo("✓ Connected Claude Code (Anthropic) key")
    return True


def _wait_for_provision(interest_id, task_id, api_key, sleep=time.sleep):
    """Poll the provision task until it completes or fails; return result."""
    waited = 0
    last_msg = None
    while waited < PROVISION_POLL_TIMEOUT:
        task = poll_provision_action(interest_id, task_id, api_key=api_key)
        status = task.get("status")
        msg = task.get("message")
        if msg and msg != last_msg:
            click.echo(f"  … {msg}")
            last_msg = msg
        if status == "completed":
            return task.get("result") or {}
        if status == "failed":
            raise click.ClickException(
                f"Provisioning failed: {task.get('error') or 'unknown error'}"
            )
        sleep(PROVISION_POLL_INTERVAL)
        waited += PROVISION_POLL_INTERVAL
    raise click.ClickException(
        f"Timed out after {PROVISION_POLL_TIMEOUT}s waiting for provisioning "
        f"(task {task_id}). It may still finish server-side."
    )


# ─── main handler ──────────────────────────────────────────────────────────

def handle_outrider_init(
    repo, interest_id, auto_interest, mode,
    anthropic_key, skip_confirm, dry_run, no_wait,
):
    """Set up Outrider on a repo via the Remyx engine. Called from
    commands.outrider_init."""
    if interest_id and auto_interest:
        raise click.UsageError(
            "--interest and --auto-interest are mutually exclusive."
        )

    # 1. API key (the only credential the CLI needs)
    api_key = os.environ.get("REMYXAI_API_KEY") or click.prompt(
        "REMYXAI_API_KEY (from engine.remyx.ai Settings)", hide_input=True
    )
    if not api_key.strip():
        raise click.ClickException("REMYXAI_API_KEY is required.")

    # 2. Resolve repo
    resolved_repo = _normalize_repo(repo) if repo else _detect_github_repo_from_cwd()
    if not resolved_repo:
        raise click.ClickException(
            "No GitHub repo specified or detected. Pass --repo owner/name."
        )
    repo_url = f"https://github.com/{resolved_repo}"

    # 3. Plan
    mode_desc = {
        "auto": "provision, merge the setup PR, and start the first run",
        "review": "provision and open a setup PR for you to review and merge",
        "off": "create the interest only (no provisioning)",
    }[mode]
    interest_desc = (
        f"use existing interest {interest_id}" if interest_id
        else "auto-create an interest from the repo" if auto_interest
        else "prompt for an interest UUID"
    )
    click.echo("")
    click.echo("Plan:")
    click.echo(f"  - Repo:      {resolved_repo}")
    click.echo(f"  - Interest:  {interest_desc}")
    click.echo(f"  - Mode:      {mode} — {mode_desc}")
    click.echo(
        "  - The engine installs everything server-side as remyx-ai[bot]; "
        "your local git is untouched."
    )
    click.echo("")

    if dry_run:
        click.secho("dry-run: no changes made.", fg="yellow")
        return

    if not skip_confirm:
        click.confirm("Proceed?", abort=True, default=False)

    # 4. Resolve interest
    resolved_interest = _resolve_interest_id(
        interest_id, auto_interest, resolved_repo, repo_url, api_key
    )

    # 5. Mode `off` stops here — interest only.
    if mode == "off":
        click.echo("")
        click.secho("✓ Interest ready.", fg="green", bold=True)
        click.echo(f"  Interest: {resolved_interest}")
        click.echo(
            f"  To provision later: remyxai outrider init "
            f"--repo {resolved_repo} --interest {resolved_interest}"
        )
        return

    # 6. Preflight: App install + model provider (only needed to provision)
    _ensure_app_installed(resolved_repo, api_key, no_wait)
    _ensure_model_provider(anthropic_key, api_key)

    # 6b. Pre-warm recommendations so the first run has picks to open a PR
    # from. A brand-new interest ranks asynchronously; firing the Outrider
    # first run before the pool populates makes it report "no recommendations"
    # (the cold-start race). Trigger a refresh now and — unless --no-wait —
    # block until the pool is populated before provisioning dispatches the run.
    click.echo("\nWarming up recommendations for the interest…")
    _kick_off_recommendations(
        resolved_interest,
        wait=(not no_wait),
        api_key=api_key,
        echo=click.echo,
    )

    # 7. Provision (server-side, bot-authored)
    auto_merge = (mode == "auto")
    click.echo("\nProvisioning Outrider via engine.remyx.ai…")
    resp = provision_action(
        resolved_interest, repo_url=repo_url,
        auto_merge=auto_merge, api_key=api_key,
    )
    task_id = resp.get("task_id")
    if not task_id:
        raise click.ClickException(
            f"provision-action did not return a task_id: {resp}"
        )
    if no_wait:
        click.echo(
            f"  Provisioning started (task {task_id}); it runs server-side."
        )
        return

    result = _wait_for_provision(resolved_interest, task_id, api_key)

    # 8. Report
    click.echo("")
    click.secho("✓ Outrider is set up.", fg="green", bold=True)
    if result.get("pr_url"):
        label = "Setup PR (merged)" if result.get("merged") else "Setup PR"
        click.echo(f"  {label}: {result['pr_url']}")
    click.echo(
        f"  Repo secret REMYX_API_KEY: "
        f"{'set' if result.get('secret_set') else 'not set'}"
    )
    if result.get("dispatched"):
        click.echo(
            "  First run: dispatched — a recommendation PR will appear shortly."
        )
    elif result.get("merged"):
        click.echo(
            "  First run: starts on schedule (or once recommendations populate)."
        )
    else:
        click.echo("  Next: merge the setup PR to activate Outrider.")
    if result.get("model_key_missing"):
        click.secho(
            "  ⚠ No model provider key set — connect Claude Code so the first "
            "run can complete.", fg="yellow",
        )
