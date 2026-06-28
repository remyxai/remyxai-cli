"""
CLI action handler for the no-App ("local") Outrider setup path.

`remyxai outrider setup-local` installs Outrider on a repo WITHOUT the Remyx
GitHub App — for enterprises that can't (or won't, yet) grant a third-party
App while a security review is pending. It uses the customer's own
authenticated `gh` CLI to set the repo secrets, write the workflow, and open
(optionally merge) the setup PR. The only Remyx dependency is the
REMYX_API_KEY the workflow uses at runtime to fetch recommendations.

This is the self-provisioning counterpart to `outrider init` (which drives the
engine + Remyx App). Same running Action; different installer + PR-author.

So the running Action can open its recommendation PRs (no bot token here), the
CLI enables the repo's "Allow Actions to create and approve PRs" setting and
the workflow uses the built-in GITHUB_TOKEN. No GitHub token is stored as a
secret — only REMYX_API_KEY and ANTHROPIC_API_KEY.

Side effects are ordered reversible-first (branch, workflow, PR) with secrets
last, and the branch + PR roll back on any post-mutation failure.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from typing import Optional

import click

# Shared helpers with the engine path (repo parsing + interest resolution).
from remyxai.cli.outrider_actions import (
    _detect_github_repo_from_cwd,
    _normalize_repo,
    _resolve_interest_id,
)

logger = logging.getLogger(__name__)

WORKFLOW_FILENAME = "outrider.yml"
WORKFLOW_PATH = f".github/workflows/{WORKFLOW_FILENAME}"
PR_TITLE = "Install Outrider — weekly arXiv → recommendation PRs"


# ─── gh helpers ─────────────────────────────────────────────────────────────

def _gh_available() -> bool:
    return shutil.which("gh") is not None


def _gh_authenticated() -> bool:
    if not _gh_available():
        return False
    return subprocess.run(
        ["gh", "api", "user", "--silent"], capture_output=True, text=True,
    ).returncode == 0


def _gh_api_json(args: list) -> dict:
    result = subprocess.run(["gh", "api", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise click.ClickException(
            f"GitHub API call failed ({' '.join(args[:2])}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    try:
        return json.loads(result.stdout) if result.stdout.strip() else {}
    except json.JSONDecodeError:
        return {}


def _gh_default_branch(repo: str) -> str:
    return _gh_api_json([f"/repos/{repo}"]).get("default_branch") or "main"


def _gh_branch_exists(repo: str, branch: str) -> bool:
    return subprocess.run(
        ["gh", "api", f"/repos/{repo}/branches/{branch}", "--silent"],
        capture_output=True, text=True,
    ).returncode == 0


def _gh_get_branch_sha(repo: str, branch: str) -> str:
    ref = _gh_api_json([f"/repos/{repo}/git/ref/heads/{branch}"])
    sha = ref.get("object", {}).get("sha")
    if not sha:
        raise click.ClickException(f"could not resolve SHA for {repo}@{branch}")
    return sha


def _gh_create_branch(repo: str, branch: str, from_sha: str) -> None:
    _gh_api_json([
        "-X", "POST", f"/repos/{repo}/git/refs",
        "-f", f"ref=refs/heads/{branch}", "-f", f"sha={from_sha}",
    ])


def _gh_delete_branch(repo: str, branch: str) -> None:
    r = subprocess.run(
        ["gh", "api", "-X", "DELETE",
         f"/repos/{repo}/git/refs/heads/{branch}", "--silent"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        logger.warning("rollback: failed to delete branch %s: %s", branch, r.stderr.strip())


def _gh_get_file_sha(repo: str, path: str, branch: str) -> Optional[str]:
    """Existing blob SHA of `path` on `branch`, or None. Required to overwrite."""
    r = subprocess.run(
        ["gh", "api", f"/repos/{repo}/contents/{path}?ref={branch}"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    return data.get("sha") if isinstance(data, dict) else None


def _gh_put_file(repo, branch, path, content, commit_message) -> None:
    """Create/update a file on `branch`. Passes the existing sha so a repo that
    already carries the workflow doesn't 422 ('sha wasn't supplied')."""
    import base64
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    args = [
        "-X", "PUT", f"/repos/{repo}/contents/{path}",
        "-f", f"message={commit_message}",
        "-f", f"content={encoded}", "-f", f"branch={branch}",
    ]
    existing = _gh_get_file_sha(repo, path, branch)
    if existing:
        args += ["-f", f"sha={existing}"]
    _gh_api_json(args)


def _gh_open_pr(repo, head, base, title, body, draft=True) -> tuple:
    args = [
        "-X", "POST", f"/repos/{repo}/pulls",
        "-f", f"title={title}", "-f", f"head={head}", "-f", f"base={base}",
        "-f", f"body={body}", "-F", f"draft={'true' if draft else 'false'}",
    ]
    pr = _gh_api_json(args)
    return pr["html_url"], pr["number"]


def _gh_close_pr(repo: str, number: int) -> None:
    r = subprocess.run(
        ["gh", "api", "-X", "PATCH", f"/repos/{repo}/pulls/{number}",
         "-f", "state=closed", "--silent"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        logger.warning("rollback: failed to close PR #%s: %s", number, r.stderr.strip())


def _gh_merge_pr(repo: str, number: int) -> bool:
    """Best-effort merge. Returns True on success; False (with a message) if the
    repo's protections block it — the PR stays open for the user to merge."""
    r = subprocess.run(
        ["gh", "api", "-X", "PUT", f"/repos/{repo}/pulls/{number}/merge",
         "-f", "merge_method=squash"],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def _gh_set_secret(repo: str, name: str, value: str) -> None:
    """Set a repo secret via stdin (never argv/logs)."""
    r = subprocess.run(
        ["gh", "secret", "set", name, "--repo", repo],
        input=value, text=True, capture_output=True,
    )
    if r.returncode != 0:
        stderr = r.stderr.strip()
        hint = ""
        if "403" in stderr or "permission" in stderr.lower():
            hint = (f"\n  Your gh token likely lacks admin scope on {repo}. "
                    f"Re-auth with `gh auth login` or a PAT with repo+workflow scopes.")
        raise click.ClickException(f"failed to set secret {name!r} on {repo}: {stderr}{hint}")


def _gh_enable_pr_creation(repo: str) -> None:
    """Allow Actions to create/approve PRs (so the workflow's GITHUB_TOKEN can
    open recommendation PRs). Requires admin on the repo."""
    _gh_api_json([
        "-X", "PUT", f"/repos/{repo}/actions/permissions/workflow",
        "-F", "default_workflow_permissions=write",
        "-F", "can_approve_pull_request_reviews=true",
    ])


def _gh_dispatch(repo: str, branch: str) -> bool:
    r = subprocess.run(
        ["gh", "api", "-X", "POST",
         f"/repos/{repo}/actions/workflows/{WORKFLOW_FILENAME}/dispatches",
         "-f", f"ref={branch}", "--silent"],
        capture_output=True, text=True,
    )
    return r.returncode == 0


# ─── workflow rendering (inline; no Remyx App / bot-token step) ─────────────

def _render_local_workflow(interest_id: str, no_cron: bool = False) -> str:
    # No github-token input → the action uses this repo's built-in
    # GITHUB_TOKEN, which setup-local authorizes to open PRs.
    #
    # When ``no_cron=True``, the schedule block is rendered commented-out
    # (not omitted) so the user can re-enable scheduled runs later by
    # uncommenting three lines, without re-running setup-local.
    if no_cron:
        schedule_block = (
            "  # schedule:\n"
            "  #   - cron: '0 14 * * 1'   # Mondays 14:00 UTC; uncomment to enable\n"
        )
    else:
        schedule_block = (
            "  schedule:\n"
            "    - cron: '0 14 * * 1'   # Mondays 14:00 UTC; pick any cadence\n"
        )
    return f"""name: Outrider

# Generated by `remyxai outrider setup-local` (no Remyx GitHub App).
# Weekly scout: queries engine.remyx.ai for a paper recommendation against
# this repo's ResearchInterest, then opens a draft PR wiring it in.
#   https://github.com/remyxai/outrider

on:
{schedule_block}  workflow_dispatch:

jobs:
  recommend:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    permissions:
      contents: write
      pull-requests: write
      issues: write
    env:
      REMYX_API_KEY: ${{{{ secrets.REMYX_API_KEY }}}}
      ANTHROPIC_API_KEY: ${{{{ secrets.ANTHROPIC_API_KEY }}}}
    steps:
      - uses: remyxai/outrider@v1
        with:
          interest-id: {interest_id}
          # Minimum days between recommendation PRs on this repo. '0' lets
          # every scheduled or manually-triggered run open a PR; raise it
          # (e.g. '7') to cap how often Outrider posts.
          rate-limit-days: '0'
"""


# ─── main handler ──────────────────────────────────────────────────────────

def handle_outrider_setup_local(
    repo, interest_id, auto_interest, mode,
    anthropic_key, skip_confirm, dry_run, no_cron=False,
):
    """Self-provision Outrider with the user's own gh token (no Remyx App)."""
    import os

    if interest_id and auto_interest:
        raise click.UsageError(
            "--interest and --auto-interest are mutually exclusive."
        )

    # 1. REMYX key (set as a repo secret + used to resolve the interest)
    remyx_key = os.environ.get("REMYXAI_API_KEY") or click.prompt(
        "REMYXAI_API_KEY (from engine.remyx.ai Settings)", hide_input=True
    )
    if not remyx_key.strip():
        raise click.ClickException("REMYXAI_API_KEY is required.")

    # 2. Anthropic key (set as a repo secret; the engine isn't involved here)
    anthropic_key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        anthropic_key = click.prompt(
            "ANTHROPIC_API_KEY (from console.anthropic.com)", hide_input=True
        )
    if not anthropic_key.strip():
        raise click.ClickException("ANTHROPIC_API_KEY is required for the workflow.")

    # 3. Repo
    resolved_repo = _normalize_repo(repo) if repo else _detect_github_repo_from_cwd()
    if not resolved_repo:
        raise click.ClickException(
            "No GitHub repo specified or detected. Pass --repo owner/name."
        )
    repo_url = f"https://github.com/{resolved_repo}"

    # 4. Plan
    click.echo("")
    click.echo("Plan (no Remyx GitHub App — uses your gh credentials):")
    click.echo(f"  - Repo:      {resolved_repo}")
    click.echo(f"  - Mode:      {mode} (auto = open + merge PR + dispatch; review = open PR only)")
    click.echo(f"  - Secrets:   REMYX_API_KEY, ANTHROPIC_API_KEY")
    click.echo("  - PR auth:   enable the repo 'Actions can create PRs' setting "
               "(PRs by github-actions[bot])")
    click.echo(f"  - Writes:    {WORKFLOW_PATH} on a branch + opens a PR")
    click.echo("")

    if dry_run:
        click.echo("--- rendered workflow ---")
        click.echo(_render_local_workflow("<interest-id>", no_cron=no_cron))
        click.secho("dry-run: no changes made.", fg="yellow")
        return

    # 5. gh preconditions
    if not _gh_available():
        raise click.ClickException(
            "`gh` (GitHub CLI) is not installed. See https://cli.github.com."
        )
    if not _gh_authenticated():
        raise click.ClickException(
            "`gh` cannot authenticate. Run `gh auth login` or set a valid "
            "$GITHUB_TOKEN with repo + workflow scopes, then re-run."
        )

    if not skip_confirm:
        click.confirm("Proceed?", abort=True, default=False)

    # 6. Resolve interest (engine call — the interest lives server-side)
    resolved_interest = _resolve_interest_id(
        interest_id, auto_interest, resolved_repo, repo_url, remyx_key
    )

    default_branch = _gh_default_branch(resolved_repo)
    branch_name = "install-outrider"
    if _gh_branch_exists(resolved_repo, branch_name):
        raise click.ClickException(
            f"branch {branch_name!r} already exists on {resolved_repo}. "
            f"Delete it or merge/close the existing setup PR, then re-run."
        )

    # 7. Execute — reversible first (branch, file, PR), secrets last; rollback
    pr_number = None
    branch_created = False
    try:
        base_sha = _gh_get_branch_sha(resolved_repo, default_branch)
        _gh_create_branch(resolved_repo, branch_name, base_sha)
        branch_created = True
        click.echo(f"✓ Created branch {branch_name}")

        workflow = _render_local_workflow(resolved_interest, no_cron=no_cron)
        _gh_put_file(resolved_repo, branch_name, WORKFLOW_PATH, workflow,
                     "Install Outrider (self-provisioned via remyxai CLI)")
        click.echo(f"✓ Wrote {WORKFLOW_PATH}")

        body = (
            f"Installs [Outrider](https://github.com/remyxai/outrider) "
            f"(self-provisioned, no Remyx GitHub App).\n\n"
            f"Research interest: `{resolved_interest}`\n\n"
            f"Generated by `remyxai outrider setup-local`."
        )
        pr_url, pr_number = _gh_open_pr(
            resolved_repo, branch_name, default_branch, PR_TITLE, body,
            draft=(mode != "auto"),
        )
        click.echo(f"✓ Opened PR: {pr_url}")

        # Let the running Action open PRs with the built-in GITHUB_TOKEN.
        _gh_enable_pr_creation(resolved_repo)
        click.echo("✓ Enabled Actions PR creation on the repo")

        # Secrets LAST (closest to success; least cleanup risk).
        _gh_set_secret(resolved_repo, "REMYX_API_KEY", remyx_key)
        click.echo("✓ Set REMYX_API_KEY")
        _gh_set_secret(resolved_repo, "ANTHROPIC_API_KEY", anthropic_key)
        click.echo("✓ Set ANTHROPIC_API_KEY")
    except Exception as e:
        if pr_number is not None:
            click.echo(f"  ↩ rolling back: closing PR #{pr_number}", err=True)
            _gh_close_pr(resolved_repo, pr_number)
        if branch_created:
            click.echo(f"  ↩ rolling back: deleting branch {branch_name}", err=True)
            _gh_delete_branch(resolved_repo, branch_name)
        if isinstance(e, click.ClickException):
            raise
        raise click.ClickException(f"setup-local failed: {e}")

    # 8. auto mode — merge + dispatch
    merged = False
    if mode == "auto":
        merged = _gh_merge_pr(resolved_repo, pr_number)
        if merged:
            click.echo("✓ Merged the setup PR")
            if _gh_dispatch(resolved_repo, default_branch):
                click.echo("✓ Dispatched the first run")
        else:
            click.secho(
                "  Could not auto-merge (branch protection?). The PR is open — "
                "merge it to activate Outrider.", fg="yellow",
            )

    # 9. Report
    click.echo("")
    click.secho("✓ Outrider set up (no App).", fg="green", bold=True)
    click.echo(f"  PR:       {pr_url}")
    if mode == "auto" and merged:
        click.echo("  Status:   active — a recommendation PR will appear shortly.")
    else:
        click.echo("  Next:     merge the PR to activate Outrider.")
    click.echo(f"  Manual:   gh workflow run {WORKFLOW_FILENAME} --repo {resolved_repo}")
