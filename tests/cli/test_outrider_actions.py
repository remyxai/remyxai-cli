"""Tests for the Outrider install action handler.

Covers:
- Repo detection (`_detect_github_repo_from_cwd`)
- Workflow template rendering + UUID validation
- `gh secret set` invariants (stdin-only credential path)
- CLI wiring via `remyxai.cli.commands` (now unblocked since myxboard
  cleanup removed the HfFolder import)
- `--dry-run` zero-mutation contract
- Rollback path on post-mutation failure
- UUID-validation rejection
"""
import subprocess
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from remyxai.cli import outrider_actions
from remyxai.cli.commands import cli


# ─── repo detection ────────────────────────────────────────────────────────

def test_detect_github_repo_https(monkeypatch):
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **k: "https://github.com/remyxai/outrider.git\n",
    )
    assert outrider_actions._detect_github_repo_from_cwd() == "remyxai/outrider"


def test_detect_github_repo_ssh(monkeypatch):
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **k: "git@github.com:remyxai/outrider.git\n",
    )
    assert outrider_actions._detect_github_repo_from_cwd() == "remyxai/outrider"


def test_detect_github_repo_non_github(monkeypatch):
    """Non-github.com remotes return None — Outrider requires GitHub."""
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **k: "https://gitlab.com/remyxai/outrider.git\n",
    )
    assert outrider_actions._detect_github_repo_from_cwd() is None


def test_detect_github_repo_no_git(monkeypatch):
    """Outside a git repo, returns None instead of raising."""
    def fake(*a, **k):
        raise subprocess.CalledProcessError(128, a)
    monkeypatch.setattr("subprocess.check_output", fake)
    assert outrider_actions._detect_github_repo_from_cwd() is None


# ─── workflow template ─────────────────────────────────────────────────────

def test_render_workflow_substitutes_interest_id():
    rendered = outrider_actions._render_workflow(
        "6a730cc4-010c-49ce-9c7f-6d9c59431739"
    )
    assert "__INTEREST_ID__" not in rendered
    assert "interest-id: 6a730cc4-010c-49ce-9c7f-6d9c59431739" in rendered


def test_render_workflow_pins_outrider_v1():
    rendered = outrider_actions._render_workflow("uuid")
    assert "uses: remyxai/outrider@v1" in rendered


def test_render_workflow_declares_both_secrets():
    rendered = outrider_actions._render_workflow("uuid")
    assert "REMYX_API_KEY: ${{ secrets.REMYX_API_KEY }}" in rendered
    assert "ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}" in rendered


def test_render_workflow_includes_workflow_dispatch():
    rendered = outrider_actions._render_workflow("uuid")
    assert "workflow_dispatch:" in rendered


# ─── UUID validation ──────────────────────────────────────────────────────

def test_uuid_re_accepts_valid_uuid():
    assert outrider_actions.UUID_RE.match(
        "6a730cc4-010c-49ce-9c7f-6d9c59431739"
    )


def test_uuid_re_rejects_obviously_wrong():
    for bad in ["", "not-a-uuid", "12345", "6a730cc4", "g" * 36]:
        assert not outrider_actions.UUID_RE.match(bad), f"matched: {bad!r}"


def test_resolve_interest_id_rejects_bad_uuid_via_flag():
    """The --interest flag must reject malformed UUIDs before any
    network call (no engine.remyx.ai roundtrip on bad input)."""
    with pytest.raises(click.UsageError):
        outrider_actions._resolve_interest_id(
            interest_id="not-a-uuid",
            auto_interest=False,
            repo="owner/repo",
            remyxai_key="k",
        )


# ─── gh secret-stdin invariant ─────────────────────────────────────────────

def test_gh_set_secret_passes_value_via_stdin(monkeypatch):
    """The secret value must go through stdin, never argv. This is the
    invariant that matters for credential safety."""
    captured = {}

    class _Done:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        return _Done()

    monkeypatch.setattr("subprocess.run", fake_run)
    outrider_actions._gh_set_secret("owner/repo", "MY_SECRET", "supersecret")
    assert "supersecret" not in captured["cmd"]
    assert captured["input"] == "supersecret"


def test_gh_set_secret_403_surfaces_scope_hint(monkeypatch):
    """When gh complains about permissions, the error explicitly
    names the required scope so the operator can re-auth correctly."""
    class _Fail:
        returncode = 1
        stderr = "HTTP 403: Resource not accessible by integration"
    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Fail())
    with pytest.raises(click.ClickException) as exc_info:
        outrider_actions._gh_set_secret("owner/repo", "MY_SECRET", "x")
    msg = str(exc_info.value.message)
    assert "admin scope" in msg
    assert "owner/repo" in msg


# ─── CLI wiring ────────────────────────────────────────────────────────────

@patch("remyxai.cli.commands.handle_outrider_init")
def test_outrider_init_passes_args_through(mock_handler):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["outrider", "init",
         "--repo", "owner/repo",
         "--interest", "6a730cc4-010c-49ce-9c7f-6d9c59431739",
         "--yes"],
    )
    assert result.exit_code == 0
    mock_handler.assert_called_once_with(
        repo="owner/repo",
        interest_id="6a730cc4-010c-49ce-9c7f-6d9c59431739",
        auto_interest=False,
        branch_name="install-outrider",
        skip_confirm=True,
        dry_run=False,
    )


@patch("remyxai.cli.commands.handle_outrider_init")
def test_outrider_init_dry_run_flag(mock_handler):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["outrider", "init", "--dry-run", "--repo", "x/y",
         "--interest", "00000000-0000-0000-0000-000000000000"],
    )
    assert result.exit_code == 0
    _, kwargs = mock_handler.call_args
    assert kwargs["dry_run"] is True
    assert kwargs["skip_confirm"] is False  # dry-run still implies opt-in confirm


def test_outrider_init_help_lists_new_options():
    runner = CliRunner()
    result = runner.invoke(cli, ["outrider", "init", "--help"])
    assert result.exit_code == 0
    assert "--repo" in result.output
    assert "--dry-run" in result.output
    assert "--interest" in result.output
    assert "--auto-interest" in result.output


def test_outrider_init_mutual_exclusion_raises(monkeypatch):
    """--interest + --auto-interest is caught by the handler before any
    GitHub API call."""
    monkeypatch.setenv("REMYXAI_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    with pytest.raises(click.UsageError):
        outrider_actions.handle_outrider_init(
            repo="owner/repo",
            interest_id="6a730cc4-010c-49ce-9c7f-6d9c59431739",
            auto_interest=True,
            branch_name="b",
            skip_confirm=True,
            dry_run=False,
        )


# ─── --dry-run zero-mutation contract ──────────────────────────────────────

def test_dry_run_performs_no_mutations(monkeypatch, capsys):
    """--dry-run prints the plan + rendered workflow, performs zero
    GitHub API mutations, and exits clean. Regression guard for the
    contract Salma's review surfaced."""
    monkeypatch.setenv("REMYXAI_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    # Stub read-only API probes.
    monkeypatch.setattr(outrider_actions, "_gh_available", lambda: True)
    monkeypatch.setattr(outrider_actions, "_gh_authenticated", lambda: True)
    monkeypatch.setattr(outrider_actions, "_gh_default_branch", lambda repo: "main")
    monkeypatch.setattr(
        "remyxai.cli.outrider_actions.get_interest",
        lambda interest_id, api_key=None: {"id": interest_id},
    )

    # Sentinel: any mutation call = test fails.
    def _no_mutation(*a, **k):
        raise AssertionError("dry-run must not call mutation APIs")

    monkeypatch.setattr(outrider_actions, "_gh_create_branch", _no_mutation)
    monkeypatch.setattr(outrider_actions, "_gh_put_file", _no_mutation)
    monkeypatch.setattr(outrider_actions, "_gh_open_pr", _no_mutation)
    monkeypatch.setattr(outrider_actions, "_gh_set_secret", _no_mutation)
    monkeypatch.setattr(outrider_actions, "_gh_branch_exists", _no_mutation)

    outrider_actions.handle_outrider_init(
        repo="owner/repo",
        interest_id="6a730cc4-010c-49ce-9c7f-6d9c59431739",
        auto_interest=False,
        branch_name="install-outrider",
        skip_confirm=True,
        dry_run=True,
    )

    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "interest-id: 6a730cc4-010c-49ce-9c7f-6d9c59431739" in out


# ─── rollback path on post-mutation failure ────────────────────────────────

def test_failure_after_branch_creation_deletes_branch(monkeypatch):
    """If `_gh_put_file` fails after the branch was created, the rollback
    handler must delete the branch (no orphan refs left behind)."""
    monkeypatch.setenv("REMYXAI_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    monkeypatch.setattr(outrider_actions, "_gh_available", lambda: True)
    monkeypatch.setattr(outrider_actions, "_gh_authenticated", lambda: True)
    monkeypatch.setattr(outrider_actions, "_gh_default_branch", lambda repo: "main")
    monkeypatch.setattr(outrider_actions, "_gh_branch_exists", lambda *a: False)
    monkeypatch.setattr(
        "remyxai.cli.outrider_actions.get_interest",
        lambda interest_id, api_key=None: {"id": interest_id},
    )
    monkeypatch.setattr(
        outrider_actions, "_gh_get_branch_sha", lambda *a: "abc1234"
    )
    monkeypatch.setattr(outrider_actions, "_gh_create_branch", lambda *a: None)

    def _put_fails(*a, **k):
        raise click.ClickException("simulated PUT failure")
    monkeypatch.setattr(outrider_actions, "_gh_put_file", _put_fails)

    delete_calls = []
    monkeypatch.setattr(
        outrider_actions,
        "_gh_delete_branch",
        lambda repo, branch: delete_calls.append((repo, branch)),
    )
    close_calls = []
    monkeypatch.setattr(
        outrider_actions,
        "_gh_close_pr",
        lambda repo, num: close_calls.append((repo, num)),
    )

    with pytest.raises(click.ClickException, match="simulated PUT failure"):
        outrider_actions.handle_outrider_init(
            repo="owner/repo",
            interest_id="6a730cc4-010c-49ce-9c7f-6d9c59431739",
            auto_interest=False,
            branch_name="install-outrider",
            skip_confirm=True,
            dry_run=False,
        )

    assert delete_calls == [("owner/repo", "install-outrider")], (
        f"branch should be deleted on PUT failure; got: {delete_calls}"
    )
    assert close_calls == [], (
        f"PR was never opened; close_pr should not have been called; got: {close_calls}"
    )


def test_failure_after_pr_opens_closes_pr_and_deletes_branch(monkeypatch):
    """If `_gh_set_secret` fails AFTER the PR was opened, BOTH the PR
    must be closed AND the branch deleted on rollback."""
    monkeypatch.setenv("REMYXAI_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    monkeypatch.setattr(outrider_actions, "_gh_available", lambda: True)
    monkeypatch.setattr(outrider_actions, "_gh_authenticated", lambda: True)
    monkeypatch.setattr(outrider_actions, "_gh_default_branch", lambda repo: "main")
    monkeypatch.setattr(outrider_actions, "_gh_branch_exists", lambda *a: False)
    monkeypatch.setattr(
        "remyxai.cli.outrider_actions.get_interest",
        lambda interest_id, api_key=None: {"id": interest_id},
    )
    monkeypatch.setattr(
        outrider_actions, "_gh_get_branch_sha", lambda *a: "abc1234"
    )
    monkeypatch.setattr(outrider_actions, "_gh_create_branch", lambda *a: None)
    monkeypatch.setattr(outrider_actions, "_gh_put_file", lambda **k: None)
    monkeypatch.setattr(
        outrider_actions, "_gh_open_pr",
        lambda **k: ("https://github.com/owner/repo/pull/42", 42),
    )

    def _set_secret_fails(*a, **k):
        raise click.ClickException("simulated secret failure")
    monkeypatch.setattr(outrider_actions, "_gh_set_secret", _set_secret_fails)

    delete_calls = []
    close_calls = []
    monkeypatch.setattr(
        outrider_actions,
        "_gh_delete_branch",
        lambda repo, branch: delete_calls.append((repo, branch)),
    )
    monkeypatch.setattr(
        outrider_actions,
        "_gh_close_pr",
        lambda repo, num: close_calls.append((repo, num)),
    )

    with pytest.raises(click.ClickException, match="simulated secret failure"):
        outrider_actions.handle_outrider_init(
            repo="owner/repo",
            interest_id="6a730cc4-010c-49ce-9c7f-6d9c59431739",
            auto_interest=False,
            branch_name="install-outrider",
            skip_confirm=True,
            dry_run=False,
        )

    assert close_calls == [("owner/repo", 42)], (
        f"PR should be closed on secret failure; got: {close_calls}"
    )
    assert delete_calls == [("owner/repo", "install-outrider")], (
        f"branch should be deleted on secret failure; got: {delete_calls}"
    )


# ─── idempotency ───────────────────────────────────────────────────────────

def test_idempotency_existing_branch_bails_before_secrets(monkeypatch):
    """If the branch already exists on origin, the handler must fail
    BEFORE any secret-set call. Prevents the case Salma flagged where
    a retry re-sets secrets then dies on the branch step."""
    monkeypatch.setenv("REMYXAI_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    monkeypatch.setattr(outrider_actions, "_gh_available", lambda: True)
    monkeypatch.setattr(outrider_actions, "_gh_authenticated", lambda: True)
    monkeypatch.setattr(outrider_actions, "_gh_default_branch", lambda repo: "main")
    monkeypatch.setattr(
        "remyxai.cli.outrider_actions.get_interest",
        lambda interest_id, api_key=None: {"id": interest_id},
    )

    # Branch ALREADY exists.
    monkeypatch.setattr(outrider_actions, "_gh_branch_exists", lambda *a: True)

    # Sentinel: any mutation = test fails.
    monkeypatch.setattr(
        outrider_actions, "_gh_set_secret",
        lambda *a, **k: pytest.fail("must not set secret when branch exists"),
    )
    monkeypatch.setattr(
        outrider_actions, "_gh_create_branch",
        lambda *a, **k: pytest.fail("must not create branch that already exists"),
    )

    with pytest.raises(click.ClickException, match="already exists"):
        outrider_actions.handle_outrider_init(
            repo="owner/repo",
            interest_id="6a730cc4-010c-49ce-9c7f-6d9c59431739",
            auto_interest=False,
            branch_name="install-outrider",
            skip_confirm=True,
            dry_run=False,
        )
