"""Tests for `remyxai outrider trigger` — REMYX-148.

Covers:
- CLI wiring (option flags, mutex enforcement, click usage errors)
- Repo resolution (explicit, auto-detect from git, missing)
- Default ref resolution (from gh API) + explicit ref override
- gh-dispatch invocation (correct path, ref, pin-method / pin-arxiv inputs)
- Failure surfacing (404 not-installed, 403 missing-scope)
- Run-URL best-effort lookup

The subprocess calls are mocked at the patch boundary; no real network.
"""
import subprocess
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from remyxai.cli import outrider_actions
from remyxai.cli.commands import cli


# ─── _gh_dispatch_outrider ────────────────────────────────────────────────


def test_gh_dispatch_includes_pin_method_input():
    """Non-empty inputs flow through to the gh-api -f args."""
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, "", "")

    with patch("subprocess.run", side_effect=fake_run):
        ok, err = outrider_actions._gh_dispatch_outrider(
            "owner/name", "main",
            {"pin-method": "knowledge distillation", "pin-arxiv": "",
             "interest-id": ""},
        )
    assert ok is True
    args = captured["args"]
    assert "ref=main" in args
    assert "pin-method=knowledge distillation" in args
    # Empty inputs are dropped, not sent as empty strings.
    assert not any(a.startswith("pin-arxiv=") for a in args)
    assert not any(a.startswith("interest-id=") for a in args)


def test_gh_dispatch_surfaces_stderr_on_failure():
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, "", "HTTP 404: Not Found")

    with patch("subprocess.run", side_effect=fake_run):
        ok, err = outrider_actions._gh_dispatch_outrider(
            "owner/name", "main", {"pin-method": "X"},
        )
    assert ok is False
    assert "404" in err


# ─── _gh_default_branch ───────────────────────────────────────────────────


def test_default_branch_from_gh_api():
    def fake_check_output(args, **kwargs):
        return "develop\n"

    with patch("subprocess.check_output", side_effect=fake_check_output):
        assert outrider_actions._gh_default_branch("owner/name") == "develop"


def test_default_branch_none_on_gh_failure():
    def fake_check_output(args, **kwargs):
        raise subprocess.CalledProcessError(1, args)

    with patch("subprocess.check_output", side_effect=fake_check_output):
        assert outrider_actions._gh_default_branch("owner/name") is None


# ─── handle_outrider_trigger — high-level flow ────────────────────────────


def test_trigger_mutex_pin_method_and_pin_arxiv():
    with pytest.raises(click.UsageError, match="mutually exclusive"):
        outrider_actions.handle_outrider_trigger(
            repo="owner/name",
            pin_method="X", pin_arxiv="2410.20305v2",
            interest_id=None, ref=None,
        )


def test_trigger_errors_when_no_repo_and_not_in_git_checkout(monkeypatch):
    monkeypatch.setattr(outrider_actions, "_detect_github_repo_from_cwd",
                        lambda: None)
    with pytest.raises(click.UsageError, match="Could not determine target repo"):
        outrider_actions.handle_outrider_trigger(
            repo=None, pin_method="X", pin_arxiv=None,
            interest_id=None, ref=None,
        )


def test_trigger_refuses_when_workflow_not_installed(monkeypatch):
    """Pre-flight: trigger refuses to dispatch on repos that haven't been
    initialized with `remyxai outrider init`. Surfaces a clear install
    hint before any dispatch attempt."""
    monkeypatch.setattr(outrider_actions, "_outrider_workflow_exists",
                        lambda repo: False)
    monkeypatch.setattr(outrider_actions, "_gh_default_branch",
                        lambda repo: pytest.fail(
                            "must short-circuit before resolving ref"))
    monkeypatch.setattr(outrider_actions, "_gh_dispatch_outrider",
                        lambda r, b, i: pytest.fail(
                            "must short-circuit before dispatch"))

    with pytest.raises(click.ClickException) as exc:
        outrider_actions.handle_outrider_trigger(
            repo="owner/name", pin_method="X", pin_arxiv=None,
            interest_id=None, ref=None,
        )
    msg = exc.value.message.lower()
    assert "not installed" in msg
    assert "outrider init" in msg


def test_trigger_403_surfaces_scope_hint(monkeypatch):
    monkeypatch.setattr(outrider_actions, "_outrider_workflow_exists",
                        lambda repo: True)
    monkeypatch.setattr(outrider_actions, "_gh_default_branch",
                        lambda repo: "main")
    monkeypatch.setattr(outrider_actions, "_gh_dispatch_outrider",
                        lambda r, b, i: (False, "HTTP 403: missing permission"))

    with pytest.raises(click.ClickException) as exc:
        outrider_actions.handle_outrider_trigger(
            repo="owner/name", pin_method="X", pin_arxiv=None,
            interest_id=None, ref=None,
        )
    assert "scope" in exc.value.message.lower()


def test_trigger_happy_path_with_pin_method(monkeypatch, capsys):
    captured_dispatch = {}

    def fake_dispatch(repo, branch, inputs):
        captured_dispatch["repo"] = repo
        captured_dispatch["branch"] = branch
        captured_dispatch["inputs"] = inputs
        return (True, "")

    monkeypatch.setattr(outrider_actions, "_outrider_workflow_exists",
                        lambda repo: True)
    monkeypatch.setattr(outrider_actions, "_gh_default_branch",
                        lambda repo: "main")
    monkeypatch.setattr(outrider_actions, "_gh_dispatch_outrider", fake_dispatch)
    monkeypatch.setattr(outrider_actions, "_gh_latest_run_url",
                        lambda repo, sleep=None:
                        "https://github.com/owner/name/actions/runs/123")

    outrider_actions.handle_outrider_trigger(
        repo="owner/name", pin_method="knowledge distillation",
        pin_arxiv=None, interest_id=None, ref=None,
    )

    assert captured_dispatch["repo"] == "owner/name"
    assert captured_dispatch["branch"] == "main"
    assert captured_dispatch["inputs"]["pin-method"] == "knowledge distillation"
    assert captured_dispatch["inputs"]["pin-arxiv"] == ""

    out = capsys.readouterr().out
    assert "Dispatched" in out
    assert "runs/123" in out


def test_trigger_uses_explicit_ref(monkeypatch):
    seen = {}

    def fake_dispatch(repo, branch, inputs):
        seen["branch"] = branch
        return (True, "")

    monkeypatch.setattr(outrider_actions, "_outrider_workflow_exists",
                        lambda repo: True)
    monkeypatch.setattr(outrider_actions, "_gh_default_branch",
                        lambda repo: pytest.fail("should not query when ref is set"))
    monkeypatch.setattr(outrider_actions, "_gh_dispatch_outrider", fake_dispatch)
    monkeypatch.setattr(outrider_actions, "_gh_latest_run_url",
                        lambda repo, sleep=None: None)

    outrider_actions.handle_outrider_trigger(
        repo="owner/name", pin_method="X", pin_arxiv=None,
        interest_id=None, ref="release/v2",
    )
    assert seen["branch"] == "release/v2"


# ─── _outrider_workflow_exists ────────────────────────────────────────────


def test_outrider_workflow_exists_true_when_gh_returns_zero():
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, "", "")
    with patch("subprocess.run", side_effect=fake_run):
        assert outrider_actions._outrider_workflow_exists("owner/name") is True


def test_outrider_workflow_exists_false_on_404():
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, "", "HTTP 404")
    with patch("subprocess.run", side_effect=fake_run):
        assert outrider_actions._outrider_workflow_exists("owner/name") is False


# ─── CLI integration via click runner ─────────────────────────────────────


def test_cli_outrider_trigger_pin_method(monkeypatch):
    monkeypatch.setattr(outrider_actions, "_outrider_workflow_exists",
                        lambda repo: True)
    monkeypatch.setattr(outrider_actions, "_gh_default_branch",
                        lambda repo: "main")
    monkeypatch.setattr(outrider_actions, "_gh_dispatch_outrider",
                        lambda r, b, i: (True, ""))
    monkeypatch.setattr(outrider_actions, "_gh_latest_run_url",
                        lambda r, sleep=None: None)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "outrider", "trigger",
        "--repo", "owner/name",
        "--pin-method", "knowledge distillation",
    ])
    assert result.exit_code == 0, result.output
    assert "Dispatched" in result.output
    assert "pin-method" in result.output


def test_cli_outrider_trigger_mutex_via_click():
    runner = CliRunner()
    result = runner.invoke(cli, [
        "outrider", "trigger",
        "--repo", "owner/name",
        "--pin-method", "X", "--pin-arxiv", "2410.20305v2",
    ])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()
