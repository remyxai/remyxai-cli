"""Tests for `remyxai outrider trigger`.

Covers:
- CLI wiring (option flags, mutex enforcement, click usage errors)
- Repo resolution (explicit, auto-detect from git, missing)
- Default ref resolution (from gh API) + explicit ref override
- gh-dispatch invocation (correct path, ref, search-method / pin-arxiv inputs)
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


def test_gh_dispatch_includes_search_method_input():
    """Non-empty inputs flow through to `gh workflow run -f <key>=<val>`."""
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, "", "")

    with patch("subprocess.run", side_effect=fake_run):
        ok, err = outrider_actions._gh_dispatch_outrider(
            "owner/name", "main",
            {"search-method": "knowledge distillation", "pin-arxiv": "",
             "interest-id": ""},
        )
    assert ok is True
    args = captured["args"]
    # gh workflow run nests inputs under `inputs.*` server-side, so the
    # raw POST never trips the "X is not a permitted key" rejection.
    assert args[:3] == ["gh", "workflow", "run"]
    assert "outrider.yml" in args
    assert "--repo" in args and "owner/name" in args
    assert "--ref" in args and "main" in args
    assert "search-method=knowledge distillation" in args
    # Empty inputs are dropped, not sent as empty strings.
    assert not any(a.startswith("pin-arxiv=") for a in args)
    assert not any(a.startswith("interest-id=") for a in args)


def test_gh_dispatch_surfaces_stderr_on_failure():
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, "", "HTTP 404: Not Found")

    with patch("subprocess.run", side_effect=fake_run):
        ok, err = outrider_actions._gh_dispatch_outrider(
            "owner/name", "main", {"search-method": "X"},
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


def test_trigger_mutex_search_method_and_pin_arxiv():
    with pytest.raises(click.UsageError, match="mutually exclusive"):
        outrider_actions.handle_outrider_trigger(
            repo="owner/name",
            search_method="X", pin_arxiv="2410.20305v2",
            interest_id=None, ref=None,
        )


def test_trigger_errors_when_no_repo_and_not_in_git_checkout(monkeypatch):
    monkeypatch.setattr(outrider_actions, "_detect_github_repo_from_cwd",
                        lambda: None)
    with pytest.raises(click.UsageError, match="Could not determine target repo"):
        outrider_actions.handle_outrider_trigger(
            repo=None, search_method="X", pin_arxiv=None,
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
            repo="owner/name", search_method="X", pin_arxiv=None,
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
            repo="owner/name", search_method="X", pin_arxiv=None,
            interest_id=None, ref=None,
        )
    assert "scope" in exc.value.message.lower()


def test_trigger_happy_path_with_search_method(monkeypatch, capsys):
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
        repo="owner/name", search_method="knowledge distillation",
        pin_arxiv=None, interest_id=None, ref=None,
    )

    assert captured_dispatch["repo"] == "owner/name"
    assert captured_dispatch["branch"] == "main"
    assert captured_dispatch["inputs"]["search-method"] == "knowledge distillation"
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
        repo="owner/name", search_method="X", pin_arxiv=None,
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


def test_cli_outrider_trigger_search_method(monkeypatch):
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
        "--search-method", "knowledge distillation",
    ])
    assert result.exit_code == 0, result.output
    assert "Dispatched" in result.output
    assert "search-method" in result.output


def test_cli_outrider_trigger_mutex_via_click():
    runner = CliRunner()
    result = runner.invoke(cli, [
        "outrider", "trigger",
        "--repo", "owner/name",
        "--search-method", "X", "--pin-arxiv", "2410.20305v2",
    ])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()


# ─── --claude-timeout forwarding ──────────────────────────────────────────


def test_trigger_forwards_claude_timeout_when_set(monkeypatch, capsys):
    """`--claude-timeout 1800` flows to the workflow_dispatch as a string."""
    captured = {}

    def fake_dispatch(repo, branch, inputs):
        captured["inputs"] = inputs
        return (True, "")

    monkeypatch.setattr(outrider_actions, "_outrider_workflow_exists",
                        lambda repo: True)
    monkeypatch.setattr(outrider_actions, "_gh_default_branch",
                        lambda repo: "main")
    monkeypatch.setattr(outrider_actions, "_gh_dispatch_outrider", fake_dispatch)
    monkeypatch.setattr(outrider_actions, "_gh_latest_run_url",
                        lambda repo, sleep=None: None)

    outrider_actions.handle_outrider_trigger(
        repo="owner/name", search_method="X", pin_arxiv=None,
        interest_id=None, ref=None, claude_timeout=1800,
    )
    # Stringified at the dispatch boundary because workflow_dispatch
    # input values are strings on the wire.
    assert captured["inputs"]["claude-timeout"] == "1800"
    out = capsys.readouterr().out
    assert "claude-timeout: 1800s" in out


def test_trigger_omits_claude_timeout_when_unset(monkeypatch):
    """No flag → empty string → `_gh_dispatch_outrider` drops it →
    the action's own default (900s) applies."""
    captured = {}

    def fake_dispatch(repo, branch, inputs):
        captured["inputs"] = inputs
        return (True, "")

    monkeypatch.setattr(outrider_actions, "_outrider_workflow_exists",
                        lambda repo: True)
    monkeypatch.setattr(outrider_actions, "_gh_default_branch",
                        lambda repo: "main")
    monkeypatch.setattr(outrider_actions, "_gh_dispatch_outrider", fake_dispatch)
    monkeypatch.setattr(outrider_actions, "_gh_latest_run_url",
                        lambda repo, sleep=None: None)

    outrider_actions.handle_outrider_trigger(
        repo="owner/name", search_method="X", pin_arxiv=None,
        interest_id=None, ref=None,
    )
    assert captured["inputs"]["claude-timeout"] == ""


def test_trigger_rejects_claude_timeout_below_minimum():
    """Catch obviously-wrong values at the CLI boundary rather than
    waiting for the action to fail on a too-tight ceiling."""
    with pytest.raises(click.UsageError, match="at least 60 seconds"):
        outrider_actions.handle_outrider_trigger(
            repo="owner/name", search_method="X", pin_arxiv=None,
            interest_id=None, ref=None, claude_timeout=30,
        )


def test_cli_claude_timeout_flag_accepted_and_dispatched(monkeypatch):
    """End-to-end through click: --claude-timeout reaches the dispatch."""
    captured = {}

    def fake_dispatch(repo, branch, inputs):
        captured["inputs"] = inputs
        return (True, "")

    monkeypatch.setattr(outrider_actions, "_outrider_workflow_exists",
                        lambda repo: True)
    monkeypatch.setattr(outrider_actions, "_gh_default_branch",
                        lambda repo: "main")
    monkeypatch.setattr(outrider_actions, "_gh_dispatch_outrider", fake_dispatch)
    monkeypatch.setattr(outrider_actions, "_gh_latest_run_url",
                        lambda r, sleep=None: None)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "outrider", "trigger",
        "--repo", "owner/name",
        "--search-method", "2410.20305v2",
        "--claude-timeout", "2700",
    ])
    assert result.exit_code == 0, result.output
    assert captured["inputs"]["claude-timeout"] == "2700"
    assert "claude-timeout: 2700s" in result.output


def test_cli_claude_timeout_must_be_integer():
    """Click's `type=int` rejects non-integer values at the boundary."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        "outrider", "trigger",
        "--repo", "owner/name",
        "--search-method", "X",
        "--claude-timeout", "nope",
    ])
    assert result.exit_code != 0
    assert "not a valid integer" in result.output.lower()


# ─── --provider forwarding ─────────────────────────────────────────────────


def test_trigger_forwards_provider_when_set(monkeypatch, capsys):
    """`--provider zai` flows to the workflow_dispatch as a string input."""
    captured = {}

    def fake_dispatch(repo, branch, inputs):
        captured["inputs"] = inputs
        return (True, "")

    monkeypatch.setattr(outrider_actions, "_outrider_workflow_exists",
                        lambda repo: True)
    monkeypatch.setattr(outrider_actions, "_gh_default_branch",
                        lambda repo: "main")
    monkeypatch.setattr(outrider_actions, "_gh_dispatch_outrider", fake_dispatch)
    monkeypatch.setattr(outrider_actions, "_gh_latest_run_url",
                        lambda repo, sleep=None: None)

    outrider_actions.handle_outrider_trigger(
        repo="owner/name", search_method="X", pin_arxiv=None,
        interest_id=None, ref=None, provider="zai",
    )
    assert captured["inputs"]["provider"] == "zai"
    out = capsys.readouterr().out
    assert "provider:       zai" in out


def test_trigger_omits_provider_when_unset(monkeypatch):
    """No flag → empty string → `_gh_dispatch_outrider` drops it →
    the workflow's own default provider applies."""
    captured = {}

    def fake_dispatch(repo, branch, inputs):
        captured["inputs"] = inputs
        return (True, "")

    monkeypatch.setattr(outrider_actions, "_outrider_workflow_exists",
                        lambda repo: True)
    monkeypatch.setattr(outrider_actions, "_gh_default_branch",
                        lambda repo: "main")
    monkeypatch.setattr(outrider_actions, "_gh_dispatch_outrider", fake_dispatch)
    monkeypatch.setattr(outrider_actions, "_gh_latest_run_url",
                        lambda repo, sleep=None: None)

    outrider_actions.handle_outrider_trigger(
        repo="owner/name", search_method="X", pin_arxiv=None,
        interest_id=None, ref=None,
    )
    assert captured["inputs"]["provider"] == ""


def test_cli_provider_flag_dispatched(monkeypatch):
    """End-to-end through click: --provider reaches the dispatch."""
    captured = {}

    def fake_dispatch(repo, branch, inputs):
        captured["inputs"] = inputs
        return (True, "")

    monkeypatch.setattr(outrider_actions, "_outrider_workflow_exists",
                        lambda repo: True)
    monkeypatch.setattr(outrider_actions, "_gh_default_branch",
                        lambda repo: "main")
    monkeypatch.setattr(outrider_actions, "_gh_dispatch_outrider", fake_dispatch)
    monkeypatch.setattr(outrider_actions, "_gh_latest_run_url",
                        lambda r, sleep=None: None)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "outrider", "trigger",
        "--repo", "owner/name",
        "--search-method", "2410.20305v2",
        "--provider", "anthropic",
    ])
    assert result.exit_code == 0, result.output
    assert captured["inputs"]["provider"] == "anthropic"
    assert "provider:       anthropic" in result.output


def test_cli_provider_combines_with_claude_timeout(monkeypatch):
    """Both new flags coexist; their inputs travel together to the
    workflow_dispatch payload."""
    captured = {}

    def fake_dispatch(repo, branch, inputs):
        captured["inputs"] = inputs
        return (True, "")

    monkeypatch.setattr(outrider_actions, "_outrider_workflow_exists",
                        lambda repo: True)
    monkeypatch.setattr(outrider_actions, "_gh_default_branch",
                        lambda repo: "main")
    monkeypatch.setattr(outrider_actions, "_gh_dispatch_outrider", fake_dispatch)
    monkeypatch.setattr(outrider_actions, "_gh_latest_run_url",
                        lambda r, sleep=None: None)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "outrider", "trigger",
        "--repo", "owner/name",
        "--search-method", "2410.20305v2",
        "--provider", "zai",
        "--claude-timeout", "1200",
    ])
    assert result.exit_code == 0, result.output
    assert captured["inputs"]["provider"] == "zai"
    assert captured["inputs"]["claude-timeout"] == "1200"
    assert captured["inputs"]["search-method"] == "2410.20305v2"


# ─── --model forwarding ───────────────────────────────────────────────────


def test_trigger_forwards_model_when_set(monkeypatch, capsys):
    """`--model glm-5.2` flows to the workflow_dispatch `model` input."""
    captured = {}

    def fake_dispatch(repo, branch, inputs):
        captured["inputs"] = inputs
        return (True, "")

    monkeypatch.setattr(outrider_actions, "_outrider_workflow_exists",
                        lambda repo: True)
    monkeypatch.setattr(outrider_actions, "_gh_default_branch",
                        lambda repo: "main")
    monkeypatch.setattr(outrider_actions, "_gh_dispatch_outrider", fake_dispatch)
    monkeypatch.setattr(outrider_actions, "_gh_latest_run_url",
                        lambda repo, sleep=None: None)

    outrider_actions.handle_outrider_trigger(
        repo="owner/name", search_method="X", pin_arxiv=None,
        interest_id=None, ref=None, provider="zai", model="glm-5.2",
    )
    assert captured["inputs"]["model"] == "glm-5.2"
    out = capsys.readouterr().out
    assert "model:          glm-5.2" in out


def test_trigger_omits_model_when_unset(monkeypatch):
    """No flag → empty string → `_gh_dispatch_outrider` drops it →
    the provider picks its own default."""
    captured = {}

    def fake_dispatch(repo, branch, inputs):
        captured["inputs"] = inputs
        return (True, "")

    monkeypatch.setattr(outrider_actions, "_outrider_workflow_exists",
                        lambda repo: True)
    monkeypatch.setattr(outrider_actions, "_gh_default_branch",
                        lambda repo: "main")
    monkeypatch.setattr(outrider_actions, "_gh_dispatch_outrider", fake_dispatch)
    monkeypatch.setattr(outrider_actions, "_gh_latest_run_url",
                        lambda repo, sleep=None: None)

    outrider_actions.handle_outrider_trigger(
        repo="owner/name", search_method="X", pin_arxiv=None,
        interest_id=None, ref=None,
    )
    assert captured["inputs"]["model"] == ""


def test_cli_model_combines_with_provider(monkeypatch):
    """End-to-end through click: --provider + --model + --search-method
    all reach the dispatch payload together."""
    captured = {}

    def fake_dispatch(repo, branch, inputs):
        captured["inputs"] = inputs
        return (True, "")

    monkeypatch.setattr(outrider_actions, "_outrider_workflow_exists",
                        lambda repo: True)
    monkeypatch.setattr(outrider_actions, "_gh_default_branch",
                        lambda repo: "main")
    monkeypatch.setattr(outrider_actions, "_gh_dispatch_outrider", fake_dispatch)
    monkeypatch.setattr(outrider_actions, "_gh_latest_run_url",
                        lambda r, sleep=None: None)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "outrider", "trigger",
        "--repo", "owner/name",
        "--search-method", "2410.20305v2",
        "--provider", "zai",
        "--model", "glm-4.6",
        "--claude-timeout", "1500",
    ])
    assert result.exit_code == 0, result.output
    assert captured["inputs"]["provider"] == "zai"
    assert captured["inputs"]["model"] == "glm-4.6"
    assert captured["inputs"]["claude-timeout"] == "1500"
    assert "model:          glm-4.6" in result.output
