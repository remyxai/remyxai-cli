"""Tests for `remyxai outrider trigger`.

Covers:
- CLI wiring (option flags, mutex enforcement, click usage errors)
- Repo resolution (explicit, auto-detect from git, missing)
- Default ref resolution (from gh API) + explicit ref override
- gh-dispatch invocation (correct path, ref, search-method / pin-arxiv inputs)
- Failure surfacing (404 not-installed, 403 missing-scope)
- Run-URL best-effort lookup
- Refinement inputs (mode/publish/start-from-ref/lead-content/…) reaching
  the dispatch, and the drop-and-retry fallback for older installs
- The pending-run warning (the workflow's static concurrency group holds one
  pending run, so a second dispatch silently cancels the first)

The subprocess calls are mocked at the patch boundary; no real network.
"""
import subprocess
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from remyxai.cli import outrider_actions
from remyxai.cli.commands import cli


# Captured before the autouse stub below shadows it, so the two unit tests
# for the probe itself still exercise the real implementation.
_REAL_GH_PENDING_RUNS = outrider_actions._gh_pending_runs


@pytest.fixture(autouse=True)
def _no_pending_runs(monkeypatch):
    """Default every test to "nothing queued" (opt in per-test below)."""
    monkeypatch.setattr(outrider_actions, "_gh_pending_runs", lambda repo: [])


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


# ─── refinement inputs (issue #49, field note 4) ───────────────────────────
#
# `trigger` used to expose only search-method / pin-arxiv / ref / provider /
# model / claude-timeout, so brief mode and "build on this existing branch"
# meant dropping to raw `gh workflow run -f …`. --ref is not a substitute:
# it picks which branch the workflow FILE comes from, not what the agent
# builds on.

def _dispatch_capture(monkeypatch):
    captured = {}

    def fake_dispatch(repo, branch, inputs):
        captured["repo"] = repo
        captured["branch"] = branch
        captured["inputs"] = dict(inputs)
        return (True, "")

    monkeypatch.setattr(outrider_actions, "_outrider_workflow_exists",
                        lambda repo: True)
    monkeypatch.setattr(outrider_actions, "_gh_default_branch",
                        lambda repo: "main")
    monkeypatch.setattr(outrider_actions, "_gh_dispatch_outrider", fake_dispatch)
    monkeypatch.setattr(outrider_actions, "_gh_latest_run_url",
                        lambda r, sleep=None: None)
    return captured


def test_trigger_forwards_the_refinement_inputs(monkeypatch, capsys):
    captured = _dispatch_capture(monkeypatch)
    outrider_actions.handle_outrider_trigger(
        repo="owner/name", search_method=None, pin_arxiv=None,
        interest_id=None, ref=None,
        mode="brief", publish="branch",
        start_from_ref="outrider/deim-draft",
        lead_content="## Gap analysis\n\n- port the loss",
        staged_synthesis=True,
        test_integration_policy="advisory", fidelity_policy="advisory",
    )
    inputs = captured["inputs"]
    assert inputs["mode"] == "brief"
    assert inputs["publish"] == "branch"
    assert inputs["start-from-ref"] == "outrider/deim-draft"
    assert inputs["lead-content"].startswith("## Gap analysis")
    assert inputs["staged-synthesis"] == "true"
    assert inputs["test-integration-policy"] == "advisory"
    assert inputs["fidelity-policy"] == "advisory"
    # start-from-ref is what the agent builds on; the workflow file still
    # comes from the default branch.
    assert captured["branch"] == "main"
    out = capsys.readouterr().out
    assert "start-from-ref: outrider/deim-draft" in out


def test_trigger_omits_refinement_inputs_when_unset(monkeypatch):
    captured = _dispatch_capture(monkeypatch)
    outrider_actions.handle_outrider_trigger(
        repo="owner/name", search_method=None, pin_arxiv="2402.02347v3",
        interest_id=None, ref=None,
    )
    inputs = captured["inputs"]
    for key in ("mode", "publish", "start-from-ref", "lead-content",
                "staged-synthesis", "test-integration-policy",
                "fidelity-policy"):
        assert inputs[key] == "", key


def test_trigger_reads_lead_content_from_a_file(monkeypatch, tmp_path):
    captured = _dispatch_capture(monkeypatch)
    gap = tmp_path / "gap-analysis.md"
    gap.write_text("# Gaps\n\n1. loss not ported\n")
    outrider_actions.handle_outrider_trigger(
        repo="owner/name", search_method=None, pin_arxiv=None,
        interest_id=None, ref=None, lead_content_file=str(gap),
    )
    assert captured["inputs"]["lead-content"] == "# Gaps\n\n1. loss not ported\n"


def test_trigger_lead_content_flags_are_mutually_exclusive(tmp_path):
    gap = tmp_path / "g.md"
    gap.write_text("x")
    with pytest.raises(click.UsageError, match="mutually exclusive"):
        outrider_actions.handle_outrider_trigger(
            repo="owner/name", search_method=None, pin_arxiv=None,
            interest_id=None, ref=None,
            lead_content="inline", lead_content_file=str(gap),
        )


def test_trigger_rejects_oversized_lead_content():
    with pytest.raises(click.UsageError, match="64KB"):
        outrider_actions.handle_outrider_trigger(
            repo="owner/name", search_method=None, pin_arxiv=None,
            interest_id=None, ref=None,
            lead_content="x" * (outrider_actions.LEAD_CONTENT_MAX_CHARS + 1),
        )


def test_trigger_rejects_more_inputs_than_github_accepts(monkeypatch):
    _dispatch_capture(monkeypatch)
    with pytest.raises(click.UsageError, match="at most 10"):
        outrider_actions.handle_outrider_trigger(
            repo="owner/name", search_method=None, pin_arxiv="2402.02347v3",
            interest_id="6a730cc4-010c-49ce-9c7f-6d9c59431739", ref=None,
            claude_timeout=1800, provider="zai", model="glm-5.2",
            mode="recommend", publish="pr", start_from_ref="b",
            lead_content="ctx", staged_synthesis=True,
            fidelity_policy="advisory",
        )


def test_cli_refinement_flags_reach_the_dispatch(monkeypatch, tmp_path):
    captured = _dispatch_capture(monkeypatch)
    gap = tmp_path / "gap.md"
    gap.write_text("# Gaps")
    runner = CliRunner()
    result = runner.invoke(cli, [
        "outrider", "trigger", "--repo", "owner/name",
        "--start-from-ref", "outrider/draft-1",
        "--lead-content-file", str(gap),
        "--staged-synthesis",
        "--fidelity-policy", "advisory",
        "--publish", "branch",
    ])
    assert result.exit_code == 0, result.output
    inputs = captured["inputs"]
    assert inputs["start-from-ref"] == "outrider/draft-1"
    assert inputs["lead-content"] == "# Gaps"
    assert inputs["staged-synthesis"] == "true"
    assert inputs["fidelity-policy"] == "advisory"
    assert inputs["publish"] == "branch"


# ─── undeclared-input fallback ─────────────────────────────────────────────


def test_dispatch_drops_undeclared_inputs_and_retries():
    """A repo installed before an input existed 422s naming it; drop exactly
    those and retry, rather than dead-ending the dispatch."""
    attempts = []

    def fake_dispatch(repo, branch, inputs):
        attempts.append(dict(inputs))
        if "lead-content" in inputs or "mode" in inputs:
            return (False, 'HTTP 422: Unexpected inputs provided: '
                           '["lead-content", "mode"]')
        return (True, "")

    with patch.object(outrider_actions, "_gh_dispatch_outrider",
                      side_effect=fake_dispatch):
        ok, stderr, dropped = outrider_actions._dispatch_with_input_fallback(
            "owner/name", "main",
            {"mode": "brief", "lead-content": "ctx", "pin-arxiv": "2402.1"},
        )
    assert ok is True
    assert dropped == ["lead-content", "mode"]
    assert len(attempts) == 2
    assert attempts[1] == {"pin-arxiv": "2402.1"}


def test_dispatch_does_not_retry_other_failures():
    calls = []

    def fake_dispatch(repo, branch, inputs):
        calls.append(inputs)
        return (False, "HTTP 403: missing permission")

    with patch.object(outrider_actions, "_gh_dispatch_outrider",
                      side_effect=fake_dispatch):
        ok, stderr, dropped = outrider_actions._dispatch_with_input_fallback(
            "owner/name", "main", {"mode": "brief"},
        )
    assert (ok, dropped) == (False, [])
    assert len(calls) == 1


def test_trigger_warns_about_dropped_inputs(monkeypatch, capsys):
    monkeypatch.setattr(outrider_actions, "_outrider_workflow_exists",
                        lambda repo: True)
    monkeypatch.setattr(outrider_actions, "_gh_default_branch",
                        lambda repo: "main")
    monkeypatch.setattr(outrider_actions, "_gh_latest_run_url",
                        lambda r, sleep=None: None)
    monkeypatch.setattr(outrider_actions, "_dispatch_with_input_fallback",
                        lambda r, b, i: (True, "", ["lead-content"]))
    outrider_actions.handle_outrider_trigger(
        repo="owner/name", search_method=None, pin_arxiv=None,
        interest_id=None, ref=None, lead_content="ctx",
    )
    out = capsys.readouterr().out
    assert "doesn't declare: lead-content" in out
    assert "--force" in out


# ─── pending-run detection (issue #49, field note 5) ───────────────────────


def test_pending_runs_filters_on_not_yet_started():
    payload = ('[{"status":"queued","url":"u1"},'
               '{"status":"in_progress","url":"u2"},'
               '{"status":"completed","url":"u3"}]')
    with patch("subprocess.check_output", return_value=payload):
        pending = _REAL_GH_PENDING_RUNS("owner/name")
    assert [r["url"] for r in pending] == ["u1"]


def test_pending_runs_empty_when_gh_fails():
    with patch("subprocess.check_output",
               side_effect=subprocess.CalledProcessError(1, "gh")):
        assert _REAL_GH_PENDING_RUNS("owner/name") == []


def test_trigger_warns_when_a_run_is_already_pending(monkeypatch, capsys):
    captured = _dispatch_capture(monkeypatch)
    monkeypatch.setattr(outrider_actions, "_gh_pending_runs",
                        lambda repo: [{"status": "queued", "url": "u1"}])
    outrider_actions.handle_outrider_trigger(
        repo="owner/name", search_method=None, pin_arxiv=None,
        interest_id=None, ref=None,
    )
    out = capsys.readouterr().out
    assert "already pending" in out
    assert "--wait-for-slot" in out
    # Advisory only — the dispatch still goes out.
    assert captured["inputs"] is not None


def test_wait_for_slot_polls_until_the_queue_clears(monkeypatch):
    states = [
        [{"status": "queued", "url": "u1"}],   # initial probe
        [{"status": "queued", "url": "u1"}],   # first poll
        [],                                    # clear
    ]
    monkeypatch.setattr(outrider_actions, "_gh_pending_runs",
                        lambda repo: states.pop(0))
    slept = []
    outrider_actions._warn_or_wait_for_queue(
        "owner/name", wait=True, sleep=slept.append, interval=1, timeout=10,
    )
    assert slept == [1, 1]


def test_wait_for_slot_times_out_rather_than_cancelling_a_run(monkeypatch):
    monkeypatch.setattr(outrider_actions, "_gh_pending_runs",
                        lambda repo: [{"status": "queued", "url": "u1"}])
    with pytest.raises(click.ClickException, match="still 1 pending"):
        outrider_actions._warn_or_wait_for_queue(
            "owner/name", wait=True, sleep=lambda s: None,
            interval=5, timeout=10,
        )
