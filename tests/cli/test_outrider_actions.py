"""Tests for the Outrider install action handler (thin engine client).

`remyxai outrider init` now drives the Remyx engine's provision-action flow
(the same "set it up for me" path as the web app) instead of mutating the
repo with a local `gh` token. These tests cover:

- repo normalization / detection
- UUID validation + interest resolution
- App-install and model-provider preflights
- the provision poll loop
- CLI wiring + the --dry-run no-mutation contract
"""
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from remyxai.cli import outrider_actions
from remyxai.cli.commands import cli


# ─── repo normalization / detection ────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    ("remyxai/outrider", "remyxai/outrider"),
    ("https://github.com/remyxai/outrider", "remyxai/outrider"),
    ("https://github.com/remyxai/outrider.git", "remyxai/outrider"),
    ("git@github.com:remyxai/outrider.git", "remyxai/outrider"),
    ("https://gitlab.com/remyxai/outrider", None),
    ("not a repo", None),
    ("", None),
])
def test_normalize_repo(value, expected):
    assert outrider_actions._normalize_repo(value) == expected


def test_detect_github_repo_from_cwd_https(monkeypatch):
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **k: "https://github.com/remyxai/outrider.git\n",
    )
    assert outrider_actions._detect_github_repo_from_cwd() == "remyxai/outrider"


def test_detect_github_repo_from_cwd_no_git(monkeypatch):
    import subprocess
    def fake(*a, **k):
        raise subprocess.CalledProcessError(128, a)
    monkeypatch.setattr("subprocess.check_output", fake)
    assert outrider_actions._detect_github_repo_from_cwd() is None


# ─── UUID validation ───────────────────────────────────────────────────────

def test_uuid_re_accepts_valid():
    assert outrider_actions.UUID_RE.match("6a730cc4-010c-49ce-9c7f-6d9c59431739")


def test_uuid_re_rejects_bad():
    for bad in ["", "not-a-uuid", "12345", "6a730cc4", "g" * 36]:
        assert not outrider_actions.UUID_RE.match(bad), f"matched: {bad!r}"


# ─── interest resolution ───────────────────────────────────────────────────

def test_resolve_interest_rejects_bad_uuid_no_network():
    """Malformed --interest is rejected before any engine roundtrip."""
    with pytest.raises(click.UsageError):
        outrider_actions._resolve_interest_id(
            interest_id="not-a-uuid", auto_interest=False,
            repo="o/r", repo_url="https://github.com/o/r", api_key="k",
        )


def test_resolve_interest_valid_uuid_probes_engine():
    uid = "6a730cc4-010c-49ce-9c7f-6d9c59431739"
    with patch.object(outrider_actions, "get_interest", return_value={"id": uid}) as gi:
        out = outrider_actions._resolve_interest_id(
            interest_id=uid, auto_interest=False,
            repo="o/r", repo_url="https://github.com/o/r", api_key="k",
        )
    assert out == uid
    gi.assert_called_once()


def test_resolve_interest_auto_creates_via_repo_analysis():
    """--auto-interest must run the analyze-repo flow (rich
    ExperimentHistory context) rather than stuffing the URL into context."""
    uid = "6a730cc4-010c-49ce-9c7f-6d9c59431739"
    with patch.object(
        outrider_actions, "create_interest_from_repo",
        return_value={"id": uid, "history_extraction_task_id": "t-1"},
    ) as ci:
        out = outrider_actions._resolve_interest_id(
            interest_id=None, auto_interest=True,
            repo="remyxai/outrider", repo_url="https://github.com/remyxai/outrider",
            api_key="k",
        )
    assert out == uid
    # Goes through the repo-analysis flow with the repo URL + the user's key,
    # and does NOT provision paper PRs here (outrider init does that later).
    assert ci.call_args.args[0] == "https://github.com/remyxai/outrider"
    assert ci.call_args.kwargs["api_key"] == "k"
    assert ci.call_args.kwargs["automate"] == "none"


def test_resolve_interest_auto_create_surfaces_analysis_failure():
    """A silent URL-stub; analysis failure must raise."""
    with patch.object(
        outrider_actions, "create_interest_from_repo",
        side_effect=outrider_actions.RepoAnalysisError("rate limited"),
    ):
        with pytest.raises(click.ClickException):
            outrider_actions._resolve_interest_id(
                interest_id=None, auto_interest=True,
                repo="o/r", repo_url="https://github.com/o/r", api_key="k",
            )


# ─── App-install preflight ──────────────────────────────────────────────────

def test_ensure_app_installed_when_already_installed():
    with patch.object(outrider_actions, "is_app_installed", return_value=True), \
         patch.object(outrider_actions, "get_app_install_url") as url:
        outrider_actions._ensure_app_installed("o/r", "k", no_wait=False)
    url.assert_not_called()  # no need to surface the link


def test_ensure_app_installed_no_wait_raises_with_link():
    with patch.object(outrider_actions, "is_app_installed", return_value=False), \
         patch.object(outrider_actions, "get_app_install_url",
                      return_value={"configured": True, "install_url": "https://x/install"}):
        with pytest.raises(click.ClickException):
            outrider_actions._ensure_app_installed("o/r", "k", no_wait=True)


def test_ensure_app_installed_polls_until_installed():
    """First check False (surfaces link), then becomes True on poll."""
    seq = iter([False, True])
    with patch.object(outrider_actions, "is_app_installed",
                      side_effect=lambda *a, **k: next(seq)), \
         patch.object(outrider_actions, "get_app_install_url",
                      return_value={"configured": True, "install_url": "https://x/install"}):
        # sleep is injected → no real delay
        outrider_actions._ensure_app_installed(
            "o/r", "k", no_wait=False, sleep=lambda s: None
        )


# ─── model-provider preflight ───────────────────────────────────────────────

def test_ensure_model_provider_already_connected():
    with patch.object(outrider_actions, "get_integration_status",
                      return_value={"connected": True}), \
         patch.object(outrider_actions, "connect_credential") as conn:
        assert outrider_actions._ensure_model_provider(None, "k") is True
    conn.assert_not_called()


def test_ensure_model_provider_connects_with_key():
    with patch.object(outrider_actions, "get_integration_status",
                      return_value={"connected": False}), \
         patch.object(outrider_actions, "connect_credential") as conn:
        assert outrider_actions._ensure_model_provider("sk-ant-x", "k") is True
    assert conn.call_args[0][1] == {"api_key": "sk-ant-x"}


def test_ensure_model_provider_warns_when_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch.object(outrider_actions, "get_integration_status",
                      return_value={"connected": False}), \
         patch.object(outrider_actions, "connect_credential") as conn:
        assert outrider_actions._ensure_model_provider(None, "k") is False
    conn.assert_not_called()


# ─── provision poll loop ─────────────────────────────────────────────────────

def test_wait_for_provision_returns_result_on_completed():
    result = {"pr_url": "https://github.com/o/r/pull/1", "merged": True}
    with patch.object(outrider_actions, "poll_provision_action",
                      return_value={"status": "completed", "result": result}):
        out = outrider_actions._wait_for_provision("iid", "t1", "k", sleep=lambda s: None)
    assert out == result


def test_wait_for_provision_raises_on_failed():
    with patch.object(outrider_actions, "poll_provision_action",
                      return_value={"status": "failed", "error": "boom"}):
        with pytest.raises(click.ClickException, match="boom"):
            outrider_actions._wait_for_provision("iid", "t1", "k", sleep=lambda s: None)


# ─── CLI wiring ──────────────────────────────────────────────────────────────

@patch("remyxai.cli.commands.handle_outrider_init")
def test_outrider_init_passes_args_through(mock_handler):
    runner = CliRunner()
    result = runner.invoke(cli, [
        "outrider", "init",
        "--repo", "owner/repo",
        "--interest", "6a730cc4-010c-49ce-9c7f-6d9c59431739",
        "--mode", "review", "--yes",
    ])
    assert result.exit_code == 0
    mock_handler.assert_called_once_with(
        repo="owner/repo",
        interest_id="6a730cc4-010c-49ce-9c7f-6d9c59431739",
        auto_interest=False,
        mode="review",
        anthropic_key=None,
        skip_confirm=True,
        dry_run=False,
        no_wait=False,
    )


@patch("remyxai.cli.commands.handle_outrider_init")
def test_outrider_init_mode_defaults_to_auto(mock_handler):
    runner = CliRunner()
    result = runner.invoke(cli, ["outrider", "init", "--repo", "o/r", "--auto-interest", "-y"])
    assert result.exit_code == 0
    assert mock_handler.call_args.kwargs["mode"] == "auto"


def test_outrider_init_help_lists_options():
    runner = CliRunner()
    result = runner.invoke(cli, ["outrider", "init", "--help"])
    assert result.exit_code == 0
    for opt in ("--repo", "--interest", "--auto-interest", "--mode",
                "--anthropic-key", "--no-wait", "--dry-run"):
        assert opt in result.output


def test_outrider_init_mutual_exclusion(monkeypatch):
    monkeypatch.setenv("REMYXAI_API_KEY", "k")
    with pytest.raises(click.UsageError):
        outrider_actions.handle_outrider_init(
            repo="o/r", interest_id="6a730cc4-010c-49ce-9c7f-6d9c59431739",
            auto_interest=True, mode="auto", anthropic_key=None,
            skip_confirm=True, dry_run=False, no_wait=False,
        )


# ─── --dry-run no-mutation contract ──────────────────────────────────────────

def test_dry_run_makes_no_api_calls(monkeypatch):
    """--dry-run prints the plan and touches no engine endpoint."""
    monkeypatch.setenv("REMYXAI_API_KEY", "k")
    with patch.object(outrider_actions, "is_app_installed") as inst, \
         patch.object(outrider_actions, "get_integration_status") as gs, \
         patch.object(outrider_actions, "create_interest_from_repo") as ci, \
         patch.object(outrider_actions, "get_interest") as gi, \
         patch.object(outrider_actions, "provision_action") as prov:
        outrider_actions.handle_outrider_init(
            repo="owner/repo", interest_id="6a730cc4-010c-49ce-9c7f-6d9c59431739",
            auto_interest=False, mode="auto", anthropic_key=None,
            skip_confirm=True, dry_run=True, no_wait=False,
        )
    for m in (inst, gs, ci, gi, prov):
        m.assert_not_called()


# ─── mode=off: interest only, no provisioning ────────────────────────────────

def test_mode_off_resolves_interest_but_never_provisions(monkeypatch):
    monkeypatch.setenv("REMYXAI_API_KEY", "k")
    uid = "6a730cc4-010c-49ce-9c7f-6d9c59431739"
    with patch.object(outrider_actions, "get_interest", return_value={"id": uid}), \
         patch.object(outrider_actions, "provision_action") as prov, \
         patch.object(outrider_actions, "is_app_installed") as inst:
        outrider_actions.handle_outrider_init(
            repo="owner/repo", interest_id=uid, auto_interest=False,
            mode="off", anthropic_key=None, skip_confirm=True,
            dry_run=False, no_wait=False,
        )
    prov.assert_not_called()
    inst.assert_not_called()  # no App preflight needed for interest-only


# ─── full auto happy path ─────────────────────────────────────────────────────

def test_full_auto_flow_provisions_and_reports(monkeypatch):
    monkeypatch.setenv("REMYXAI_API_KEY", "k")
    uid = "6a730cc4-010c-49ce-9c7f-6d9c59431739"
    completed = {
        "status": "completed",
        "result": {"pr_url": "https://github.com/o/r/pull/7", "merged": True,
                   "secret_set": True, "dispatched": True, "model_key_missing": False},
    }
    calls = []
    with patch.object(outrider_actions, "get_interest", return_value={"id": uid}), \
         patch.object(outrider_actions, "is_app_installed", return_value=True), \
         patch.object(outrider_actions, "get_integration_status", return_value={"connected": True}), \
         patch.object(outrider_actions, "_kick_off_recommendations",
                      side_effect=lambda *a, **k: calls.append("warm")) as warm, \
         patch.object(outrider_actions, "provision_action",
                      side_effect=lambda *a, **k: calls.append("provision") or {"task_id": "t1"}) as prov, \
         patch.object(outrider_actions, "poll_provision_action", return_value=completed):
        outrider_actions.handle_outrider_init(
            repo="owner/repo", interest_id=uid, auto_interest=False,
            mode="auto", anthropic_key=None, skip_confirm=True,
            dry_run=False, no_wait=False,
        )
    assert prov.call_args.kwargs["auto_merge"] is True
    # Recommendations are warmed up before the first run is provisioned.
    warm.assert_called_once()
    assert warm.call_args.kwargs.get("wait") is True
    assert calls == ["warm", "provision"]


def test_no_wait_warms_without_blocking(monkeypatch):
    """--no-wait still triggers the warm-up refresh, but doesn't block on it."""
    monkeypatch.setenv("REMYXAI_API_KEY", "k")
    uid = "6a730cc4-010c-49ce-9c7f-6d9c59431739"
    with patch.object(outrider_actions, "get_interest", return_value={"id": uid}), \
         patch.object(outrider_actions, "is_app_installed", return_value=True), \
         patch.object(outrider_actions, "get_integration_status", return_value={"connected": True}), \
         patch.object(outrider_actions, "_kick_off_recommendations") as warm, \
         patch.object(outrider_actions, "provision_action", return_value={"task_id": "t1"}):
        outrider_actions.handle_outrider_init(
            repo="owner/repo", interest_id=uid, auto_interest=False,
            mode="auto", anthropic_key=None, skip_confirm=True,
            dry_run=False, no_wait=True,
        )
    warm.assert_called_once()
    assert warm.call_args.kwargs.get("wait") is False
