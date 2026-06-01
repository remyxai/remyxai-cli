"""Tests for the Outrider install action handler.

Covers pure-function logic (git parsing, template rendering, gh argv
shape) without touching real git / gh / engine APIs. Tests deliberately
import from the leaf module (`remyxai.cli.outrider_actions`) directly
rather than via `remyxai.cli.commands`, since the latter pulls in
`evaluation_actions` → `client/myxboard.py` which has a pre-existing
`huggingface_hub.HfFolder` import that's broken on `huggingface_hub>=1.0`.
The CLI-wiring tests can be re-added once that upstream issue is fixed.
"""
import subprocess
from pathlib import Path

import click
import pytest

from remyxai.cli import outrider_actions


# ─── git parsing ───────────────────────────────────────────────────────────

def test_find_git_root_locates_repo(tmp_path):
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / "subdir" / "deeper").mkdir(parents=True)
    assert outrider_actions._find_git_root(root / "subdir" / "deeper") == root


def test_find_git_root_returns_none_outside_repo(tmp_path):
    assert outrider_actions._find_git_root(tmp_path) is None


def test_detect_github_repo_https(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **k: "https://github.com/remyxai/outrider.git\n",
    )
    assert outrider_actions._detect_github_repo(tmp_path) == "remyxai/outrider"


def test_detect_github_repo_ssh(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **k: "git@github.com:remyxai/outrider.git\n",
    )
    assert outrider_actions._detect_github_repo(tmp_path) == "remyxai/outrider"


def test_detect_github_repo_non_github(monkeypatch, tmp_path):
    """Non-github.com remotes return None — Outrider requires GitHub."""
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **k: "https://gitlab.com/remyxai/outrider.git\n",
    )
    assert outrider_actions._detect_github_repo(tmp_path) is None


def test_detect_default_branch_from_origin_head(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **k: "refs/remotes/origin/master\n",
    )
    assert outrider_actions._detect_default_branch(tmp_path) == "master"


def test_detect_default_branch_falls_back_to_main(monkeypatch, tmp_path):
    def fake(*a, **k):
        raise subprocess.CalledProcessError(128, a)
    monkeypatch.setattr("subprocess.check_output", fake)
    assert outrider_actions._detect_default_branch(tmp_path) == "main"


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


# ─── gh secret-stdin invariant ─────────────────────────────────────────────

def test_gh_set_secret_passes_value_via_stdin(monkeypatch):
    """The secret value must go through stdin, never argv. This is the
    one invariant that matters for credential safety."""
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


def test_gh_set_secret_raises_clickexception_on_failure(monkeypatch):
    class _Fail:
        returncode = 1
        stderr = "permission denied"
    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Fail())
    with pytest.raises(click.ClickException, match="permission denied"):
        outrider_actions._gh_set_secret("owner/repo", "MY_SECRET", "x")


# ─── gh PR creation argv shape ─────────────────────────────────────────────

def test_gh_create_pr_returns_url(monkeypatch):
    class _Done:
        returncode = 0
        stdout = "https://github.com/owner/repo/pull/42\n"
        stderr = ""
    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Done())
    url = outrider_actions._gh_create_pr(
        repo="owner/repo", head="b", base="main",
        title="t", body="b", draft=True,
    )
    assert url == "https://github.com/owner/repo/pull/42"


def test_gh_create_pr_draft_flag_toggle(monkeypatch):
    captured = {}

    class _Done:
        returncode = 0
        stdout = "https://github.com/x/y/pull/1"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _Done()

    monkeypatch.setattr("subprocess.run", fake_run)

    outrider_actions._gh_create_pr(
        repo="x/y", head="b", base="m", title="t", body="b", draft=True,
    )
    assert "--draft" in captured["cmd"]

    outrider_actions._gh_create_pr(
        repo="x/y", head="b", base="m", title="t", body="b", draft=False,
    )
    assert "--draft" not in captured["cmd"]


# ─── handler validation ────────────────────────────────────────────────────

def test_outrider_init_mutual_exclusion_raises(tmp_path, monkeypatch):
    """--interest + --auto-interest must be caught by the handler."""
    # Make the handler runnable up to the mutual-exclusion check by
    # neutralizing the other early failures.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(click.UsageError):
        outrider_actions.handle_outrider_init(
            interest_id="x", auto_interest=True,
            branch_name="b", skip_confirm=True,
        )
