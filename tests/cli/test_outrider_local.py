"""Tests for the no-App ("local") Outrider setup path.

`remyxai outrider setup-local` self-provisions with the user's own gh token —
no Remyx GitHub App. Covers workflow rendering (GITHUB_TOKEN vs PAT), the
gh-secret stdin invariant, the sha-on-update fix, rollback ordering, the
dry-run contract, and CLI wiring.
"""
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from remyxai.cli import outrider_local
from remyxai.cli.commands import cli


# ─── workflow rendering ─────────────────────────────────────────────────────

def test_render_uses_builtin_github_token():
    wf = outrider_local._render_local_workflow("uuid-123")
    assert "interest-id: uuid-123" in wf
    assert "github-token:" not in wf                      # uses the built-in GITHUB_TOKEN
    assert "workflow_dispatch:" in wf
    assert "rate-limit-days: '0'" in wf                   # don't suppress manual/scheduled runs


def test_render_declares_remyx_and_model_secrets():
    """REMYX_API_KEY is set as a job-level env var (it's needed across
    every step); ANTHROPIC_API_KEY and ZAI_API_KEY are accessed via
    ``secrets.*`` inside the Configure-backend-auth step rather than
    set unconditionally, so a non-default backend dispatch doesn't get
    both auth vars set (which Claude Code would resolve in favor of
    the x-api-key path that non-Anthropic backends reject)."""
    wf = outrider_local._render_local_workflow("uuid")
    assert "REMYX_API_KEY: ${{ secrets.REMYX_API_KEY }}" in wf
    # Configure step references both Anthropic + z.ai secrets via env.
    assert "ANTHROPIC_API_KEY_SECRET: ${{ secrets.ANTHROPIC_API_KEY }}" in wf
    assert "ZAI_API_KEY_SECRET: ${{ secrets.ZAI_API_KEY }}" in wf
    # The legacy unconditional job-level ANTHROPIC_API_KEY env line is
    # GONE — its presence would trip the action's startup auth-guard
    # mutual-exclusion warning on every backend=glm dispatch.
    assert "      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}" not in wf


def test_render_declares_backend_input_and_configure_step():
    """Workflow exposes a `backend` workflow_dispatch input + a step
    that writes the right auth env var to $GITHUB_ENV based on it."""
    wf = outrider_local._render_local_workflow("uuid")

    # workflow_dispatch input declaration
    assert "      backend:" in wf
    assert "type: choice" in wf
    for opt in ("- anthropic", "- glm"):
        assert opt in wf, f"missing backend option: {opt}"
    assert "default: 'anthropic'" in wf

    # Configure backend auth step
    assert "name: Configure backend auth" in wf
    # Picks the right env var based on inputs.backend.
    assert "if [ \"${{ inputs.backend }}\" = \"glm\" ]; then" in wf
    assert 'echo "ANTHROPIC_AUTH_TOKEN=$ZAI_API_KEY_SECRET" >> "$GITHUB_ENV"' in wf
    assert 'echo "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY_SECRET" >> "$GITHUB_ENV"' in wf


def test_render_forwards_model_base_url_for_glm_backend():
    """The action's model-base-url input is set to z.ai's endpoint
    when backend=glm, empty otherwise (default Anthropic)."""
    wf = outrider_local._render_local_workflow("uuid")
    assert (
        "model-base-url: ${{ inputs.backend == 'glm' "
        "&& 'https://api.z.ai/api/anthropic' || '' }}"
    ) in wf


def test_render_declares_workflow_dispatch_inputs():
    """The generated workflow exposes pin-method / pin-arxiv /
    claude-timeout as workflow_dispatch inputs so `remyxai outrider
    trigger` and manual `gh workflow run -f ...` can forward them
    without the workflow rejecting them as 'not a permitted key'."""
    wf = outrider_local._render_local_workflow("uuid")
    # Inputs block under workflow_dispatch.
    assert "workflow_dispatch:" in wf
    assert "    inputs:" in wf
    # Each declared input is present.
    for name in ("pin-method:", "pin-arxiv:", "claude-timeout:"):
        assert name in wf, f"missing input declaration: {name}"
    # claude-timeout's default matches the action's documented 900s.
    assert "default: '900'" in wf


def test_render_forwards_workflow_dispatch_inputs_to_action():
    """Each declared workflow_dispatch input is forwarded into the
    action's `with:` block via ${{ inputs.<name> }}."""
    wf = outrider_local._render_local_workflow("uuid")
    for name in ("pin-method", "pin-arxiv", "claude-timeout"):
        assert f"{name}: ${{{{ inputs.{name} }}}}" in wf, (
            f"missing forwarding for {name}"
        )


# ─── gh secret stdin invariant ───────────────────────────────────────────────

def test_gh_set_secret_value_via_stdin(monkeypatch):
    captured = {}

    class _Done:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["input"] = kw.get("input")
        return _Done()

    monkeypatch.setattr("subprocess.run", fake_run)
    outrider_local._gh_set_secret("o/r", "REMYX_API_KEY", "supersecret")
    assert "supersecret" not in captured["cmd"]
    assert captured["input"] == "supersecret"


def test_gh_set_secret_403_hint(monkeypatch):
    class _Fail:
        returncode = 1
        stderr = "HTTP 403: permission"
    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Fail())
    with pytest.raises(click.ClickException, match="admin scope"):
        outrider_local._gh_set_secret("o/r", "X", "v")


# ─── sha-on-update (regression: PUT over existing file 422'd) ────────────────

def test_gh_put_file_includes_sha_when_file_exists(monkeypatch):
    calls = []

    class _R:
        def __init__(self, rc=0, out=""):
            self.returncode, self.stdout, self.stderr = rc, out, ""

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if "PUT" not in cmd:                       # the GET in _gh_get_file_sha
            import json
            return _R(out=json.dumps({"sha": "abc123"}))
        return _R(out="{}")

    monkeypatch.setattr("subprocess.run", fake_run)
    outrider_local._gh_put_file("o/r", "b", outrider_local.WORKFLOW_PATH, "x", "msg")
    put_cmd = next(c for c in calls if "PUT" in c)
    assert "sha=abc123" in put_cmd


# ─── handler: rollback ordering ──────────────────────────────────────────────

def _base_patches(monkeypatch):
    monkeypatch.setenv("REMYXAI_API_KEY", "rk")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")


def test_rollback_deletes_branch_and_skips_secrets_when_put_fails(monkeypatch):
    _base_patches(monkeypatch)
    uid = "6a730cc4-010c-49ce-9c7f-6d9c59431739"
    with patch.object(outrider_local, "_resolve_interest_id", return_value=uid), \
         patch.object(outrider_local, "_gh_available", return_value=True), \
         patch.object(outrider_local, "_gh_authenticated", return_value=True), \
         patch.object(outrider_local, "_gh_default_branch", return_value="main"), \
         patch.object(outrider_local, "_gh_branch_exists", return_value=False), \
         patch.object(outrider_local, "_gh_get_branch_sha", return_value="sha"), \
         patch.object(outrider_local, "_gh_create_branch"), \
         patch.object(outrider_local, "_gh_put_file", side_effect=click.ClickException("422")), \
         patch.object(outrider_local, "_gh_open_pr") as open_pr, \
         patch.object(outrider_local, "_gh_delete_branch") as del_branch, \
         patch.object(outrider_local, "_gh_set_secret") as set_secret:
        with pytest.raises(click.ClickException):
            outrider_local.handle_outrider_setup_local(
                repo="o/r", interest_id=uid, auto_interest=False, mode="review",
                anthropic_key=None, skip_confirm=True, dry_run=False,
            )
    del_branch.assert_called_once()      # branch rolled back
    open_pr.assert_not_called()          # never got to the PR
    set_secret.assert_not_called()       # secrets never set (last step)


# ─── handler: happy path (review, default GITHUB_TOKEN auth) ─────────────────

def test_review_mode_enables_pr_creation_and_sets_secrets(monkeypatch):
    _base_patches(monkeypatch)
    uid = "6a730cc4-010c-49ce-9c7f-6d9c59431739"
    with patch.object(outrider_local, "_resolve_interest_id", return_value=uid), \
         patch.object(outrider_local, "_gh_available", return_value=True), \
         patch.object(outrider_local, "_gh_authenticated", return_value=True), \
         patch.object(outrider_local, "_gh_default_branch", return_value="main"), \
         patch.object(outrider_local, "_gh_branch_exists", return_value=False), \
         patch.object(outrider_local, "_gh_get_branch_sha", return_value="sha"), \
         patch.object(outrider_local, "_gh_create_branch"), \
         patch.object(outrider_local, "_gh_put_file"), \
         patch.object(outrider_local, "_gh_open_pr", return_value=("https://x/pull/1", 1)), \
         patch.object(outrider_local, "_gh_enable_pr_creation") as enable, \
         patch.object(outrider_local, "_gh_set_secret") as set_secret, \
         patch.object(outrider_local, "_gh_merge_pr") as merge:
        outrider_local.handle_outrider_setup_local(
            repo="o/r", interest_id=uid, auto_interest=False, mode="review",
            anthropic_key=None, skip_confirm=True, dry_run=False,
        )
    enable.assert_called_once()          # enables the repo's Actions-PR setting
    merge.assert_not_called()            # review mode doesn't merge
    names = {c.args[1] for c in set_secret.call_args_list}
    assert {"REMYX_API_KEY", "ANTHROPIC_API_KEY"} <= names
    # no GitHub token is ever stored as a secret
    assert not any("TOKEN" in n and n not in {"REMYX_API_KEY", "ANTHROPIC_API_KEY"}
                   for n in names)


# ─── dry-run + wiring ─────────────────────────────────────────────────────────

def test_dry_run_makes_no_gh_calls(monkeypatch):
    _base_patches(monkeypatch)
    with patch.object(outrider_local, "_gh_authenticated") as auth, \
         patch.object(outrider_local, "_resolve_interest_id") as ri, \
         patch.object(outrider_local, "_gh_create_branch") as cb, \
         patch.object(outrider_local, "_gh_set_secret") as ss:
        outrider_local.handle_outrider_setup_local(
            repo="o/r", interest_id="6a730cc4-010c-49ce-9c7f-6d9c59431739",
            auto_interest=False, mode="auto", anthropic_key=None,
            skip_confirm=True, dry_run=True,
        )
    for m in (auth, ri, cb, ss):
        m.assert_not_called()


@patch("remyxai.cli.commands.handle_outrider_setup_local")
def test_setup_local_wiring(mock_handler):
    runner = CliRunner()
    result = runner.invoke(cli, [
        "outrider", "setup-local", "--repo", "o/r",
        "--interest", "6a730cc4-010c-49ce-9c7f-6d9c59431739",
        "--mode", "review", "-y",
    ])
    assert result.exit_code == 0
    kwargs = mock_handler.call_args.kwargs
    assert kwargs["mode"] == "review"
    assert kwargs["skip_confirm"] is True


def test_setup_local_help_lists_options():
    runner = CliRunner()
    result = runner.invoke(cli, ["outrider", "setup-local", "--help"])
    assert result.exit_code == 0
    for opt in ("--repo", "--interest", "--auto-interest", "--mode",
                "--anthropic-key", "--dry-run"):
        assert opt in result.output
    assert "--gh-pat" not in result.output   # dropped for v1
