"""Tests for `remyxai outrider set-backend-secret`.

Covers:
- Backend name → secret name mapping (anthropic → ANTHROPIC_API_KEY,
  glm → ZAI_API_KEY).
- Rejection of unknown backend names at the CLI boundary.
- File-input safety: trailing newline stripped, but no other mutation.
- Empty / literal-"-" / unknown-backend / nonexistent-file all rejected
  before any `gh` call.
- Short-but-non-empty key warns but proceeds (the action's startup
  auth-guard is the hard-fail line).
- The value never appears in the `gh secret set` command line — it's
  always piped via stdin.
- End-to-end through the click runner.

Run with: pytest tests/cli/test_set_backend_secret.py -q
"""
from pathlib import Path
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from remyxai.cli import outrider_actions
from remyxai.cli.commands import cli


# ─── handler-level checks ────────────────────────────────────────────────


def _write_key(tmp_path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def test_rejects_unknown_backend(tmp_path):
    k = _write_key(tmp_path, "k", "sk-ant-fakebutlongenough12345")
    with pytest.raises(click.UsageError, match="must be one of"):
        outrider_actions.handle_set_backend_secret(
            repo="owner/name", backend="bedrock", key_from=str(k),
        )


def test_rejects_missing_key_file():
    with pytest.raises(click.ClickException, match="does not exist"):
        outrider_actions.handle_set_backend_secret(
            repo="owner/name", backend="glm",
            key_from="/nonexistent/path/to/key",
        )


def test_rejects_literal_dash_value(tmp_path):
    """A file containing only '-' is the `gh secret set --body -`
    footprint — refuse to forward it."""
    k = _write_key(tmp_path, "k", "-")
    with pytest.raises(click.ClickException, match="literal '-'"):
        outrider_actions.handle_set_backend_secret(
            repo="owner/name", backend="glm", key_from=str(k),
        )


def test_rejects_empty_value(tmp_path):
    k = _write_key(tmp_path, "k", "")
    with pytest.raises(click.ClickException, match="empty"):
        outrider_actions.handle_set_backend_secret(
            repo="owner/name", backend="glm", key_from=str(k),
        )


def test_rejects_empty_after_newline_strip(tmp_path):
    """A file with just a newline is still empty after the rstrip."""
    k = _write_key(tmp_path, "k", "\n")
    with pytest.raises(click.ClickException, match="empty"):
        outrider_actions.handle_set_backend_secret(
            repo="owner/name", backend="anthropic", key_from=str(k),
        )


def test_strips_single_trailing_newline(tmp_path, monkeypatch):
    """`printf '%s\\n' "$KEY" > /tmp/key` leaves a trailing \\n; strip
    it so the secret doesn't carry the newline through."""
    k = _write_key(tmp_path, "k", "sk-ant-fakebutlongenough12345\n")
    captured = {}

    def fake_gh_set_secret(repo, name, value):
        captured["repo"] = repo
        captured["name"] = name
        captured["value"] = value

    import remyxai.cli.outrider_local as outrider_local
    monkeypatch.setattr(outrider_local, "_gh_set_secret", fake_gh_set_secret)

    outrider_actions.handle_set_backend_secret(
        repo="owner/name", backend="anthropic", key_from=str(k),
    )
    assert captured["value"] == "sk-ant-fakebutlongenough12345"
    assert not captured["value"].endswith("\n")


def test_anthropic_backend_maps_to_anthropic_api_key(tmp_path, monkeypatch):
    k = _write_key(tmp_path, "k", "sk-ant-fakebutlongenough12345")
    captured = {}

    def fake_gh_set_secret(repo, name, value):
        captured["name"] = name

    import remyxai.cli.outrider_local as outrider_local
    monkeypatch.setattr(outrider_local, "_gh_set_secret", fake_gh_set_secret)

    outrider_actions.handle_set_backend_secret(
        repo="owner/name", backend="anthropic", key_from=str(k),
    )
    assert captured["name"] == "ANTHROPIC_API_KEY"


def test_glm_backend_maps_to_zai_api_key(tmp_path, monkeypatch):
    k = _write_key(tmp_path, "k", "zai-fakebutlongenough1234567890")
    captured = {}

    def fake_gh_set_secret(repo, name, value):
        captured["name"] = name

    import remyxai.cli.outrider_local as outrider_local
    monkeypatch.setattr(outrider_local, "_gh_set_secret", fake_gh_set_secret)

    outrider_actions.handle_set_backend_secret(
        repo="owner/name", backend="glm", key_from=str(k),
    )
    assert captured["name"] == "ZAI_API_KEY"


def test_short_value_warns_but_proceeds(tmp_path, monkeypatch, capsys):
    """A 10-char value gets a yellow warning (the action's startup
    auth-guard hard-fails below 8); the CLI still forwards it so the
    customer can override our heuristic when they know what they're
    doing."""
    k = _write_key(tmp_path, "k", "short-key1")  # 10 chars
    called = {"count": 0}

    def fake_gh_set_secret(repo, name, value):
        called["count"] += 1

    import remyxai.cli.outrider_local as outrider_local
    monkeypatch.setattr(outrider_local, "_gh_set_secret", fake_gh_set_secret)

    outrider_actions.handle_set_backend_secret(
        repo="owner/name", backend="glm", key_from=str(k),
    )
    assert called["count"] == 1                   # still proceeded
    out = capsys.readouterr().out
    assert "unusually short" in out


def test_value_is_piped_not_in_argv(tmp_path, monkeypatch):
    """End-to-end privacy: the secret value must NOT appear in argv
    when `gh secret set` is invoked. The outrider_local helper pipes
    via stdin; we just need to confirm the integration uses it."""
    secret_value = "sk-ant-uniquemarker-deadbeefcafe-1234"
    k = _write_key(tmp_path, "k", secret_value)
    captured_cmds = []

    import subprocess as _subprocess
    real_run = _subprocess.run

    def fake_run(cmd, **kw):
        captured_cmds.append(cmd)
        # Simulate gh success.
        class _Done:
            returncode = 0
            stderr = ""
        return _Done()

    monkeypatch.setattr(
        "remyxai.cli.outrider_local.subprocess.run", fake_run,
    )

    outrider_actions.handle_set_backend_secret(
        repo="owner/name", backend="anthropic", key_from=str(k),
    )
    # The secret value never appears in any captured command line.
    for cmd in captured_cmds:
        joined = " ".join(cmd)
        assert secret_value not in joined


# ─── CLI integration via click runner ────────────────────────────────────


def test_cli_set_backend_secret_happy_path(tmp_path, monkeypatch):
    k = _write_key(tmp_path, "k", "sk-ant-fakebutlongenough12345")
    captured = {}

    def fake_gh_set_secret(repo, name, value):
        captured["repo"] = repo
        captured["name"] = name
        captured["value"] = value

    import remyxai.cli.outrider_local as outrider_local
    monkeypatch.setattr(outrider_local, "_gh_set_secret", fake_gh_set_secret)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "outrider", "set-backend-secret",
        "--repo", "owner/name",
        "--backend", "anthropic",
        "--key-from", str(k),
    ])
    assert result.exit_code == 0, result.output
    assert captured["repo"] == "owner/name"
    assert captured["name"] == "ANTHROPIC_API_KEY"
    assert "✓ Set ANTHROPIC_API_KEY" in result.output


def test_cli_set_backend_secret_requires_backend(tmp_path):
    k = _write_key(tmp_path, "k", "sk-ant-fakebutlongenough12345")
    runner = CliRunner()
    result = runner.invoke(cli, [
        "outrider", "set-backend-secret",
        "--repo", "owner/name",
        "--key-from", str(k),
    ])
    assert result.exit_code != 0
    assert "backend" in result.output.lower()


def test_cli_set_backend_secret_requires_key_from(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, [
        "outrider", "set-backend-secret",
        "--repo", "owner/name",
        "--backend", "anthropic",
    ])
    assert result.exit_code != 0
    assert "key-from" in result.output.lower()


def test_cli_set_backend_secret_nonexistent_file_via_click():
    """Click's `type=click.Path(exists=True)` rejects at the boundary."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        "outrider", "set-backend-secret",
        "--repo", "owner/name",
        "--backend", "glm",
        "--key-from", "/nonexistent/file",
    ])
    assert result.exit_code != 0
    assert "does not exist" in result.output.lower() or \
           "no such file" in result.output.lower()
