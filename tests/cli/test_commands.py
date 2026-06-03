"""Tests for top-level CLI command wiring.

The top-level `cli` group exposes only the current product surface
(`papers`, `interests`, `outrider`, `search`). Per-group wiring is
tested in the dedicated test_*.py files (test_interests_actions,
test_outrider_actions, test_search_actions, etc.).
"""
from click.testing import CliRunner

from remyxai.cli.commands import cli


def test_cli_help_lists_current_command_groups():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    for cmd in ("papers", "interests", "outrider", "search"):
        assert cmd in result.output, f"expected '{cmd}' in top-level help"


def test_cli_help_omits_deprecated_commands():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    for gone in ("list-models", "summarize-model", "deploy-model", "dataset"):
        assert gone not in result.output, f"deprecated '{gone}' still listed"
