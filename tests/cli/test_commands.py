"""Tests for top-level CLI command wiring.

Click renames underscores to dashes in command names by default
(`list_models` → `list-models`), so the runner invokes the kebab-case
form here.
"""
from unittest.mock import patch

from click.testing import CliRunner

from remyxai.cli.commands import cli


@patch("remyxai.cli.commands.handle_model_action")
def test_list_models(mock_handle_model_action):
    runner = CliRunner()
    result = runner.invoke(cli, ["list-models"])

    mock_handle_model_action.assert_called_once_with({"subaction": "list"})
    assert result.exit_code == 0


@patch("remyxai.cli.commands.handle_model_action")
def test_summarize_model(mock_handle_model_action):
    runner = CliRunner()
    result = runner.invoke(cli, ["summarize-model", "model_name"])

    mock_handle_model_action.assert_called_once_with(
        {"subaction": "summarize", "model_name": "model_name"}
    )
    assert result.exit_code == 0


@patch("remyxai.cli.commands.handle_deployment_action")
def test_deploy_model_up(mock_handle_deployment_action):
    runner = CliRunner()
    result = runner.invoke(cli, ["deploy-model", "model_name", "up"])

    mock_handle_deployment_action.assert_called_once_with(
        {"model_name": "model_name", "action": "up"}
    )
    assert result.exit_code == 0


@patch("remyxai.cli.commands.handle_deployment_action")
def test_deploy_model_down(mock_handle_deployment_action):
    runner = CliRunner()
    result = runner.invoke(cli, ["deploy-model", "model_name", "down"])

    mock_handle_deployment_action.assert_called_once_with(
        {"model_name": "model_name", "action": "down"}
    )
    assert result.exit_code == 0


def test_deploy_model_invalid_action():
    """Invalid action must surface as an error with a non-zero exit code.

    The command uses `click.ClickException`, which prefixes the message
    with `Error: ` and exits 1. This is the contract this test enforces.
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["deploy-model", "model_name", "invalid_action"])

    assert "deploying model failed" in result.output
    assert result.exit_code != 0
