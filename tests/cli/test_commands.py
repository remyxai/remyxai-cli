import pytest
from unittest.mock import patch, call
from click.testing import CliRunner
from remyxai.cli.commands import cli


@patch("remyxai.cli.commands.handle_model_action")
def test_list_models(mock_handle_model_action):
    runner = CliRunner()
    result = runner.invoke(cli, ["list_models"])
    
    mock_handle_model_action.assert_called_once_with({"subaction": "list"})
    assert result.exit_code == 0


@patch("remyxai.cli.commands.handle_model_action")
def test_summarize_model(mock_handle_model_action):
    runner = CliRunner()
    result = runner.invoke(cli, ["summarize_model", "model_name"])
    
    mock_handle_model_action.assert_called_once_with({"subaction": "summarize", "model_name": "model_name"})
    assert result.exit_code == 0


@patch("remyxai.cli.commands.handle_evaluation_action")
def test_evaluate_myxboard(mock_handle_evaluation_action):
    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate_myxboard", "gpt-3", "gpt-neo", "myxmatch", "lighteval_arithmetic"])

    mock_handle_evaluation_action.assert_called_once_with({
        "models": ("gpt-3", "gpt-neo"),
        "tasks": ("myxmatch", "lighteval_arithmetic")
    })

    assert result.exit_code == 0


@patch("remyxai.cli.commands.handle_deployment_action")
def test_deploy_model_up(mock_handle_deployment_action):
    runner = CliRunner()
    result = runner.invoke(cli, ["deploy_model", "model_name", "up"])

    mock_handle_deployment_action.assert_called_once_with({"model_name": "model_name", "action": "up"})
    assert result.exit_code == 0


@patch("remyxai.cli.commands.handle_deployment_action")
def test_deploy_model_down(mock_handle_deployment_action):
    runner = CliRunner()
    result = runner.invoke(cli, ["deploy_model", "model_name", "down"])

    mock_handle_deployment_action.assert_called_once_with({"model_name": "model_name", "action": "down"})
    assert result.exit_code == 0


def test_deploy_model_invalid_action():
    runner = CliRunner()
    result = runner.invoke(cli, ["deploy_model", "model_name", "invalid_action"])

    assert "Error deploying model" in result.output
    assert result.exit_code != 0
