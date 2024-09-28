import pytest
from unittest.mock import patch
from remyxai.cli.deployment_actions import handle_deployment_action

@patch("remyxai.cli.deployment_actions.RemyxAPI.deploy_model")
def test_handle_deployment_action_up(mock_deploy_model):
    args = {"model_name": "gpt-3", "action": "up"}
    handle_deployment_action(args)
    
    # Ensure that deploy_model was called with correct args
    mock_deploy_model.assert_called_once_with("gpt-3", "up")


@patch("remyxai.cli.deployment_actions.RemyxAPI.deploy_model")
def test_handle_deployment_action_down(mock_deploy_model):
    args = {"model_name": "gpt-3", "action": "down"}
    handle_deployment_action(args)
    
    # Ensure that deploy_model was called with correct args
    mock_deploy_model.assert_called_once_with("gpt-3", "down")


def test_handle_deployment_action_invalid():
    args = {"model_name": "gpt-3", "action": "invalid"}
    
    with pytest.raises(ValueError):
        handle_deployment_action(args)

