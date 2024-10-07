import pytest
from unittest.mock import patch
from remyxai.cli.evaluation_actions import handle_model_action, handle_evaluation_action

@patch("remyxai.cli.evaluation_actions.RemyxAPI.list_models")
def test_handle_model_action_list(mock_list_models):
    mock_list_models.return_value = ["gpt-3", "gpt-neo"]
    args = {"subaction": "list"}
    handle_model_action(args)
    mock_list_models.assert_called_once()


@patch("remyxai.cli.evaluation_actions.RemyxAPI.get_model_summary")
def test_handle_model_action_summarize(mock_get_model_summary):
    mock_get_model_summary.return_value = {"name": "gpt-3", "summary": "A large language model."}
    args = {"subaction": "summarize", "model_name": "gpt-3"}
    handle_model_action(args)
    mock_get_model_summary.assert_called_once_with("gpt-3")


@patch("remyxai.cli.evaluation_actions.RemyxAPI.evaluate")
@patch("remyxai.cli.evaluation_actions.MyxBoard")
def test_handle_evaluation_action(mock_myx_board, mock_evaluate):
    mock_myx_board.return_value.get_results.return_value = {
        "gpt-3": {"myxmatch": 0.9}
    }
    
    args = {"models": ["gpt-3"], "tasks": ["myxmatch"]}
    handle_evaluation_action(args)
    mock_evaluate.assert_called_once()
    mock_myx_board.return_value.get_results.assert_called_once()
