"""Tests for the model-action CLI handler.

`handle_evaluation_action` was removed when MyxBoard was retired —
its test cases (mock_myx_board, mock_evaluate) went with it.
"""
from unittest.mock import patch

from remyxai.cli.evaluation_actions import handle_model_action


@patch("remyxai.cli.evaluation_actions.list_models")
def test_handle_model_action_list(mock_list_models):
    mock_list_models.return_value = ["gpt-3", "gpt-neo"]
    handle_model_action({"subaction": "list"})
    mock_list_models.assert_called_once()


@patch("remyxai.cli.evaluation_actions.get_model_summary")
def test_handle_model_action_summarize(mock_get_model_summary):
    mock_get_model_summary.return_value = {"name": "gpt-3", "summary": "..."}
    handle_model_action({"subaction": "summarize", "model_name": "gpt-3"})
    mock_get_model_summary.assert_called_once_with("gpt-3")


@patch("remyxai.cli.evaluation_actions.delete_model")
def test_handle_model_action_delete(mock_delete_model):
    handle_model_action({"subaction": "delete", "model_name": "gpt-3"})
    mock_delete_model.assert_called_once_with("gpt-3")


@patch("remyxai.cli.evaluation_actions.download_model")
def test_handle_model_action_download(mock_download_model):
    handle_model_action(
        {"subaction": "download", "model_name": "gpt-3", "model_format": "onnx"}
    )
    mock_download_model.assert_called_once_with("gpt-3", "onnx")
