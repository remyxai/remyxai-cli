import pytest
from unittest.mock import patch, MagicMock
from remyxai.api.evaluations import (
    MyxBoard,
    evaluate_myxboard,
    evaluate_task,
    handle_long_running_task,
    EvaluationTask,
)


@patch("remyxai.api.evaluations.requests.post")
def test_evaluate_myx_board(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "results": {"gpt-3": 2.0, "gpt-neo": 1.5}
    }

    myx_board = MyxBoard(["gpt-3", "gpt-neo"])
    evaluate_myxboard(myx_board, [EvaluationTask.MYXMATCH])

    assert myx_board.results["gpt-3"]["MYXMATCH"] == 2.0
    assert myx_board.results["gpt-neo"]["MYXMATCH"] == 1.5


@patch("remyxai.api.evaluations.requests.post")
def test_evaluate_task(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"results": {"gpt-3": 0.9}}

    myx_board = MyxBoard(["gpt-3", "gpt-neo"])
    evaluate_task(myx_board, EvaluationTask.MYXMATCH)

    assert myx_board.results["gpt-3"]["MYXMATCH"] == 0.9


@patch("remyxai.api.evaluations.requests.get")
@patch("remyxai.api.evaluations.time.sleep", return_value=None)
def test_handle_long_running_task(mock_sleep, mock_get):
    mock_get.side_effect = [
        MagicMock(status_code=200, json=MagicMock(return_value={"status": "running"})),
        MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={"status": "completed", "results": {"gpt-3": 3.0}}
            ),
        ),
    ]

    myx_board = MyxBoard(["gpt-3"])
    handle_long_running_task("job_1234", myx_board, EvaluationTask.MYXMATCH)

    assert myx_board.results["gpt-3"]["MYXMATCH"] == 3.0
