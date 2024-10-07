import pytest
from unittest.mock import patch
from remyxai.api.user import get_user_profile, get_user_credits


@patch("remyxai.api.user.requests.get")
def test_get_user_profile(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"name": "test_user"}
    profile = get_user_profile()
    assert profile["name"] == "test_user"


@patch("remyxai.api.user.requests.get")
def test_get_user_credits(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"credits": 100}
    credits = get_user_credits()
    assert credits["credits"] == 100
