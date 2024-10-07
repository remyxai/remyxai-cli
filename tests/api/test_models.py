import shutil
import pytest
from io import BytesIO
from unittest.mock import patch, mock_open
from remyxai.api.models import (
    list_models,
    get_model_summary,
    delete_model,
    download_model,
)


@patch("remyxai.api.models.requests.get")
def test_list_models(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = ["model_1", "model_2"]
    models = list_models()
    assert models == ["model_1", "model_2"]


@patch("remyxai.api.models.requests.get")
def test_get_model_summary(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"name": "model_1"}
    summary = get_model_summary("model_1")
    assert summary["name"] == "model_1"


@patch("remyxai.api.models.requests.post")
def test_delete_model(mock_post):
    mock_post.return_value.status_code = 200
    response = delete_model("model_1")
    assert response == mock_post.return_value.json()


@patch("remyxai.api.models.requests.post")
def test_download_model_success(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.raw = BytesIO(b"binary content")

    with patch("builtins.open", mock_open()) as mock_file:
        response = download_model("model_1", "onnx")
        mock_file.assert_called_once_with("model_1.zip", "wb")
        assert response.status_code == 200
