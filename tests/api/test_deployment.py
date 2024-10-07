import pytest
from unittest.mock import patch, mock_open
from io import BytesIO
from remyxai.api.deployment import download_deployment_package, deploy_model
from remyxai.api.models import download_model


@patch("remyxai.api.models.requests.post")
def test_download_model_success(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.raw = BytesIO(b"binary content")

    with patch("builtins.open", mock_open()) as mock_file:
        response = download_model("model_1", "onnx")
        mock_file.assert_called_once_with("model_1.zip", "wb")
        assert response.status_code == 200


@patch("remyxai.api.deployment.os.path.exists", return_value=True)
@patch("remyxai.api.deployment.os.chdir")
@patch("remyxai.api.deployment.subprocess.run")
def test_deploy_model_down(mock_subprocess, mock_chdir, mock_exists):
    deploy_model("model_name", action="down")
    mock_subprocess.assert_called_once_with(["docker", "compose", "down"], check=True)
