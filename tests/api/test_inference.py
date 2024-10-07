import pytest
from unittest.mock import patch
from remyxai.api.inference import run_inference


@patch("remyxai.api.inference.InferenceServerClient")
def test_run_inference(mock_triton_client):
    mock_client_instance = mock_triton_client.return_value
    mock_client_instance.infer.return_value.get_response.return_value = {
        "outputs": [{"data": ["output_data"]}]
    }

    results, elapsed_time = run_inference("model_name", "test_prompt")
    assert results == "output_data"
