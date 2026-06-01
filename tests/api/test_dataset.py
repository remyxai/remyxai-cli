from unittest.mock import patch

from remyxai.api.datasets import BASE_URL, delete_dataset, download_dataset, list_datasets


@patch("remyxai.api.datasets.requests.get")
def test_list_datasets(mock_get):
    """list_datasets reads from the response's 'message' field."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"message": ["dataset1", "dataset2"]}

    datasets = list_datasets()

    assert isinstance(datasets, list)
    assert datasets == ["dataset1", "dataset2"]


@patch("remyxai.api.datasets.requests.delete")
def test_delete_dataset(mock_delete):
    """delete_dataset requires (dataset_type, dataset_name)."""
    mock_delete.return_value.status_code = 200
    mock_delete.return_value.json.return_value = {"message": "Dataset deleted successfully"}

    result = delete_dataset("eval", "test_dataset")

    assert result == "Dataset deleted successfully"
    args, _ = mock_delete.call_args
    assert args[0] == f"{BASE_URL}/datasets/delete/eval/test_dataset"


@patch("remyxai.api.datasets.requests.get")
def test_download_dataset(mock_get):
    """download_dataset requires (dataset_type, dataset_name); returns an error
    when no presigned_url is in the response (avoids touching the network)."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"presigned_url": ""}

    result = download_dataset("eval", "test_dataset")

    # With an empty presigned URL the function returns an explicit error dict
    # rather than attempting a download. Keep the test hermetic.
    assert isinstance(result, dict)
    assert "error" in result
    args, _ = mock_get.call_args
    assert args[0] == f"{BASE_URL}/datasets/download/eval/test_dataset"
