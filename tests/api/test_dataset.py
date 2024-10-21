from remyxai.api.datasets import list_datasets, delete_dataset, download_dataset
from unittest.mock import patch
from remyxai.api.datasets import BASE_URL

@patch("remyxai.api.datasets.requests.get")
def test_list_datasets(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"datasets": ["dataset1", "dataset2"]}
    datasets = list_datasets()
    assert isinstance(datasets, list)
    assert len(datasets) > 0

@patch("remyxai.api.datasets.requests.delete")
def test_delete_dataset(mock_delete):
    mock_delete.return_value.status_code = 200
    mock_delete.return_value.json.return_value = {"message": "Dataset deleted successfully"}
    dataset_name = "test_dataset"
    delete_dataset(dataset_name)
    assert mock_delete.called_once_with(f"{BASE_URL}/datasets/delete/{dataset_name}")

@patch("remyxai.api.datasets.requests.get")
def test_download_dataset(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"url": "https://example.com/dataset.zip"}
    dataset_name = "test_dataset"
    download_dataset(dataset_name)
    assert mock_get.called_once_with(f"{BASE_URL}/datasets/download/{dataset_name}")



