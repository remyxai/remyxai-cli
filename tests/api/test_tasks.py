import pytest
from unittest.mock import patch
from remyxai.api.tasks import train_classifier, train_detector, train_generator


@patch("remyxai.api.tasks.requests.post")
def test_train_classifier(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"task_id": "123"}
    response = train_classifier("model_name", ["label1", "label2"], "model_selector")
    assert response["task_id"] == "123"


@patch("remyxai.api.tasks.requests.post")
def test_train_detector(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"task_id": "456"}
    response = train_detector("model_name", ["label1", "label2"], "model_selector")
    assert response["task_id"] == "456"


@patch("remyxai.api.tasks.requests.post")
def test_train_generator(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"task_id": "789"}
    response = train_generator("model_name", "hf_dataset")
    assert response["task_id"] == "789"
