import requests
from . import BASE_URL, HEADERS, log_api_response

def train_classifier(model_name: str, labels: list, model_selector: str, hf_dataset=None):
    url = f"{BASE_URL}task/classify/{model_name}/{','.join(labels)}/{model_selector}"
    params = {"hf_dataset": hf_dataset} if hf_dataset else None
    response = requests.post(url, headers=HEADERS, params=params)
    return response.json()

def train_detector(model_name: str, labels: list, model_selector: str, hf_dataset=None):
    url = f"{BASE_URL}task/detect/{model_name}/{','.join(labels)}/{model_selector}"
    params = {"hf_dataset": hf_dataset} if hf_dataset else None
    response = requests.post(url, headers=HEADERS, params=params)
    return response.json()

def train_generator(model_name: str, hf_dataset: str):
    url = f"{BASE_URL}task/generate/{model_name}"
    params = {"hf_dataset": hf_dataset}
    response = requests.post(url, headers=HEADERS, params=params)
    return response.json()

