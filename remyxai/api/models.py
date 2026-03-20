import logging
import shutil
import requests
from typing import Optional
from functools import lru_cache
from . import BASE_URL, HEADERS, get_headers, log_api_response


def _resolve_headers(api_key: Optional[str] = None) -> dict:
    """Return headers using explicit key if provided, else module-level default."""
    if api_key:
        return get_headers(api_key)
    return HEADERS


@lru_cache(maxsize=1)
def fetch_available_architectures():
    url = f"{BASE_URL}/model/architectures"
    response = requests.get(url)
    architectures = response.json()
    return architectures


def list_models(api_key: Optional[str] = None):
    url = f"{BASE_URL}/model/list"
    response = requests.get(url, headers=_resolve_headers(api_key))
    return response.json()


def get_model_summary(model_name: str, api_key: Optional[str] = None):
    url = f"{BASE_URL}/model/summary/{model_name}"
    response = requests.get(url, headers=_resolve_headers(api_key))
    return response.json()


def delete_model(model_name: str, api_key: Optional[str] = None):
    url = f"{BASE_URL}/model/delete/{model_name}"
    response = requests.post(url, headers=_resolve_headers(api_key))
    return response.json()


def download_model(
    model_name: str, model_format: str, api_key: Optional[str] = None
):
    url = f"{BASE_URL}/model/download/{model_name}/{model_format}"
    response = requests.post(url, headers=_resolve_headers(api_key), stream=True)

    if response.status_code == 200:
        filename = f"{model_name}.zip"
        with open(filename, "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
        return response
    else:
        return response
