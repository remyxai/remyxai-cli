import os
import shutil
import requests
from io import BytesIO
from . import BASE_URL, HEADERS, log_api_response


def list_models():
    url = f"{BASE_URL}model/list"
    response = requests.get(url, headers=HEADERS)
    return response.json()


def get_model_summary(model_name):
    url = f"{BASE_URL}model/summary/{model_name}"
    response = requests.get(url, headers=HEADERS)
    return response.json()


def delete_model(model_name: str):
    url = f"{BASE_URL}model/delete/{model_name}"
    response = requests.post(url, headers=HEADERS)
    return response.json()


def download_model(model_name: str, model_format: str):
    url = f"{BASE_URL}model/download/{model_name}/{model_format}"
    response = requests.post(url, headers=HEADERS, stream=True)

    if response.status_code == 200:
        filename = f"{model_name}.zip"
        with open(filename, "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
        return response  # Return the full response object
    else:
        return response  # Return the response even if there's an error
