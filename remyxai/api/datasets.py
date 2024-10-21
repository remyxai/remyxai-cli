import logging
import requests
import urllib.parse
from . import BASE_URL, HEADERS, log_api_response
import shutil

def list_datasets() -> list:
    """List all datasets from the server."""
    url = f"{BASE_URL}/datasets/list"
    response = requests.get(url, headers=HEADERS) 

    log_api_response(response)  

    if response.status_code == 200:
        return response.json().get("message", [])
    else:
        logging.error(f"Failed to fetch datasets list: {response.status_code}")
        return {"error": f"Failed to fetch datasets list: {response.text}"}


def download_dataset(dataset_type: str, dataset_name: str):
    """Download dataset by generating a presigned URL."""
    url = f"{BASE_URL}/datasets/download/{dataset_type}/{dataset_name}"
    response = requests.get(url, headers=HEADERS, stream=True)

    log_api_response(response)

    if response.status_code == 200:
        presigned_url = response.json().get("presigned_url", "")
        if presigned_url:
            filename = f"{dataset_name}.csv"
            with requests.get(presigned_url, stream=True) as r:
                with open(filename, "wb") as out_file:
                    shutil.copyfileobj(r.raw, out_file)
            return {"message": f"Dataset {dataset_name} downloaded successfully"}
        else:
            logging.error("Presigned URL not found in the response")
            return {"error": "Presigned URL not found in the response"}
    else:
        logging.error(f"Failed to download dataset: {response.status_code}")
        return {"error": f"Failed to download dataset: {response.text}"}


def delete_dataset(dataset_type: str, dataset_name: str) -> str:
    """Delete a dataset."""
    url = f"{BASE_URL}/datasets/delete/{dataset_type}/{dataset_name}"
    response = requests.delete(url, headers=HEADERS)  

    log_api_response(response)  

    if response.status_code == 200:
        return response.json().get("message", "")

    else:
        logging.error(f"Failed to delete dataset: {response.status_code}")
        return {"error": f"Failed to delete dataset: {response.text}"}
    
