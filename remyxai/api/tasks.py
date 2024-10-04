import logging
import requests
import urllib.parse
from . import BASE_URL, HEADERS, log_api_response


def run_myxmatch(name: str, prompt: str, models: list) -> dict:
    """Submit a MyxMatch task to the server."""
    mapped_models = []
    for model in models:
        # Map Hugging Face model repos to supported names
        if "/" in model:
            mapped_model = model.split("/")[1]
        else:
            mapped_model = model

        mapped_models.append(mapped_model)

    models_str = ",".join(mapped_models)

    # Ensure name and prompt are URL-encoded for safe API usage
    encoded_name = urllib.parse.quote(name.replace("/", "--"), safe="")
    encoded_prompt = urllib.parse.quote(prompt, safe="")

    url = f"{BASE_URL}/task/myxmatch/{encoded_name}/{encoded_prompt}/{models_str}"
    logging.info(f"POST request to {url}")

    response = requests.post(url, headers=HEADERS)

    if response.status_code == 202:
        try:
            return response.json()
        except (requests.JSONDecodeError, ValueError) as e:
            logging.error(f"Error decoding JSON response: {e}")
            return {"error": "Invalid JSON response"}
    else:
        logging.error(f"Failed to create MyxMatch task: {response.status_code}")
        return {"error": f"Failed to create MyxMatch task: {response.text}"}


def get_job_status(job_name: str) -> dict:
    """
    Get the status of a specific job by job name.

    :param job_name: The name of the job to check.
    :return: A dictionary containing the status of the job.
    """
    url = f"{BASE_URL}/task/job-status/{job_name}"
    logging.info(f"GET request to {url}")

    try:
        response = requests.get(url, headers=HEADERS)
        logging.debug(f"Raw response from server: {response.text}")

        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        return {"status": "error", "message": str(http_err)}
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error occurred: {req_err}")
        return {"status": "error", "message": str(req_err)}
    except ValueError as json_err:
        logging.error(f"JSON parse error: {json_err}")
        return {"status": "error", "message": "Failed to parse JSON response"}


def train_classifier(
    model_name: str, labels: list, model_selector: str, hf_dataset=None
):
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
