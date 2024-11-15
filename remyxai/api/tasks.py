import logging
import requests
import urllib.parse
from . import BASE_URL, HEADERS, log_api_response
from typing import Optional



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

def run_benchmark(name: str, models: list, evals: list) -> dict:
    """Submit a benchmark task to the server."""

    headers = {"Authorization": HEADERS["Authorization"]}

    models_str = ",".join(models)
    evals_str = ",".join(evals)
    encoded_name = urllib.parse.quote(name.replace("/", "--"), safe="")

    # Endpoint URL
    url = f"{BASE_URL}/task/benchmark"

    payload = {
        "name": encoded_name,
        "models": models_str,
        "evals": evals_str
    }

    logging.info(f"POST request to {url} with payload: {payload}")

    response = requests.post(url, headers=headers, data=payload)

    if response.status_code == 202:
        try:
            return response.json()
        except (requests.JSONDecodeError, ValueError) as e:
            logging.error(f"Error decoding JSON response: {e}")
            return {"error": "Invalid JSON response"}
    else:
        logging.error(f"Failed to create benchmark task: {response.status_code}")
        return {"error": f"Failed to create benchmark task: {response.text}"}


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


def run_datacomposer(dataset_name: str, num_samples: int, context: Optional[str] = None, dataset_file: Optional[str] = None) -> dict:
    """
    Submit a Data Composer task to the server. Supports file upload, Hugging Face dataset, or text prompt.
    Args:
        dataset_name (str): The name of the dataset to compose or extend.
        num_samples (int): The number of samples to create.
        context (Optional[str]): The context, which could be a Hugging Face dataset or a text prompt.
        dataset_file (Optional[str]): Path to the dataset file to upload, if any.
    Returns:
        dict: The server's response.
    """
    url = f"{BASE_URL}/task/datacomposer/{urllib.parse.quote(dataset_name)}/{num_samples}"
    headers = {"Authorization": HEADERS["Authorization"]}
    logging.info(f"POST request to {url}")
    data = {}
    files = None
    if context:
        data['context'] = context
    if dataset_file:
        try:
            files = {'dataset-file': open(dataset_file, 'rb')}
        except FileNotFoundError as e:
            logging.error(f"Dataset file not found: {e}")
            return {"error": "Dataset file not found."}
    try:
        if files:
            response = requests.post(url, headers=headers, data=data, files=files)
        else:
            response = requests.post(url, headers=headers, data=data)
        if response.status_code == 202:
            try:
                return response.json()
            except (requests.JSONDecodeError, ValueError) as e:
                logging.error(f"Error decoding JSON response: {e}")
                return {"error": "Invalid JSON response"}
        else:
            logging.error(f"Failed to create Data Composer task: {response.status_code}")
            return {"error": f"Failed to create Data Composer task: {response.text}"}
    except Exception as e:
        logging.error(f"An error occurred while making the request: {e}")
        return {"error": str(e)}
    finally:
        if files:
            files['dataset-file'].close()
