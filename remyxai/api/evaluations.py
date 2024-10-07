import logging
import requests
from enum import Enum
from . import BASE_URL, HEADERS


# Define the EvaluationTask enum
class EvaluationTask(Enum):
    MYXMATCH = "myxmatch"


def list_evaluations() -> list:
    """List all evaluations from the server."""
    url = f"{BASE_URL}/evaluation/list"
    logging.info(f"GET request to {url}")
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:
        try:
            return response.json().get("message", [])
        except (requests.JSONDecodeError, ValueError) as e:
            logging.error(f"Error decoding JSON response: {e}")
            return {"error": "Invalid JSON response"}
    else:
        logging.error(f"Failed to fetch evaluations: {response.status_code}")
        return {"error": f"Failed to fetch evaluations: {response.text}"}

def download_evaluation(task_name: str, eval_name: str) -> dict:
    """Download evaluation results using the task name and eval name."""
    # Construct the correct URL for downloading evaluation results
    # Ensure the name is not sanitized incorrectly here
    url = f"{BASE_URL}/evaluation/download/{task_name}/{eval_name}"
    logging.info(f"GET request to {url}")

    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:
        try:
            result = response.json()
            logging.info(f"Downloaded evaluation result: {result}")
            return result
        except (requests.JSONDecodeError, ValueError) as e:
            logging.error(f"Error decoding JSON response: {e}")
            return {"error": "Invalid JSON response"}
    else:
        logging.error(f"Failed to download evaluation result: {response.status_code}")
        return {"error": f"Failed to download evaluation result: {response.text}"}

def delete_evaluation(eval_type: str, eval_name: str) -> dict:
    """Delete an evaluation from the server."""
    url = f"{BASE_URL}/evaluation/delete/{eval_type}/{eval_name}"
    logging.info(f"POST request to {url}")
    response = requests.post(url, headers=HEADERS)

    if response.status_code == 200:
        try:
            return response.json()
        except (requests.JSONDecodeError, ValueError) as e:
            logging.error(f"Error decoding JSON response: {e}")
            return {"error": "Invalid JSON response"}
    else:
        logging.error(f"Failed to delete evaluation: {response.status_code}")
        return {"error": f"Failed to delete evaluation: {response.text}"}
