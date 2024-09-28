import logging
import requests
from enum import Enum
from typing import List, Dict
from remyxai.client.myxboard import MyxBoard
from .tasks import get_job_status 
from . import BASE_URL, HEADERS

class EvaluationTask(Enum):
    MYXMATCH = "myxmatch"

def list_evaluations() -> List[Dict[str, str]]:
    """List all evaluations for the current user."""
    url = f"{BASE_URL}/api/v1.0/evaluation/list"
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            return response.json().get("message", [])
        else:
            logging.error(f"Failed to list evaluations: {response.status_code}")
            return []
    except Exception as e:
        logging.error(f"Error listing evaluations: {e}")
        raise

def delete_evaluation(eval_type: str, name: str) -> bool:
    """Delete a specific evaluation."""
    url = f"{BASE_URL}/api/v1.0/evaluation/delete/{eval_type}/{name}"
    try:
        response = requests.post(url, headers=HEADERS)
        if response.status_code == 200:
            logging.info(f"Evaluation {eval_type} for {name} deleted successfully.")
            return True
        else:
            logging.error(f"Failed to delete evaluation: {response.status_code}")
            return False
    except Exception as e:
        logging.error(f"Error deleting evaluation: {e}")
        raise

def download_evaluation(eval_type: str, name: str) -> Dict[str, dict]:
    """Download evaluation results by evaluation type and name."""
    url = f"{BASE_URL}/api/v1.0/evaluation/download/{eval_type}/{name}"
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            return response.json().get("message", {})
        else:
            logging.error(f"Failed to download evaluation: {response.status_code}")
            return {}
    except Exception as e:
        logging.error(f"Error downloading evaluation: {e}")
        raise
