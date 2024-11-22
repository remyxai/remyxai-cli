import logging
import requests
from typing import List
from enum import Enum
from remyxai.api.models import fetch_available_architectures
from . import BASE_URL, HEADERS


class AvailableArchitectures:
    """
    Class to interact with the list of available model architectures.
    """
    def __init__(self):
        self.architectures = self._load_architectures()

    def _load_architectures(self):
        architectures = fetch_available_architectures()
        if not architectures:
            logging.warning("Using empty model list due to fetch error.")
        return architectures

    def list_architectures(self):
        return self.architectures

    def is_architecture_available(self, architecture_name):
        return architecture_name in self.architectures

class EvaluationTask(Enum):
    MYXMATCH = "myxmatch"
    BENCHMARK = "benchmark"


class BenchmarkTask(Enum):
    BIGBENCH_ANALOGICAL_SIMILARITY = "bigbench|analogical_similarity|0|0"
    BIGBENCH_AUTHORSHIP_VERIFICATION = "bigbench|authorship_verification|0|0"
    BIGBENCH_CODE_LINE_DESCRIPTION = "bigbench|code_line_description|0|0"
    BIGBENCH_CONCEPTUAL_COMBINATIONS = "bigbench|conceptual_combinations|0|0"
    BIGBENCH_LOGICAL_DEDUCTION = "bigbench|logical_deduction|0|0"
    HARNESS_CAUSAL_JUDGMENT = "harness|bbh:causal_judgment|0|0"
    HARNESS_DATE_UNDERSTANDING = "harness|bbh:date_understanding|0|0"
    HARNESS_DISAMBIGUATION_QA = "harness|bbh:disambiguation_qa|0|0"
    HARNESS_GEOMETRIC_SHAPES = "harness|bbh:geometric_shapes|0|0"
    HARNESS_LOGICAL_DEDUCTION_FIVE_OBJECTS = (
        "harness|bbh:logical_deduction_five_objects|0|0"
    )
    HELM_BABI_QA = "helm|babi_qa|0|0"
    HELM_BBQ = "helm|bbq|0|0"
    HELM_BOOLQ = "helm|boolq|0|0"
    HELM_COMMONSENSEQA = "helm|commonsenseqa|0|0"
    HELM_MMLU_PHILOSOPHY = "helm|mmlu:philosophy|0|0"
    LEADERBOARD_ARC_CHALLENGE = "leaderboard|arc:challenge|0|0"
    LEADERBOARD_GSM8K = "leaderboard|gsm8k|0|0"
    LEADERBOARD_HELLASWAG = "leaderboard|hellaswag|0|0"
    LEADERBOARD_TRUTHFULQA_MC = "leaderboard|truthfulqa:mc|0|0"
    LEADERBOARD_MMLU_WORLD_RELIGIONS = "leaderboard|mmlu:world_religions|0|0"
    LIGHTEVAL_ARC_EASY = "lighteval|arc:easy|0|0"
    LIGHTEVAL_ASDIV = "lighteval|asdiv|0|0"
    LIGHTEVAL_BIGBENCH_MOVIE_RECOMMENDATION = (
        "lighteval|bigbench:movie_recommendation|0|0"
    )
    LIGHTEVAL_GLUE_COLA = "lighteval|glue:cola|0|0"
    LIGHTEVAL_TRUTHFULQA_GEN = "lighteval|truthfulqa:gen|0|0"

    @classmethod
    def list_tasks(cls) -> List[str]:
        """Return a list of available benchmark tasks as strings."""
        return [task.value for task in cls]


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
