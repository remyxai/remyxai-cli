import logging
import requests
from typing import List
from enum import Enum
from . import BASE_URL, HEADERS


class AvailableModels(Enum):
    PHI_3_MINI_4K_INSTRUCT = "microsoft/Phi-3-mini-4k-instruct"
    BIOMISTRAL_7B = "BioMistral/BioMistral-7B"
    CODELLAMA_7B_INSTRUCT_HF = "codellama/CodeLlama-7b-Instruct-hf"
    GORILLA_OPENFUNCTIONS_V2 = "gorilla-llm/gorilla-openfunctions-v2"
    LLAMA_2_7B_HF = "meta-llama/Llama-2-7b-hf"
    MISTRAL_7B_INSTRUCT_V0_3 = "mistralai/Mistral-7B-Instruct-v0.3"
    META_LLAMA_3_8B = "meta-llama/Meta-Llama-3-8B"
    META_LLAMA_3_8B_INSTRUCT = "meta-llama/Meta-Llama-3-8B-Instruct"
    QWEN2_1_5B = "Qwen/Qwen2-1.5B"
    QWEN2_1_5B_INSTRUCT = "Qwen/Qwen2-1.5B-Instruct"

    @classmethod
    def list_models(cls) -> List[str]:
        """Return a list of supported model names as strings."""
        return [model.value for model in cls]


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
