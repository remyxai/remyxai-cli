import time
import logging
from enum import Enum
from typing import List, Dict
import requests
from . import BASE_URL, HEADERS, log_api_response
from huggingface_hub import get_collection, create_collection, add_collection_item
from typing import Any, List, Dict, Optional

class MyxBoard:
    def __init__(self, model_repo_ids: List[str]):
        self.models: List[str] = model_repo_ids  # List of model identifiers
        self.results: Dict[str, Dict[str, Optional[float]]] = self.initialize_results_dataframe(model_repo_ids)

    def initialize_results_dataframe(self, model_repo_ids: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
        # Initialize a dictionary to store evaluation results for each model
        return {model_id: {"status": "candidate"} for model_id in model_repo_ids}

    def update_results(self, results: Dict[str, Dict[str, float]]) -> None:
        # Update the self.results dictionary with new evaluation metrics
        for model_id, task_results in results.items():
            for task, value in task_results.items():
                self.results[model_id][task] = value

    def get_results(self) -> Dict[str, Dict[str, Optional[float]]]:
        # Return results as a dictionary
        return self.results

    @staticmethod
    def from_huggingface_collection(collection_name: str) -> "MyxBoard":
        # Fetch the collection from Hugging Face using the huggingface_hub API
        collection = get_collection(collection_name)

        # Filter to only include items with item_type='model'
        model_repo_ids = [item.modelId for item in collection.items if item.item_type == 'model']

        # Return an instance of MyxBoard initialized with model_repo_ids
        return MyxBoard(model_repo_ids)

    def to_huggingface_collection(self, collection_title: str, namespace: str, notes: Optional[str] = None) -> str:
        # Create a new collection on Hugging Face
        collection = create_collection(title=collection_title, namespace=namespace)

        # Add models from the MyxBoard to the collection
        for model_id in self.models:
            add_collection_item(
                collection.slug,
                item_id=model_id,
                item_type="model",
                note=notes or f"Added {model_id} from MyxBoard"
            )

        # Return the slug of the created collection for reference
        return collection.slug

class EvaluationTask(Enum):
    MYXMATCH = "myxmatch"
    LIGHTEVAL_ARITHMETIC = "lighteval_arithmetic"
    LIGHTEVAL_TRUTHFULQA = "lighteval_truthfulqa"

def myxmatch_evaluation(myx_board: MyxBoard, task: EvaluationTask) -> Dict[str, Dict[str, float]]:
    payload = {"models": myx_board.models, "task": task.value}
    response = requests.post(f"{BASE_URL}/evaluate", json=payload, headers=HEADERS)
    if response.status_code == 200:
        return response.json().get('results', {})
    else:
        raise Exception(f"Evaluation failed: {response.status_code}")

def evaluate_myx_board(myx_board: MyxBoard, tasks: List[EvaluationTask]) -> None:
    for task in tasks:
        results = myxmatch_evaluation(myx_board, task)
        myx_board.update_results({task.name: results})


def evaluate_task(board: MyxBoard, task: EvaluationTask) -> None:
    """
    Evaluate all models for a specific task. If it's a long-running job, it will handle the job asynchronously.
    Args:
        board (MyxBoard): The board containing models to be evaluated.
        task (EvaluationTask): The evaluation task to run.
    """
    payload: Dict[str, Any] = {
        "models": board.models,  # Send model identifiers
        "task": task.value        # Send task type from enum
    }

    # Send request to the evaluation API
    response = requests.post(f"{BASE_URL}/evaluate", json=payload, headers=HEADERS)

    if response.status_code == 200:
        data = response.json()
        # Check if the task is a long-running one (e.g., job ID is provided)
        if 'job_id' in data:
            handle_long_running_task(data['job_id'], board, task)
        else:
            # If results are returned immediately, update the board
            board.update_results(data['results'])
    else:
        logging.error(f"Error during task evaluation: {response.status_code}")
        raise Exception(f"Evaluation failed with status code: {response.status_code}")

def handle_long_running_task(job_id: str, board, task) -> None:
    """
    Poll for job completion and update the MyxBoard results when the job is finished.
    Args:
        job_id (str): The job ID for the long-running task.
        board (MyxBoard): The MyxBoard object tracking models and results.
        task (EvaluationTask): The evaluation task.
    """
    while True:
        response = requests.get(f"{BASE_URL}/job_status/{job_id}", headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'completed':
                # Update board results when the job is completed
                board.update_results(data['results'])
                logging.info(f"Task {task.value} for job {job_id} completed successfully.")
                break
            elif data['status'] == 'failed':
                logging.error(f"Job {job_id} for task {task.value} failed.")
                break
            else:
                logging.info(f"Job {job_id} for task {task.value} is still running...")
                time.sleep(5)  # Poll every 5 seconds
        else:
            logging.error(f"Failed to poll job status for {job_id}")
            raise Exception(f"Error polling job status: {response.status_code}")

