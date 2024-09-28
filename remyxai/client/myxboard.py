import logging
import requests
from typing import List, Dict, Optional, Union
from huggingface_hub import get_collection, create_collection, add_collection_item
from remyxai.api.tasks import get_job_status

class MyxBoard:
    def __init__(self, model_repo_ids: List[str], name: str, from_hf_collection: bool = False):
        self.name: str = name  # Name of the MyxBoard
        self.models: List[str] = model_repo_ids  # List of model identifiers
        self.from_hf_collection: bool = from_hf_collection  # Track if the MyxBoard is from a Hugging Face collection
        self.results: Dict[str, Dict[str, Optional[Union[float, dict]]]] = self.initialize_results()  # Task results
        self.job_status: Dict[str, str] = {}  # To track the status of running jobs per task
        self.hf_collection_name: Optional[str] = None  # Track the HF collection name (if applicable)

    def initialize_results(self) -> Dict[str, Dict[str, Optional[Union[float, dict]]]]:
        """
        Initializes results for each model with no evaluations yet.
        """
        return {model_id: {} for model_id in self.models}

    def update_results(self, task_name: str, results: Dict[str, Union[float, dict]]) -> None:
        """
        Updates results for a specific task. The results dictionary will contain model ids as keys
        and their corresponding evaluation scores or detailed results as values.
        """
        if "results" not in self.results:
            self.results["results"] = {}

        self.results["results"][task_name] = results
        self._reorder_models_by_results(task_name)

    def _reorder_models_by_results(self, task_name: str) -> None:
        """
        Reorder the models list based on their ranking from the results of a specific task.
        This is particularly useful for tasks like MYXMATCH that return ranked results.
        """
        ranked_models = []
        task_results = self.results["results"].get(task_name, {})

        for model_info in task_results.get("models", []):
            ranked_models.append((model_info["model"], model_info["rank"]))

        ranked_models.sort(key=lambda x: x[1])  # Sort by rank
        self.models = [model_id for model_id, _ in ranked_models]

    def track_job_status(self, job_id: str, task_name: str) -> None:
        """
        Tracks the job status of an ongoing evaluation.
        """
        self.job_status[task_name] = job_id

    def get_job_status(self, task_name: str) -> Optional[str]:
        """
        Get the job status of an evaluation for a specific task.
        """
        return self.job_status.get(task_name)

    def get_results(self) -> Dict[str, Union[str, dict]]:
        """
        Get the current results of the MyxBoard.
        Returns results if tasks are completed or job status if tasks are still running.
        """
        final_result = {
            "name": self.name,
            "models": self.models,
            "from_hf_collection": self.from_hf_collection,
            "hf_collection_name": self.hf_collection_name,
            "results": {}
        }

        for task, job_id in self.job_status.items():
            job_status = self._check_job_status(job_id)
            if job_status != "completed":
                return {"status": job_status, "job_id": job_id}

        final_result["results"] = self.results["results"]
        return final_result

    def _check_job_status(self, job_id: str) -> str:
        """
        Helper method to poll the job status from the server. This is used in get_results() to check if tasks are still running.
        """
        try:
            job_status_response = get_job_status(job_id)
            return job_status_response.get("status", "unknown")
        except Exception as e:
            logging.error(f"Error fetching job status for job {job_id}: {e}")
            return "error"

    @staticmethod
    def from_huggingface_collection(collection_name: str) -> "MyxBoard":
        """
        Create a MyxBoard from a Hugging Face collection.
        """
        collection = get_collection(collection_name)
        model_repo_ids = [item.modelId for item in collection.items if item.item_type == 'model']
        myx_board = MyxBoard(model_repo_ids=model_repo_ids, name=collection_name, from_hf_collection=True)
        myx_board.hf_collection_name = collection_name
        return myx_board

    def to_huggingface_collection(self, collection_title: str, namespace: str, notes: Optional[str] = None) -> str:
        """
        Push the MyxBoard results as a Hugging Face collection, preserving the model ordering.
        """
        try:
            # Check if the collection already exists
            collection = get_collection(collection_title)
            if collection:
                logging.warning(f"Collection {collection_title} already exists. Adding models.")
            else:
                collection = create_collection(title=collection_title, namespace=namespace)

            # Update the Hugging Face collection with the MyxBoard's ranked models
            for model_id in self.models:
                add_collection_item(
                    collection.slug,
                    item_id=model_id,
                    item_type="model",
                    note=notes or f"Added {model_id} from MyxBoard"
                )

            return collection.slug
        except Exception as e:
            logging.error(f"Error pushing MyxBoard to Hugging Face collection: {e}")
            raise

    def update_ordering_from_collection(self, collection_name: str) -> None:
        """
        Update the ordering of the models based on the specified Hugging Face collection.
        """
        try:
            collection = get_collection(collection_name)
            ordered_model_ids = [item.modelId for item in collection.items if item.item_type == 'model']
            self.models = ordered_model_ids
            logging.info(f"Model ordering updated from Hugging Face collection: {collection_name}")
        except Exception as e:
            logging.error(f"Error updating model ordering from Hugging Face collection: {e}")
            raise

