from typing import List, Dict, Optional, Union
import logging
from remyxai.api.tasks import get_job_status
from remyxai.api.myxboard import (
    list_myxboards,  # Updated import
    store_myxboard,
    update_myxboard,
    download_myxboard,
    delete_myxboard,
)
from remyxai.api.evaluations import EvaluationTask
from huggingface_hub import get_collection


class MyxBoard:
    def __init__(
        self,
        model_repo_ids: Optional[List[str]] = None,
        name: Optional[str] = None,
        hf_collection_name: Optional[str] = None,
    ):
        """
        Initialize a MyxBoard either with a list of model repository IDs or from a Hugging Face collection.

        :param model_repo_ids: List of model repository IDs to initialize the MyxBoard.
        :param name: Name of the MyxBoard.
        :param hf_collection_name: Optional name of the Hugging Face collection to use if initializing from a collection.
        """
        self.hf_collection_name = hf_collection_name

        if self.hf_collection_name:
            self.name = hf_collection_name
            self.models = self._initialize_from_hf_collection(hf_collection_name)
            self.from_hf_collection = True
        else:
            self.name = name
            self.models = model_repo_ids or []
            self.from_hf_collection = False

        self.results = self._initialize_results()

        # Check if this MyxBoard already exists
        existing_myxboard = self._get_existing_myxboard()
        if existing_myxboard:
            self._populate_from_existing(existing_myxboard)
        else:
            self._store_new_myxboard()

    def _initialize_results(self) -> Dict[str, Union[str, dict]]:
        """Initialize the MyxBoard's result structure."""
        return {
            "name": self.name,
            "models": self.models,
            "from_hf_collection": self.from_hf_collection,
            "hf_collection_name": self.hf_collection_name,
            "results": {},
            "job_status": {},
        }

    def _initialize_from_hf_collection(self, collection_name: str) -> List[str]:
        """Fetch models from a Hugging Face collection."""
        try:
            collection = get_collection(collection_name)
            model_repo_ids = [
                item.modelId for item in collection.items if item.item_type == "model"
            ]
            logging.info(
                f"MyxBoard initialized from Hugging Face collection: {collection_name}"
            )
            return model_repo_ids
        except Exception as e:
            logging.error(f"Error initializing from Hugging Face collection: {e}")
            raise

    def _get_existing_myxboard(self) -> Optional[Dict]:
        """Check if a MyxBoard already exists."""
        try:
            myxboard_list = list_myxboards()  # Updated to use list_myxboards
            return next(
                (
                    myxboard
                    for myxboard in myxboard_list
                    if myxboard["name"] == self.name
                ),
                None,
            )
        except Exception as e:
            logging.error(f"Error fetching MyxBoard list: {e}")
            return None

    def _populate_from_existing(self, myxboard_data: Dict) -> None:
        """Populate this MyxBoard with data stored."""
        self.models = myxboard_data["models"]
        self.results = download_myxboard(self.name).get("results", {})

    def _store_new_myxboard(self) -> None:
        """Store a new MyxBoard."""
        try:
            store_myxboard(self.name, self.models, self.results)
            logging.info(f"MyxBoard {self.name} created.")
        except Exception as e:
            logging.error(f"Error storing MyxBoard: {e}")
            raise

    def update_results(
        self, task_name: EvaluationTask, results: Dict[str, Union[float, dict]]
        ) -> None:
        """Update results for a specific task and push the updated MyxBoard."""
        self.results["results"][task_name.value] = results

        # If the task is MYXMATCH, reorder the models by rank
        if task_name == EvaluationTask.MYXMATCH:
            self._reorder_models_by_results(task_name)

        self._update_myxboard()

    def _reorder_models_by_results(self, task_name: EvaluationTask) -> None:
        """
        Reorder the models list based on their ranking from the results of a specific task.
        This is particularly useful for tasks like MYXMATCH that return ranked results.
        """
        task_results = self.results["results"].get(task_name.value, {})
        ranked_models = []

        # Extract and sort models by rank
        for model_info in task_results.get("models", []):
            ranked_models.append((model_info["model"], model_info["rank"]))

        ranked_models.sort(key=lambda x: x[1])  # Sort by rank
        self.models = [model_id for model_id, _ in ranked_models]

    def _update_myxboard(self) -> None:
        """Push the updated MyxBoard."""
        try:
            update_myxboard(self.name, self.models, self.results)
            logging.info(f"MyxBoard {self.name} updated.")
        except Exception as e:
            logging.error(f"Error updating MyxBoard: {e}")
            raise

    def delete(self) -> None:
        """Delete this MyxBoard."""
        try:
            delete_myxboard(self.name)
            logging.info(f"MyxBoard {self.name} deleted.")
        except Exception as e:
            logging.error(f"Error deleting MyxBoard: {e}")
            raise

    def get_results(self) -> Dict[str, Union[str, dict]]:
        """Return the current MyxBoard results or job status if tasks are still running."""
        for task, job_info in self.results.get("job_status", {}).items():
            job_status = self._check_job_status(job_info["job_name"])
            if job_status != "completed":
                return {"status": job_status, "job_name": job_info["job_name"]}

        return self.results

    def _check_job_status(self, job_name: str) -> str:
        """Check the job status of an evaluation task."""
        try:
            job_status_response = get_job_status(job_name)  # Use job_name to check status
            return job_status_response.get("status", "unknown")
        except Exception as e:
            logging.error(f"Error checking job status for {job_name}: {e}")
            return "error"

    @staticmethod
    def from_huggingface_collection(collection_name: str) -> "MyxBoard":
        """Create a MyxBoard from a Hugging Face collection."""
        return MyxBoard(hf_collection_name=collection_name)

