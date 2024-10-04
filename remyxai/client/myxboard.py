from typing import List, Dict, Optional, Union
import logging
import urllib.parse
from remyxai.api.tasks import get_job_status
from remyxai.api.myxboard import (
    list_myxboards,
    store_myxboard,
    update_myxboard,
    download_myxboard,
    delete_myxboard,
)
from remyxai.api.evaluations import EvaluationTask, download_evaluation
from huggingface_hub import get_collection


class MyxBoard:
    MYXMATCH_SUPPORTED_MODELS = [
        "Phi-3-mini-4k-instruct",
        "BioMistral-7B",
        "CodeLlama-7b-Instruct-hf",
        "gorilla-openfunctions-v2",
        "Llama-2-7b-hf",
        "Mistral-7B-Instruct-v0.3",
        "Meta-Llama-3-8B",
        "Meta-Llama-3-8B-Instruct",
        "Qwen2-1.5B",
        "Qwen2-1.5B-Instruct",
    ]

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
        self.name = name or hf_collection_name
        self._sanitized_name = self._sanitize_name(self.name)

        if self.hf_collection_name:
            self.models = self._initialize_from_hf_collection(hf_collection_name)
            self.from_hf_collection = True
        else:
            self.models = model_repo_ids or []
            self.from_hf_collection = False

        self._validate_models(self.models)
        self.results = self._initialize_results()

        existing_myxboard = self._get_existing_myxboard()
        if existing_myxboard:
            self._populate_from_existing(existing_myxboard)
        else:
            self._store_new_myxboard()

    @classmethod
    def get_supported_models(cls) -> List[str]:
        """
        Return the list of supported models for MyxMatch evaluations.
        This allows users to see which models they can use when creating a MyxBoard.
        """
        return cls.MYXMATCH_SUPPORTED_MODELS

    def _sanitize_name(self, name: str) -> str:
        """Sanitize the MyxBoard name by replacing slashes with underscores for API use."""
        return urllib.parse.quote(name.replace("/", "--"), safe="")

    def _validate_models(self, models: List[str]) -> None:
        """
        Validate that the models in the MyxBoard are supported.
        This does not modify the models list.
        """
        for model in models:
            # If HF repo, map it to the supported name for validation
            if "/" in model:
                mapped_model = model.split("/")[1]
            else:
                mapped_model = model

            if mapped_model not in self.MYXMATCH_SUPPORTED_MODELS:
                raise ValueError(
                    f"Model '{model}' is not supported for MYXMATCH evaluation. "
                    f"Supported models: {self.get_supported_models()}"
                )

        logging.info(f"Validated models: {models}")

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
                item.item_id for item in collection.items if item.item_type == "model"
            ]
            logging.info(
                f"MyxBoard initialized from Hugging Face collection: {collection_name}"
            )
            return model_repo_ids
        except Exception as e:
            logging.error(f"Error initializing from Hugging Face collection: {e}")
            raise

    def _get_existing_myxboard(self) -> Optional[Dict]:
        """Check if a MyxBoard with this name already exists on the server."""
        try:
            logging.info(
                f"Checking if MyxBoard '{self.name}' already exists on the server."
            )
            myxboard_list = list_myxboards()
            existing_myxboard = next(
                (
                    myxboard
                    for myxboard in myxboard_list
                    if myxboard["name"] == self._sanitized_name
                ),
                None,
            )
            if existing_myxboard:
                logging.info(f"Existing MyxBoard '{self.name}' found on the server.")
            else:
                logging.info(f"No MyxBoard named '{self.name}' found on the server.")
            return existing_myxboard
        except Exception as e:
            logging.error(f"Error fetching MyxBoard list from the server: {e}")
            return None

    def _populate_from_existing(self, myxboard_data: Dict) -> None:
        """Populate this MyxBoard with data stored on the server."""
        logging.info(
            f"Populating MyxBoard '{self.name}' with existing data from the server."
        )
        self.models = myxboard_data["models"]
        self.results = download_myxboard(self._sanitized_name)

    def _store_new_myxboard(self) -> None:
        """Store a new MyxBoard."""
        try:
            logging.info(f"Storing new MyxBoard '{self.name}'.")
            store_myxboard(self._sanitized_name, self.models, self.results)
            logging.info(f"MyxBoard '{self.name}' created and stored.")
        except Exception as e:
            logging.error(f"Error storing new MyxBoard '{self.name}': {e}")
            raise

    def update_results(
        self, task_name: EvaluationTask, results: Dict[str, Union[float, dict]]
    ) -> None:
        """Update results for a specific task, push the updates to the server, and cache locally."""
        self.results["results"][task_name.value] = results

        if task_name == EvaluationTask.MYXMATCH:
            self._reorder_models_by_results(task_name)

        self._save_updates()

    def _reorder_models_by_results(self, task_name: EvaluationTask) -> None:
        """Reorder the models list based on their ranking from the results."""
        task_results = self.results["results"].get(task_name.value, {})
        ranked_models = [
            (model_info["model"], model_info["rank"])
            for model_info in task_results.get("models", [])
        ]
        ranked_models.sort(key=lambda x: x[1])  # Sort by rank
        self.models = [model_id for model_id, _ in ranked_models]

    def get_results(
        self, task_names: Optional[List[EvaluationTask]] = None
    ) -> Dict[str, Union[str, dict]]:
        """
        Return the current MyxBoard results or job status if tasks are still running.
        - If `task_names` is provided, return results for the specific tasks.
        - If not, return results for all tasks.
        """
        if task_names:
            task_results = {}
            for task_name in task_names:
                task_key = task_name.value

                if task_key in self.results.get("results", {}):
                    logging.info(f"Returning cached results for task: {task_key}")
                    task_results[task_key] = self.results["results"][task_key]
                else:
                    job_info = self.results["job_status"].get(task_key)
                    if job_info:
                        job_status = self._check_job_status(job_info["job_name"])
                        if job_status != "COMPLETED":
                            logging.info(
                                f"Job for task '{task_key}' is still running: {job_status}"
                            )
                            task_results[task_key] = {
                                "status": job_status,
                                "job_name": job_info["job_name"],
                            }
                        else:
                            task_results[task_key] = self._fetch_evaluation_results(
                                task_name
                            )

            self._save_updates()
            return task_results

        for task_key, job_info in self.results.get("job_status", {}).items():
            job_status = self._check_job_status(job_info["job_name"])

            if job_status == "COMPLETED" and task_key not in self.results.get(
                "results", {}
            ):
                task_name = EvaluationTask(
                    task_key
                )  # Convert key back to EvaluationTask
                self._fetch_evaluation_results(task_name)

        logging.info("Returning all cached results")
        self._save_updates()
        return self.results

    def _fetch_evaluation_results(
        self, task_name: EvaluationTask
    ) -> Dict[str, Union[float, dict]]:
        """Fetch and return evaluation results for a completed job. Cache the results in self.results."""
        try:
            logging.info(f"Fetching evaluation results for task: {task_name.value}")
            eval_results = download_evaluation(task_name.value, self._sanitized_name)

            if eval_results:
                self.results["results"][task_name.value] = eval_results
                logging.info(
                    f"Fetched results for task '{task_name.value}' and cached them"
                )

                self.results["job_status"][task_name.value]["status"] = "COMPLETED"

                self._save_updates()
                return eval_results
            else:
                logging.warning(f"No results returned for task '{task_name.value}'")

        except Exception as e:
            logging.error(
                f"Error fetching evaluation results for {task_name.value}: {e}"
            )
            return {"error": f"Error fetching results for task {task_name.value}"}

    def _save_updates(self) -> None:
        """Push updates after any changes to the MyxBoard."""
        try:
            logging.info(f"Pushing MyxBoard updates for {self.name}")
            response = update_myxboard(self._sanitized_name, self.models, self.results)

            if response.get("error"):
                logging.error(f"Error updating MyxBoard remotely: {response['error']}")
            else:
                logging.info(f"MyxBoard {self.name} successfully updated.")
        except Exception as e:
            logging.error(f"Error updating MyxBoard: {e}")
            raise

    def save(self) -> None:
        """
        Manually save the current state of the MyxBoard (including models, results, and job statuses)
        to the studio
        """
        logging.info(f"Manually saving MyxBoard '{self.name}'.")
        self._save_updates()

    def _check_job_status(self, job_name: str) -> str:
        """Check the job status of an evaluation task."""
        try:
            job_status_response = get_job_status(job_name)
            return job_status_response.get("status", "unknown")
        except Exception as e:
            logging.error(f"Error checking job status for {job_name}: {e}")
            return "error"

    def delete(self) -> None:
        """Delete this MyxBoard."""
        try:
            delete_myxboard(self._sanitized_name)
            logging.info(f"MyxBoard {self.name} deleted.")
        except Exception as e:
            logging.error(f"Error deleting MyxBoard: {e}")
            raise

    @staticmethod
    def from_huggingface_collection(collection_name: str) -> "MyxBoard":
        """Create a MyxBoard from a Hugging Face collection."""
        return MyxBoard(hf_collection_name=collection_name)
