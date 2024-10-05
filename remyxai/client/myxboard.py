import os
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
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import create_repo, upload_file, get_collection, update_collection_item, add_collection_item, HfFolder, DatasetCard, HfApi

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
        self.results = {}  # Only for evaluation results
        self.job_status = {}  # Track pending, running, and completed jobs

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
        return {}  # Start with an empty dictionary for results and job statuses

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
        """Populate MyxBoard with data from the server."""
        logging.info(f"Populating MyxBoard '{self.name}' with existing data.")
        self.models = myxboard_data["models"]

        # Download and split results from job statuses
        downloaded_results = download_myxboard(self._sanitized_name)
        if "message" in downloaded_results:
            results_data = downloaded_results["message"]
        else:
            results_data = downloaded_results

        # Only store evaluation results in self.results, track job status separately
        self.results = results_data.get("results", {})
        self.job_status = results_data.get("job_status", {})

    def _store_new_myxboard(self) -> None:
        """Store a new MyxBoard with a flat results structure."""
        try:
            logging.info(f"Storing new MyxBoard '{self.name}'.")
            flat_results = self._flatten_results(self.results)  # Flatten results before storing
            store_myxboard(self._sanitized_name, self.models, flat_results)
            logging.info(f"MyxBoard '{self.name}' created and stored.")
        except Exception as e:
            logging.error(f"Error storing new MyxBoard '{self.name}': {e}")
            raise

    def update_results(
        self, task_name: EvaluationTask, results: Dict[str, Union[float, dict]]
    ) -> None:
        """Update results for a specific task, push the updates to the server, and cache locally."""
        self.results[task_name.value] = results.get("message", results)

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
        task_results = {}

        # Initialize job_status if it's not already present in self.results
        if "job_status" not in self.results:
            self.results["job_status"] = {}

        # If specific tasks are provided, return results only for those tasks
        if task_names:
            for task_name in task_names:
                task_key = task_name.value

                # Check if results are already cached
                if task_key in self.results:
                    logging.info(f"Returning cached results for task: {task_key}")
                    task_results[task_key] = self.results[task_key]
                else:
                    # Check job status if results are not cached
                    job_info = self.results.get("job_status", {}).get(task_key)
                    if job_info:
                        job_status = self._check_job_status(job_info["job_name"])

                        # If job is still running, return the job status
                        if job_status != "COMPLETED":
                            logging.info(f"Job for task '{task_key}' is still running: {job_status}")
                            task_results[task_key] = {
                                "status": job_status,
                                "job_name": job_info["job_name"],
                            }
                        else:
                            # If job is completed, fetch and update the results
                            task_results[task_key] = self._fetch_evaluation_results(task_name)
                    else:
                        logging.warning(f"No job info found for task: {task_key}")

            # Save updates to MyxBoard after fetching the results
            self._save_updates()
            return task_results

        # If no specific tasks are provided, check all tasks for job completion and fetch results
        for task_key, job_info in self.results.get("job_status", {}).items():
            job_status = self._check_job_status(job_info["job_name"])

            # Fetch results for completed jobs
            if job_status == "COMPLETED" and task_key not in self.results:
                task_name = EvaluationTask(task_key)
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
                # Extract and clean the results
                eval_results = eval_results.get("message", eval_results)

                # Store the results in a flat structure within self.results
                self.results[task_name.value] = eval_results

                # Mark the job as completed in job_status
                self.results["job_status"][task_name.value]["status"] = "COMPLETED"

                logging.info(f"Fetched results for task '{task_name.value}' and cached them")

                # Save the updated results
                self._save_updates()
                return eval_results
            else:
                logging.warning(f"No results returned for task '{task_name.value}'")

        except Exception as e:
            logging.error(f"Error fetching evaluation results for {task_name.value}: {e}")
            return {"error": f"Error fetching results for task {task_name.value}"}

    def _flatten_results(self, results: dict) -> dict:
        """Flatten the results structure before storing to avoid unnecessary nesting."""
        if "results" in results and "results" in results["results"]:
            flattened_results = results["results"]
            flattened_results["job_status"] = results.get("job_status", {})
            return flattened_results
        return results

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
    def from_hf_collection(collection_name: str) -> "MyxBoard":
        """Create a MyxBoard from a Hugging Face collection."""
        return MyxBoard(hf_collection_name=collection_name)

    def push_to_hf(self) -> None:
        """
        Push the evaluation results to Hugging Face by creating a dataset, tagging it,
        and adding the dataset to the original collection the MyxBoard is made from.
        """
        if not self.from_hf_collection:
            raise ValueError("This MyxBoard is not created from a Hugging Face collection.")

        if not self.results or not any(k for k in self.results if k != "job_status"):
            raise ValueError("No evaluation results found to push to Hugging Face.")

        try:
            # 1. Parse MyxBoard results into a readable dataset format
            dataset_dict = self._create_dataset_from_results()

            # 2. Generate a name for the dataset using the collection name (remove trailing digits)
            dataset_name = self.hf_collection_name.rsplit("-", 1)[0]

            # 3. Create or update the dataset on Hugging Face
            self._push_dataset_to_hf(dataset_name, dataset_dict)

            # 4. Add dataset to the original collection
            self._add_dataset_to_collection(dataset_name)

            # 5. Tag the dataset with 'remyx'
            self._tag_dataset(dataset_name)

        except Exception as e:
            logging.error(f"Error pushing to Hugging Face: {e}")
            raise

    def _create_dataset_from_results(self) -> DatasetDict:
        """
        Parse the evaluation results from the MyxBoard and format them into a dataset.
        The dataset will have columns like 'task_name' and 'result' which now includes
        all details (models, prompt, etc.) for each evaluation task.
        """
        parsed_data = []

        for task_name, task_result in self.results.items():
            if task_name == "job_status":
                continue  # Skip the job_status key

            # Include all results (models and prompt, etc.) in the 'result' column
            parsed_data.append({
                "task_name": task_name,
                "result": task_result  # Add the full result for this task (models, prompt, etc.)
            })

        # Convert the parsed data to a Hugging Face Dataset
        dataset = Dataset.from_pandas(pd.DataFrame(parsed_data))
        return DatasetDict({"results": dataset})

    def _tag_dataset(self, dataset_name: str) -> None:
        """
        Tag the dataset on Hugging Face with 'remyx' in its metadata.
        """
        try:
            # Load the existing dataset card content from the Hub
            card = DatasetCard.load(dataset_name)

            # Ensure the 'tags' field is initialized
            if 'tags' not in card.data:
                card.data['tags'] = []

            # Add the 'remyx' tag if not already present
            if 'remyx' not in card.data['tags']:
                card.data['tags'].append('remyx')

            code_snippet = f"""
## How to Load and Parse the Results Dataset
The dataset contains evaluation results, with columns for `task_name` and `result`. Each row corresponds to an evaluation task result. The `result` field contains details such as model rankings, prompts, and any other task-specific information.

### Example Code to Load the Dataset:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{dataset_name}")

# Iterate over and view each evaluation result
for example in dataset:
    task_name = example['myxmatch'] # example evaluation task
    result = example['result']  # Contains all task-specific details
    print(f"Task: {{task_name}}, Result: {{result}}")
```
            """
            # Push the updated card with the new tag to the Hub
            card_content = card.content if card.content else ""

            if "## How to Load and Parse the Results Dataset" not in card_content:
                card_content += f"\n\n{code_snippet}"

            # Update the DatasetCard content
            card.content = card_content

            # Push the updated metadata and card content back to the Hub
            card.push_to_hub(dataset_name)

            logging.info(f"Successfully updated dataset '{dataset_name}'.")

        except Exception as e:
            logging.error(f"Error updating dataset: {e}")
            raise

    def _push_dataset_to_hf(self, dataset_name: str, dataset_dict: DatasetDict) -> None:
        """
        Create or update a dataset on Hugging Face using the dataset name, and tag it with 'remyx'.
        """
        try:
            # Attempt to retrieve the token from the environment
            token = HfFolder.get_token() or os.getenv("HF_TOKEN")

            if not token:
                raise EnvironmentError("No Hugging Face token found. Please login using `huggingface-cli login` or set your token in the HF_TOKEN environment variable.")

            # Create a new repository on Hugging Face for the dataset if it doesn't exist
            create_repo(repo_id=dataset_name, repo_type="dataset", private=False, exist_ok=True)

            # Push the dataset to the Hugging Face dataset repository using the retrieved token
            dataset_dict.push_to_hub(repo_id=dataset_name, token=token)

            logging.info(f"Successfully created or updated dataset '{dataset_name}' on Hugging Face.")

        except Exception as e:
            logging.error(f"Error creating or updating dataset '{dataset_name}' on Hugging Face: {e}")
            raise

    def _add_dataset_to_collection(self, dataset_name: str) -> None:
        """
        Add the newly created dataset to the original Hugging Face collection.
        """
        try:
            collection_slug = self.hf_collection_name

            # Retrieve the dataset ID (the name of the created dataset)
            dataset_id = dataset_name

            # Add the dataset to the collection using the Hugging Face API
            add_collection_item(
                collection_slug=collection_slug,
                item_type="dataset",
                item_id=dataset_id,  # Corrected argument
                exists_ok=True
            )

            logging.info(f"Successfully added dataset '{dataset_id}' to collection '{collection_slug}'.")

        except Exception as e:
            logging.error(f"Error adding dataset '{dataset_name}' to collection: {e}")
            raise

