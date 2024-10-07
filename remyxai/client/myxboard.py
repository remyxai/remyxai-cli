import os
import time
import logging
from typing import List, Dict, Optional, Union
import urllib.parse
from remyxai.api.evaluations import EvaluationTask, download_evaluation
from remyxai.api.tasks import get_job_status
from remyxai.api.myxboard import (
    list_myxboards,
    store_myxboard,
    update_myxboard,
    download_myxboard,
    delete_myxboard,
)
from remyxai.utils.myxboard import (
    _reorder_models_by_results,
    _validate_models,
    add_code_snippet_to_card,
    format_results_for_storage,
)
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import (
    create_repo,
    get_collection,
    add_collection_item,
    HfFolder,
    DatasetCard,
)


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
        self.hf_collection_name = hf_collection_name
        self.name = name or hf_collection_name
        self._sanitized_name = self._sanitize_name(self.name)

        if self.hf_collection_name:
            self.models = self._initialize_from_hf_collection(hf_collection_name)
            self.from_hf_collection = True
        else:
            self.models = model_repo_ids or []
            self.from_hf_collection = False

        _validate_models(self.models, self.MYXMATCH_SUPPORTED_MODELS)
        self.results = {}
        self.job_status = {}

        existing_myxboard = self._get_existing_myxboard()
        if existing_myxboard:
            self._populate_from_existing(existing_myxboard)
        else:
            self._store_new_myxboard()

    def poll_and_store_results(self) -> bool:
        """
        Poll for job completion and store results.
        Return True if all jobs are completed; otherwise, return False.
        """
        completed = True
        for task_name, job_info in self.results.get("job_status", {}).items():
            if job_info["status"] != "COMPLETED":
                status = get_job_status(job_info["job_name"]).get("status", "unknown")
                job_info["status"] = status
                logging.info(f"Polling task: {task_name} | Current status: {status}")

                if status == "COMPLETED":
                    eval_results = self._fetch_evaluation_results(task_name)
                    logging.info(
                        f"Fetched eval_results for task {task_name}: {eval_results}"
                    )

                    if isinstance(eval_results, dict) and "models" in eval_results:
                        logging.info(f"Formatting results for task: {task_name}")
                        formatted_results = format_results_for_storage(
                            eval_results, task_name, job_info["start_time"], time.time()
                        )
                        logging.info(
                            f"Formatted results for task {task_name}: {formatted_results}"
                        )
                        self.results[task_name] = formatted_results
                    else:
                        logging.error(
                            f"Unexpected format for eval_results: {eval_results}"
                        )
                else:
                    completed = False
        self._save_updates()
        return completed

    def fetch_results(self) -> Dict[str, Union[str, dict]]:
        """
        Poll the server for completed job results. If the job is completed, fetch the results,
        update the results and job status fields, and return the updated results.
        """
        try:
            updated_results = {}

            for task_name, job_info in self.results.get("job_status", {}).items():
                job_name = job_info.get("job_name")
                current_status = job_info.get("status")

                if current_status != "COMPLETED":
                    job_status_response = get_job_status(job_name)
                    new_status = job_status_response.get("status")

                    self.results["job_status"][task_name]["status"] = new_status

                    if new_status == "COMPLETED":
                        eval_results = self._fetch_evaluation_results(task_name)

                        self.results[task_name] = eval_results
                        self.results["job_status"][task_name]["status"] = "COMPLETED"

                        updated_results[task_name] = eval_results

            self._save_updates()

            return updated_results if updated_results else self.results

        except Exception as e:
            logging.error(f"Error fetching results: {e}")
            raise

    def view_results(self) -> dict:
        """
        View results in a simplified format without polling the server.
        """
        return {
            task_name: result
            for task_name, result in self.results.items()
            if task_name != "job_status"
        }

    def _fetch_evaluation_results(self, task_name: str) -> Dict[str, Union[str, dict]]:
        """
        Fetch evaluation results from the server for a completed job and return them.
        """
        try:
            logging.info(f"Fetching evaluation results for task: {task_name}")
            eval_results = download_evaluation(task_name, self._sanitized_name)

            logging.info(
                f"Raw eval_results fetched for task {task_name}: {eval_results}"
            )

            if isinstance(eval_results, dict):
                eval_results = eval_results.get("message", eval_results)
            else:
                logging.error(f"Invalid eval_results format: {eval_results}")
                return {}

            logging.info(f"Successfully fetched results for task: {task_name}")
            return eval_results

        except Exception as e:
            logging.error(
                f"Error fetching evaluation results for task {task_name}: {e}"
            )
            return {}

    def get_results(self, verbose: bool = False) -> dict:
        """
        Return the evaluation results.
        - If `verbose` is False (default), return the simplified version of the results.
        - If `verbose` is True, return the detailed backend structure.
        """
        if verbose:
            return self.results

        simplified_results = {}

        for task_name, task_results in self.results.items():
            if task_name == "job_status":
                continue

            simplified_task_results = []
            for result in task_results:
                model_name = result["config_general"]["model_name"]
                rank = (
                    result["results"]
                    .get(f"{task_name}|general|0", {})
                    .get("rank", None)
                )
                prompt = result["details"].get("full_prompt", "")

                simplified_task_results.append(
                    {"model": model_name, "rank": rank, "prompt": prompt}
                )

            simplified_results[task_name] = simplified_task_results

        return simplified_results

    def _save_updates(self) -> None:
        """
        Save the current state of the MyxBoard to the server (results, job statuses, etc.).
        """
        try:
            update_myxboard(self._sanitized_name, self.models, self.results)
            logging.info(f"MyxBoard '{self.name}' successfully updated.")
        except Exception as e:
            logging.error(f"Error updating MyxBoard '{self.name}': {e}")
            raise

    def push_to_hf(self) -> None:
        """
        Push the evaluation results to Hugging Face by creating a dataset, tagging it,
        and adding the dataset to the original collection the MyxBoard is made from.
        """
        if not self.results or not any(k for k in self.results if k != "job_status"):
            raise ValueError("No evaluation results found to push to Hugging Face.")

        try:
            dataset_dict = self._create_dataset_from_results()
            dataset_name = self.hf_collection_name.rsplit("-", 1)[0]

            self._push_dataset_to_hf(dataset_name, dataset_dict)
            self._add_dataset_to_collection(dataset_name)
            self._tag_dataset(dataset_name)
        except Exception as e:
            logging.error(f"Error pushing to Hugging Face: {e}")
            raise

    def _create_dataset_from_results(self) -> DatasetDict:
        parsed_data = []

        for task_name, task_result in self.results.items():
            if task_name == "job_status":
                continue

            parsed_data.append({"task_name": task_name, "result": task_result})

        dataset = Dataset.from_pandas(pd.DataFrame(parsed_data))
        return DatasetDict({"results": dataset})

    def _push_dataset_to_hf(self, dataset_name: str, dataset_dict: DatasetDict) -> None:
        token = HfFolder.get_token() or os.getenv("HF_TOKEN")

        if not token:
            raise EnvironmentError("No Hugging Face token found.")

        create_repo(
            repo_id=dataset_name, repo_type="dataset", private=False, exist_ok=True
        )
        dataset_dict.push_to_hub(repo_id=dataset_name, token=token)

    def _add_dataset_to_collection(self, dataset_name: str) -> None:
        collection_slug = self.hf_collection_name
        add_collection_item(
            collection_slug=collection_slug,
            item_type="dataset",
            item_id=dataset_name,
            exists_ok=True,
        )

    def _tag_dataset(self, dataset_name: str) -> None:
        card = DatasetCard.load(dataset_name)
        if "tags" not in card.data:
            card.data["tags"] = []
        if "remyx" not in card.data["tags"]:
            card.data["tags"].append("remyx")
        card = add_code_snippet_to_card(card, dataset_name)
        card.push_to_hub(dataset_name)

    def _check_job_status(self, job_name: str) -> str:
        job_status_response = get_job_status(job_name)
        return job_status_response.get("status", "unknown")

    def _sanitize_name(self, name: str) -> str:
        return urllib.parse.quote(name.replace("/", "--"), safe="")

    def _initialize_from_hf_collection(self, collection_name: str) -> List[str]:
        """Fetch models from a Hugging Face collection."""
        collection = get_collection(collection_name)
        model_repo_ids = [
            item.item_id for item in collection.items if item.item_type == "model"
        ]
        logging.info(
            f"MyxBoard initialized from Hugging Face collection: {collection_name}"
        )
        return model_repo_ids

    def _get_existing_myxboard(self) -> Optional[Dict]:
        myxboard_list = list_myxboards()
        existing_myxboard = next(
            (
                myxboard
                for myxboard in myxboard_list
                if myxboard["name"] == self._sanitized_name
            ),
            None,
        )
        return existing_myxboard

    def _populate_from_existing(self, myxboard_data: Dict) -> None:
        self.models = myxboard_data["models"]
        downloaded_results = download_myxboard(self._sanitized_name)
        self.results = downloaded_results.get("results", {})
        self.job_status = downloaded_results.get("job_status", {})

    def _store_new_myxboard(self) -> None:
        store_myxboard(self._sanitized_name, self.models, self.results)

    def delete(self) -> None:
        delete_myxboard(self._sanitized_name)
