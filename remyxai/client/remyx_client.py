import time
import logging
import threading
from datetime import datetime
from typing import List, Optional
from remyxai.api.models import (
    list_models,
    get_model_summary,
    delete_model,
    download_model,
)
from remyxai.api.tasks import run_myxmatch, get_job_status
from remyxai.api.deployment import deploy_model, download_deployment_package
from remyxai.api.inference import run_inference
from remyxai.api.user import get_user_profile, get_user_credits
from remyxai.api.evaluations import (
    list_evaluations,
    download_evaluation,
    delete_evaluation,
    EvaluationTask,
)
from remyxai.utils.myxboard import format_results_for_storage, notify_completion


class RemyxAPI:
    def evaluate(
        self,
        myx_board,
        tasks: List[EvaluationTask],
        prompt: Optional[str] = None,
        on_complete: Optional[callable] = notify_completion,
    ) -> None:
        """
        Run evaluations for a MyxBoard on specific tasks and start asynchronous polling.
        :param myx_board: The MyxBoard to evaluate.
        :param tasks: List of tasks to evaluate.
        :param prompt: Optional prompt for tasks like MYXMATCH that require it.
        :param on_complete: Callback function to call once evaluations are complete.
        """
        try:
            if "job_status" not in myx_board.results:
                myx_board.results["job_status"] = {}

            for task in tasks:
                task_name = task.value
                if task == EvaluationTask.MYXMATCH:
                    if not prompt:
                        raise ValueError(f"Task '{task_name}' requires a prompt.")

                    job_response = run_myxmatch(
                        myx_board.name, prompt, myx_board.models
                    )

                    job_name = job_response.get("job_name")
                    start_time = time.time()

                    myx_board.results["job_status"][task_name] = {
                        "job_name": job_name,
                        "status": "pending",
                        "start_time": start_time,
                    }

            myx_board._save_updates()
            print("Starting evaluation...")
            polling_thread = threading.Thread(
                target=self._poll_for_completion, args=(myx_board, on_complete)
            )
            polling_thread.start()

        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            raise

    def _poll_for_completion(self, myx_board, on_complete: Optional[callable]) -> None:
        """
        Poll in the background to check job completion status, and fetch results when done.
        :param myx_board: The MyxBoard instance to poll for.
        :param on_complete: Function to call when polling completes.
        """
        while not myx_board.poll_and_store_results():
            logging.info("Jobs still running, polling again in 30 seconds.")
            time.sleep(120)

        logging.info("All jobs completed, results have been stored.")
        if on_complete:
            on_complete()

    def list_evaluations(self):
        """List available evaluations."""
        try:
            return list_evaluations()
        except Exception as e:
            logging.error(f"Error listing evaluations: {e}")
            raise

    def download_evaluation(self, eval_type: str, eval_name: str):
        """Download results of a specific evaluation."""
        try:
            return download_evaluation(eval_type, eval_name)
        except Exception as e:
            logging.error(f"Error downloading evaluation for {eval_name}: {e}")
            raise

    def delete_evaluation(self, eval_type: str, eval_name: str):
        """Delete a specific evaluation."""
        try:
            return delete_evaluation(eval_type, eval_name)
        except Exception as e:
            logging.error(f"Error deleting evaluation {eval_name}: {e}")
            raise

    def list_models(self):
        """List available models."""
        try:
            return list_models()
        except Exception as e:
            logging.error(f"Error listing models: {e}")
            raise

    def get_model_summary(self, model_name: str):
        """Get a summary of a specific model."""
        try:
            return get_model_summary(model_name)
        except Exception as e:
            logging.error(f"Error getting model summary for {model_name}: {e}")
            raise

    def delete_model(self, model_name: str):
        """Delete a model."""
        try:
            return delete_model(model_name)
        except Exception as e:
            logging.error(f"Error deleting model {model_name}: {e}")
            raise

    def download_model(self, model_name: str, model_format: str):
        """Download a specific model."""
        try:
            return download_model(model_name, model_format)
        except Exception as e:
            logging.error(f"Error downloading model {model_name}: {e}")
            raise

    def deploy_model(self, model_name: str, action="up"):
        """Deploy or tear down a model."""
        try:
            response = deploy_model(model_name, action)
            logging.info(f"Model {model_name} deployment action '{action}' succeeded.")
            return response
        except Exception as e:
            logging.error(f"Error deploying model {model_name}: {e}")
            raise

    def run_inference(
        self,
        model_name: str,
        prompt: str,
        server_url="localhost:8000",
        model_version="1",
    ):
        """Run inference on a model."""
        try:
            return run_inference(model_name, prompt, server_url, model_version)
        except Exception as e:
            logging.error(f"Error running inference for {model_name}: {e}")
            raise

    def get_user_profile(self):
        """Get the user profile."""
        try:
            return get_user_profile()
        except Exception as e:
            logging.error(f"Error retrieving user profile: {e}")
            raise

    def get_user_credits(self):
        """Get the user's credits."""
        try:
            return get_user_credits()
        except Exception as e:
            logging.error(f"Error retrieving user credits: {e}")
            raise
