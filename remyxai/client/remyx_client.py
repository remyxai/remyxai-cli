import logging
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
from remyxai.api.evaluations import list_evaluations, download_evaluation, delete_evaluation, EvaluationTask


class RemyxAPI:
    def evaluate(
        self,
        myx_board,
        tasks: List[EvaluationTask],
        prompt: Optional[str] = None,
    ) -> None:
        """
        Run evaluations for a MyxBoard on specific tasks.

        :param myx_board: The MyxBoard to evaluate.
        :param tasks: List of tasks to evaluate the models on.
        :param prompt: Optional prompt for tasks like MYXMATCH that require it.
        """
        try:
            # Ensure "job_status" is initialized in the MyxBoard results
            if "job_status" not in myx_board.results:
                myx_board.results["job_status"] = {}

            for task in tasks:
                task_name = task.value

                if task == EvaluationTask.MYXMATCH:
                    if not prompt:
                        raise ValueError(f"Task '{task_name}' requires a prompt.")

                    # Run MYXMATCH task using the provided prompt
                    job_response = run_myxmatch(
                        myx_board.name, prompt, myx_board.models
                    )

                    # Store the job name in the job_status dictionary
                    job_name = job_response.get("job_name")
                    myx_board.results["job_status"][task_name] = {
                        "job_name": job_name,
                        "status": "pending"  # Initial status is 'pending'
                    }

                else:
                    # Handle other tasks (future extensions)
                    pass

            logging.info(
                f"Evaluations submitted for MyxBoard '{myx_board.name}' with tasks: {tasks}"
            )

        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            raise

    def check_job_status(self, myx_board) -> None:
        """
        Check the status of all jobs associated with the MyxBoard.

        :param myx_board: The MyxBoard for which to check job statuses.
        """
        try:
            for task, job_info in myx_board.results.get("job_status", {}).items():
                job_name = job_info.get("job_name")
                status_response = get_job_status(job_name)  # Use job_name to query status
                status = status_response.get("status")

                # Update the status in the results
                myx_board.results["job_status"][task]["status"] = status

                logging.info(f"Task '{task}' job status: {status}")

                if status == "completed":
                    logging.info(f"Task '{task}' completed.")
                else:
                    logging.info(f"Task '{task}' still running with status: {status}")

        except Exception as e:
            logging.error(f"Error checking job status: {e}")
            raise

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

