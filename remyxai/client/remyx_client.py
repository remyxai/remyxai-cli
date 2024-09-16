import logging
from api.evaluations import evaluate_task, MyxBoard, EvaluationTask
from api.models import list_models, get_model_summary, delete_model, download_model
from api.engines import train_classifier, train_detector, train_generator
from api.deployment import deploy_model, download_deployment_package
from api.inference import run_inference
from api.user import get_user_profile, get_user_credits

class RemyxAPI:
    def evaluate(self, myx_board: MyxBoard, tasks: list[EvaluationTask]) -> None:
        """Run evaluations for a MyxBoard on specific tasks."""
        try:
            for task in tasks:
                evaluate_task(myx_board, task)  # Use the imported function
            logging.info(f"Evaluation completed for tasks: {tasks}")
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
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

    def train_classifier(self, model_name: str, labels: list, model_selector: str, hf_dataset=None):
        """Train a classification model."""
        try:
            return train_classifier(model_name, labels, model_selector, hf_dataset)
        except Exception as e:
            logging.error(f"Error training classifier for {model_name}: {e}")
            raise

    def train_detector(self, model_name: str, labels: list, model_selector: str, hf_dataset=None):
        """Train a detection model."""
        try:
            return train_detector(model_name, labels, model_selector, hf_dataset)
        except Exception as e:
            logging.error(f"Error training detector for {model_name}: {e}")
            raise

    def train_generator(self, model_name: str, hf_dataset: str):
        """Train a generative model."""
        try:
            return train_generator(model_name, hf_dataset)
        except Exception as e:
            logging.error(f"Error training generator for {model_name}: {e}")
            raise

    def deploy_model(self, model_name: str, action='up'):
        """Deploy or tear down a model."""
        try:
            response = deploy_model(model_name, action)
            logging.info(f"Model {model_name} deployment action '{action}' succeeded.")
            return response
        except Exception as e:
            logging.error(f"Error deploying model {model_name}: {e}")
            raise

    def run_inference(self, model_name: str, prompt: str, server_url="localhost:8000", model_version="1"):
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
            logging.error("Error retrieving user profile: {e}")
            raise

    def get_user_credits(self):
        """Get the user's credits."""
        try:
            return get_user_credits()
        except Exception as e:
            logging.error("Error retrieving user credits: {e}")
            raise

