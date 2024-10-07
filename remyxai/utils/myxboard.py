import logging
from typing import List
from datetime import datetime, timezone


def notify_completion():
    print("Evaluations are done! You can now view the results.")


def format_results_for_storage(
    eval_results: dict, task_name: str, start_time: float, end_time: float
) -> list:
    """
    Format evaluation results into the internal schema used for storage.
    """
    formatted_results = []

    # Ensure eval_results contains the "models" key
    if not isinstance(eval_results, dict) or "models" not in eval_results:
        logging.error(f"Invalid format for eval_results: {eval_results}")
        return formatted_results

    # Process the models and store their evaluation results
    logging.info(f"Formatting results for task: {task_name}")
    for model_info in eval_results["models"]:
        model_name = model_info.get("model")
        rank = model_info.get("rank")

        # Print intermediate model processing
        logging.info(f"Processing model {model_name} with rank {rank}")

        formatted_results.append(
            {
                "config_general": {
                    "model_name": model_name,
                    "start_time": start_time,
                    "end_time": end_time,
                },
                "results": {f"{task_name}|general|0": {"rank": rank}},
                "details": {
                    "full_prompt": eval_results.get("prompt", ""),
                    "evaluation_type": task_name,
                },
            }
        )

    logging.info(f"Formatted results: {formatted_results}")
    return formatted_results


def _reorder_models_by_results(results: dict, task_name: str) -> list:
    """
    Reorder the models list based on their ranking from the results.
    """
    task_results = results.get(task_name, {})
    ranked_models = [
        (model_info["model"], model_info["rank"])
        for model_info in task_results.get("models", [])
    ]
    ranked_models.sort(key=lambda x: x[1])  # Sort by rank
    return [model_id for model_id, _ in ranked_models]


def _validate_models(models: List[str], supported_models: List[str]) -> None:
    """
    Validate that the models in the MyxBoard are supported.
    This function does not modify the models list but raises an exception for unsupported models.
    :param models: List of model names to validate.
    :param supported_models: List of supported model names.
    """
    for model in models:
        # If HF repo, map it to the supported name for validation
        if "/" in model:
            mapped_model = model.split("/")[1]
        else:
            mapped_model = model

        if mapped_model not in supported_models:
            raise ValueError(
                f"Model '{model}' is not supported. Supported models: {supported_models}"
            )
    logging.info(f"Validated models: {models}")


def get_start_end_times(job_status_response):
    """
    Extract start and end times from job status response.
    """

    start_time = job_status_response.get("start_time", datetime.now(timezone.utc).isoformat())
    end_time = job_status_response.get("end_time", datetime.now(timezone.utc).isoformat())
    return start_time, end_time


def add_code_snippet_to_card(card, dataset_name):
    """
    Add a code snippet section to the DatasetCard content if it's not already present.

    Parameters:
    card (DatasetCard): The dataset card object.
    code_snippet (str): The code snippet to be added. Default is "<CODE HERE>".

    Returns:
    DatasetCard: The updated dataset card object.
    """
    code_snippet = f"""
## Load and parse Remyx evaluation results
The dataset contains evaluation results, with columns for task_name and result. Each row corresponds to an evaluation task result. The result field contains details such as model rankings, prompts, and any other task-specific information.

### Example:

```
python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{dataset_name}")

# Iterate over each example in the dataset
for example in dataset:
    task_name = example['task_name']
    results = example['result']  # List of result entries

    print(f"Task Name: {{task_name}}")

    for result_entry in results:
        model_name = result_entry["config_general"]["model_name"]
        full_prompt = result_entry["details"]["full_prompt"]
        rank = result_entry["results"]["myxmatch|general|0"]["rank"]

        print(f"  Model: {{model_name}}")
        print(f"  Prompt: {{full_prompt}}")
        print(f"  Rank: {{rank}}")
        print("-" * 40)
```"""
    card_content = card.content if card.content else ""
    if "## Load and parse Remyx evaluation results" not in card_content:
        card_content += f"\n\n{code_snippet}"
    card.content = card_content
    return card
