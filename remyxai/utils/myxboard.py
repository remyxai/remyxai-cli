import math
import logging
from typing import List
from datetime import datetime, timezone


def notify_completion():
    print("Evaluations are done! You can now view the results.")

def sanitize_float(value):
    """
    Replace NaN, Infinity, and -Infinity with a valid default value (e.g., None or 0.0).
    """
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 0.0  # Replace with 0.0 if you prefer numerical default
    return value

def format_myxmatch_results_for_storage(
    eval_results: dict, task_name: str, start_time: float, end_time: float
) -> list:
    """
    Format myxmatch evaluation results into the internal schema used for storage.
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

def format_results_for_storage(
    eval_results: dict, task_name: str, start_time: float, end_time: float
) -> list:
    """
    Dispatch to the correct formatter based on the task type.
    """
    if task_name == "myxmatch":
        return format_myxmatch_results_for_storage(eval_results, task_name, start_time, end_time)
    elif task_name == "benchmark":
        return format_benchmark_results_for_storage(eval_results, task_name, start_time, end_time)
    else:
        logging.error(f"Unsupported task type: {task_name}")
        return []

def format_benchmark_results_for_storage(
    eval_results: dict, task_name: str, start_time: float, end_time: float
) -> list:
    """
    Format evaluation results for 'benchmark' tasks into the internal schema used for storage.
    """
    formatted_results = []

    # Ensure eval_results contains the "benchmark" key
    if not isinstance(eval_results, dict) or "benchmark" not in eval_results:
        logging.error(f"Invalid format for eval_results: {eval_results}")
        return formatted_results

    benchmark_results = eval_results.get("benchmark", [])
    if not isinstance(benchmark_results, list):
        logging.error(f"Unexpected format for benchmark results: {benchmark_results}")
        return formatted_results

    for model_eval in benchmark_results:
        model_name = model_eval.get("model")
        if not model_name:
            logging.warning(f"Skipping entry with missing 'model': {model_eval}")
            continue

        # Sanitize and dynamically extract metrics
        model_results = {
            key: {k: sanitize_float(v) for k, v in value.items()}
            if isinstance(value, dict)
            else sanitize_float(value)
            for key, value in model_eval.items()
            if key != "model"  # Exclude the "model" key itself
        }

        formatted_results.append(
            {
                "config_general": {
                    "model_name": model_name,
                    "start_time": start_time,
                    "end_time": end_time,
                },
                "results": model_results,
                "details": {
                    "evaluation_type": task_name,
                    "eval_tasks": eval_results.get("eval_tasks", ""),
                    "execution_time": eval_results.get("execution_time", ""),
                    "run_id": eval_results.get("run_id", ""),
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


def _validate_models(models: list, supported_models: list) -> None:
    """
    Validate that all models in `models` are in the supported models list.
    Raise a ValueError if any model is not supported.
    """
    # Ensure models are valid
    unsupported_models = [model for model in models if model not in supported_models]

    if unsupported_models:
        raise ValueError(
            f"The following models are not supported: {unsupported_models}. "
            f"Supported models: {supported_models}"
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
