"""
Action handlers for model-related CLI commands (list, summarize).

Note: legacy MyxBoard-based evaluation (`evaluate_myxboard`,
`handle_evaluation_action`, `handle_task_mapping`) was removed when
the MyxBoard service was retired. The remaining handlers wrap the
`remyxai.api.models` functions directly — the older RemyxAPI class
wrapper was deleted in the same cleanup since it added no value over
direct API calls.
"""
from remyxai.api.models import (
    list_models,
    get_model_summary,
    delete_model,
    download_model,
)


def handle_model_action(args):
    """Handle model-related actions: list, summarize, delete, download."""
    subaction = args["subaction"]

    if subaction == "list":
        models = list_models()
        print(f"Available models: {models}")

    elif subaction == "summarize":
        model_name = args["model_name"]
        summary = get_model_summary(model_name)
        print(f"Summary for model {model_name}: {summary}")

    elif subaction == "delete":
        model_name = args["model_name"]
        response = delete_model(model_name)
        print(f"Deleted model {model_name}: {response}")

    elif subaction == "download":
        model_name = args["model_name"]
        model_format = args["model_format"]
        response = download_model(model_name, model_format)
        print(f"Downloaded model {model_name} in format {model_format}: {response}")

    else:
        raise ValueError(f"Unknown model subaction: {subaction}")
