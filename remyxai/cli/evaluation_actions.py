from remyxai.client.remyx_client import RemyxAPI
from remyxai.client.myxboard import MyxBoard
from remyxai.api.evaluations import EvaluationTask


def handle_model_action(args):
    """
    Handle model-related actions like listing models, summarizing models, deleting, or downloading models.
    Args:
        args (dict): Dictionary containing the subaction and related parameters.
    """
    api = RemyxAPI()

    if args["subaction"] == "list":
        models = api.list_models()
        print(f"Available models: {models}")

    elif args["subaction"] == "summarize":
        model_name = args["model_name"]
        summary = api.get_model_summary(model_name)
        print(f"Summary for model {model_name}: {summary}")

    elif args["subaction"] == "delete":
        model_name = args["model_name"]
        response = api.delete_model(model_name)
        print(f"Deleted model {model_name}: {response}")

    elif args["subaction"] == "download":
        model_name = args["model_name"]
        model_format = args["model_format"]
        response = api.download_model(model_name, model_format)
        print(f"Downloaded model {model_name} in format {model_format}: {response}")

    else:
        raise ValueError(f"Unknown model subaction: {args['subaction']}")


def handle_evaluation_action(args):
    """
    Handle evaluation actions using the MyxBoard and RemyxAPI.
    Args:
        args (dict): Dictionary containing the models to evaluate and tasks to perform.
    """
    api = RemyxAPI()

    # Initialize the MyxBoard with provided models
    model_ids = args["models"]
    myx_board = MyxBoard(model_repo_ids=model_ids)

    # Map tasks to EvaluationTask enum
    try:
        tasks = [EvaluationTask[task.upper()] for task in args["tasks"]]
    except KeyError as e:
        raise ValueError(f"Invalid task specified: {e}")

    # Perform evaluation via RemyxAPI
    api.evaluate(myx_board, tasks)

    # Get and display results
    results = myx_board.get_results()
    print("Evaluation Results:")
    for model, model_results in results.items():
        print(f"Model: {model}")
        for task, result in model_results.items():
            print(f"  Task: {task}, Result: {result}")


def handle_task_mapping(task_name):
    """
    Helper function to map task names to the EvaluationTask enum.
    Args:
        task_name (str): The name of the task to map.
    Returns:
        EvaluationTask: Mapped task enum.
    Raises:
        ValueError: If the task name does not match a valid task.
    """
    try:
        return EvaluationTask[task_name.upper()]
    except KeyError:
        raise ValueError(f"Invalid task name: {task_name}")
