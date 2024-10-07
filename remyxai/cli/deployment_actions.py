from remyxai.client.remyx_client import RemyxAPI


def handle_deployment_action(args):
    """Handle deployment actions (up/down) for a model."""
    api = RemyxAPI()
    model_name = args["model_name"]
    action = args["action"]

    if action not in ["up", "down"]:
        raise ValueError(
            "Invalid action. Use 'up' to deploy or 'down' to tear down the model."
        )

    response = api.deploy_model(model_name, action)
    if response:
        print(f"Model {model_name} deployment {action} action succeeded.")
    else:
        print(f"Model {model_name} deployment {action} action failed.")
