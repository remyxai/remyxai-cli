import click
from remyxai.cli.deployment_actions import handle_deployment_action
from remyxai.cli.evaluation_actions import handle_model_action, handle_evaluation_action


@click.group()
def cli():
    """RemyxAI CLI."""
    pass


@cli.command()
def list_models():
    """List all available models."""
    try:
        handle_model_action({"subaction": "list"})
    except Exception as e:
        click.echo(f"Error listing models: {e}")


@cli.command()
@click.argument("model_name")
def summarize_model(model_name):
    """Summarize a model."""
    try:
        handle_model_action({"subaction": "summarize", "model_name": model_name})
    except Exception as e:
        click.echo(f"Error summarizing model: {e}")


@cli.command()
@click.argument("models", nargs=-1)
@click.argument("tasks", nargs=-1)
def evaluate_myxboard(models, tasks):
    """Evaluate the MyxBoard with the given models and tasks."""
    try:
        handle_evaluation_action({"models": models, "tasks": tasks})
    except Exception as e:
        click.echo(f"Error evaluating MyxBoard: {e}")


@cli.command()
@click.argument("model_name")
@click.argument("action")
def deploy_model(model_name, action):
    """Deploy or tear down a model."""
    try:
        handle_deployment_action({"model_name": model_name, "action": action})
    except Exception as e:
        click.echo(f"Error deploying model: {e}")


if __name__ == "__main__":
    cli()
