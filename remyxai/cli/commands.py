"""
RemyxAI CLI - Main command interface
All commands use "search" convention for asset discovery
"""
import click
from remyxai.cli.deployment_actions import handle_deployment_action
from remyxai.cli.evaluation_actions import handle_model_action, handle_evaluation_action
from remyxai.cli.dataset_actions import handle_dataset_action
from remyxai.cli.search_actions import (
    handle_search,
    handle_info,
    handle_list,
    handle_stats,
)

@click.group()
def cli():
    """
    RemyxAI CLI - ExperimentOps for AI Development
    
    """
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


@cli.command()
@click.argument("action")
@click.argument("dataset_name", required=False)
def dataset(action, dataset_name=None):
    """Manage datasets."""
    try:
        handle_dataset_action({"action": action, "dataset_name": dataset_name})
    except Exception as e:
        click.echo(f"Error managing dataset: {e}")


@cli.group()
def search():
    """
    Search and discover research assets (papers + Docker images).
    
    Find research papers, containerized implementations, and related assets.
    """
    pass


@search.command("query")
@click.argument("query_text")
@click.option("--max-results", "-n", default=10, help="Maximum results to return")
@click.option("--category", "-c", multiple=True, help="Filter by arXiv category")
@click.option("--docker/--no-docker", default=None, help="Filter by Docker availability")
def search_query_cmd(query_text, max_results, category, docker):
    """
    Search for research assets in the Remyx catalog.
    
    Examples:
    
      # Search all assets
      remyxai search query "data synthesis"
      
      # Search only assets with Docker images
      remyxai search query "data synthesis" --docker
      
      # Search with filters
      remyxai search query "machine learning" --docker -c cs.LG -n 5
    """
    try:
        categories = list(category) if category else None
        
        has_docker_filter = None
        if docker is True:
            has_docker_filter = True
        elif docker is False:
            has_docker_filter = False
        
        handle_search(
            query_text, 
            max_results=max_results, 
            categories=categories,
            has_docker=has_docker_filter
        )
    except Exception as e:
        click.echo(f"❌ Error searching assets: {e}", err=True)


@search.command("info")
@click.argument("arxiv_id")
@click.option("--format", "-f", default="text", type=click.Choice(["text", "json"]), 
              help="Output format")
def info_cmd(arxiv_id, format):
    """
    Get detailed information about a specific asset.
    
    Examples:
    
      remyxai search info 2010.11929v2
      remyxai search info 2010.11929v2 --format json
    """
    try:
        handle_info(arxiv_id, output_format=format)
    except Exception as e:
        click.echo(f"❌ Error getting asset info: {e}", err=True)


@search.command("list")
@click.option("--limit", "-n", default=20, help="Number of assets to list")
@click.option("--offset", "-o", default=0, help="Pagination offset")
@click.option("--category", "-c", multiple=True, help="Filter by arXiv category")
@click.option("--docker/--no-docker", default=None, help="Filter by Docker availability")
def list_cmd(limit, offset, category, docker):
    """
    List recently added research assets.
    
    Examples:
    
      remyxai search list
      remyxai search list --docker
      remyxai search list -n 10 -c cs.CV
    """
    try:
        categories = list(category) if category else None
        
        has_docker_filter = None
        if docker is True:
            has_docker_filter = True
        elif docker is False:
            has_docker_filter = False
        
        handle_list(
            limit=limit, 
            offset=offset, 
            categories=categories,
            has_docker=has_docker_filter
        )
    except Exception as e:
        click.echo(f"❌ Error listing assets: {e}", err=True)


@search.command("stats")
def stats_cmd():
    """
    Show statistics about available research assets.
    
    Example:
      remyxai search stats
    """
    try:
        handle_stats()
    except Exception as e:
        click.echo(f"❌ Error getting stats: {e}", err=True)


if __name__ == "__main__":
    cli()
