# remyxai/cli/commands.py
import click
from remyxai.cli.deployment_actions import handle_deployment_action
from remyxai.cli.evaluation_actions import handle_model_action, handle_evaluation_action
from remyxai.cli.dataset_actions import handle_dataset_action
from remyxai.cli.paper_actions import (
    handle_paper_search,
    handle_paper_info,
    handle_paper_list,
    handle_paper_stats,
)

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
def papers():
    """Manage containerized research papers for AG2 integration."""
    pass


@papers.command("search")
@click.argument("query")
@click.option("--max-results", "-n", default=10, help="Maximum results to return")
@click.option("--category", "-c", multiple=True, help="Filter by arXiv category (can specify multiple)")
@click.option("--docker/--no-docker", default=None, help="Filter by Docker availability")
@click.option("--all", "search_all", is_flag=True, help="Search all papers (default)")
def search_papers_cmd(query, max_results, category, docker, search_all):
    """
    Search for research papers in the Remyx catalog.
    
    By default, searches ALL papers. Use --docker or --no-docker to filter.
    
    Examples:
    
      # Search all papers
      remyxai papers search "vision transformers"
      
      # Search only papers with Docker images
      remyxai papers search "vision transformers" --docker
      
      # Search only papers without Docker images
      remyxai papers search "vision transformers" --no-docker
      
      # Search with multiple filters
      remyxai papers search "machine learning" --docker -c cs.LG -n 5
    """
    try:
        categories = list(category) if category else None
        
        has_docker_filter = None
        if docker is True:
            has_docker_filter = True
        elif docker is False:
            has_docker_filter = False
        # If docker is None (not specified), has_docker_filter stays None (search all)
        
        handle_paper_search(
            query, 
            max_results=max_results, 
            categories=categories,
            has_docker=has_docker_filter
        )
    except Exception as e:
        click.echo(f"❌ Error searching papers: {e}", err=True)

@papers.command("info")
@click.argument("arxiv_id")
@click.option("--format", "-f", default="text", type=click.Choice(["text", "json"]), 
              help="Output format (text or json)")
def paper_info_cmd(arxiv_id, format):
    """
    Get detailed information about a specific paper.
    
    Examples:
    
      remyxai papers info 2010.11929v2
      
      remyxai papers info 2010.11929v2 --format json
    """
    try:
        handle_paper_info(arxiv_id, output_format=format)
    except Exception as e:
        click.echo(f"❌ Error getting paper info: {e}", err=True)


@papers.command("list")
@click.option("--limit", "-n", default=20, help="Number of papers to list")
@click.option("--offset", "-o", default=0, help="Pagination offset")
@click.option("--category", "-c", multiple=True, help="Filter by arXiv category")
def list_papers_cmd(limit, offset, category):
    """
    List recently containerized papers.
    
    Examples:
    
      remyxai papers list
      
      remyxai papers list -n 10
      
      remyxai papers list -c cs.CV --offset 20
    """
    try:
        categories = list(category) if category else None
        handle_paper_list(limit=limit, offset=offset, categories=categories)
    except Exception as e:
        click.echo(f"❌ Error listing papers: {e}", err=True)


@papers.command("stats")
def stats_cmd():
    """
    Show statistics about available papers.
    
    Example:
    
      remyxai papers stats
    """
    try:
        handle_paper_stats()
    except Exception as e:
        click.echo(f"❌ Error getting stats: {e}", err=True)


if __name__ == "__main__":
    cli()
