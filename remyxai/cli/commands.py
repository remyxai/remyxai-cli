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
from remyxai.cli.recommendation_actions import (
    handle_papers_digest,
    handle_papers_list,
    handle_papers_refresh,
    handle_refresh_status,
)
from remyxai.cli.interest_actions import (
    handle_interests_list,
    handle_interests_get,
    handle_interests_create,
    handle_interests_update,
    handle_interests_delete,
    handle_interests_toggle,
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

# ═════════════════════════════════════════════════════════════════════════════
# papers — daily recommendations
# ═════════════════════════════════════════════════════════════════════════════
 
@cli.group()
def papers():
    """
    Daily recommendations from Remyx AI GitRank.
 
    Gemini-ranked arXiv papers (and soon GitHub repos) matched to your
    Research Interests.
    """
    pass
 
 
@papers.command("digest")
@click.option("--limit", "-n", default=5, show_default=True,
              help="Max items per Research Interest.")
@click.option("--period", "-p", default="today", show_default=True,
              type=click.Choice(["today", "week", "all"]))
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def papers_digest(limit, period, output_format):
    """
    Show recommendations grouped by Research Interest.
 
    Examples:
 
      remyxai papers digest
      remyxai papers digest --period week --limit 3
      remyxai papers digest --format json | jq .interests[0].recommendations
    """
    handle_papers_digest(limit=limit, period=period, output_format=output_format)
 
 
@papers.command("list")
@click.option("--interest-id", "-i", default=None,
              help="Filter by Research Interest UUID.")
@click.option("--limit", "-n", default=20, show_default=True,
              help="Max results (1-50).")
@click.option("--period", "-p", default="all", show_default=True,
              type=click.Choice(["today", "week", "all"]))
@click.option("--source-type", "-s", default=None,
              help="Filter by source type: arxiv_paper, github_repo, etc.")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def papers_list(interest_id, limit, period, source_type, output_format):
    """
    List recommendations (flat view, optionally filtered).
 
    Examples:
 
      remyxai papers list --period today
      remyxai papers list --source-type arxiv_paper --period week
      remyxai papers list --interest-id <uuid> --format json
    """
    handle_papers_list(
        interest_id=interest_id,
        limit=limit,
        period=period,
        source_type=source_type,
        output_format=output_format,
    )
 
 
@papers.command("refresh")
@click.option("--interest-id", "-i", default=None,
              help="Refresh one interest (omit to refresh all active).")
@click.option("--num-results", "-n", default=None, type=int,
              help="Items per interest (defaults to interest.daily_count).")
@click.option("--wait", "-w", is_flag=True, default=False,
              help="Block until all tasks complete.")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def papers_refresh(interest_id, num_results, wait, output_format):
    """
    Trigger a fresh Gemini re-ranking run for your Research Interests.
 
    Cold-start (first run, empty pool) takes 40-120s. Subsequent runs
    are served from the pre-ranked pool in ~200ms.
 
    Examples:
 
      remyxai papers refresh --wait
      remyxai papers refresh --interest-id <uuid> --num-results 5 --wait
    """
    handle_papers_refresh(
        interest_id=interest_id,
        num_results=num_results,
        wait=wait,
        output_format=output_format,
    )
 
 
@papers.command("refresh-status")
@click.argument("task_id")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def papers_refresh_status(task_id, output_format):
    """
    Poll the status of a refresh task.
 
    Example:
 
      remyxai papers refresh-status a1b2c3d4-0000-0000-0000-000000000000
    """
    handle_refresh_status(task_id=task_id, output_format=output_format)
 
 
# ═════════════════════════════════════════════════════════════════════════════
# interests — Research Interest profile management
# ═════════════════════════════════════════════════════════════════════════════
 
@cli.group()
def interests():
    """
    Manage Research Interest profiles.
 
    Each profile has a name, a natural-language context describing what to
    track, and a daily recommendation count. The recommendation pipeline
    uses the context to match papers, GitHub repos, and future sources —
    no changes needed here when new sources are added to GitRank.
    """
    pass
 
 
@interests.command("list")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def interests_list(output_format):
    """
    List all Research Interest profiles.
 
    Examples:
 
      remyxai interests list
      remyxai interests list --format json
    """
    handle_interests_list(output_format=output_format)
 
 
@interests.command("get")
@click.argument("interest_id")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def interests_get(interest_id, output_format):
    """
    Show a single Research Interest by ID.
 
    Example:
 
      remyxai interests get <uuid>
    """
    handle_interests_get(interest_id=interest_id, output_format=output_format)
 
 
@interests.command("create")
@click.option("--name", "-n", default=None,
              help="Short label, e.g. 'RAG & Retrieval'.")
@click.option("--context", "-c", default=None,
              help="Natural language description, or a HuggingFace/GitHub URL.")
@click.option("--daily-count", "-d", default=2, show_default=True,
              help="Recommendations per day (1-10).")
@click.option("--inactive", is_flag=True, default=False,
              help="Create as inactive (excluded from daily digest until toggled).")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def interests_create(name, context, daily_count, inactive, output_format):
    """
    Create a new Research Interest profile.
 
    Prompts interactively if --name or --context are not supplied.
 
    Examples:
 
      remyxai interests create
 
      remyxai interests create \\
        --name "LLM Efficiency" \\
        --context "Quantization, speculative decoding, KV cache compression" \\
        --daily-count 3
    """
    handle_interests_create(
        name=name,
        context=context,
        daily_count=daily_count,
        inactive=inactive,
        output_format=output_format,
    )
 
 
@interests.command("update")
@click.argument("interest_id")
@click.option("--name", "-n", default=None, help="New name.")
@click.option("--context", "-c", default=None,
              help="New context — invalidates the recommendation pool.")
@click.option("--daily-count", "-d", default=None, type=int,
              help="New daily recommendation count.")
@click.option("--activate", "is_active", flag_value=True, default=None,
              help="Set active.")
@click.option("--deactivate", "is_active", flag_value=False,
              help="Set inactive.")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def interests_update(interest_id, name, context, daily_count, is_active, output_format):
    """
    Update a Research Interest.
 
    Changing --context or --name clears the cached recommendation pool;
    run `remyxai papers refresh` afterwards to rebuild it.
 
    Examples:
 
      remyxai interests update <uuid> --daily-count 5
      remyxai interests update <uuid> --context "New research focus..."
      remyxai interests update <uuid> --deactivate
    """
    handle_interests_update(
        interest_id=interest_id,
        name=name,
        context=context,
        daily_count=daily_count,
        is_active=is_active,
        output_format=output_format,
    )
 
 
@interests.command("delete")
@click.argument("interest_id")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Skip confirmation prompt.")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def interests_delete(interest_id, yes, output_format):
    """
    Delete a Research Interest and all its recommendations.
 
    Examples:
 
      remyxai interests delete <uuid>
      remyxai interests delete <uuid> --yes
    """
    handle_interests_delete(
        interest_id=interest_id,
        yes=yes,
        output_format=output_format,
    )
 
 
@interests.command("toggle")
@click.argument("interest_id")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def interests_toggle(interest_id, output_format):
    """
    Toggle a Research Interest between active and inactive.
 
    Example:
 
      remyxai interests toggle <uuid>
    """
    handle_interests_toggle(
        interest_id=interest_id,
        output_format=output_format,
    )


if __name__ == "__main__":
    cli()
