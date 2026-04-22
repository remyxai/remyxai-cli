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
    handle_interests_regenerate,
    handle_interests_list_repos,
)
from remyxai.cli.experiment_actions import (
    handle_experiments_list,
    handle_experiments_get,
    handle_experiments_validate,
    handle_experiments_validate_status,
)
from remyxai.cli.project_actions import (
    handle_projects_list,
    handle_projects_get,
    handle_projects_configure_eval,
    handle_projects_set_policy,
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


# =============================================================================
# papers — daily recommendations
# =============================================================================

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
@click.option("--full", is_flag=True, default=False,
              help="Show full reasoning text without truncation.")
def papers_digest(limit, period, output_format, full):
    """
    Show recommendations grouped by Research Interest.

    Examples:

      remyxai papers digest
      remyxai papers digest --period week --limit 3
      remyxai papers digest --full
      remyxai papers digest --format json | jq .interests[0].recommendations
    """
    handle_papers_digest(limit=limit, period=period, output_format=output_format, full=full)


@papers.command("list")
@click.option("--interest", "-i", default=None,
              help="Filter by Research Interest name or UUID.")
@click.option("--limit", "-n", default=20, show_default=True,
              help="Max results (1-50).")
@click.option("--period", "-p", default="all", show_default=True,
              type=click.Choice(["today", "week", "all"]))
@click.option("--source-type", "-s", default=None,
              help="Filter by source type: arxiv_paper, github_repo, etc.")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
@click.option("--full", is_flag=True, default=False,
              help="Show full reasoning text without truncation.")
def papers_list(interest, limit, period, source_type, output_format, full):
    """
    List recommendations (flat view, optionally filtered).

    Examples:

      remyxai papers list --period today
      remyxai papers list --interest "Reinforcement Learning" --period today
      remyxai papers list --source-type arxiv_paper --period week
      remyxai papers list --format json
      remyxai papers list --full
    """
    handle_papers_list(
        interest_id=interest,
        limit=limit,
        period=period,
        source_type=source_type,
        output_format=output_format,
        full=full,
    )


@papers.command("refresh")
@click.option("--interest", "-i", default=None,
              help="Refresh one interest by name or UUID (omit to refresh all active).")
@click.option("--num-results", "-n", default=None, type=int,
              help="Items per interest (defaults to interest.daily_count).")
@click.option("--wait", "-w", is_flag=True, default=False,
              help="Block until all tasks complete.")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def papers_refresh(interest, num_results, wait, output_format):
    """
    Trigger a fresh Gemini re-ranking run for your Research Interests.

    Cold-start (first run, empty pool) takes 40-120s. Subsequent runs
    are served from the pre-ranked pool in ~200ms.

    Examples:

      remyxai papers refresh --wait
      remyxai papers refresh --interest "Reinforcement Learning" --wait
      remyxai papers refresh --interest "Reinforcement Learning" --num-results 5 --wait
    """
    handle_papers_refresh(
        interest_id=interest,
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


# =============================================================================
# interests — Research Interest profile management
# =============================================================================

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
@click.option("--interest", "-i", required=True,
              help="Interest name or UUID.")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def interests_get(interest, output_format):
    """
    Show a single Research Interest.

    Example:

      remyxai interests get --interest "Reinforcement Learning"
    """
    handle_interests_get(interest_id=interest, output_format=output_format)


@interests.command("create")
@click.option("--name", "-n", default=None,
              help="Short label, e.g. 'RAG & Retrieval'.")
@click.option("--context", "-c", default=None,
              help="Natural language description, or a HuggingFace/GitHub URL.")
@click.option("--repo", default=None,
              help="GitHub repo URL to generate the interest profile from "
                   "(REMYX-28). Server runs Gemini analysis and the markdown "
                   "report becomes the default context.")
@click.option("--daily-count", "-d", default=2, show_default=True,
              help="Recommendations per day (1-10).")
@click.option("--inactive", is_flag=True, default=False,
              help="Create as inactive (excluded from daily digest until toggled).")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def interests_create(name, context, repo, daily_count, inactive, output_format):
    """
    Create a new Research Interest profile.

    Prompts interactively if --name or --context are not supplied.

    Examples:

      remyxai interests create

      remyxai interests create \\
        --name "LLM Efficiency" \\
        --context "Quantization, speculative decoding, KV cache compression" \\
        --daily-count 3

      # Generate from a repo (public or via connected GitHub integration)
      remyxai interests create --repo https://github.com/owner/repo
    """
    handle_interests_create(
        name=name,
        context=context,
        daily_count=daily_count,
        inactive=inactive,
        output_format=output_format,
        repo=repo,
    )


@interests.command("regenerate")
@click.option("--interest", "-i", required=True,
              help="Interest name or UUID.")
@click.option("--repo-url", default=None,
              help="Override the stored source_repo_url (optional).")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def interests_regenerate(interest, repo_url, output_format):
    """
    Re-run repo analysis for an existing repo-sourced Research Interest.

    Blocks on the async task and applies the refreshed payload to the
    interest. Use after a significant change in the source repo.

    Example:

      remyxai interests regenerate --interest "remyxai/remyx"
    """
    handle_interests_regenerate(
        interest_id=interest,
        repo_url=repo_url,
        output_format=output_format,
    )


@interests.command("list-repos")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def interests_list_repos(output_format):
    """
    List GitHub repos you can pick from for repo-sourced interests.

    Requires a connected GitHub integration. When not connected, the
    command prints a hint rather than erroring.

    Example:

      remyxai interests list-repos
    """
    handle_interests_list_repos(output_format=output_format)


@interests.command("update")
@click.option("--interest", "-i", required=True,
              help="Interest name or UUID.")
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
def interests_update(interest, name, context, daily_count, is_active, output_format):
    """
    Update a Research Interest.

    Changing --context or --name clears the cached recommendation pool;
    run `remyxai papers refresh` afterwards to rebuild it.

    Examples:

      remyxai interests update --interest "Reinforcement Learning" --daily-count 5
      remyxai interests update --interest "Reinforcement Learning" --context "New research focus..."
      remyxai interests update --interest "Reinforcement Learning" --deactivate
    """
    handle_interests_update(
        interest_id=interest,
        name=name,
        context=context,
        daily_count=daily_count,
        is_active=is_active,
        output_format=output_format,
    )


@interests.command("delete")
@click.option("--interest", "-i", required=True,
              help="Interest name or UUID.")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Skip confirmation prompt.")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def interests_delete(interest, yes, output_format):
    """
    Delete a Research Interest and all its recommendations.

    Examples:

      remyxai interests delete --interest "Reinforcement Learning"
      remyxai interests delete --interest "Reinforcement Learning" --yes
    """
    handle_interests_delete(
        interest_id=interest,
        yes=yes,
        output_format=output_format,
    )


@interests.command("toggle")
@click.option("--interest", "-i", required=True,
              help="Interest name or UUID.")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def interests_toggle(interest, output_format):
    """
    Toggle a Research Interest between active and inactive.

    Example:

      remyxai interests toggle --interest "Reinforcement Learning"
    """
    handle_interests_toggle(
        interest_id=interest,
        output_format=output_format,
    )


# =============================================================================
# experiments — experiment board + validation (REMYX-19)
# =============================================================================

@cli.group()
def experiments():
    """
    Browse experiments and launch validation runs.

    The `validate` subcommand wraps the REMYX-24 eval-env pipeline to run
    baseline-vs-feature evaluations for an experiment against a locked
    EvalTemplate.
    """
    pass


@experiments.command("list")
@click.option("--project-id", "-p", default=None,
              help="Filter to a specific project UUID.")
@click.option("--status", "-s", default=None,
              help="Filter by status (backlog, implementing, validating, ...).")
@click.option("--initiative", "-i", default=None,
              help="Filter by initiative name.")
@click.option("--limit", "-n", default=20, show_default=True,
              help="Max results (capped server-side at 100).")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def experiments_list(project_id, status, initiative, limit, output_format):
    """
    List experiments with optional filters.

    Examples:

      remyxai experiments list
      remyxai experiments list --status validating
      remyxai experiments list --initiative "Customer Support AI"
    """
    handle_experiments_list(
        project_id=project_id,
        status=status,
        initiative=initiative,
        limit=limit,
        output_format=output_format,
    )


@experiments.command("get")
@click.argument("experiment_id")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def experiments_get(experiment_id, output_format):
    """
    Show a single experiment.

    Example:

      remyxai experiments get 7a9c...e1
    """
    handle_experiments_get(
        experiment_id=experiment_id,
        output_format=output_format,
    )


@experiments.command("validate")
@click.argument("experiment_id")
@click.option("--template-id", required=True,
              help="UUID of a LOCKED EvalTemplate.")
@click.option("--variant", "variants", multiple=True, required=True,
              help="Variant spec: 'name=commit_sha' or 'name=ref:commit_sha'. "
                   "Pass multiple times (at least one — typically baseline + feature).")
@click.option("--seeds", default=1, show_default=True, type=int,
              help="Per-variant seed count.")
@click.option("--github-url", default=None,
              help="Target repo URL. Falls back to the experiment's stored "
                   "validation_config.github_url / repo if omitted.")
@click.option("--pr-number", default=None, type=int,
              help="PR number for lineage (optional).")
@click.option("--pr-url", default=None,
              help="PR URL for lineage (optional).")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def experiments_validate(
    experiment_id, template_id, variants, seeds,
    github_url, pr_number, pr_url, output_format,
):
    """
    Launch a validation run for an experiment.

    Wraps POST /api/v1.0/eval-env/runs. The engine builds per-variant
    Docker images, submits Modal Sandboxes, collects webhook results, and
    computes a Pass/Warn/Fail verdict against the locked template's
    decision criteria.

    Example:

      remyxai experiments validate 7a9c...e1 \\
        --template-id a1b2c3...f0 \\
        --variant baseline=9f8a1d2 \\
        --variant feature=c4e5b7a
    """
    handle_experiments_validate(
        experiment_id=experiment_id,
        template_id=template_id,
        github_url=github_url,
        variants=list(variants),
        seeds=seeds,
        pr_number=pr_number,
        pr_url=pr_url,
        output_format=output_format,
    )


@experiments.command("validate-status")
@click.argument("run_id")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def experiments_validate_status(run_id, output_format):
    """
    Poll a validation run.

    Example:

      remyxai experiments validate-status 3b4d...77
    """
    handle_experiments_validate_status(
        run_id=run_id,
        output_format=output_format,
    )


# =============================================================================
# projects — project configuration (REMYX-19)
# =============================================================================

@cli.group()
def projects():
    """
    List, inspect, and configure Remyx projects.

    Projects store the eval templates and decision policies that drive
    validation and automated disposition for experiments.
    """
    pass


@projects.command("list")
@click.option("--team-id", default=None,
              help="Required when using a service token; otherwise "
                   "resolved from team membership.")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def projects_list(team_id, output_format):
    """
    List non-archived projects for the caller's team.

    Example:

      remyxai projects list
    """
    handle_projects_list(team_id=team_id, output_format=output_format)


@projects.command("get")
@click.argument("project_id")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def projects_get(project_id, output_format):
    """
    Show a project's config (eval templates, decision policies, etc.).

    Example:

      remyxai projects get a1b2c3...
    """
    handle_projects_get(project_id=project_id, output_format=output_format)


@projects.command("configure-eval")
@click.argument("template_name")
@click.option("--project-id", "-p", required=True,
              help="UUID of the target project.")
@click.option("--template", "-t", "template_file", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Path to a JSON file with the template definition.")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def projects_configure_eval(template_name, project_id, template_file, output_format):
    """
    Create or replace an eval template on a project.

    The template body (JSON file) should include provider, image,
    entry_point, dataset_ref, metrics_map, compute, etc.

    Example:

      remyxai projects configure-eval default \\
        --project-id a1b2c3... \\
        --template ./eval-templates/default.json
    """
    handle_projects_configure_eval(
        project_id=project_id,
        template_name=template_name,
        template_file=template_file,
        output_format=output_format,
    )


@projects.command("set-policy")
@click.argument("policy_name")
@click.option("--project-id", "-p", required=True,
              help="UUID of the target project.")
@click.option("--policy", "policy_file", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Path to a JSON file with the decision policy.")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def projects_set_policy(policy_name, project_id, policy_file, output_format):
    """
    Create or replace a decision policy on a project.

    The policy body (JSON file) should contain rules keyed by disposition
    (ship, reject, iterate) combining predicates over metric deltas,
    confidence bands, and sample sizes. See REMYX-14.

    Example:

      remyxai projects set-policy default \\
        --project-id a1b2c3... \\
        --policy ./policies/default.json
    """
    handle_projects_set_policy(
        project_id=project_id,
        policy_name=policy_name,
        policy_file=policy_file,
        output_format=output_format,
    )


if __name__ == "__main__":
    cli()
