"""
RemyxAI CLI — entry point and command-group registrations.
"""
import click
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
    handle_interests_create_from_repo,
    handle_interests_create_from_project,
    handle_interests_update,
    handle_interests_delete,
    handle_interests_toggle,
)
from remyxai.cli.outrider_actions import (
    _parse_bulk_repos_tsv,
    _run_bulk,
    handle_outrider_init,
    handle_outrider_trigger,
)
from remyxai.cli.outrider_local import handle_outrider_setup_local


@click.group()
def cli():
    """RemyxAI CLI — ExperimentOps for AI development."""
    pass


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
@click.option("--daily-count", "-d", default=2, show_default=True,
              help="Recommendations per day (1-10).")
@click.option("--inactive", is_flag=True, default=False,
              help="Create as inactive (excluded from daily digest until toggled).")
@click.option("--refresh/--no-refresh", default=True, show_default=True,
              help="Kick off a recommendations refresh after creation.")
@click.option("--wait", "-w", is_flag=True, default=False,
              help="Block until the recommendations refresh completes.")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def interests_create(name, context, daily_count, inactive, refresh, wait,
                     output_format):
    """
    Create a new Research Interest profile.

    Prompts interactively if --name or --context are not supplied. By
    default it also kicks off a recommendations refresh so the interest is
    populated; pass --no-refresh to skip, or --wait to block until ready.

    Examples:

      remyxai interests create

      remyxai interests create \\
        --name "LLM Efficiency" \\
        --context "Quantization, speculative decoding, KV cache compression" \\
        --daily-count 3 --wait
    """
    handle_interests_create(
        name=name,
        context=context,
        daily_count=daily_count,
        inactive=inactive,
        refresh=refresh,
        wait=wait,
        output_format=output_format,
    )


@interests.command("from-repo")
@click.argument("repo_url")
@click.option("--name", "-n", default=None,
              help="Interest name (defaults to the repo name).")
@click.option("--daily-count", "-d", default=2, show_default=True,
              help="Recommendations per day (1-10).")
@click.option("--inactive", is_flag=True, default=False,
              help="Create as inactive (excluded from daily digest until toggled).")
@click.option("--automate", default="none", show_default=True,
              type=click.Choice(["none", "review", "auto"]),
              help="Automate paper PRs on this repo: 'none' (Not now), "
                   "'review' (open a setup PR to review), or "
                   "'auto' (set it up + merge + first run).")
@click.option("--timeout", default=300, show_default=True,
              help="Seconds to wait for repo analysis before giving up.")
@click.option("--refresh/--no-refresh", default=True, show_default=True,
              help="Kick off a recommendations refresh after creation.")
@click.option("--wait", "-w", is_flag=True, default=False,
              help="Block until the recommendations refresh completes.")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def interests_from_repo(repo_url, name, daily_count, inactive, automate,
                        timeout, refresh, wait, output_format):
    """
    Create a Research Interest from a GitHub repo.

    Analyzes REPO_URL to generate a profile, then creates the interest
    linked to the repo — which dispatches experiment-history extraction.
    Paper-PR automation defaults to "Not now"; opt in with --automate.
    A recommendations refresh is kicked off by default (--no-refresh to
    skip, --wait to block until ready).

    Examples:

      remyxai interests from-repo https://github.com/remyxai/outrider

      remyxai interests from-repo https://github.com/remyxai/outrider \\
        --name "Outrider" --daily-count 3 --automate review --wait
    """
    handle_interests_create_from_repo(
        repo_url=repo_url,
        name=name,
        daily_count=daily_count,
        inactive=inactive,
        automate=automate,
        timeout=timeout,
        refresh=refresh,
        wait=wait,
        output_format=output_format,
    )


@interests.command("from-project")
@click.argument("project")
@click.option("--name", "-n", default=None,
              help="Interest name (defaults to a project-derived name).")
@click.option("--daily-count", "-d", default=2, show_default=True,
              help="Recommendations per day (1-10).")
@click.option("--inactive", is_flag=True, default=False,
              help="Create as inactive (excluded from daily digest until toggled).")
@click.option("--include-experiment", "-e", multiple=True,
              help="Experiment name or UUID to track (repeatable). "
                   "Omit to track all experiments on the project.")
@click.option("--no-auto-update", is_flag=True, default=False,
              help="Do not auto-refresh context as new experiments land.")
@click.option("--refresh/--no-refresh", default=True, show_default=True,
              help="Kick off a recommendations refresh after creation.")
@click.option("--wait", "-w", is_flag=True, default=False,
              help="Block until context build + recommendations refresh complete "
                   "(recommended for project interests).")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def interests_from_project(project, name, daily_count, inactive,
                           include_experiment, no_auto_update, refresh, wait,
                           output_format):
    """
    Create a Research Interest from a project.

    PROJECT is a project name or UUID. The server builds the interest
    context from the project's experiments. By default it tracks every
    experiment and keeps the context fresh as new ones are added. A
    recommendations refresh is kicked off by default — use --wait so it
    runs after the async context build finishes.

    Examples:

      remyxai interests from-project "Spatial VQA" --wait

      remyxai interests from-project 1a2b3c4d-... \\
        -e "baseline-run" -e "dpo-v2" --no-auto-update
    """
    handle_interests_create_from_project(
        project=project,
        name=name,
        daily_count=daily_count,
        inactive=inactive,
        include_experiment=include_experiment,
        no_auto_update=no_auto_update,
        refresh=refresh,
        wait=wait,
        output_format=output_format,
    )


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
# outrider — install the Outrider GitHub Action on the current repo
# =============================================================================

@cli.group()
def outrider():
    """
    Manage the Outrider GitHub Action.

    Outrider is a GitHub Action that scouts arXiv weekly for your repo,
    picks the most implementable paper for your codebase, and opens a
    draft PR wiring it into an existing call site. See:

      https://github.com/remyxai/outrider
    """
    pass


@outrider.command("init")
@click.option("--repo", "repo", default=None,
              help=(
                  "Target repo as owner/name or a GitHub URL. Defaults to "
                  "detecting from the current directory's git remote."
              ))
@click.option("--interest", "-i", "interest_id", default=None,
              help="Remyx ResearchInterest UUID. Get it from engine.remyx.ai.")
@click.option("--auto-interest", is_flag=True, default=False,
              help=(
                  "Auto-create the ResearchInterest from this repo (calls "
                  "engine.remyx.ai). Mutually exclusive with --interest."
              ))
@click.option("--mode", type=click.Choice(["auto", "review", "off"]),
              default="auto", show_default=True,
              help=(
                  "auto: provision + merge the setup PR + start the first run "
                  "(\"set it up for me\"). review: provision + open a setup PR "
                  "to review. off: create the interest only."
              ))
@click.option("--anthropic-key", "anthropic_key", default=None,
              help=(
                  "Anthropic API key to connect as the model provider "
                  "(Claude Code). Falls back to $ANTHROPIC_API_KEY. Only used "
                  "if one isn't already connected."
              ))
@click.option("--no-wait", is_flag=True, default=False,
              help=(
                  "Don't block polling for the App install or provisioning to "
                  "finish; print next steps and return."
              ))
@click.option("--bulk-repos", "bulk_repos", type=click.Path(), default=None,
              help=(
                  "Path to a TSV mapping repos to ResearchInterest UUIDs "
                  "(\"owner/name<TAB>uuid\"). Loops `init` over each row "
                  "sequentially. Mutually exclusive with --repo / --interest "
                  "/ --auto-interest."
              ))
@click.option("--pace", "pace_s", type=int, default=3, show_default=True,
              help="Seconds to wait between repos in --bulk-repos mode.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print the plan and exit without making any changes.")
@click.option("--yes", "-y", "skip_confirm", is_flag=True, default=False,
              help="Skip the confirmation prompt (default is opt-in).")
def outrider_init(
    repo, interest_id, auto_interest, mode, anthropic_key,
    no_wait, bulk_repos, pace_s, dry_run, skip_confirm,
):
    """
    Set up Outrider on a GitHub repo.

    Drives the Remyx engine to do the same "set it up for me" flow as the
    web app: the Remyx GitHub App (remyx-ai[bot]) sets the repo secrets,
    writes the workflow, opens a bot-authored setup PR, and — in auto mode
    — merges it and fires the first run. Your local git is never touched
    and no personal `gh` token is needed; only your REMYXAI_API_KEY.

    Requires: the Remyx GitHub App installed on the repo (the command
    surfaces the install link if it isn't) and a connected model provider —
    connect Claude Code once on engine.remyx.ai/integrations and the CLI
    uses it automatically. (--anthropic-key / ANTHROPIC_API_KEY is only a
    one-time fallback if you haven't connected one there yet.)

    Examples:

      remyxai outrider init --repo remyxai/RepoRanger --auto-interest

      remyxai outrider init --repo owner/name --interest <uuid> --mode review

      remyxai outrider init --bulk-repos repos.tsv --mode review --yes
    """
    if bulk_repos:
        if repo or interest_id or auto_interest:
            raise click.UsageError(
                "--bulk-repos is mutually exclusive with "
                "--repo / --interest / --auto-interest."
            )
        rows = _parse_bulk_repos_tsv(bulk_repos)
        _run_bulk(
            handle_outrider_init,
            rows,
            common_kwargs=dict(
                auto_interest=False,
                mode=mode,
                anthropic_key=anthropic_key,
                skip_confirm=skip_confirm,
                dry_run=dry_run,
                no_wait=no_wait,
            ),
            pace_s=pace_s,
        )
        return
    handle_outrider_init(
        repo=repo,
        interest_id=interest_id,
        auto_interest=auto_interest,
        mode=mode,
        anthropic_key=anthropic_key,
        skip_confirm=skip_confirm,
        dry_run=dry_run,
        no_wait=no_wait,
    )


@outrider.command("setup-local")
@click.option("--repo", "repo", default=None,
              help="Target repo as owner/name or a GitHub URL. Defaults to "
                   "the current directory's git remote.")
@click.option("--interest", "-i", "interest_id", default=None,
              help="Remyx ResearchInterest UUID. Get it from engine.remyx.ai.")
@click.option("--auto-interest", is_flag=True, default=False,
              help="Auto-create the ResearchInterest from this repo. Mutually "
                   "exclusive with --interest.")
@click.option("--mode", type=click.Choice(["auto", "review"]),
              default="auto", show_default=True,
              help="auto: open + merge the setup PR + dispatch the first run. "
                   "review: open the setup PR for you to merge.")
@click.option("--anthropic-key", "anthropic_key", default=None,
              help="Anthropic API key to set as the ANTHROPIC_API_KEY repo "
                   "secret. Falls back to $ANTHROPIC_API_KEY, then prompts.")
@click.option("--no-cron", "no_cron", is_flag=True, default=False,
              help=(
                  "Render the workflow with the scheduled cron commented "
                  "out. Manual `workflow_dispatch` / `remyxai outrider "
                  "trigger` only. Re-enable later by uncommenting the "
                  "schedule block."
              ))
@click.option("--bulk-repos", "bulk_repos", type=click.Path(), default=None,
              help=(
                  "Path to a TSV mapping repos to ResearchInterest UUIDs "
                  "(\"owner/name<TAB>uuid\"). Loops `setup-local` over each "
                  "row sequentially. Mutually exclusive with --repo / "
                  "--interest / --auto-interest."
              ))
@click.option("--pace", "pace_s", type=int, default=3, show_default=True,
              help="Seconds to wait between repos in --bulk-repos mode.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print the plan + rendered workflow and exit; no changes.")
@click.option("--yes", "-y", "skip_confirm", is_flag=True, default=False,
              help="Skip the confirmation prompt (default is opt-in).")
def outrider_setup_local(
    repo, interest_id, auto_interest, mode, anthropic_key,
    no_cron, bulk_repos, pace_s, dry_run, skip_confirm,
):
    """
    Set up Outrider WITHOUT the Remyx GitHub App.

    For enterprises that can't grant a third-party App yet (e.g. a pending
    security review). Uses your own authenticated `gh` CLI to set the repo
    secrets, write the workflow, and open (in auto mode, merge) the setup
    PR — no Remyx App, nothing new to security-review. The only Remyx
    dependency is the REMYX_API_KEY the workflow uses at runtime.

    The running Action opens its PRs with the repo's built-in GITHUB_TOKEN;
    the command enables the repo's "Actions can create PRs" setting for that.

    Requires: an authenticated `gh` (run `gh auth login`, or set $GITHUB_TOKEN
    with repo + workflow scopes) and admin on the target repo.

    Examples:

      remyxai outrider setup-local --repo owner/name --auto-interest

      remyxai outrider setup-local --repo owner/name --interest <uuid> \\
        --mode review

      remyxai outrider setup-local --bulk-repos repos.tsv --mode review --yes \\
        --no-cron
    """
    if bulk_repos:
        if repo or interest_id or auto_interest:
            raise click.UsageError(
                "--bulk-repos is mutually exclusive with "
                "--repo / --interest / --auto-interest."
            )
        rows = _parse_bulk_repos_tsv(bulk_repos)
        _run_bulk(
            handle_outrider_setup_local,
            rows,
            common_kwargs=dict(
                auto_interest=False,
                mode=mode,
                anthropic_key=anthropic_key,
                skip_confirm=skip_confirm,
                dry_run=dry_run,
                no_cron=no_cron,
            ),
            pace_s=pace_s,
        )
        return
    handle_outrider_setup_local(
        repo=repo,
        interest_id=interest_id,
        auto_interest=auto_interest,
        mode=mode,
        anthropic_key=anthropic_key,
        skip_confirm=skip_confirm,
        dry_run=dry_run,
        no_cron=no_cron,
    )


@outrider.command("trigger")
@click.option("--repo", "repo", default=None,
              help=(
                  "Target repo as owner/name or a GitHub URL. Defaults to "
                  "detecting from the current directory's git remote."
              ))
@click.option("--pin-method", "pin_method", default=None,
              help=(
                  "Method/technique query (e.g. \"knowledge distillation\") "
                  "or a literal arxiv_id. The action's selection pass is "
                  "bypassed and the resolved paper is implemented directly. "
                  "Mutually exclusive with --pin-arxiv."
              ))
@click.option("--pin-arxiv", "pin_arxiv", default=None,
              help=(
                  "arxiv_id of a paper in the repo's candidate pool to "
                  "implement directly. Use --pin-method for arxiv_ids not "
                  "in the pool (it does a direct asset lookup)."
              ))
@click.option("--interest", "-i", "interest_id", default=None,
              help="Override the ResearchInterest UUID for this run.")
@click.option("--ref", "ref", default=None,
              help="Git ref to dispatch on. Defaults to the repo's default "
                   "branch.")
def outrider_trigger(repo, pin_method, pin_arxiv, interest_id, ref):
    """
    Dispatch a one-shot Outrider run on a repo via workflow_dispatch.

    Useful for kicking off ad-hoc method-targeted PRs without waiting for
    the next scheduled run. The repo must already have an Outrider workflow
    installed (see `remyxai outrider init` / `setup-local`). Authenticates
    via your local `gh` CLI — no Remyx engine round-trip.

    Examples:

      # Implement a specific method on a target repo
      remyxai outrider trigger --repo owner/name \\
        --pin-method "knowledge distillation"

      # Re-run a previously surfaced paper by arxiv_id (works even if it's
      # no longer in the candidate pool)
      remyxai outrider trigger --repo owner/name --pin-method 2410.20305v2

      # Plain trigger — let the normal selection pass run
      remyxai outrider trigger --repo owner/name
    """
    handle_outrider_trigger(
        repo=repo,
        pin_method=pin_method,
        pin_arxiv=pin_arxiv,
        interest_id=interest_id,
        ref=ref,
    )


if __name__ == "__main__":
    cli()
