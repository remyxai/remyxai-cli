"""
CLI action handlers for Research Interest management.
Called by the `remyxai interests` command group in commands.py.
"""
from __future__ import annotations

import json
import re
import sys
import textwrap
import time
from typing import List, Optional

import click

from remyxai.api.interests import (
    analyze_repo,
    create_interest,
    delete_interest,
    get_interest,
    list_interests,
    poll_analyze_repo,
    toggle_interest,
    update_interest,
)
from remyxai.api.projects import list_experiments, list_projects
from remyxai.api.recommendations import (
    poll_refresh_task,
    trigger_recommendations_refresh,
)

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


# ─── formatting helpers ──────────────────────────────────────────────────────

def _print_interest(i: dict, verbose: bool = False) -> None:
    icon = "✅" if i.get("is_active") else "⏸ "
    last = i.get("last_recommendation_at")
    last_str = f"  last reco: {last[:10]}" if last else ""

    click.echo(f"\n  {icon}  {i['name']}")
    click.echo(f"       id:          {i['id']}")
    click.echo(f"       daily_count: {i.get('daily_count', 2)}")
    click.echo(f"       active:      {i.get('is_active', True)}{last_str}")

    if verbose:
        ctx = (i.get("context") or "").strip()
        if ctx:
            wrapped = textwrap.fill(
                ctx, width=72,
                initial_indent="       context:    ",
                subsequent_indent="                   ",
            )
            click.echo(wrapped)


# ─── recommendations auto-refresh ────────────────────────────────────────────

_REFRESH_TERMINAL = {"completed", "failed"}


def _poll_task(
    task_id: str,
    timeout: int = 600,
    poll_interval: int = 4,
    api_key: Optional[str] = None,
) -> Optional[dict]:
    """Poll any background task to a terminal state via the shared
    refresh-poll endpoint (build, extraction, and refresh tasks all live in
    the same task store). Returns the final task dict, or None on
    timeout/error."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            task = poll_refresh_task(task_id, api_key=api_key)
        except Exception:
            return None
        if (task.get("status") or "").lower() in _REFRESH_TERMINAL:
            return task
        if time.monotonic() >= deadline:
            return None
        time.sleep(poll_interval)


def _kick_off_recommendations(
    interest_id: str,
    *,
    num_results: Optional[int] = None,
    wait: bool = False,
    await_task_id: Optional[str] = None,
    api_key: Optional[str] = None,
    echo=None,
) -> Optional[dict]:
    """Trigger a recommendations refresh for a freshly-created interest so it
    has actual recommendations.

    ``await_task_id`` (project-mode context build) is polled to completion
    first when ``wait`` is set, so recommendations rank against the built
    context rather than the placeholder. ``echo`` is an optional
    callable(str) for progress lines (text mode). Returns the refresh
    trigger response, or None if it could not be started.
    """
    def _say(msg: str) -> None:
        if echo:
            echo(msg)

    # Project-mode interests build their context asynchronously; let that
    # finish before generating recommendations.
    if wait and await_task_id:
        _say("  ⏳ waiting for interest context to finish building…")
        built = _poll_task(await_task_id, api_key=api_key)
        if built and (built.get("status") or "").lower() == "failed":
            _say(
                "  ⚠️  context build failed "
                f"({built.get('error') or 'unknown error'}); "
                "recommendations may be weak."
            )

    try:
        res = trigger_recommendations_refresh(
            interest_id=interest_id,
            num_results=num_results,
            api_key=api_key,
        )
    except Exception as e:
        _say(f"  ⚠️  could not start recommendations refresh: {e}")
        return None

    tasks = res.get("tasks", [])
    if not tasks:
        _say("  ⚠️  recommendations refresh returned no tasks.")
        return res

    for t in tasks:
        _say(f"  🔄 recommendations refresh started (task: {t.get('task_id')})")

    if not wait:
        _say("     Track with:  remyxai papers refresh-status <task_id>")
        return res

    _say("  ⏳ generating recommendations…")
    for t in tasks:
        final = _poll_task(t.get("task_id"), api_key=api_key)
        if not final:
            _say("  ⚠️  timed out waiting for recommendations.")
            continue
        status = (final.get("status") or "").lower()
        if status == "completed":
            count = (final.get("result") or {}).get("count")
            label = f"{count} recommendation(s)" if count is not None else "recommendations"
            _say(f"  ✅ {label} ready.")
        else:
            _say(
                "  ❌ recommendations refresh failed: "
                f"{final.get('error') or final.get('message') or 'unknown error'}"
            )
    return res


# ─── name-or-id resolver ─────────────────────────────────────────────────────

def _resolve_interest_id(name_or_id: str) -> str:
    """Accept either a UUID or an interest name and return the UUID.

    If name_or_id looks like a UUID (36 chars, hyphen-separated),
    return it directly. Otherwise fetch all interests and match by name
    case-insensitively, returning the first match found.
    """
    if re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        name_or_id, re.IGNORECASE
    ):
        return name_or_id

    try:
        interests = list_interests()
    except Exception as e:
        click.echo("Failed to fetch interests: {}".format(e), err=True)
        sys.exit(1)

    needle = name_or_id.lower()
    for interest in interests:
        if interest.get("name", "").lower() == needle:
            return interest["id"]

    names = [i.get("name", "") for i in interests]
    available = ", ".join(names) if names else "none"
    click.echo(
        "No interest found with name {!r}. Available: {}".format(name_or_id, available),
        err=True,
    )
    sys.exit(1)


# ─── list ────────────────────────────────────────────────────────────────────

def handle_interests_list(output_format: str = "text") -> None:
    try:
        interests = list_interests()
    except Exception as e:
        click.echo(f"❌ Failed to fetch interests: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(interests, indent=2))
        return

    if not interests:
        click.echo(
            "\n  No Research Interests found.\n"
            "  Create one:  remyxai interests create\n"
        )
        return

    active = sum(1 for i in interests if i.get("is_active"))
    click.echo(
        f"\n🎯  Research Interests  "
        f"({len(interests)} total, {active} active)"
    )
    click.echo("━" * 60)
    for i in interests:
        _print_interest(i, verbose=True)
    click.echo()


# ─── get ─────────────────────────────────────────────────────────────────────

def handle_interests_get(interest_id: str, output_format: str = "text") -> None:
    interest_id = _resolve_interest_id(interest_id)
    try:
        interest = get_interest(interest_id)
    except Exception as e:
        click.echo(f"❌ Failed to fetch interest: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(interest, indent=2))
        return

    click.echo("\n🎯  Research Interest")
    click.echo("━" * 60)
    _print_interest(interest, verbose=True)
    click.echo()


# ─── create ──────────────────────────────────────────────────────────────────

def handle_interests_create(
    name: Optional[str],
    context: Optional[str],
    daily_count: int,
    inactive: bool,
    output_format: str,
    refresh: bool = True,
    wait: bool = False,
) -> None:
    if not name:
        name = click.prompt("  Interest name (e.g. 'RAG & Retrieval')")
    if not context:
        click.echo(
            "  Context: describe what you want to track in natural language.\n"
            "  Can also be a HuggingFace or GitHub URL.\n"
            "  Examples:\n"
            "    'Retrieval-augmented generation, hybrid search, re-ranking'\n"
            "    'https://huggingface.co/ibm-granite/granitelib-rag-r1.0'\n"
        )
        context = click.prompt("  Context")

    try:
        result = create_interest(
            name=name,
            context=context,
            daily_count=daily_count,
            is_active=not inactive,
        )
    except Exception as e:
        click.echo(f"❌ Failed to create interest: {e}", err=True)
        sys.exit(1)

    # Populate the interest with actual recommendations.
    refresh_res = None
    if refresh:
        refresh_res = _kick_off_recommendations(
            result["id"],
            num_results=daily_count,
            wait=wait,
            echo=(None if output_format == "json" else click.echo),
        )

    if output_format == "json":
        if refresh_res is not None:
            result["recommendation_refresh"] = refresh_res
        click.echo(json.dumps(result, indent=2))
        return

    click.echo(f"\n✅  Created '{result['name']}'  (id: {result['id']})")
    click.echo(
        f"   daily_count: {result['daily_count']}  |  "
        f"active: {result['is_active']}"
    )
    if not refresh:
        click.echo(
            "\n  Trigger your first recommendations:\n"
            f"  remyxai papers refresh --interest {result['name']!r} --wait"
        )
    click.echo()


# ─── update ──────────────────────────────────────────────────────────────────

def handle_interests_update(
    interest_id: str,
    name: Optional[str],
    context: Optional[str],
    daily_count: Optional[int],
    is_active: Optional[bool],
    output_format: str,
) -> None:
    interest_id = _resolve_interest_id(interest_id)
    if not any(v is not None for v in [name, context, daily_count, is_active]):
        click.echo(
            "❌ Provide at least one field to update:\n"
            "   --name, --context, --daily-count, --activate, --deactivate",
            err=True,
        )
        sys.exit(1)

    try:
        result = update_interest(
            interest_id=interest_id,
            name=name,
            context=context,
            daily_count=daily_count,
            is_active=is_active,
        )
    except Exception as e:
        click.echo(f"❌ Failed to update interest: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
        return

    click.echo(f"\n✅  Updated '{result['name']}'")
    if result.get("pool_invalidated"):
        click.echo(
            "   ℹ️  Context changed — recommendation pool cleared.\n"
            f"   Run:  remyxai papers refresh --interest {result['name']!r} --wait"
        )
    click.echo()


# ─── delete ──────────────────────────────────────────────────────────────────

def handle_interests_delete(
    interest_id: str,
    yes: bool,
    output_format: str,
) -> None:
    interest_id = _resolve_interest_id(interest_id)
    if not yes:
        try:
            i = get_interest(interest_id)
            label = i.get("name", interest_id)
        except Exception:
            label = interest_id

        click.confirm(
            f"  Delete Research Interest '{label}'? "
            "This removes all associated recommendations.",
            abort=True,
        )

    try:
        result = delete_interest(interest_id)
    except Exception as e:
        click.echo(f"❌ Failed to delete interest: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
        return

    click.echo(f"\n🗑️  Deleted interest {interest_id}\n")


# ─── toggle ──────────────────────────────────────────────────────────────────

def handle_interests_toggle(interest_id: str, output_format: str) -> None:
    interest_id = _resolve_interest_id(interest_id)
    try:
        result = toggle_interest(interest_id)
    except Exception as e:
        click.echo(f"❌ Failed to toggle interest: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
        return

    state = "active ✅" if result.get("is_active") else "paused ⏸"
    click.echo(f"\n  '{result['name']}' is now {state}\n")


# ─── create from repo ──────────────────────────────────────────────────────

# "Automate paper PRs on this repo" choices → provisioning kwargs.
#   none   → "Not now"                  (no provisioning)
#   review → "Open a PR for me to review" (setup PR, no auto-merge)
#   auto   → "Set it up for me"            (setup PR + auto-merge + first run)
AUTOMATE_CHOICES = ("none", "review", "auto")


class RepoAnalysisError(RuntimeError):
    """Raised when repo analysis fails, times out, or returns no report."""


def _run_repo_analysis(
    repo_url: str,
    timeout: int = 300,
    poll_interval: int = 5,
    api_key: Optional[str] = None,
    echo=None,
) -> dict:
    """Kick off analyze-repo and poll to completion.

    Returns the completed task's ``result`` dict (report_markdown,
    source_repo_url, repo_analysis, …). ``echo`` is an optional
    callable(str) for progress lines. Raises RepoAnalysisError on
    failure, timeout, or a completed-but-empty report.
    """
    started = analyze_repo(repo_url, api_key=api_key)
    task_id = started.get("task_id")
    if not task_id:
        raise RepoAnalysisError(f"analyze-repo returned no task_id: {started}")

    deadline = time.monotonic() + timeout
    last_message = None

    while True:
        task = poll_analyze_repo(task_id, api_key=api_key)
        status = (task.get("status") or "").lower()

        message = task.get("message")
        if echo and message and message != last_message:
            pct = task.get("progress")
            prefix = f"  [{pct}%] " if pct is not None else "  "
            echo(f"{prefix}{message}")
            last_message = message

        if status == "completed":
            result = task.get("result") or {}
            if not result.get("report_markdown"):
                raise RepoAnalysisError(
                    f"analysis completed but returned no report: {result}"
                )
            return result

        if status == "failed":
            err = task.get("error") or task.get("message") or "unknown error"
            raise RepoAnalysisError(f"repo analysis failed: {err}")

        if time.monotonic() >= deadline:
            raise RepoAnalysisError(
                f"repo analysis timed out after {timeout}s "
                f"(analyze-repo task {task_id})"
            )

        time.sleep(poll_interval)


def create_interest_from_repo(
    repo_url: str,
    *,
    name: Optional[str] = None,
    daily_count: int = 2,
    is_active: bool = True,
    automate: str = "none",
    timeout: int = 300,
    poll_interval: int = 5,
    api_key: Optional[str] = None,
    echo=None,
) -> dict:
    """Analyze a GitHub repo and create a Research Interest from it.

    This is the shared core behind ``interests from-repo`` and
    ``outrider init --auto-interest``: it generates the repo profile
    report (used as the rich interest context) and carries the repo
    fields so the server links the interest to its ExperimentHistory and
    dispatches experiment-history extraction.

    Returns the created interest dict. Raises RepoAnalysisError on
    analysis problems (caller decides how to surface it); ``create_interest``
    network errors propagate as-is.

    ``automate`` controls the "Automate paper PRs on this repo" option:
    "none" (Not now), "review" (setup PR), or "auto" (PR + merge + run).
    """
    if automate not in AUTOMATE_CHOICES:
        raise ValueError(f"automate must be one of {AUTOMATE_CHOICES}")

    result = _run_repo_analysis(
        repo_url,
        timeout=timeout,
        poll_interval=poll_interval,
        api_key=api_key,
        echo=echo,
    )
    report = result["report_markdown"]
    source_repo_url = result.get("source_repo_url") or repo_url

    provision_kwargs: dict = {}
    if automate in ("review", "auto"):
        provision_kwargs = {
            "provision_action": True,
            "provision_auto_merge": (automate == "auto"),
            "provision_repo_url": source_repo_url,
        }

    interest_name = name or (result.get("full_name") or repo_url).split("/")[-1]

    return create_interest(
        name=interest_name,
        context=report,
        daily_count=daily_count,
        is_active=is_active,
        source_repo_url=source_repo_url,
        source_repo_metadata=result.get("source_repo_metadata"),
        repo_analysis=result.get("repo_analysis"),
        generated_report=report,
        report_generated_at=result.get("report_generated_at"),
        api_key=api_key,
        **provision_kwargs,
    )


def handle_interests_create_from_repo(
    repo_url: str,
    name: Optional[str],
    daily_count: int,
    inactive: bool,
    automate: str,
    timeout: int,
    output_format: str,
    refresh: bool = True,
    wait: bool = False,
) -> None:
    """Create a Research Interest from a GitHub repo (CLI handler).

    Thin wrapper over ``create_interest_from_repo`` that handles progress
    output, error reporting, and result formatting.
    """
    if automate not in AUTOMATE_CHOICES:
        click.echo(
            f"❌ --automate must be one of {', '.join(AUTOMATE_CHOICES)}",
            err=True,
        )
        sys.exit(1)

    if output_format != "json":
        click.echo(f"\n🔍  Analyzing {repo_url} … (this can take 30–90s)")

    try:
        created = create_interest_from_repo(
            repo_url,
            name=name,
            daily_count=daily_count,
            is_active=not inactive,
            automate=automate,
            timeout=timeout,
            echo=(None if output_format == "json" else click.echo),
        )
    except RepoAnalysisError as e:
        click.echo(f"\n❌ {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Failed to create interest: {e}", err=True)
        sys.exit(1)

    source_repo_url = created.get("source_repo_url") or repo_url

    # The repo profile is already the interest's context, so recommendations
    # can be generated immediately (no need to await the deeper extraction).
    refresh_res = None
    if refresh:
        refresh_res = _kick_off_recommendations(
            created["id"],
            num_results=daily_count,
            wait=wait,
            echo=(None if output_format == "json" else click.echo),
        )

    if output_format == "json":
        if refresh_res is not None:
            created["recommendation_refresh"] = refresh_res
        click.echo(json.dumps(created, indent=2))
        return

    click.echo(f"\n✅  Created '{created['name']}'  (id: {created['id']})")
    click.echo(f"       source repo: {source_repo_url}")
    click.echo(
        f"       daily_count: {created.get('daily_count')}  |  "
        f"active: {created.get('is_active')}"
    )
    if created.get("history_extraction_task_id"):
        click.echo(
            "       🧪 experiment-history extraction dispatched "
            f"(task: {created['history_extraction_task_id']})"
        )
    if created.get("provision_task_id"):
        mode_label = (
            "auto-merge + first run" if automate == "auto" else "setup PR for review"
        )
        click.echo(
            f"       🤖 paper-PR automation provisioning ({mode_label}) "
            f"(task: {created['provision_task_id']})"
        )
    else:
        click.echo("       paper-PR automation: not now")
    click.echo()


# ─── create from project ───────────────────────────────────────────────────

def _resolve_project_id(name_or_id: str) -> str:
    """Accept a project UUID or name and return the UUID."""
    if UUID_RE.match(name_or_id):
        return name_or_id

    try:
        projects = list_projects()
    except Exception as e:
        click.echo(f"❌ Failed to fetch projects: {e}", err=True)
        sys.exit(1)

    needle = name_or_id.lower()
    for p in projects:
        if (p.get("name") or "").lower() == needle:
            return p["id"]

    names = [p.get("name", "") for p in projects]
    available = ", ".join(n for n in names if n) or "none"
    click.echo(
        f"No project found with name {name_or_id!r}. Available: {available}",
        err=True,
    )
    sys.exit(1)


def _resolve_experiment_ids(
    project_id: str,
    selectors: List[str],
) -> List[str]:
    """Map experiment names-or-UUIDs to UUIDs within a project."""
    resolved: List[str] = []
    experiments = None
    by_name = None

    for sel in selectors:
        if UUID_RE.match(sel):
            resolved.append(sel)
            continue
        # Lazy-load the project's experiments only when a name is given.
        if experiments is None:
            try:
                experiments = list_experiments(project_id=project_id)
            except Exception as e:
                click.echo(f"❌ Failed to fetch experiments: {e}", err=True)
                sys.exit(1)
            by_name = {
                (e.get("name") or "").lower(): e["id"] for e in experiments
            }
        match = by_name.get(sel.lower())
        if not match:
            click.echo(
                f"No experiment named {sel!r} in this project.", err=True
            )
            sys.exit(1)
        resolved.append(match)
    return resolved


def handle_interests_create_from_project(
    project: str,
    name: Optional[str],
    daily_count: int,
    inactive: bool,
    include_experiment: tuple,
    no_auto_update: bool,
    output_format: str,
    refresh: bool = True,
    wait: bool = False,
) -> None:
    """Create a project-structured Research Interest.

    The server builds the interest context from the project's experiments
    (asynchronously) and returns a ``build_task_id``. By default it tracks
    all experiments on the project and auto-updates as new ones land; pass
    ``include_experiment`` to curate a subset.
    """
    project_id = _resolve_project_id(project)

    included_ids = None
    if include_experiment:
        included_ids = _resolve_experiment_ids(project_id, list(include_experiment))

    try:
        created = create_interest(
            name=name,
            daily_count=daily_count,
            is_active=not inactive,
            project_id=project_id,
            mode="project_structured",
            included_experiment_ids=included_ids,
            auto_update_from_experiments=(False if no_auto_update else None),
        )
    except Exception as e:
        click.echo(f"❌ Failed to create interest: {e}", err=True)
        sys.exit(1)

    # Project-mode context is built asynchronously (build_task_id). Pass it
    # as await_task_id so a --wait refresh sequences behind the build and
    # recommendations rank against the real context, not the placeholder.
    refresh_res = None
    if refresh:
        refresh_res = _kick_off_recommendations(
            created["id"],
            num_results=daily_count,
            wait=wait,
            await_task_id=created.get("build_task_id"),
            echo=(None if output_format == "json" else click.echo),
        )

    if output_format == "json":
        if refresh_res is not None:
            created["recommendation_refresh"] = refresh_res
        click.echo(json.dumps(created, indent=2))
        return

    click.echo(f"\n✅  Created '{created['name']}'  (id: {created['id']})")
    click.echo(f"       project: {project_id}")
    if included_ids:
        click.echo(f"       tracking {len(included_ids)} selected experiment(s)")
    else:
        click.echo("       tracking: all experiments on the project")
    click.echo(
        f"       daily_count: {created.get('daily_count')}  |  "
        f"active: {created.get('is_active')}"
    )
    if created.get("build_task_id"):
        click.echo(
            "       ⏳ building recommendation context from experiments "
            f"(task: {created['build_task_id']})"
        )
    if refresh and not wait:
        click.echo(
            "\n  ℹ️  For project interests, run the refresh with --wait so "
            "recommendations\n      generate after the context finishes building."
        )
    click.echo()
