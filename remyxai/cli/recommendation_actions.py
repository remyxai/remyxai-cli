"""
CLI action handlers for paper recommendations.
Called by the `remyxai papers` command group in commands.py.

─── Adding a new source type ─────────────────────────────────────────────────

When a new source_type ships (e.g. "github_repo"):

1. Add a renderer function:
       def _render_github_repo(rec: dict) -> None: ...

2. Add a Slack formatter:
       def _slack_github_repo(rec: dict) -> list[str]: ...
       (returns a list of lines to join, same pattern as _slack_arxiv_paper)

3. Register both in _RENDERERS and _SLACK_FORMATTERS:
       _RENDERERS["github_repo"] = _render_github_repo
       _SLACK_FORMATTERS["github_repo"] = _slack_github_repo

Nothing else changes. All display paths call through these registries.
"""
from __future__ import annotations

import json
import sys
import textwrap
import time
from datetime import date
from typing import Any, Callable, Dict, List, Optional

import click

from remyxai.api.recommendations import (
    get_recommendations_digest,
    list_recommended,
    poll_refresh_task,
    trigger_recommendations_refresh,
)


# ─── shared helpers ──────────────────────────────────────────────────────────

def _relevance_bar(score: float, width: int = 10) -> str:
    filled = round(score * width)
    return "▓" * filled + "░" * (width - filled) + f" {score:.0%}"


def _fmt_authors(authors: List[str], max_n: int = 3) -> str:
    if not authors:
        return "Unknown"
    if len(authors) <= max_n:
        return ", ".join(authors)
    return ", ".join(authors[:max_n]) + " et al."


def _clip(text: str, limit: int = 120) -> str:
    """Clip text at a word boundary and append ellipsis if truncated."""
    if len(text) <= limit:
        return text
    clipped = text[:limit].rstrip()
    last_space = clipped.rfind(" ")
    if last_space > limit - 25:
        clipped = clipped[:last_space]
    return clipped + "…"

def _wrap(text: str, limit: int = 120, indent: str = "     ") -> str:
    """Clip text at a word boundary and wrap to fit a narrow terminal/TUI viewport."""
    return textwrap.fill(
        _clip(text, limit), width=72,
        initial_indent=indent,
        subsequent_indent=indent,
    )



# ─── terminal renderers (one per source_type) ────────────────────────────────

def _render_arxiv_paper(rec: dict) -> None:
    r = rec["resource"]
    reasoning = rec.get("reasoning", "").strip()
    summary = reasoning or (r.get("abstract_summary") or "").strip()

    click.echo(f"     {_fmt_authors(r.get('authors', []))}")
    if summary:
        click.echo(_wrap(summary))

    extras = []
    if r.get("has_docker") and r.get("docker_image"):
        extras.append(f"🐳 {r['docker_image']}")
    if r.get("github_url"):
        extras.append(f"⑂  {r['github_url']}")
    if extras:
        click.echo(f"     {' | '.join(extras)}")


def _render_github_repo(rec: dict) -> None:
    """Stub for future github_repo source type."""
    r = rec["resource"]
    reasoning = rec.get("reasoning", "").strip()

    meta = []
    if r.get("language"):
        meta.append(r["language"])
    if r.get("stars") is not None:
        meta.append(f"★ {r['stars']:,}")
    if meta:
        click.echo(f"     {' · '.join(meta)}")
    if reasoning:
        click.echo(_wrap(reasoning))
    if r.get("has_docker") and r.get("docker_image"):
        click.echo(f"     🐳 {r['docker_image']}")


def _render_unknown(rec: dict, full: bool = False) -> None:
    reasoning = rec.get("reasoning", "").strip()
    if reasoning:
        click.echo(f"     {_clip(reasoning, full=full)}")


_RENDERERS: Dict[str, Callable[[dict], None]] = {
    "arxiv_paper": _render_arxiv_paper,
    "github_repo":  _render_github_repo,
}


def _render_recommendation(rec: dict, rank: int) -> None:
    source_type = rec.get("source_type", "unknown")
    score = rec.get("relevance_score", 0.0)
    click.echo(f"\n  {rank}. {rec['title']}")
    click.echo(f"     {rec['url']}")
    _RENDERERS.get(source_type, _render_unknown)(rec)
    click.echo(f"     Relevance: {_relevance_bar(score)}")


# ─── Slack formatters (one per source_type) ──────────────────────────────────

def _slack_arxiv_paper(rec: dict) -> List[str]:
    r = rec["resource"]
    reasoning = rec.get("reasoning", "").strip()
    summary = reasoning or (r.get("abstract_summary") or "").strip()

    lines = [f"*<{rec['url']}|{rec['title']}>*"]
    lines.append(f"_{_fmt_authors(r.get('authors', []))}_")
    if summary:
        lines.append(_clip(summary))
    extras = []
    if r.get("has_docker"):
        extras.append("🐳 Docker image available")
    if r.get("github_url"):
        extras.append(f"⑂ <{r['github_url']}|GitHub>")
    if extras:
        lines.append("  ".join(extras))
    return lines


def _slack_github_repo(rec: dict) -> List[str]:
    """Stub for future github_repo source type."""
    r = rec["resource"]
    reasoning = rec.get("reasoning", "").strip()

    lines = [f"*<{rec['url']}|{rec['title']}>*"]
    meta = []
    if r.get("language"):
        meta.append(r["language"])
    if r.get("stars") is not None:
        meta.append(f"★ {r['stars']:,}")
    if meta:
        lines.append("  ".join(meta))
    if reasoning:
        lines.append(_clip(reasoning))
    return lines


def _slack_unknown(rec: dict) -> List[str]:
    lines = [f"*<{rec['url']}|{rec['title']}>*"]
    reasoning = rec.get("reasoning", "").strip()
    if reasoning:
        lines.append(_clip(reasoning))
    return lines


_SLACK_FORMATTERS: Dict[str, Callable[[dict], List[str]]] = {
    "arxiv_paper": _slack_arxiv_paper,
    "github_repo":  _slack_github_repo,
}


def _slack_entry(rec: dict, rank: int) -> str:
    source_type = rec.get("source_type", "unknown")
    score = rec.get("relevance_score", 0.0)
    formatter = _SLACK_FORMATTERS.get(source_type, _slack_unknown)
    lines = formatter(rec)
    lines.append(f"Relevance: {_relevance_bar(score)}")
    body = "\n".join(f"   {ln}" for ln in lines)
    return f"{rank}. {body}"


def build_slack_digest(data: dict) -> str:
    """
    Build the full Slack message from a digest API response.
    Called by the OpenClaw skill when posting to #research.
    """
    today_str = date.today().strftime("%B %-d")
    interests = data.get("interests", [])
    # Engine returns total_papers (not total)
    total = data.get("total_papers", 0)

    blocks = [f"🔬 *Remyx Daily Recommendations* — {today_str}", "━" * 44]

    for interest in interests:
        recs = interest.get("recommendations", [])
        iname = interest["name"]
        if not recs:
            blocks.append(f"\n_{iname}: no new recommendations today._")
            continue
        n = len(recs)
        blocks.append(f"\n*{iname}*  _{n} item{'s' if n != 1 else ''}_")
        for i, rec in enumerate(recs, start=1):
            blocks.append(f"\n{_slack_entry(rec, i)}")

    blocks.append("\n" + "━" * 44)
    blocks.append(
        f"_Total: {total} · "
        "Powered by <https://remyx.ai|Remyx AI> · ExperimentOps_"
    )
    return "\n".join(blocks)


# ─── handle_papers_digest ────────────────────────────────────────────────────

def handle_papers_digest(
    limit: int = 5,
    period: str = "today",
    output_format: str = "text",
) -> None:
    try:
        data = get_recommendations_digest(limit=limit, period=period)
    except Exception as e:
        click.echo(f"❌ Failed to fetch digest: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(data, indent=2))
        return

    interests = data.get("interests", [])
    # Engine returns total_papers (not total)
    total = data.get("total_papers", 0)
    source_types = data.get("source_types", [])

    today_str = date.today().strftime("%B %-d")
    click.echo(f"\n🔬  Remyx Daily Recommendations — {today_str}")
    if source_types:
        click.echo(f"    Sources: {', '.join(source_types)}")
    click.echo("━" * 60)

    if not interests or total == 0:
        click.echo(f"\n  No recommendations found for period '{period}'.")
        click.echo("  Run:  remyxai papers refresh --wait")
        click.echo(f"  Or:   remyxai papers digest --period week\n")
        return

    for interest in interests:
        recs = interest.get("recommendations", [])
        if not recs:
            click.echo(f"\n  {interest['name']}: no new items today.\n")
            continue
        n = len(recs)
        click.echo(f"\n  {interest['name']}  ({n} item{'s' if n != 1 else ''})")
        for i, rec in enumerate(recs, start=1):
            _render_recommendation(rec, i)
        click.echo()

    click.echo("━" * 60)
    click.echo(
        f"  Total: {total} item{'s' if total != 1 else ''} "
        f"across {len(interests)} interest(s)"
    )
    click.echo("  Powered by Remyx AI · ExperimentOps\n")


# ─── handle_papers_list ──────────────────────────────────────────────────────

def handle_papers_list(
    interest_id: Optional[str],
    limit: int,
    period: str,
    source_type: Optional[str],
    output_format: str,
) -> None:
    try:
        data = list_recommended(
            interest_id=interest_id,
            limit=limit,
            period=period,
            source_type=source_type,
        )
    except Exception as e:
        click.echo(f"❌ Failed to fetch recommendations: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(data, indent=2))
        return

    # Engine returns "papers" as the list key on this endpoint
    recs = data.get("papers", data.get("recommendations", []))
    if not recs:
        click.echo(f"\n  No recommendations found (period={period}).")
        return

    click.echo(f"\n📚  Recommendations ({period})  — {len(recs)} result(s)")
    click.echo("━" * 60)
    for i, rec in enumerate(recs, start=1):
        _render_recommendation(rec, i)
    click.echo()


# ─── handle_papers_refresh ───────────────────────────────────────────────────

def handle_papers_refresh(
    interest_id: Optional[str],
    num_results: Optional[int],
    wait: bool,
    output_format: str,
) -> None:
    try:
        result = trigger_recommendations_refresh(
            interest_id=interest_id,
            num_results=num_results,
        )
    except Exception as e:
        click.echo(f"❌ Failed to start refresh: {e}", err=True)
        sys.exit(1)

    tasks = result.get("tasks", [])

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
        return

    click.echo(f"\n⚡  Started {len(tasks)} refresh task(s)")
    for t in tasks:
        click.echo(f"   • {t['interest_name']}  (task: {t['task_id']})")

    if not wait:
        click.echo(
            "\n  Use --wait to block until complete, or poll:\n"
            "  remyxai papers refresh-status <task_id>\n"
        )
        return

    # Poll until all tasks reach a terminal state
    click.echo("\n  Waiting", nl=False)
    terminal = {"completed", "failed"}
    pending = {t["task_id"]: t["interest_name"] for t in tasks}
    final: Dict[str, Any] = {}

    while pending:
        time.sleep(3)
        click.echo(".", nl=False)
        for task_id in list(pending):
            try:
                status = poll_refresh_task(task_id)
                if status.get("status") in terminal:
                    final[task_id] = status
                    del pending[task_id]
            except Exception:
                pass  # transient — keep polling

    click.echo(" done!\n")
    for status in final.values():
        s = status.get("status")
        icon = "✅" if s == "completed" else "❌"
        click.echo(f"  {icon}  {status.get('message', s)}")
    click.echo()


# ─── handle_refresh_status ───────────────────────────────────────────────────

def handle_refresh_status(task_id: str, output_format: str) -> None:
    try:
        data = poll_refresh_task(task_id)
    except Exception as e:
        click.echo(f"❌ Failed to poll task: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(data, indent=2))
        return

    status = data.get("status", "unknown")
    icons = {"completed": "✅", "failed": "❌", "running": "⚙️ ", "pending": "⏳"}
    click.echo(
        f"\n  {icons.get(status, '❓')}  {status.upper()}  "
        f"[{data.get('progress', 0)}%]  {data.get('message', '')}"
    )
    if r := data.get("result"):
        click.echo(f"  Items found: {r.get('count', 0)}")
    if err := data.get("error"):
        click.echo(f"  Error: {err}", err=True)
    click.echo()
