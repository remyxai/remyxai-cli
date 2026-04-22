"""
CLI action handlers for the Experiments group.
Called by the `remyxai experiments` commands in commands.py.
"""
from __future__ import annotations

import json
import sys
import textwrap
from typing import List, Optional

import click

from remyxai.api.experiments import (
    get_experiment,
    get_validation_run,
    list_experiments,
    start_validation_run,
)


# ── formatting helpers ───────────────────────────────────────────────────────

def _print_experiment_row(e: dict) -> None:
    status_icon = {
        "backlog": "📋 ",
        "implementing": "🔨 ",
        "validating": "🧪 ",
        "validated": "✅ ",
        "completed": "✅ ",
        "failed": "❌ ",
    }.get(e.get("status", ""), "  ")

    line = f"  {status_icon} {e.get('name') or e.get('hypothesis') or '(unnamed)'}"
    click.echo(line)
    click.echo(f"       id:       {e['id']}")
    if e.get("status"):
        click.echo(f"       status:   {e['status']}")
    if e.get("initiative"):
        click.echo(f"       init'iv:  {e['initiative']}")
    if e.get("validation_status"):
        click.echo(f"       val:      {e['validation_status']}")


def _print_experiment_detail(e: dict) -> None:
    click.echo(f"\n  {e.get('name') or '(unnamed)'}")
    click.echo("━" * 60)

    click.echo(f"  id:                {e['id']}")
    for field in (
        "status", "outcome", "initiative", "target_metric",
        "observed_delta", "delta_confidence",
        "validation_status", "pr_url", "ticket_url",
    ):
        val = e.get(field)
        if val is not None and val != "":
            click.echo(f"  {field}:{' ' * max(1, 18 - len(field))}{val}")

    hyp = (e.get("hypothesis") or "").strip()
    if hyp:
        wrapped = textwrap.fill(
            hyp, width=72,
            initial_indent="  hypothesis:        ",
            subsequent_indent="                     ",
        )
        click.echo(wrapped)


# ── list ─────────────────────────────────────────────────────────────────────

def handle_experiments_list(
    project_id: Optional[str],
    status: Optional[str],
    initiative: Optional[str],
    limit: int,
    output_format: str,
) -> None:
    try:
        experiments = list_experiments(
            project_id=project_id,
            status=status,
            initiative=initiative,
            limit=limit,
        )
    except Exception as e:
        click.echo(f"❌ Failed to list experiments: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(experiments, indent=2))
        return

    if not experiments:
        click.echo("\n  No experiments match those filters.\n")
        return

    click.echo(f"\n🧪  Experiments  ({len(experiments)} shown)")
    click.echo("━" * 60)
    for e in experiments:
        _print_experiment_row(e)
    click.echo()


# ── get ──────────────────────────────────────────────────────────────────────

def handle_experiments_get(
    experiment_id: str,
    output_format: str,
) -> None:
    try:
        e = get_experiment(experiment_id)
    except Exception as exc:
        click.echo(f"❌ Failed to fetch experiment: {exc}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(e, indent=2))
        return

    _print_experiment_detail(e)
    click.echo()


# ── validate ─────────────────────────────────────────────────────────────────

def _parse_variants(variant_args: List[str]) -> List[dict]:
    """Parse repeated --variant "name=sha" / "name=ref:sha" flags."""
    parsed: List[dict] = []
    for raw in variant_args:
        if "=" not in raw:
            click.echo(
                f"❌ Invalid --variant {raw!r}; expected name=commit_sha "
                "or name=ref:commit_sha",
                err=True,
            )
            sys.exit(1)
        name, rest = raw.split("=", 1)
        name = name.strip()
        rest = rest.strip()
        if not name or not rest:
            click.echo(f"❌ --variant {raw!r} must have both name and sha", err=True)
            sys.exit(1)

        if ":" in rest:
            ref, sha = rest.split(":", 1)
            parsed.append({"name": name, "ref": ref.strip(), "commit_sha": sha.strip()})
        else:
            parsed.append({"name": name, "commit_sha": rest})
    return parsed


def handle_experiments_validate(
    experiment_id: str,
    template_id: str,
    github_url: Optional[str],
    variants: List[str],
    seeds: int,
    pr_number: Optional[int],
    pr_url: Optional[str],
    output_format: str,
) -> None:
    """Launch a REMYX-24 eval run for an experiment."""
    variant_list = _parse_variants(variants)
    if len(variant_list) < 1:
        click.echo(
            "❌ At least one --variant is required (e.g. "
            "--variant baseline=abc123 --variant feature=def456).",
            err=True,
        )
        sys.exit(1)

    # If caller didn't supply github_url, try to read it off the experiment.
    if not github_url:
        try:
            exp = get_experiment(experiment_id)
        except Exception as e:
            click.echo(f"❌ Failed to fetch experiment for repo lookup: {e}", err=True)
            sys.exit(1)
        github_url = (
            (exp.get("validation_config") or {}).get("github_url")
            or exp.get("repo")
            or ""
        )
        if not github_url:
            click.echo(
                "❌ This experiment has no stored repo URL; pass --github-url.",
                err=True,
            )
            sys.exit(1)

    try:
        result = start_validation_run(
            experiment_id=experiment_id,
            template_id=template_id,
            github_url=github_url,
            variants=variant_list,
            seeds=seeds,
            pr_number=pr_number,
            pr_url=pr_url,
        )
    except Exception as e:
        click.echo(f"❌ Failed to start validation run: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
        return

    run = result.get("run") or {}
    run_id = run.get("id") or result.get("run_id")
    click.echo(f"\n🧪  Validation run started")
    click.echo(f"   run_id:      {run_id}")
    click.echo(f"   status:      {result.get('status') or run.get('status')}")
    click.echo(f"   variants:    {len(variant_list)} × {seeds} seed(s)")
    click.echo(
        f"\n   Poll:  remyxai experiments validate-status {run_id}\n"
    )


# ── validate-status ──────────────────────────────────────────────────────────

def handle_experiments_validate_status(
    run_id: str,
    output_format: str,
) -> None:
    try:
        result = get_validation_run(run_id)
    except Exception as e:
        click.echo(f"❌ Failed to fetch run: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
        return

    run = result.get("run") or {}
    click.echo(f"\n🧪  Run {run_id}")
    click.echo(f"   status:            {run.get('status')}")
    if run.get("verdict"):
        click.echo(f"   verdict:           {run.get('verdict')}")
    click.echo(
        f"   results:           "
        f"{result.get('results_received', 0)}/{result.get('results_expected', '?')}"
    )
    click.echo()
