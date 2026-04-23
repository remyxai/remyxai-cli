"""
CLI action handlers for the Projects group.
Called by the `remyxai projects` commands in commands.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click

from remyxai.api.projects import (
    configure_eval_template,
    get_project,
    list_projects,
    set_decision_policy,
)


def _load_json_file(path: str) -> dict:
    """Read + parse a JSON file, emit a clean CLI error on failure."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        click.echo(f"❌ File not found: {path}", err=True)
        sys.exit(1)
    except json.JSONDecodeError as e:
        click.echo(f"❌ Invalid JSON in {path}: {e}", err=True)
        sys.exit(1)

    if not isinstance(data, dict):
        click.echo(f"❌ {path} must contain a JSON object", err=True)
        sys.exit(1)
    return data


# ── list ─────────────────────────────────────────────────────────────────────

def handle_projects_list(
    team_id: Optional[str],
    output_format: str,
) -> None:
    try:
        projects = list_projects(team_id=team_id)
    except Exception as e:
        click.echo(f"❌ Failed to list projects: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(projects, indent=2))
        return

    if not projects:
        click.echo("\n  No projects found.\n")
        return

    click.echo(f"\n📁  Projects  ({len(projects)})")
    click.echo("━" * 60)
    for p in projects:
        click.echo(f"  {p.get('name', '(unnamed)')}")
        click.echo(f"       id:          {p['id']}")
        if p.get("description"):
            click.echo(f"       desc:        {p['description']}")
    click.echo()


# ── get ──────────────────────────────────────────────────────────────────────

def handle_projects_get(
    project_id: str,
    output_format: str,
) -> None:
    try:
        project = get_project(project_id)
    except Exception as e:
        click.echo(f"❌ Failed to fetch project: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(project, indent=2))
        return

    click.echo(f"\n📁  {project.get('name', '(unnamed)')}")
    click.echo("━" * 60)
    click.echo(f"  id:          {project['id']}")
    if project.get("description"):
        click.echo(f"  desc:        {project['description']}")
    config = project.get("config") or {}
    templates = list((config.get("eval_templates") or {}).keys())
    policies = list((config.get("decision_policies") or {}).keys())
    click.echo(f"  templates:   {', '.join(templates) if templates else '(none)'}")
    click.echo(f"  policies:    {', '.join(policies) if policies else '(none)'}")
    click.echo()


# ── configure-eval ───────────────────────────────────────────────────────────

def handle_projects_configure_eval(
    project_id: str,
    template_name: str,
    template_file: str,
    output_format: str,
) -> None:
    """Upsert an eval template loaded from a JSON file."""
    template = _load_json_file(template_file)

    try:
        result = configure_eval_template(
            project_id=project_id,
            template_name=template_name,
            template=template,
        )
    except Exception as e:
        click.echo(f"❌ Failed to upsert eval template: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
        return

    click.echo(
        f"\n✅  Eval template '{template_name}' saved on project "
        f"{project_id}\n"
    )


# ── set-policy ───────────────────────────────────────────────────────────────

def handle_projects_set_policy(
    project_id: str,
    policy_name: str,
    policy_file: str,
    output_format: str,
) -> None:
    """Upsert a decision policy loaded from a JSON file."""
    policy = _load_json_file(policy_file)

    try:
        result = set_decision_policy(
            project_id=project_id,
            policy_name=policy_name,
            policy=policy,
        )
    except Exception as e:
        click.echo(f"❌ Failed to upsert decision policy: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
        return

    click.echo(
        f"\n✅  Decision policy '{policy_name}' saved on project "
        f"{project_id}\n"
    )
