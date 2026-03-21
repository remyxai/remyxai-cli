"""
remyxai/cli/interest_actions.py

CLI action handlers for Research Interest management.
Called by the `remyxai interests` command group in commands.py.
"""
from __future__ import annotations

import json
import sys
import textwrap
from typing import Optional

import click

from remyxai.api.interests import (
    create_interest,
    delete_interest,
    get_interest,
    list_interests,
    toggle_interest,
    update_interest,
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

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
        return

    click.echo(f"\n✅  Created '{result['name']}'  (id: {result['id']})")
    click.echo(
        f"   daily_count: {result['daily_count']}  |  "
        f"active: {result['is_active']}\n"
    )
    click.echo(
        "  Trigger your first recommendations:\n"
        f"  remyxai papers refresh --interest-id {result['id']} --wait\n"
    )


# ─── update ──────────────────────────────────────────────────────────────────

def handle_interests_update(
    interest_id: str,
    name: Optional[str],
    context: Optional[str],
    daily_count: Optional[int],
    is_active: Optional[bool],
    output_format: str,
) -> None:
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
            f"   Run:  remyxai papers refresh --interest-id {interest_id} --wait"
        )
    click.echo()


# ─── delete ──────────────────────────────────────────────────────────────────

def handle_interests_delete(
    interest_id: str,
    yes: bool,
    output_format: str,
) -> None:
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
