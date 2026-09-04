"""
CLI action handlers for Outrider lifecycle management.

`remyxai outrider init` sets up Outrider on a GitHub repo by driving the
Remyx engine — the same server-side "set it up for me" flow as the web app.
The engine, via the Remyx GitHub App (remyx-ai[bot]), sets the repo secrets,
writes the workflow, opens a bot-authored setup PR, and (in `auto` mode)
merges it and fires the first run.

Nothing touches the user's local git, and the ordinary install needs only the
user's REMYX_API_KEY. Flow:

  1. Resolve the target repo (owner/name).
  2. Resolve the ResearchInterest (provided UUID, auto-created, or prompted).
  3. Resolve the tier config + how each tier's provider key reaches the repo,
     and refuse now if one can't (see ProviderKeyPlan).
  4. Ensure the Remyx GitHub App is installed on the repo (surface the install
     link + poll — installing is an interactive browser step).
  5. Ensure a model provider is connected, and push the secrets for any tier
     provider the engine won't cover (this is the one step that uses `gh`).
  6. Kick off provisioning and report the bot-authored setup PR.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import NamedTuple, Optional

import click

from remyxai.api import BASE_URL, DEFAULT_BASE_URL
from remyxai.api.interests import (
    get_interest,
    provision_action,
    poll_provision_action,
)
from remyxai.cli.interest_actions import (
    RepoAnalysisError,
    create_interest_from_repo,
    _kick_off_recommendations,
)
from remyxai.api.integrations import connect_credential, get_integration_status
from remyxai.api.github_app import (
    get_app_install_url,
    get_app_installation,
    is_app_installed,
)

logger = logging.getLogger(__name__)

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# Model providers, in the engine's connection-priority order. Each maps its
# integration id (used by the integrations API) to the workflow provider value
# the Outrider action + engine phases understand. All providers are equal —
# this order is only the fallback used when the caller doesn't name one. The
# engine's MODEL_PROVIDERS registry is the source of truth.
MODEL_PROVIDERS = [
    ("claude_code", "anthropic"),   # Claude Code (Anthropic)
    ("zai", "zai"),                 # Z.ai (GLM)
    ("moonshot", "moonshot"),       # Moonshot AI (Kimi)
]
# `claude_code` is the id the inline --anthropic-key / $ANTHROPIC_API_KEY
# shortcut connects; it is not a preferred provider, just the one with a
# key flag on this command today.
ANTHROPIC_INTEGRATION = "claude_code"

# The `--provider` / `--drafter-provider` / `--refiner-provider` choice list.
# Derived from MODEL_PROVIDERS so adding a provider in one place reaches every
# call site (the four hand-maintained click.Choice lists this replaced were
# how `moonshot` ended up accepted by `init` but rejected by
# `set-provider-secret`).
PROVIDER_CHOICES = [workflow_value for _, workflow_value in MODEL_PROVIDERS]

# Integration id (integrations API / the engine's `model_provider` field) for
# a workflow provider value.
PROVIDER_INTEGRATION_IDS = {
    workflow_value: integration_id
    for integration_id, workflow_value in MODEL_PROVIDERS
}

# Maps a provider name (matching the workflow's `provider` input choices) to
# the GitHub Actions secret name the workflow's `Configure provider auth` step
# reads. Each provider has a conventional env var name; we mirror that
# convention here — and read the same names from the environment when `init`
# has to push a tier's key itself — so customers don't have to look it up.
#
# Twin of ``outrider_local._BACKEND_REGISTRY[*]["secret_env"]``; kept separate
# only because that module imports this one (no cycle allowed). A test asserts
# the two agree.
# The user-facing name of the "keys live only in GitHub Actions secrets"
# flag. Held in one place so the messages that teach it can't drift from the
# option that implements it (commands.outrider_init).
BYOK_FLAG = "--github-secrets-only"

_PROVIDER_SECRET_NAMES = {
    "anthropic": "ANTHROPIC_API_KEY",
    "zai": "ZAI_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
}

INSTALL_POLL_INTERVAL = 5     # seconds between App-install checks
INSTALL_POLL_TIMEOUT = 300    # stop waiting for the browser install after 5 min
PROVISION_POLL_INTERVAL = 3
PROVISION_POLL_TIMEOUT = 300
# `trigger --wait-for-slot`: a queued Outrider run usually starts within a
# minute or two, but a busy Actions account can hold it much longer.
QUEUE_POLL_INTERVAL = 15
QUEUE_POLL_TIMEOUT = 900
# GitHub's per-dispatch input ceiling (workflow_dispatch accepts at most 10
# top-level inputs) and a conservative bound on a single input's size — the
# whole payload has to stay under ~64KB.
GH_MAX_DISPATCH_INPUTS = 10
LEAD_CONTENT_MAX_CHARS = 60000


# ─── repo resolution (read-only; never mutates the working tree) ───────────

def _normalize_repo(value: str) -> Optional[str]:
    """Accept owner/name, an https URL, or an ssh URL → 'owner/name'."""
    value = (value or "").strip()
    if value.endswith(".git"):
        value = value[:-4]
    for pat in (
        r"^git@github\.com:([\w.-]+/[\w.-]+)$",
        r"^https?://github\.com/([\w.-]+/[\w.-]+)$",
        r"^([\w.-]+/[\w.-]+)$",
    ):
        m = re.match(pat, value)
        if m:
            return m.group(1)
    return None


def _detect_github_repo_from_cwd() -> Optional[str]:
    """Parse `origin` of the cwd's git repo → 'owner/name', or None."""
    try:
        out = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return _normalize_repo(out)


# ─── interest resolution ───────────────────────────────────────────────────

def _resolve_interest_id(interest_id, auto_interest, repo, repo_url, api_key):
    """Resolve to a validated interest UUID via flag / auto-create / prompt."""
    if interest_id:
        if not UUID_RE.match(interest_id):
            raise click.UsageError(
                f"--interest must be a UUID, got: {interest_id!r}"
            )
        try:
            get_interest(interest_id, api_key=api_key)
        except Exception as e:
            raise click.ClickException(
                f"interest {interest_id} could not be fetched from "
                f"engine.remyx.ai: {e}. Check the UUID and your REMYX_API_KEY."
            )
        return interest_id

    if auto_interest:
        click.echo(
            "Creating a Research Interest from this repo (may take 30-90s)…"
        )
        # Use the analyze-repo flow so the interest gets a rich,
        # ExperimentHistory-derived context (and the server dispatches
        # extraction) instead of a URL-only stub. Paper-PR
        # provisioning is handled separately below by `outrider init`,
        # so we don't provision here (automate="none").
        try:
            created = create_interest_from_repo(
                repo_url,
                name=repo.split("/")[-1],
                daily_count=3,
                is_active=True,
                automate="none",
                api_key=api_key,
                echo=click.echo,
            )
        except RepoAnalysisError as e:
            raise click.ClickException(
                f"interest creation failed during repo analysis: {e}\n"
                f"  Try again, or create one at engine.remyx.ai and re-run "
                f"with --interest <uuid>."
            )
        except Exception as e:
            raise click.ClickException(
                f"interest creation failed: {e}\n"
                f"  Create one at engine.remyx.ai and re-run with "
                f"--interest <uuid>."
            )
        new_id = created.get("id")
        if not new_id or not UUID_RE.match(new_id):
            raise click.ClickException(
                f"interest creation did not return a UUID: {created}"
            )
        click.echo(f"✓ Created interest: {new_id}")
        if created.get("history_extraction_task_id"):
            click.echo(
                "  🧪 experiment-history extraction dispatched; "
                "interest context will keep deepening as it completes."
            )
        # The profile is derived from a recency-weighted commit sample, so on a
        # large multi-domain repo it can describe the last few months of work
        # as if it were the whole repo — and nothing downstream fails, the
        # recommendations just skew. Worth 20 seconds of the operator's eyes.
        click.echo(
            f"  ⓘ Skim the generated profile before the first run — on a "
            f"large multi-domain repo it can under-represent subsystems that "
            f"haven't changed lately:\n"
            f"      remyxai interests get -i {new_id}\n"
            f"      remyxai interests update -i {new_id} -c \"<accurate "
            f"description>\"   # then: remyxai papers refresh"
        )
        return new_id

    typed = click.prompt("Remyx interest UUID (from engine.remyx.ai)").strip()
    if not UUID_RE.match(typed):
        raise click.UsageError(f"interest UUID is malformed: {typed!r}")
    return typed


# ─── preflight helpers ─────────────────────────────────────────────────────

def _install_action_link(repo, api_key):
    """(url, instruction, needs_click) for getting the App onto `repo`.

    Three states look identical from here, and telling them apart is the whole
    point:
      - App installed nowhere            → the install link
      - installed, repo not in its set   → the installation's repo-access page
                                           (the install link is a no-op then,
                                           which is what produces "I already
                                           installed it" + a timeout)
      - installed on all repos           → nothing to click; GitHub is still
                                           propagating a just-created repo, or
                                           the repo isn't visible to the App

    ``needs_click`` is False for that last state — telling someone to grant
    access they've already granted is how the timeout got misread as an
    install problem in the first place.
    """
    try:
        status = get_app_installation(repo, api_key=api_key)
    except Exception:  # status is a nicety; never block the install on it
        status = {}

    if status.get("account_installed") and status.get("manage_url"):
        account = status.get("account") or repo.split("/", 1)[0]
        reason = status.get("reason")
        if reason == "suspended":
            return status["manage_url"], (
                f"The Remyx GitHub App is installed on {account} but is "
                f"suspended. Unsuspend it, then come back here:"
            ), True
        if reason == "repo_unavailable":
            # All-repos install: nothing for the user to click. Usually a
            # just-created fork GitHub hasn't propagated into the
            # installation yet, which resolves on its own within a minute.
            return status["manage_url"], (
                f"The Remyx GitHub App is installed on {account} with access "
                f"to all repositories, so {repo} should be covered — GitHub "
                f"may still be catching up on a just-created repo. If this "
                f"doesn't clear, check that {repo} exists and is visible to "
                f"the App:"
            ), False
        return status["manage_url"], (
            f"The Remyx GitHub App is installed on {account}, but {repo} "
            f"isn't in the repos it can access. Add it here "
            f"(Repository access → select {repo} → Save):"
        ), True

    info = get_app_install_url(api_key=api_key)
    if not info.get("configured", True) or not info.get("install_url"):
        raise click.ClickException(
            "The Remyx GitHub App isn't configured on the server. "
            "Contact Remyx support."
        )
    return info["install_url"], (
        "Action needed — install the Remyx GitHub App on this repo:"
    ), True


def _ensure_app_installed(repo, api_key, no_wait, sleep=time.sleep):
    """Confirm the Remyx App is installed on `repo`; otherwise surface the
    right link and poll until it is (installing is a browser step)."""
    if is_app_installed(repo, api_key=api_key):
        click.echo(f"✓ Remyx GitHub App is installed on {repo}")
        return

    url, instruction, needs_click = _install_action_link(repo, api_key)
    click.echo("")
    click.secho(instruction, fg="yellow", bold=True)
    click.echo(f"  {url}")
    if needs_click:
        click.echo("  (grant it access to the repo, then come back here)")
    if no_wait:
        raise click.ClickException(
            f"App access to {repo} isn't live yet. {instruction}\n  {url}\n"
            f"  Then re-run."
        )
    click.echo(
        "\nWaiting for the install to complete…" if needs_click
        else "\nWaiting for GitHub to report access…"
    )
    waited = 0
    while waited < INSTALL_POLL_TIMEOUT:
        sleep(INSTALL_POLL_INTERVAL)
        waited += INSTALL_POLL_INTERVAL
        if is_app_installed(repo, api_key=api_key):
            click.echo(f"✓ Remyx GitHub App is now installed on {repo}")
            return
    raise click.ClickException(
        f"Timed out after {INSTALL_POLL_TIMEOUT}s waiting for App access to "
        f"{repo}.\n  {instruction}\n  {url}\n"
        f"  Re-run once that reads as granted — the check is live, so a "
        f"re-run picks it up immediately."
    )


def _connected_providers(api_key):
    """Workflow values of every connected model provider, in priority order.

    Empty when none is connected or status can't be read. Vendor-neutral:
    reports whatever the user actually connected, so a tier with no explicit
    provider follows the account's own setup rather than a hardcoded default.
    """
    connected = []
    for integration_id, workflow_value in MODEL_PROVIDERS:
        try:
            if get_integration_status(
                integration_id, api_key=api_key
            ).get("connected"):
                connected.append(workflow_value)
        except Exception:
            continue
    return connected


def _resolve_connected_provider(api_key):
    """Workflow value of the first connected model provider, or ``None``.

    The engine resolves the same way when no ``model_provider`` is named, so
    this is "the provider a tier follows when the caller doesn't pick one".
    """
    connected = _connected_providers(api_key)
    return connected[0] if connected else None


def _ensure_model_provider(anthropic_key, api_key, connected=None, byok=False):
    """Ensure *some* model provider is connected. Returns True if so.

    Any provider counts equally — Claude Code (Anthropic), Z.ai, or
    Moonshot AI. Non-fatal when absent: provisioning still proceeds, but the
    first run can't complete until a key is connected, so we warn loudly.

    ``connected`` accepts an already-resolved list from ``_connected_providers``
    so the init flow doesn't re-query integration status it just read.

    ``byok`` — never create a Remyx-side credential. Under --github-secrets-only the key in
    this shell is sealed to the repo instead, so the inline connect below
    (which is what puts a copy on Remyx's servers) must not run. Warn only.
    """
    connected = (
        _connected_providers(api_key) if connected is None else list(connected)
    )
    if connected:
        click.echo(f"✓ Model provider connected ({', '.join(connected)})")
        return True

    if byok:
        # The key plan already routed this shell's keys to the sealed lane;
        # a missing one is reported by _validate_provider_keys, not here.
        click.echo(
            f"  No Remyx-side provider credential ({BYOK_FLAG}) — the key is\n"
            "  sealed from this shell straight into the repo's GitHub secrets."
        )
        return False

    # No provider connected. The inline --anthropic-key / $ANTHROPIC_API_KEY
    # shortcut connects an Anthropic key (the only key flag on this command
    # today); any provider can otherwise be connected in Integrations.
    key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        click.secho(
            "⚠ No model provider connected. Provisioning will proceed, but the "
            "first run can't complete until you connect one at "
            "engine.remyx.ai/integrations — Claude Code (Anthropic), Z.ai, or "
            "Moonshot AI. Shortcut: pass --anthropic-key or set "
            "ANTHROPIC_API_KEY to connect an Anthropic key inline.",
            fg="yellow",
        )
        return False
    try:
        connect_credential(
            ANTHROPIC_INTEGRATION, {"api_key": key}, api_key=api_key
        )
    except Exception as e:
        raise click.ClickException(f"Failed to connect the provided key: {e}")
    click.echo("✓ Connected an Anthropic key")
    return True


# ─── tier provider keys ────────────────────────────────────────────────────
#
# A two-tier install can name a different provider per tier, but the engine
# pushes exactly ONE model-provider secret (the connected credential it
# resolves for the install — see resolve_model_provider_key). So a repo whose
# drafter runs at z.ai while the account has no z.ai credential provisions clean
# and then dies on its first run in a few hundred milliseconds
# ("ANTHROPIC_AUTH_TOKEN is not set").
#
# The engine (remyxai/remyx#558) pushes a key for EVERY provider named across
# `phases` that the account has connected, so a provider you've connected needs
# nothing from this machine. What's left for the CLI is the provider you haven't
# connected: its key comes from this shell, or the install is refused before
# anything is provisioned.

class ProviderKeyPlan(NamedTuple):
    """How each tier provider's API key reaches the repo.

    engine_providers: workflow values the engine pushes from connected
        credentials (every tier provider you've connected).
    preferred: the provider to name as ``model_provider`` — it decides which
        credential the engine bakes as the workflow's default.
    pushes: ``(provider, secret_name, key)`` this CLI must push with `gh` —
        only providers that aren't connected.
    sealed: ``(provider, secret_name, key)`` sealed on this machine and
        relayed to the repo through the engine (--github-secrets-only). Same tuple shape as
        ``pushes``; the difference is transport — no `gh`, and Remyx handles
        only ciphertext. Mutually exclusive with ``pushes`` for a given
        provider.
    missing: providers with no key anywhere — a hard failure.
    connected: the account's connected providers, for diagnosis.
    """
    engine_providers: tuple
    preferred: Optional[str]
    pushes: list
    missing: list
    connected: tuple = ()
    sealed: tuple = ()


def _provider_env_key(provider, anthropic_key=None):
    """This machine's API key for ``provider``, or ``None``.

    Read from the same env var the generated workflow reads as a repo secret
    (ANTHROPIC_API_KEY / ZAI_API_KEY / MOONSHOT_API_KEY), so "the key that
    works locally" is the key that gets installed. ``--anthropic-key`` wins
    for the anthropic provider.
    """
    if provider == "anthropic" and (anthropic_key or "").strip():
        return anthropic_key.strip()
    secret_name = _PROVIDER_SECRET_NAMES.get(provider)
    if not secret_name:
        return None
    return (os.environ.get(secret_name) or "").strip() or None


def _tier_providers(phases):
    """Distinct providers a phases config names, in tier order."""
    out = []
    for tier in ("drafter", "refiner", "main"):
        prov = ((phases or {}).get(tier) or {}).get("provider")
        if prov and prov not in out:
            out.append(prov)
    return out


def _plan_provider_secrets(phases, connected, anthropic_key=None, byok=False):
    """Route each tier provider's key to the repo; report what's unroutable.

    ``connected`` is the account's connected providers (workflow values, from
    ``_connected_providers``). When ``phases`` pins no provider, the tier list
    falls back to what the engine would bake into the workflow: the first
    connected provider, else anthropic.

    ``byok`` — a tier provider whose key is in this shell goes to the SEALED
    lane: encrypted here against the repo's Actions public key and relayed as
    ciphertext, so Remyx never holds it. Two consequences worth stating
    plainly: nothing gets connected inline (that is the copy --github-secrets-only exists to
    avoid), and a provider you have *already* connected still rides the engine
    lane when this shell has no key for it — --github-secrets-only declines to create new
    Remyx-side credentials, it doesn't disown the ones you chose to make.
    """
    tiers = _tier_providers(phases) or (
        [connected[0]] if connected else ["anthropic"]
    )
    env = {p: _provider_env_key(p, anthropic_key) for p in tiers}

    # Under --github-secrets-only this shell's keys are sealed, so they never fall to the
    # engine lane even for a provider that happens to be connected.
    sealed = (
        [(p, _PROVIDER_SECRET_NAMES[p], key)
         for p, key in env.items() if key]
        if byok else []
    )
    sealed_providers = {p for p, _, _ in sealed}

    # Every tier provider you've connected is covered server-side.
    engine_providers = [
        p for p in tiers if p in connected and p not in sealed_providers
    ]
    if (not byok and not engine_providers and not connected
            and env.get("anthropic")):
        # Nothing connected, but --anthropic-key / $ANTHROPIC_API_KEY gets
        # connected inline during preflight — the engine pushes that one.
        # Never under --github-secrets-only: that inline connect is the stored copy.
        engine_providers = ["anthropic"]

    # `model_provider` decides which credential the engine bakes as the
    # workflow's default, so name the capable tier (refiner/main) when it's
    # covered; otherwise the first covered tier.
    covered = set(engine_providers) | sealed_providers
    late_tiers = [p for p in reversed(tiers) if p in covered]
    preferred = late_tiers[0] if late_tiers else None

    pushes = [] if byok else [
        (p, _PROVIDER_SECRET_NAMES[p], key)
        for p, key in env.items() if key and p not in engine_providers
    ]
    missing = [
        p for p in tiers
        if not env[p] and p not in engine_providers and p not in sealed_providers
    ]
    return ProviderKeyPlan(
        tuple(engine_providers), preferred, pushes, missing, tuple(connected),
        tuple(sealed),
    )


def _describe_key_plan(plan):
    """One plan line per provider: how its key reaches the repo."""
    lines = []
    for provider in plan.engine_providers:
        default = " (workflow default)" if provider == plan.preferred else ""
        lines.append(
            f"{provider} — pushed by the engine from your connected "
            f"credential{default}"
        )
    for provider, secret_name, _ in plan.sealed:
        default = " (workflow default)" if provider == plan.preferred else ""
        lines.append(
            f"{provider} — {secret_name} sealed here → GitHub secret on the "
            f"repo only; Remyx can't read it{default}"
        )
    for provider, secret_name, _ in plan.pushes:
        lines.append(
            f"{provider} — {secret_name} from this shell → repo secret "
            f"(not connected server-side)"
        )
    for provider in plan.missing:
        lines.append(
            f"{provider} — NO KEY ({_PROVIDER_SECRET_NAMES[provider]} unset, "
            f"not connected)"
        )
    return lines


def _validate_provider_keys(plan, skip_key_check=False, byok=False):
    """Hard-fail on a tier whose provider has no key — at plan time.

    Provisioning succeeds regardless (it's just secrets + a workflow file), so
    without this the failure lands minutes later inside a GitHub run: the
    action's auth guard exits in ~260ms and every dispatch repeats it. Cheaper
    to refuse here.
    """
    if not plan.missing:
        return
    fixes = []
    for provider in plan.missing:
        secret_name = _PROVIDER_SECRET_NAMES[provider]
        how = ("seals it and relays it to the repo" if byok
               else "pushes it to the repo as a secret")
        fixes.append(f"    - export {secret_name}=… and re-run (init {how})")
    if byok:
        fixes.append(
            f"    - or drop {BYOK_FLAG} and connect the provider at "
            "engine.remyx.ai/integrations (that keeps a copy of the key on "
            f"Remyx, which is what {BYOK_FLAG} avoids)"
        )
    else:
        fixes.append(
            "    - connect the provider at engine.remyx.ai/integrations, then "
            "re-run (the engine pushes a key for every provider you've "
            "connected)"
        )
    fixes.append(
        "    - point that tier at a provider you have connected (--provider / "
        "--drafter-provider / --refiner-provider)"
    )
    message = (
        f"no API key for: {', '.join(plan.missing)}.\n"
        f"  The install would provision cleanly and then fail auth on its "
        f"first run, so init stops here. Fix any one of:\n"
        + "\n".join(fixes)
        + "\n  Or pass --skip-key-check to provision anyway."
    )
    if skip_key_check:
        click.secho(
            f"⚠ --skip-key-check: {message.splitlines()[0]} The first run "
            f"can't complete until those secrets are set.",
            fg="yellow",
        )
        return
    raise click.ClickException(message)


def _require_gh_for_pushes(pushes):
    """Refuse at plan time when a tier's key needs `gh` and `gh` can't push."""
    if not pushes:
        return
    from remyxai.cli.outrider_local import _gh_authenticated, _gh_available

    secrets = ", ".join(name for _, name, _ in pushes)
    if not _gh_available():
        raise click.ClickException(
            f"this install needs {secrets} pushed to the repo, which requires "
            f"the `gh` CLI (only the account's connected provider is pushed "
            f"server-side).\n"
            f"  Install gh (https://cli.github.com), or connect that provider "
            f"at engine.remyx.ai/integrations and use it for every tier."
        )
    if not _gh_authenticated():
        raise click.ClickException(
            f"this install needs {secrets} pushed to the repo, but `gh` isn't "
            f"authenticated. Run `gh auth login` (admin scope on the repo) and "
            f"re-run."
        )


def _seal_provider_secrets(interest_id, repo_url, sealed, api_key):
    """Seal each provider key against the repo's Actions public key.

    GitHub Actions secrets are libsodium sealed boxes and only GitHub holds
    the private half, so a key encrypted here is one Remyx is mathematically
    unable to read — it relays the ciphertext and keeps nothing. That is the
    whole point of --github-secrets-only: not "deleted after", but "never readable".

    Returns the ``sealed_provider_secrets`` payload for ``provision_action``.
    """
    if not sealed:
        return []
    try:
        from nacl import encoding, public
    except ImportError:
        raise click.ClickException(
            f"{BYOK_FLAG} needs PyNaCl to encrypt your key locally. Install it "
            "with `pip install pynacl` (or reinstall remyxai) and re-run."
        )
    from remyxai.api.interests import get_actions_public_key

    try:
        pk = get_actions_public_key(
            interest_id, repo_url=repo_url, api_key=api_key
        )
    except Exception as e:
        raise click.ClickException(
            f"could not read the repo's Actions public key, so there is "
            f"nothing to seal against: {e}"
        )
    sealable = set(pk.get("sealable_secret_names") or [])
    box = public.SealedBox(
        public.PublicKey(pk["key"].encode("utf-8"), encoding.Base64Encoder)
    )

    payload = []
    for provider, secret_name, key in sealed:
        if sealable and secret_name not in sealable:
            # The engine validates against its own provider registry; a name
            # it won't accept would fail mid-install with an opaque 400.
            raise click.ClickException(
                f"the engine won't accept {secret_name} as a sealed secret "
                f"(it accepts: {', '.join(sorted(sealable))}). Upgrade the "
                f"engine or drop {BYOK_FLAG} for {provider}."
            )
        if len(key) < _SECRET_MIN_LENGTH_WARN:
            click.secho(
                f"⚠ {secret_name} is {len(key)} chars — unusually short for "
                f"an API key. Sealing it anyway, but the action's auth guard "
                f"may reject it.",
                fg="yellow",
            )
        ciphertext = base64.b64encode(
            box.encrypt(key.encode("utf-8"))
        ).decode("utf-8")
        payload.append({
            "secret_name": secret_name,
            "key_id": pk["key_id"],
            "encrypted_value": ciphertext,
        })
        click.echo(
            f"  Sealed {secret_name} for {pk.get('repo') or repo_url} "
            f"(provider={provider}) — ciphertext only leaves this machine."
        )
    return payload


def _push_provider_secrets(repo, pushes):
    """Push the tier secrets `gh`-side, before provisioning dispatches a run."""
    from remyxai.cli.outrider_local import _gh_set_secret

    for provider, secret_name, key in pushes:
        if len(key) < _SECRET_MIN_LENGTH_WARN:
            click.secho(
                f"⚠ {secret_name} is {len(key)} chars — unusually short for an "
                f"API key. Proceeding, but the action's auth guard may reject "
                f"it.",
                fg="yellow",
            )
        click.echo(f"  Setting {secret_name} on {repo} (provider={provider})…")
        _gh_set_secret(repo, secret_name, key)


def _wait_for_provision(interest_id, task_id, api_key, sleep=time.sleep):
    """Poll the provision task until it completes or fails; return result."""
    waited = 0
    last_msg = None
    while waited < PROVISION_POLL_TIMEOUT:
        task = poll_provision_action(interest_id, task_id, api_key=api_key)
        status = task.get("status")
        msg = task.get("message")
        if msg and msg != last_msg:
            click.echo(f"  … {msg}")
            last_msg = msg
        if status == "completed":
            return task.get("result") or {}
        if status == "failed":
            raise click.ClickException(
                f"Provisioning failed: {task.get('error') or 'unknown error'}"
            )
        sleep(PROVISION_POLL_INTERVAL)
        waited += PROVISION_POLL_INTERVAL
    raise click.ClickException(
        f"Timed out after {PROVISION_POLL_TIMEOUT}s waiting for provisioning "
        f"(task {task_id}). It may still finish server-side."
    )


# ─── main handler ──────────────────────────────────────────────────────────

def _validate_init_tier_flags(single_tier, drafter_provider, drafter_model,
                              refiner_provider, refiner_model):
    """Fail fast on incompatible tier flags (before any work / on dry-run)."""
    if single_tier and any([drafter_provider, drafter_model,
                            refiner_provider, refiner_model]):
        raise click.UsageError(
            "--drafter-* / --refiner-* configure the two-tier setup; "
            "drop --single-tier to use them."
        )


def _build_init_phases(
    single_tier, provider, model,
    drafter_provider, drafter_model,
    refiner_provider, refiner_model,
    default_provider=None,
):
    """Translate the init tier flags into the engine ``phases`` config.

    Default (no flags): the two-tier setup — a daily *drafter* plus a weekly
    *refiner*. A tier's provider resolves in order: its own ``--drafter-*`` /
    ``--refiner-*`` flag → the shared ``--provider`` → ``default_provider``
    (the caller's connected provider). When none is known the phase leaves
    ``provider`` unset and the engine uses the connected provider. Every
    provider is equal here — none is hardcoded. ``single_tier`` opts out to
    the plain single-file workflow.

    Returns the ``phases`` dict (or ``None`` for a plain single-file install
    with no per-phase pins).
    """
    _validate_init_tier_flags(single_tier, drafter_provider, drafter_model,
                              refiner_provider, refiner_model)
    shared_provider = provider or default_provider

    def _tier(tier_provider, tier_model):
        prov = tier_provider or shared_provider
        cfg = {"model": tier_model if tier_model is not None else (model or "")}
        if prov:                       # omit → engine uses the connected provider
            cfg["provider"] = prov
        return cfg

    if single_tier:
        # Plain single-file. Only pin a phase when the caller named a
        # provider/model; otherwise send nothing and let the engine decide.
        if provider or model:
            return {"mode": "single", "main": _tier(None, None)}
        return None

    return {
        "mode": "two_tier",
        "drafter": _tier(drafter_provider, drafter_model),
        "refiner": _tier(refiner_provider, refiner_model),
    }


def handle_outrider_init(
    repo, interest_id, auto_interest, mode,
    anthropic_key, skip_confirm, dry_run, no_wait,
    single_tier=False, provider=None, model=None,
    drafter_provider=None, drafter_model=None,
    refiner_provider=None, refiner_model=None,
    force=False, skip_key_check=False, byok=False,
):
    """Set up Outrider on a repo via the Remyx engine. Called from
    commands.outrider_init.

    ``byok`` — seal this shell's provider keys against the repo's Actions
    public key and hand the engine only ciphertext, so no copy is stored on
    Remyx. For customers whose policy forbids giving model-provider keys to a
    third party.
    """
    if interest_id and auto_interest:
        raise click.UsageError(
            "--interest and --auto-interest are mutually exclusive."
        )

    # Two-tier is the default; single_tier opts out. Validate tier flags now
    # (fail fast, before any work); the phases config itself is built in step
    # 3, once the account's connected providers are known.
    _validate_init_tier_flags(single_tier, drafter_provider, drafter_model,
                              refiner_provider, refiner_model)

    # 1. API key (the only credential the CLI needs)
    api_key = os.environ.get("REMYXAI_API_KEY") or click.prompt(
        "REMYXAI_API_KEY (from engine.remyx.ai Settings)", hide_input=True
    )
    if not api_key.strip():
        raise click.ClickException("REMYXAI_API_KEY is required.")

    # 2. Resolve repo
    resolved_repo = _normalize_repo(repo) if repo else _detect_github_repo_from_cwd()
    if not resolved_repo:
        raise click.ClickException(
            "No GitHub repo specified or detected. Pass --repo owner/name."
        )
    repo_url = f"https://github.com/{resolved_repo}"

    # 3. Resolve the tier config + how each tier's provider key reaches the
    # repo. Both are read-only (integration status + the environment), so the
    # plan below — and --dry-run — show the real routing, and a tier with no
    # key anywhere fails before anything is provisioned.
    connected = [] if mode == "off" else _connected_providers(api_key)
    default_provider = provider or (connected[0] if connected else None)
    phases = _build_init_phases(
        single_tier, provider, model,
        drafter_provider, drafter_model, refiner_provider, refiner_model,
        default_provider=default_provider,
    )
    key_plan = _plan_provider_secrets(
        phases, connected, anthropic_key, byok=byok
    )

    # 4. Plan
    mode_desc = {
        "auto": "provision, merge the setup PR, and start the first run",
        "review": "provision and open a setup PR for you to review and merge",
        "off": "create the interest only (no provisioning)",
    }[mode]
    interest_desc = (
        f"use existing interest {interest_id}" if interest_id
        else "auto-create an interest from the repo" if auto_interest
        else "prompt for an interest UUID"
    )
    click.echo("")
    if single_tier:
        setup_desc = "single-file workflow"
    else:
        _fallback = default_provider or "connected provider"
        _d = drafter_provider or provider or _fallback
        _r = refiner_provider or provider or _fallback
        _dm = drafter_model or model
        _rm = refiner_model or model
        setup_desc = (
            f"two-tier (default) — drafter {_d}{':' + _dm if _dm else ''}"
            f" + refiner {_r}{':' + _rm if _rm else ''}, cron off"
        )
    click.echo("Plan:")
    click.echo(f"  - Repo:      {resolved_repo}")
    if BASE_URL != DEFAULT_BASE_URL:
        # Non-production engine (REMYXAI_API_URL). Say so — a test-server run
        # that looks identical to a production one is how a "why didn't
        # anything change" hour gets spent.
        click.echo(f"  - Engine:    {BASE_URL}  (REMYXAI_API_URL override)")
    click.echo(f"  - Interest:  {interest_desc}")
    click.echo(f"  - Mode:      {mode} — {mode_desc}")
    click.echo(f"  - Setup:     {setup_desc}")
    if mode != "off":
        for i, line in enumerate(_describe_key_plan(key_plan)):
            click.echo(f"  {'- Keys:     ' if i == 0 else '              '}{line}")
    if force:
        click.echo(
            "  - Force:     re-drive provisioning on an already-installed "
            "repo so the workflow files get rewritten with this tier config"
        )
    click.echo(
        "  - The engine installs everything server-side as remyx-ai[bot]; "
        "your local git is untouched."
    )
    click.echo("")

    # 4b. Refuse now if a tier's key can't reach the repo — provisioning
    # itself would succeed and the failure would surface one dispatch later.
    if mode != "off":
        _validate_provider_keys(
            key_plan, skip_key_check=skip_key_check, byok=byok
        )
        # Sealed keys never touch `gh` — they ride the provision-action body.
        _require_gh_for_pushes(key_plan.pushes)

    if dry_run:
        click.secho("dry-run: no changes made.", fg="yellow")
        return

    if not skip_confirm:
        click.confirm("Proceed?", abort=True, default=False)

    # 5. Resolve interest
    resolved_interest = _resolve_interest_id(
        interest_id, auto_interest, resolved_repo, repo_url, api_key
    )

    # 5b. Mode `off` stops here — interest only.
    if mode == "off":
        click.echo("")
        click.secho("✓ Interest ready.", fg="green", bold=True)
        click.echo(f"  Interest: {resolved_interest}")
        click.echo(
            f"  To provision later: remyxai outrider init "
            f"--repo {resolved_repo} --interest {resolved_interest}"
        )
        return

    # 6. Preflight: App install + model provider (only needed to provision)
    _ensure_app_installed(resolved_repo, api_key, no_wait)
    _ensure_model_provider(
        anthropic_key, api_key, connected=connected, byok=byok
    )

    # 6b. Push the tier provider secrets the engine won't. Done BEFORE
    # provisioning because `auto` mode fires the first run as its last step —
    # a secret that lands after the dispatch is a secret that missed the run.
    if key_plan.pushes:
        click.echo("\nSetting tier provider secrets on the repo…")
        _push_provider_secrets(resolved_repo, key_plan.pushes)

    # 6b-bis. Seal the BYOK keys. Done here (not at plan time) because the
    # public key is fetched per interest+repo, and the interest may only have
    # been created a step ago. The ciphertext rides the provision-action body
    # so the engine writes it in the same pass that sets REMYX_API_KEY.
    sealed_payload = []
    if key_plan.sealed:
        click.echo(
            "\nSealing provider keys into the repo's GitHub secrets…"
        )
        sealed_payload = _seal_provider_secrets(
            resolved_interest, repo_url, key_plan.sealed, api_key
        )

    # 6c. Pre-warm recommendations so the first run has picks to open a PR
    # from. A brand-new interest ranks asynchronously; firing the Outrider
    # first run before the pool populates makes it report "no recommendations"
    # (the cold-start race). Trigger a refresh now and — unless --no-wait —
    # block until the pool is populated before provisioning dispatches the run.
    click.echo("\nWarming up recommendations for the interest…")
    _kick_off_recommendations(
        resolved_interest,
        wait=(not no_wait),
        api_key=api_key,
        echo=click.echo,
    )

    # 7. Provision (server-side, bot-authored). `phases` + the key routing were
    # resolved in step 3 from the account's connected providers — no vendor is
    # hardcoded.
    auto_merge = (mode == "auto")
    click.echo("\nProvisioning Outrider via engine.remyx.ai…")
    resp = provision_action(
        resolved_interest, repo_url=repo_url,
        auto_merge=auto_merge, phases=phases, force=force,
        # Names which connected credential the engine bakes as the workflow's
        # default — it pushes a key for every connected tier provider either
        # way. Unset, it would take the first connected provider, which on a
        # mixed install isn't necessarily the capable tier.
        model_provider=PROVIDER_INTEGRATION_IDS.get(key_plan.preferred),
        sealed_provider_secrets=sealed_payload,
        api_key=api_key,
    )
    task_id = resp.get("task_id")
    if not task_id:
        raise click.ClickException(
            f"provision-action did not return a task_id: {resp}"
        )
    if no_wait:
        click.echo(
            f"  Provisioning started (task {task_id}); it runs server-side."
        )
        return

    result = _wait_for_provision(resolved_interest, task_id, api_key)

    # 8. Report
    click.echo("")
    click.secho("✓ Outrider is set up.", fg="green", bold=True)
    if result.get("pr_url"):
        label = "Setup PR (merged)" if result.get("merged") else "Setup PR"
        click.echo(f"  {label}: {result['pr_url']}")
    click.echo(
        f"  Repo secret REMYX_API_KEY: "
        f"{'set' if result.get('secret_set') else 'not set'}"
    )
    if result.get("dispatched"):
        click.echo(
            "  First run: dispatched — a recommendation PR will appear shortly."
        )
    elif result.get("merged"):
        click.echo(
            "  First run: starts on schedule (or once recommendations populate)."
        )
    else:
        click.echo("  Next: merge the setup PR to activate Outrider.")
    for provider, secret_name, _ in key_plan.sealed:
        click.echo(
            f"  Repo secret {secret_name}: set from your sealed key "
            f"(provider={provider}) — Remyx holds no copy"
        )
    for provider, secret_name, _ in key_plan.pushes:
        click.echo(f"  Repo secret {secret_name}: set (provider={provider})")
    if (result.get("model_key_missing")
            and not key_plan.pushes and not key_plan.sealed):
        click.secho(
            "  ⚠ No model provider key set — connect a provider at "
            "engine.remyx.ai/integrations (or export its API key and re-run) "
            "so the first run can complete.", fg="yellow",
        )
    if result.get("already_provisioned"):
        click.secho(
            "  ⓘ This repo was already provisioned, so the workflow files were "
            "left as they are. Re-run with --force to rewrite them with this "
            "tier config.", fg="yellow",
        )


# ─── bulk-repos onboarding ────────────────────────────────────────────────

def _parse_bulk_repos_tsv(path: str) -> list:
    """Read a TSV mapping repos to ResearchInterest UUIDs.

    Format: one row per repo, two tab-separated columns:

        owner/name<TAB>interest-uuid

    Blank lines and ``#``-prefixed comments are skipped. Lines with the
    wrong column count or a malformed UUID are surfaced with their line
    number so the caller can fix-then-retry without partial state.

    Returns: list of ``(repo, interest_uuid)`` tuples in file order.
    """
    if not os.path.exists(path):
        raise click.UsageError(f"--bulk-repos file not found: {path}")
    rows = []
    errors = []
    with open(path) as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.rstrip("\n").rstrip("\r")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                errors.append(
                    f"line {line_no}: expected 2 tab-separated columns "
                    f"(repo<TAB>interest_uuid), got {len(parts)}: {line!r}"
                )
                continue
            repo_raw, uuid_raw = parts[0].strip(), parts[1].strip()
            repo = _normalize_repo(repo_raw)
            if not repo:
                errors.append(
                    f"line {line_no}: not a valid GitHub repo: {repo_raw!r}"
                )
                continue
            if not UUID_RE.match(uuid_raw):
                errors.append(
                    f"line {line_no}: not a valid UUID: {uuid_raw!r}"
                )
                continue
            rows.append((repo, uuid_raw))
    if errors:
        raise click.UsageError(
            "--bulk-repos parse errors:\n  " + "\n  ".join(errors)
        )
    if not rows:
        raise click.UsageError(
            f"--bulk-repos file {path!r} contains no installable rows."
        )
    return rows


def _run_bulk(
    handler, rows, common_kwargs, pace_s=3,
    echo=click.echo,
):
    """Run ``handler(repo=..., interest_id=..., **common_kwargs)`` per row.

    Per-row exceptions are captured and reported in a summary at the end —
    one failure does not abort the remaining rows, since the most common
    error class (an already-installed fork, a permission edge case) is
    independent across repos.

    Returns: list of ``(repo, status)`` tuples where status is ``"ok"`` or
    the exception message. Caller can post-filter for retry.
    """
    import time

    results = []
    for i, (repo, uuid) in enumerate(rows, start=1):
        echo(f"\n── [{i}/{len(rows)}] {repo} ──")
        try:
            handler(repo=repo, interest_id=uuid, **common_kwargs)
            results.append((repo, "ok"))
        except click.ClickException as e:
            echo(f"  ✗ {e.message}")
            results.append((repo, e.message))
        except Exception as e:
            echo(f"  ✗ {type(e).__name__}: {e}")
            results.append((repo, f"{type(e).__name__}: {e}"))
        if i < len(rows) and pace_s > 0:
            time.sleep(pace_s)

    # Summary
    ok = [r for r, s in results if s == "ok"]
    failed = [(r, s) for r, s in results if s != "ok"]
    echo("")
    click.secho(f"== summary: {len(ok)}/{len(rows)} ok ==", bold=True)
    for repo, msg in failed:
        echo(f"  ✗ {repo}: {msg}")
    return results


# ─── outrider trigger ─────────────────────────────────────────────────────

WORKFLOW_FILENAME = "outrider.yml"


def _gh_default_branch(repo: str) -> Optional[str]:
    """Return the repo's default branch via `gh api`, or None on failure."""
    try:
        out = subprocess.check_output(
            ["gh", "api", f"/repos/{repo}", "--jq", ".default_branch"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out or None


def _outrider_workflow_exists(repo: str) -> bool:
    """Return True iff the repo has an Outrider workflow registered.

    Probes the Actions API rather than the Contents API: the Contents
    endpoint reports the file as soon as it's committed, but the Actions
    endpoint only knows about a workflow after GitHub has indexed it for
    dispatch — which is the precise capability we need for the trigger
    call to succeed.
    """
    r = subprocess.run(
        ["gh", "api", f"/repos/{repo}/actions/workflows/{WORKFLOW_FILENAME}",
         "--silent"],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def _gh_dispatch_outrider(repo, branch, inputs):
    """Dispatch the Outrider workflow with the supplied inputs.

    Uses ``gh workflow run`` (not raw ``gh api``) because workflow_dispatch
    expects inputs nested under ``inputs.*`` in the request body. The raw
    POST endpoint accepts ``ref`` and ``inputs`` at the top level and
    rejects any extra top-level keys with ``HTTP 422 "X is not a permitted
    key"``; ``gh workflow run`` handles that wrapping for us.

    Returns (ok, stderr) so the caller can map errors to user-facing hints.
    """
    args = [
        "gh", "workflow", "run", WORKFLOW_FILENAME,
        "--repo", repo, "--ref", branch,
    ]
    for k, v in inputs.items():
        if v is None or v == "":
            continue
        args.extend(["-f", f"{k}={v}"])
    r = subprocess.run(args, capture_output=True, text=True)
    return (r.returncode == 0, (r.stderr or "").strip())


_UNEXPECTED_INPUTS_RE = re.compile(r'"([^"]+)"')


def _dispatch_with_input_fallback(repo, branch, inputs):
    """Dispatch, dropping inputs the installed workflow doesn't declare.

    GitHub rejects a dispatch naming an undeclared input with
    ``HTTP 422 Unexpected inputs provided: ["mode", "lead-content"]``. Repos
    installed before an input existed would dead-end there, so — mirroring the
    engine's own trigger endpoint — drop exactly the inputs GitHub named and
    retry once. The workflow's baked defaults apply for the dropped ones.

    Returns ``(ok, stderr, dropped)``.
    """
    ok, stderr = _gh_dispatch_outrider(repo, branch, inputs)
    if ok or "Unexpected inputs" not in (stderr or ""):
        return ok, stderr, []
    undeclared = {
        name for name in _UNEXPECTED_INPUTS_RE.findall(stderr)
        if name in inputs
    }
    if not undeclared:
        return ok, stderr, []
    pruned = {k: v for k, v in inputs.items() if k not in undeclared}
    ok, stderr = _gh_dispatch_outrider(repo, branch, pruned)
    return ok, stderr, sorted(undeclared)


# Run statuses that mean "this run has not started yet". The generated
# workflows use a static concurrency group (`group: outrider`) with
# cancel-in-progress: false, which allows exactly ONE pending run — so
# dispatching while another run is pending silently cancels the older one.
_PENDING_RUN_STATUSES = ("queued", "waiting", "pending", "requested")


def _gh_pending_runs(repo):
    """Runs of the dispatch target that haven't started yet.

    Returns a list of ``{"status", "url"}`` dicts (empty when gh fails — this
    is an advisory check, never a reason to block a dispatch).
    """
    try:
        out = subprocess.check_output(
            ["gh", "run", "list", "--repo", repo,
             "--workflow", WORKFLOW_FILENAME, "--limit", "20",
             "--json", "status,url"],
            text=True, stderr=subprocess.DEVNULL,
        )
        runs = json.loads(out or "[]")
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return []
    return [r for r in runs if r.get("status") in _PENDING_RUN_STATUSES]


def _warn_or_wait_for_queue(repo, wait, sleep=time.sleep,
                            interval=QUEUE_POLL_INTERVAL,
                            timeout=QUEUE_POLL_TIMEOUT):
    """Handle a run that's already pending on the repo before dispatching.

    Without ``wait``, warn: the new dispatch will silently cancel the pending
    run (the concurrency group holds one). With ``wait``, poll until the queue
    drains so a batch of dispatches serializes instead of eating itself.
    """
    pending = _gh_pending_runs(repo)
    if not pending:
        return
    if not wait:
        click.secho(
            f"⚠ {len(pending)} Outrider run(s) already pending on {repo}. The "
            f"workflow's concurrency group holds one pending run, so this "
            f"dispatch will cancel the queued one (it looks like a flake, not "
            f"an error). Pass --wait-for-slot to serialize instead.",
            fg="yellow",
        )
        for run in pending:
            click.echo(f"    pending: {run.get('url')}")
        return

    click.echo(
        f"Waiting for {len(pending)} pending run(s) on {repo} to start "
        f"(--wait-for-slot)…"
    )
    waited = 0
    while waited < timeout:
        sleep(interval)
        waited += interval
        pending = _gh_pending_runs(repo)
        if not pending:
            click.echo(f"  ✓ Queue clear after {waited}s.")
            return
    raise click.ClickException(
        f"still {len(pending)} pending run(s) on {repo} after {timeout}s. "
        f"Dispatching now would cancel one — re-run later, or drop "
        f"--wait-for-slot to dispatch anyway."
    )


def _gh_latest_run_url(repo, sleep=time.sleep):
    """Best-effort lookup of the most recent outrider.yml run URL."""
    for _ in range(5):
        try:
            url = subprocess.check_output(
                ["gh", "run", "list", "--repo", repo,
                 "--workflow", WORKFLOW_FILENAME, "--limit", "1",
                 "--json", "url", "--jq", ".[0].url"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            if url:
                return url
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        sleep(2)
    return None


def _resolve_lead_content(lead_content, lead_content_file):
    """Inline markdown for the run's leading context, from a flag or a file."""
    if lead_content and lead_content_file:
        raise click.UsageError(
            "--lead-content and --lead-content-file are mutually exclusive."
        )
    if lead_content_file:
        path = Path(lead_content_file)
        if not path.is_file():
            raise click.UsageError(
                f"--lead-content-file is not a file: {lead_content_file}"
            )
        lead_content = path.read_text()
    if not lead_content:
        return ""
    if len(lead_content) > LEAD_CONTENT_MAX_CHARS:
        raise click.UsageError(
            f"lead content is {len(lead_content)} chars; GitHub caps a "
            f"workflow_dispatch payload at ~64KB, so keep it under "
            f"{LEAD_CONTENT_MAX_CHARS}. Trim the gap analysis, or commit it to "
            f"the branch and reference it from a shorter brief."
        )
    return lead_content


def handle_outrider_trigger(
    repo, search_method, pin_arxiv, interest_id, ref, claude_timeout=None,
    provider=None, model=None, base_url=None, mode=None, publish=None,
    start_from_ref=None, lead_content=None, lead_content_file=None,
    staged_synthesis=False, test_integration_policy=None,
    fidelity_policy=None, wait_for_slot=False,
):
    """Dispatch a one-shot Outrider run on a repo via workflow_dispatch.

    Called from `commands.outrider_trigger`. Mirrors `outrider setup-local`
    by using the user's authenticated `gh` to POST the dispatch — no Remyx
    engine round-trip, no Remyx App requirement. The repo must already
    have an Outrider workflow installed (set up via `remyxai outrider init`
    or `setup-local`).

    ``claude_timeout`` (seconds) overrides the action's default 900s
    implementation-call ceiling on a per-dispatch basis. Useful for very
    large monorepos where the default trips before the agent completes
    (especially when routing at slower non-Anthropic backends).

    The refinement inputs — ``mode``, ``publish``, ``start_from_ref``,
    ``lead_content``/``lead_content_file``, ``staged_synthesis``,
    ``test_integration_policy``, ``fidelity_policy`` — reach the same workflow
    inputs the two-tier refiner uses, so building on an existing branch (a
    stalled third-party PR, yesterday's drafter output) or running a
    brief-mode pass no longer means dropping to raw `gh workflow run`.
    """
    # Inputs validation
    if search_method and pin_arxiv:
        raise click.UsageError(
            "--search-method and --pin-arxiv are mutually exclusive."
        )
    if claude_timeout is not None and claude_timeout < 60:
        raise click.UsageError(
            "--claude-timeout must be at least 60 seconds (a tighter value "
            "trips before the agent can finish even a small task)."
        )
    lead = _resolve_lead_content(lead_content, lead_content_file)

    # Repo resolution
    resolved_repo = _normalize_repo(repo) if repo else _detect_github_repo_from_cwd()
    if not resolved_repo:
        raise click.UsageError(
            "Could not determine target repo. Pass --repo owner/name or run "
            "from inside a GitHub-origin git checkout."
        )

    # Pre-flight: refuse to dispatch on repos that aren't initialized.
    # Surfaces a clear install hint instead of a generic 404 from the
    # workflow_dispatch call below — the error is structurally about the
    # repo's setup state, not the dispatch attempt itself.
    if not _outrider_workflow_exists(resolved_repo):
        raise click.ClickException(
            f"Outrider is not installed on {resolved_repo}. Install it "
            f"first:\n"
            f"  remyxai outrider init --repo {resolved_repo}\n"
            f"or, if your org can't grant the Remyx GitHub App yet:\n"
            f"  remyxai outrider setup-local --repo {resolved_repo}"
        )

    # Ref defaults to the repo's default branch (so trigger works without
    # the user knowing whether the repo's default is main, master, develop).
    branch = ref or _gh_default_branch(resolved_repo)
    if not branch:
        raise click.ClickException(
            f"Could not resolve a ref for {resolved_repo}. Pass --ref "
            f"explicitly (e.g. --ref main)."
        )

    inputs = {
        "search-method": search_method or "",
        "pin-arxiv": pin_arxiv or "",
        "interest-id": interest_id or "",
        # Forward as a string — workflow_dispatch input values are
        # always strings on the wire. The action's INPUT_CLAUDE_TIMEOUT
        # parser handles the int conversion (and validates it).
        "claude-timeout": str(claude_timeout) if claude_timeout else "",
        # The target workflow must declare `provider` + `model` as
        # workflow_dispatch inputs for these to take effect. The
        # current CLI-generated template does; older templates and
        # hand-rolled workflows may need updating.
        "provider": provider or "",
        "model": model or "",
        # `base-url` routes the coding agent at a self-hosted or on-prem
        # Anthropic-compatible endpoint (litellm proxy, vLLM shim,
        # Cloudflare Access, etc). Overrides the per-provider default.
        "base-url": base_url or "",
        # Refinement inputs. `--ref` picks which branch the *workflow file*
        # comes from; `start-from-ref` is what the agent builds ON — the two
        # are not interchangeable. `publish=branch` produces the drafter
        # branch without opening a PR (dogfood / review-before-publish).
        # Undeclared inputs are dropped and retried (see
        # _dispatch_with_input_fallback) so older installs still dispatch.
        "mode": mode or "",
        "publish": publish or "",
        "start-from-ref": start_from_ref or "",
        "lead-content": lead,
        "staged-synthesis": "true" if staged_synthesis else "",
        "test-integration-policy": test_integration_policy or "",
        "fidelity-policy": fidelity_policy or "",
    }
    supplied = {k: v for k, v in inputs.items() if v}
    if len(supplied) > GH_MAX_DISPATCH_INPUTS:
        raise click.UsageError(
            f"{len(supplied)} inputs supplied ({', '.join(sorted(supplied))}) "
            f"but GitHub accepts at most {GH_MAX_DISPATCH_INPUTS} per "
            f"workflow_dispatch. Drop the ones the workflow's own defaults "
            f"already cover."
        )

    # A pending run means this dispatch cancels it (static concurrency group).
    _warn_or_wait_for_queue(resolved_repo, wait_for_slot)

    click.echo(f"Dispatching Outrider on {resolved_repo} (ref={branch})…")
    if provider:
        click.echo(f"  provider:       {provider}")
    if model:
        click.echo(f"  model:          {model}")
    if base_url:
        click.echo(f"  base-url:       {base_url}")
    if publish:
        click.echo(f"  publish:        {publish}")
    if search_method:
        click.echo(f"  search-method:  {search_method!r}")
    if pin_arxiv:
        click.echo(f"  pin-arxiv:      {pin_arxiv!r}")
    if interest_id:
        click.echo(f"  interest:       {interest_id}")
    if claude_timeout:
        click.echo(f"  claude-timeout: {claude_timeout}s")
    if mode:
        click.echo(f"  mode:           {mode}")
    if start_from_ref:
        click.echo(f"  start-from-ref: {start_from_ref}")
    if lead:
        click.echo(f"  lead-content:   {len(lead)} chars")
    if staged_synthesis:
        click.echo("  staged-synthesis: true")
    if test_integration_policy:
        click.echo(f"  test-integration-policy: {test_integration_policy}")
    if fidelity_policy:
        click.echo(f"  fidelity-policy: {fidelity_policy}")

    ok, stderr, dropped = _dispatch_with_input_fallback(
        resolved_repo, branch, inputs
    )
    if not ok:
        hint = ""
        if "403" in stderr or "permission" in stderr.lower():
            hint = ("\n  Your gh token lacks the scope to dispatch workflows "
                    "on this repo. Re-auth: `gh auth login --scopes workflow`.")
        raise click.ClickException(f"workflow dispatch failed: {stderr}{hint}")
    if dropped:
        click.secho(
            f"⚠ {resolved_repo}'s installed workflow doesn't declare: "
            f"{', '.join(dropped)}. Dispatched without them (their baked "
            f"defaults apply). Update the install to pick them up:\n"
            f"    remyxai outrider init --repo {resolved_repo} --force",
            fg="yellow",
        )

    click.secho("✓ Dispatched.", fg="green", bold=True)
    url = _gh_latest_run_url(resolved_repo)
    if url:
        click.echo(f"  Run: {url}")
    else:
        click.echo(f"  Run: https://github.com/{resolved_repo}/actions")


# ─── set-provider-secret ──────────────────────────────────────────────────

# Length below which a "secret" is almost certainly truncated or a
# placeholder. The action's startup auth-guard hard-fails below 8 chars;
# we warn earlier (16) because real API keys are typically 40+ chars
# and 8–16 is already deep in the "you probably copy-pasted wrong"
# territory.
_SECRET_MIN_LENGTH_WARN = 16


def handle_set_provider_secret(repo, provider, key_from):
    """Set the per-provider API-key secret on a repo via stdin.

    Wraps ``gh secret set`` with the pitfalls handled. Reads the key
    from ``--key-from FILE`` (never argv, never literal stdin pipes),
    avoiding the ``gh secret set --body -`` trap that stores a literal
    ``"-"`` as the secret value when stdin is disconnected. Validates
    length before sending so a clearly-truncated value is rejected at
    the CLI boundary rather than after a wasted workflow run.

    Provider name → secret name map (``_PROVIDER_SECRET_NAMES``):

    - ``anthropic`` → ``ANTHROPIC_API_KEY``
    - ``zai`` → ``ZAI_API_KEY``
    - ``moonshot`` → ``MOONSHOT_API_KEY``
    """
    if provider not in _PROVIDER_SECRET_NAMES:
        choices = ", ".join(sorted(_PROVIDER_SECRET_NAMES))
        raise click.UsageError(
            f"--provider must be one of: {choices} (got {provider!r})"
        )

    resolved_repo = _normalize_repo(repo) if repo else _detect_github_repo_from_cwd()
    if not resolved_repo:
        raise click.UsageError(
            "Could not determine target repo. Pass --repo owner/name or run "
            "from inside a GitHub-origin git checkout."
        )

    key_path = Path(key_from)
    if not key_path.exists() or not key_path.is_file():
        raise click.ClickException(
            f"--key-from path does not exist or is not a file: {key_from}"
        )
    # Strip a single trailing newline (the common shape from
    # `printf '%s\n' "$KEY" > /tmp/key`); preserve everything else
    # exactly so we don't quietly mutate the customer's secret value.
    value = key_path.read_text().rstrip("\n")

    if not value:
        raise click.ClickException(
            f"{key_from} is empty after reading; nothing to set."
        )
    if value == "-":
        raise click.ClickException(
            f"{key_from} contains the literal '-' character — that's the "
            f"`gh secret set --body -` truncation footprint, not a real "
            f"secret. Refusing to set."
        )
    if len(value) < _SECRET_MIN_LENGTH_WARN:
        click.secho(
            f"⚠ key in {key_from} is {len(value)} chars — unusually short for "
            f"an API key. Proceeding, but the action's startup auth-guard may "
            f"hard-fail if it's actually truncated.",
            fg="yellow",
        )

    secret_name = _PROVIDER_SECRET_NAMES[provider]

    click.echo(
        f"Setting secret {secret_name} on {resolved_repo} "
        f"(provider={provider}, length={len(value)})…"
    )
    # Reuse the safe stdin-piped helper from outrider_local — it
    # already handles 403 → admin-scope hints and never logs the
    # value into argv.
    from remyxai.cli.outrider_local import _gh_set_secret
    _gh_set_secret(resolved_repo, secret_name, value)
    click.secho(
        f"✓ Set {secret_name} on {resolved_repo}.", fg="green", bold=True,
    )
    default_model = {"zai": "glm-5.2", "moonshot": "kimi-k3"}.get(provider)
    if default_model:
        click.echo(
            "  Next: `remyxai outrider trigger --repo "
            f"{resolved_repo} --provider {provider} --model {default_model} "
            f"--pin-arxiv <arxiv-id>` to test."
        )
