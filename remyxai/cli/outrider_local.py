"""
CLI action handler for the no-App ("local") Outrider setup path.

`remyxai outrider setup-local` installs Outrider on a repo WITHOUT the Remyx
GitHub App — for enterprises that can't (or won't, yet) grant a third-party
App while a security review is pending. It uses the customer's own
authenticated `gh` CLI to set the repo secrets, write the workflow, and open
(optionally merge) the setup PR. The only Remyx dependency is the
REMYX_API_KEY the workflow uses at runtime to fetch recommendations.

This is the self-provisioning counterpart to `outrider init` (which drives the
engine + Remyx App). Same running Action; different installer + PR-author.

So the running Action can open its recommendation PRs (no bot token here), the
CLI enables the repo's "Allow Actions to create and approve PRs" setting and
the workflow uses the built-in GITHUB_TOKEN. No GitHub token is stored as a
secret — only REMYX_API_KEY and ANTHROPIC_API_KEY.

Side effects are ordered reversible-first (branch, workflow, PR) with secrets
last, and the branch + PR roll back on any post-mutation failure.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from typing import Optional

import click

from remyxai import agent_matrix

# Shared helpers with the engine path (repo parsing + interest resolution).
from remyxai.cli.outrider_actions import (
    _detect_github_repo_from_cwd,
    _normalize_repo,
    _resolve_interest_id,
)

logger = logging.getLogger(__name__)

WORKFLOW_FILENAME = "outrider.yml"
WORKFLOW_PATH = f".github/workflows/{WORKFLOW_FILENAME}"

# Two-tier setup paths — companions to WORKFLOW_PATH under --two-tier.
DRAFTER_WORKFLOW_PATH = ".github/workflows/outrider-daily.yml"
REFINER_WORKFLOW_PATH = ".github/workflows/outrider-weekly-refine.yml"

PR_TITLE = "Install Outrider — weekly arXiv → recommendation PRs"
PR_TITLE_TWO_TIER = "Install Outrider (two-tier drafter/refiner setup)"


# ─── gh helpers ─────────────────────────────────────────────────────────────

def _gh_available() -> bool:
    return shutil.which("gh") is not None


def _gh_authenticated() -> bool:
    if not _gh_available():
        return False
    return subprocess.run(
        ["gh", "api", "user", "--silent"], capture_output=True, text=True,
    ).returncode == 0


def _gh_api_json(args: list) -> dict:
    result = subprocess.run(["gh", "api", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise click.ClickException(
            f"GitHub API call failed ({' '.join(args[:2])}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    try:
        return json.loads(result.stdout) if result.stdout.strip() else {}
    except json.JSONDecodeError:
        return {}


def _gh_default_branch(repo: str) -> str:
    return _gh_api_json([f"/repos/{repo}"]).get("default_branch") or "main"


def _gh_branch_exists(repo: str, branch: str) -> bool:
    return subprocess.run(
        ["gh", "api", f"/repos/{repo}/branches/{branch}", "--silent"],
        capture_output=True, text=True,
    ).returncode == 0


def _gh_get_branch_sha(repo: str, branch: str) -> str:
    ref = _gh_api_json([f"/repos/{repo}/git/ref/heads/{branch}"])
    sha = ref.get("object", {}).get("sha")
    if not sha:
        raise click.ClickException(f"could not resolve SHA for {repo}@{branch}")
    return sha


def _gh_create_branch(repo: str, branch: str, from_sha: str) -> None:
    _gh_api_json([
        "-X", "POST", f"/repos/{repo}/git/refs",
        "-f", f"ref=refs/heads/{branch}", "-f", f"sha={from_sha}",
    ])


def _gh_delete_branch(repo: str, branch: str) -> None:
    r = subprocess.run(
        ["gh", "api", "-X", "DELETE",
         f"/repos/{repo}/git/refs/heads/{branch}", "--silent"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        logger.warning("rollback: failed to delete branch %s: %s", branch, r.stderr.strip())


def _gh_get_file_sha(repo: str, path: str, branch: str) -> Optional[str]:
    """Existing blob SHA of `path` on `branch`, or None. Required to overwrite."""
    r = subprocess.run(
        ["gh", "api", f"/repos/{repo}/contents/{path}?ref={branch}"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    return data.get("sha") if isinstance(data, dict) else None


def _gh_put_file(repo, branch, path, content, commit_message) -> None:
    """Create/update a file on `branch`. Passes the existing sha so a repo that
    already carries the workflow doesn't 422 ('sha wasn't supplied')."""
    import base64
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    args = [
        "-X", "PUT", f"/repos/{repo}/contents/{path}",
        "-f", f"message={commit_message}",
        "-f", f"content={encoded}", "-f", f"branch={branch}",
    ]
    existing = _gh_get_file_sha(repo, path, branch)
    if existing:
        args += ["-f", f"sha={existing}"]
    _gh_api_json(args)


def _gh_open_pr(repo, head, base, title, body, draft=True) -> tuple:
    args = [
        "-X", "POST", f"/repos/{repo}/pulls",
        "-f", f"title={title}", "-f", f"head={head}", "-f", f"base={base}",
        "-f", f"body={body}", "-F", f"draft={'true' if draft else 'false'}",
    ]
    pr = _gh_api_json(args)
    return pr["html_url"], pr["number"]


def _gh_close_pr(repo: str, number: int) -> None:
    r = subprocess.run(
        ["gh", "api", "-X", "PATCH", f"/repos/{repo}/pulls/{number}",
         "-f", "state=closed", "--silent"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        logger.warning("rollback: failed to close PR #%s: %s", number, r.stderr.strip())


def _gh_merge_pr(repo: str, number: int) -> bool:
    """Best-effort merge. Returns True on success; False (with a message) if the
    repo's protections block it — the PR stays open for the user to merge."""
    r = subprocess.run(
        ["gh", "api", "-X", "PUT", f"/repos/{repo}/pulls/{number}/merge",
         "-f", "merge_method=squash"],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def _gh_set_secret(repo: str, name: str, value: str) -> None:
    """Set a repo secret via stdin (never argv/logs)."""
    r = subprocess.run(
        ["gh", "secret", "set", name, "--repo", repo],
        input=value, text=True, capture_output=True,
    )
    if r.returncode != 0:
        stderr = r.stderr.strip()
        hint = ""
        if "403" in stderr or "permission" in stderr.lower():
            hint = (f"\n  Your gh token likely lacks admin scope on {repo}. "
                    f"Re-auth with `gh auth login` or a PAT with repo+workflow scopes.")
        raise click.ClickException(f"failed to set secret {name!r} on {repo}: {stderr}{hint}")


def _gh_enable_pr_creation(repo: str) -> None:
    """Allow Actions to create/approve PRs (so the workflow's GITHUB_TOKEN can
    open recommendation PRs). Requires admin on the repo."""
    _gh_api_json([
        "-X", "PUT", f"/repos/{repo}/actions/permissions/workflow",
        "-F", "default_workflow_permissions=write",
        "-F", "can_approve_pull_request_reviews=true",
    ])


def _gh_dispatch(repo: str, branch: str) -> bool:
    r = subprocess.run(
        ["gh", "api", "-X", "POST",
         f"/repos/{repo}/actions/workflows/{WORKFLOW_FILENAME}/dispatches",
         "-f", f"ref={branch}", "--silent"],
        capture_output=True, text=True,
    )
    return r.returncode == 0


# ─── backend registry ──────────────────────────────────────────────────────
#
# Facts about a provider — its secret env var, its endpoint, its display name,
# its default model — are NOT written here. They come from the action's
# published compatibility matrix via :mod:`remyxai.agent_matrix`, because this
# used to be a third hand-kept copy of that data (after the action's own
# registry and the engine's MODEL_PROVIDERS) and it drifted exactly as you
# would expect the hand-edited copy to: it had z.ai defaulting to `glm-5.2`
# long after the action moved to `glm-5.3`, and it never learned `openai`,
# `openrouter` or `custom` at all.
#
# What stays here is *policy this CLI chooses*, which the matrix has no
# opinion about:
#
#   default_model           what to render when the caller names none, for a
#                           provider the matrix has no default for
#   default_claude_timeout  per-provider wall-clock budget for the generated
#                           workflow — a slower backend needs a bigger one
#   model_prefixes          how a bare ``--drafter-model`` / ``--refiner-model``
#                           names its provider without a separate flag
#
# The keys also bound which providers the **two-tier local install** supports.
# That path rewrites an Anthropic-Messages workflow template in place (model,
# Bearer auth, base URL), so it is Claude-Code-only by construction; the wider
# provider set the matrix knows is reachable through `provider:` on a normal
# install, not through here.
_STAGE_POLICY: dict = {
    "anthropic": {
        "default_model": "claude-opus-4-8",
        "default_claude_timeout": "900",
        "model_prefixes": ("claude",),
    },
    "zai": {
        # GLM's thinking mode adds per-turn latency similar to Kimi's; bumped
        # from the historical 900s to give the coding session enough headroom
        # before hitting the timeout.
        "default_claude_timeout": "3600",
        "model_prefixes": ("glm",),
    },
    "moonshot": {
        # Kimi's thinking mode runs slower per turn than Anthropic Opus; the
        # bumped default matches the action's own docs/backends.md table.
        "default_claude_timeout": "3600",
        "model_prefixes": ("kimi", "moonshot"),
    },
}


def _build_backend_registry() -> dict:
    """Join this CLI's policy onto the action's published provider facts.

    ``base_url`` is what makes a provider PROXIED: its key goes in
    ANTHROPIC_AUTH_TOKEN with ANTHROPIC_BASE_URL pointing at it, rather than
    in ANTHROPIC_API_KEY (native). That distinction now comes from the matrix
    rather than from a hand-maintained field.
    """
    registry = {}
    for provider, policy in _STAGE_POLICY.items():
        if agent_matrix.provider_info(provider) is None:
            raise RuntimeError(
                f"the vendored agent matrix has no provider {provider!r}, "
                f"which this CLI's two-tier install depends on. Refresh it: "
                f"python scripts/sync_agent_matrix.py"
            )
        registry[provider] = dict(
            policy,
            secret_env=agent_matrix.secret_env("claude", provider),
            base_url=agent_matrix.endpoint("claude", provider) or None,
            display_name=agent_matrix.provider_display_name(provider),
            # Prefer the action's own default so the two cannot disagree
            # about what `--provider zai` with no `--model` actually runs.
            default_model=(
                agent_matrix.default_model("claude", provider)
                or policy.get("default_model", "")
            ),
        )
    return registry


_BACKEND_REGISTRY: dict = _build_backend_registry()

#: The `--backend` choice list for the local install path.
#:
#: Bounded by what `_render_local_workflow` can actually render, NOT by every
#: provider serving anthropic-messages. Deriving it from the matrix instead
#: put `openrouter` on the flag — click accepted it and the renderer then
#: raised ValueError, because there is no `_STAGE_POLICY` row to render from.
#: A choice list has to promise exactly what the code behind it supports.
TWO_TIER_BACKEND_CHOICES = sorted(_BACKEND_REGISTRY)

# The provider a stage falls back to when no model override names one — it's
# what the @v1 two-tier templates ship with.
_TEMPLATE_DEFAULT_PROVIDER = "anthropic"


# ─── workflow rendering (inline; no Remyx App / bot-token step) ─────────────

# cocoindex-code (AST semantic search) is installed by the ACTION, not here.
#
# This template used to carry its own install + ENVIRONMENTS.md steps, which
# was wrong in three ways once a run could use something other than Claude
# Code:
#
#   * it symlinked into ~/.claude/skills unconditionally, so a Codex or R-CLI
#     run cloned a skill into a directory that agent never reads;
#   * the ENVIRONMENTS.md it wrote told *every* agent that `ccc` was "a Claude
#     Code skill" invoked as a skill — a route two of the three agents do not
#     have, whose failure looks like the model ignoring an instruction;
#   * the action installs cocoindex itself when `enable-cocoindex` is true
#     (its default), so every setup-local install did the ~1GB install twice
#     and wrote two different ENVIRONMENTS.md files.
#
# The action does all of it agent-aware — it asks the backend for its
# skills_home and generates the surface text from tool_invocation_hint — so
# the only thing to do here is forward the flag.


def _workflow_secret_names() -> list:
    """Every secret the generated workflow should reference, from the matrix.

    Each provider's conventional ``<VENDOR>_API_KEY`` plus each agent's own
    credential — Backboard's key is both its agent credential and its
    model-routing credential, so it arrives via the agent side. REMYX_API_KEY
    leads because the run cannot fetch a recommendation without it.

    Derived rather than listed so a provider added to the action reaches new
    installs without a CLI release. The generated workflow references them
    all; the action reads only the ones its `agent` / `provider` select.
    """
    names = ["REMYX_API_KEY"]
    for provider in agent_matrix.known_providers():
        secret = agent_matrix.provider_info(provider)["secret_env"]
        if secret and secret not in names:
            names.append(secret)
    for agent in agent_matrix.known_agents():
        key = agent_matrix.agent_info(agent)["key_env"]
        if key and key not in names:
            names.append(key)
    return names


def _render_local_workflow(
    interest_id: str,
    no_cron: bool = False,
    no_cocoindex: bool = False,
    backend: str = "anthropic",
    agent: str = "",
) -> str:
    # No github-token input → the action uses this repo's built-in
    # GITHUB_TOKEN, which setup-local authorizes to open PRs.
    #
    # When ``no_cron=True``, the schedule block is rendered commented-out
    # (not omitted) so the user can re-enable scheduled runs later by
    # uncommenting three lines, without re-running setup-local.
    #
    # When ``no_cocoindex=False`` (the default), the workflow includes two
    # extra steps that install cocoindex-code and write an ENVIRONMENTS.md
    # advertising it — see outrider's docs/environments.md for why this is
    # the recommended default.
    #
    # ``backend`` picks the default value of the workflow_dispatch ``provider``
    # input and the baked ``claude-timeout``. ``agent`` picks the default of
    # the ``agent`` input — which coding-agent CLI does the work. Per-dispatch
    # switching stays available on both axes, as long as the corresponding
    # secret is set on the repo (setup-local writes only the selected
    # backend's secret; add others with
    # ``remyxai outrider set-provider-secret`` for cross-backend dispatch).
    #
    # INPUT BUDGET: workflow_dispatch accepts at most 10 top-level inputs, and
    # GitHub rejects the whole workflow past that ("maximum number of inputs
    # for workflow_dispatch event is 10"). This template shipped 11 — the nine
    # the action's own canonical outrider.yml declares, plus ``search-method``
    # and ``claude-timeout`` — so every setup-local install wrote a workflow
    # GitHub would not run. Adding ``agent`` needed two slots back:
    #
    #   search-method   dropped. The canonical template never declared it
    #                   either, so ``trigger --search-method`` already warned
    #                   on App-provisioned installs; this only makes the two
    #                   templates agree. ``--pin-arxiv`` covers the manual
    #                   case.
    #   claude-timeout  no longer an input; baked into ``with:`` from the
    #                   provider's own default. The install still gets the
    #                   right budget — which matters more than ever, since
    #                   neither Codex nor R-CLI has a round cap and the
    #                   timeout is their only spend bound — you just cannot
    #                   override it per dispatch. ``trigger --claude-timeout``
    #                   warns via the undeclared-input path.
    #
    # The result is canonical parity plus the new axis, which is a better
    # place to be than the ad-hoc set it had drifted into.
    if backend not in _BACKEND_REGISTRY:
        raise ValueError(
            f"unknown backend {backend!r}; must be one of: "
            f"{sorted(_BACKEND_REGISTRY)}"
        )
    agent = agent_matrix.resolve_agent(agent)
    if agent not in agent_matrix.known_agents():
        raise ValueError(
            f"unknown agent {agent!r}; must be one of: "
            f"{agent_matrix.known_agents()}"
        )
    reg = _BACKEND_REGISTRY[backend]
    default_timeout = reg["default_claude_timeout"]
    provider_options = "\n".join(
        f"          - {name}" for name in _BACKEND_REGISTRY
    )
    agent_options = "\n".join(
        f"          - {name}" for name in agent_matrix.known_agents()
    )
    secret_env_block = "\n".join(
        f"          {name}: ${{{{ secrets.{name} }}}}"
        for name in _workflow_secret_names()
    )

    if no_cron:
        schedule_block = (
            "  # schedule:\n"
            "  #   - cron: '0 14 * * 1'   # Mondays 14:00 UTC; uncomment to enable\n"
        )
    else:
        schedule_block = (
            "  schedule:\n"
            "    - cron: '0 14 * * 1'   # Mondays 14:00 UTC; pick any cadence\n"
        )
    # Forwarded to the action, which does the install agent-aware.
    enable_cocoindex = "false" if no_cocoindex else "true"
    return f"""name: Outrider

# Generated by `remyxai outrider setup-local` (no Remyx GitHub App).
# Weekly scout: queries engine.remyx.ai for a paper recommendation against
# this repo's ResearchInterest, then opens a draft PR wiring it in.
#   https://github.com/remyxai/outrider

on:
{schedule_block}  workflow_dispatch:
    inputs:
      provider:
        description: 'Which model backend to route the coding agent at. A separate axis from agent, which picks the agent CLI itself. The action picks the matching secret, endpoint and auth style from its own registry.'
        type: choice
        required: false
        default: '{backend}'
        options:
{provider_options}
      agent:
        description: 'Which coding-agent CLI runs the implementation. A separate axis from provider, which picks the model. Not every pair is valid — the action rejects an impossible one up front and names the agent that does serve your provider.'
        type: choice
        required: false
        default: '{agent}'
        options:
{agent_options}
      model:
        description: 'Specific model name (e.g. claude-opus-4-8, glm-5.3, kimi-k3). Use the id your provider lists. Empty = provider default.'
        required: false
        default: ''
      base-url:
        description: 'Optional Anthropic-compatible endpoint (self-hosted model, litellm proxy, vLLM Anthropic shim, on-prem gateway). Overrides the per-provider default when set. Empty = provider default.'
        required: false
        default: ''
      pin-arxiv:
        description: 'Optional exact arxiv_id. Bypasses selection and implements this specific paper.'
        required: false
        default: ''
      # The five inputs below let outrider-weekly-refine.yml (the refiner)
      # dispatch this runner: it pins the picked draft branch (start-from-ref),
      # pipes a gap analysis in (lead-content), turns on staged synthesis, and
      # selects mode/publish.
      mode:
        description: 'Run mode. recommend (default) runs the full scout→implement flow.'
        required: false
        default: 'recommend'
      publish:
        description: 'pr (default) opens a PR/Issue; branch produces a fork branch without opening one (drafter behavior).'
        required: false
        default: 'pr'
      start-from-ref:
        description: 'Optional base branch to build on top of (the refiner passes the picked drafter branch here).'
        required: false
        default: ''
      lead-content:
        description: 'Optional inline markdown (e.g. a gap analysis) fed to the agent as leading context.'
        required: false
        default: ''
      staged-synthesis:
        description: 'Enable the multi-pass staged-synthesis flow (the refiner sets true).'
        required: false
        default: 'false'

jobs:
  recommend:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    permissions:
      contents: write
      pull-requests: write
      issues: write
    steps:
      - uses: remyxai/outrider@v1
        env:
          # Every provider secret and every agent credential the action might
          # read, generated from its published matrix so a provider added
          # there reaches new installs without a CLI release. The Configure
          # step reads only the ones its `agent` / `provider` select; the rest
          # are ignored. A missing secret evaluates to an empty string and
          # that step fails clean with a specific ::error:: naming the one it
          # needed.
{secret_env_block}
        with:
          interest-id: {interest_id}
          # AST semantic search for the agent. The action installs it
          # where the selected agent can actually reach it.
          enable-cocoindex: '{enable_cocoindex}'
          # Minimum days between recommendation PRs. '0' lets every run open
          # a PR; raise (e.g. '7') to cap cadence.
          rate-limit-days: '0'
          # Forwarded from workflow_dispatch inputs so a manual `gh workflow
          # run` (or `remyxai outrider trigger`) can pin a paper and switch
          # either backend axis per-dispatch.
          agent: ${{{{ inputs.agent }}}}
          provider: ${{{{ inputs.provider }}}}
          model: ${{{{ inputs.model }}}}
          model-base-url: ${{{{ inputs.base-url }}}}
          pin-arxiv: ${{{{ inputs.pin-arxiv }}}}
          # Baked rather than dispatched: workflow_dispatch allows only 10
          # inputs and this one lost the tie-break (see the input-budget note
          # in _render_local_workflow). Neither Codex nor R-CLI has a round
          # cap, so this is their only spend bound — keep it tight.
          claude-timeout: '{default_timeout}'
          # Forwarded so outrider-weekly-refine.yml can dispatch a refinement
          # run (mode + start-from-ref + lead-content + staged-synthesis).
          mode: ${{{{ inputs.mode }}}}
          publish: ${{{{ inputs.publish }}}}
          start-from-ref: ${{{{ inputs.start-from-ref }}}}
          lead-content: ${{{{ inputs.lead-content }}}}
          staged-synthesis: ${{{{ inputs.staged-synthesis }}}}
"""


# ─── two-tier templates (drafter + refiner) ────────────────────────────────
#
# When ``--two-tier`` is set on ``setup-local``, the CLI fetches the drafter +
# refiner templates from ``remyxai/outrider@v1`` at install time rather than
# embedding them inline. This treats the outrider repo as the canonical source
# of truth for the workflow shape — bugfixes to the picker heuristic, updates
# to the gap-analysis prompt, changes to the model tier defaults, all land on
# the outrider repo first and propagate to new customer installs automatically
# without requiring a CLI release.
#
# The drafter template has one substitution point (the ``interest-id``); the
# refiner has none (it dispatches outrider.yml which reads interest-id from
# its own configuration).
#
# See remyxai/outrider docs/customization.md §5 for the design rationale.

_OUTRIDER_TEMPLATE_REPO = "remyxai/outrider"
_OUTRIDER_TEMPLATE_REF = "v1"  # moves with each Outrider action release
_DRAFTER_TEMPLATE_PATH = ".github/workflows/outrider-daily.yml"
_REFINER_TEMPLATE_PATH = ".github/workflows/outrider-weekly-refine.yml"

# The outrider repo's own drafter has a hardcoded interest-id for its self-test.
# We rewrite that specific string to the customer's interest-id at render time.
_OUTRIDER_SELF_INTEREST_ID = "29ca03e7-454d-446c-9941-32c96c53d95d"

# The outrider repo's templates reference the action locally (``uses: ./``),
# which only resolves from inside the outrider repo itself. On a customer
# install that path points at the target repo's root (no action.yml there),
# so we rewrite it to the published action ref at render time.
_LOCAL_ACTION_USES = "uses: ./"
_PUBLISHED_ACTION_USES = f"uses: {_OUTRIDER_TEMPLATE_REPO}@{_OUTRIDER_TEMPLATE_REF}"

# ─── optional per-stage model overrides ─────────────────────────────────────
#
# By default the two-tier templates are single-provider (all three stages run
# on Anthropic) so an install needs only ANTHROPIC_API_KEY. The optional
# --drafter-model / --refiner-model / --refine-model flags let a caller retune
# any stage — including routing it at z.ai's GLM — without hand-editing the
# installed workflow files. Provider is inferred from the model name: GLM
# models route at z.ai (Bearer auth via ZAI_API_KEY), everything else stays on
# Anthropic. See remyxai/outrider docs/backends.md for the routing details.

_ZAI_BASE_URL = "https://api.z.ai/api/anthropic"

# Structural anchors in the @v1 templates that the overrides rewrite. Kept as
# explicit constants so a template shape change fails loud (matching the
# interest-id / uses-./ guards) rather than silently skipping a rewrite.
_DRAFTER_ANTHROPIC_ENV = "ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}"
_DRAFTER_ZAI_ENV = "ANTHROPIC_AUTH_TOKEN: ${{ secrets.ZAI_API_KEY }}"
_DRAFTER_PUBLISH_ANCHOR = "publish: branch"
_GAP_ANTHROPIC_URL = '"https://api.anthropic.com/v1/messages"'
_GAP_ZAI_URL = '"https://api.z.ai/api/anthropic/v1/messages"'
_GAP_ANTHROPIC_AUTH = '"x-api-key": os.environ["ANTHROPIC_API_KEY"],'
_GAP_ZAI_AUTH = '"Authorization": f"Bearer {os.environ[\'ZAI_API_KEY\']}",'
_GAP_ENV_ANCHOR = "REPO: ${{ github.repository }}"


def is_gateway_model(model: str) -> bool:
    """True for a namespaced ``<vendor>/<model>`` id.

    A gateway addresses models this way — OpenRouter's ``z-ai/glm-5.3``,
    R-CLI's ``<provider>/<model>`` — and the namespace names the vendor
    *behind* the gateway, not the endpoint the request goes to. So the id
    cannot be resolved to a direct provider by inspection: ``z-ai/glm-5.3``
    is served by OpenRouter, not by z.ai.

    That distinction matters because the prefix heuristic below reads the
    leading characters, and would have matched nothing for every namespaced
    id, quietly handing them to the template default.
    """
    return "/" in (model or "")


def infer_provider(model: str) -> Optional[str]:
    """Backend a model name belongs to, or ``None`` when nothing matches.

    Driven by ``_BACKEND_REGISTRY[*]["model_prefixes"]``, so adding a backend
    teaches every stage of the two-tier install to route it. ``None`` is the
    honest answer for an unrecognized name — the caller warns rather than
    silently assuming Anthropic, which is how a Kimi drafter ended up rendered
    against ANTHROPIC_API_KEY and dead on arrival.

    A namespaced gateway id is ``None`` for a stronger reason than "no prefix
    matched": there is no direct provider to infer at all. See
    :func:`is_gateway_model`.
    """
    name = (model or "").strip().lower()
    if not name or is_gateway_model(name):
        return None
    for provider, cfg in _BACKEND_REGISTRY.items():
        if any(name.startswith(p) for p in cfg["model_prefixes"]):
            return provider
    return None


def _provider_for_model(model: str) -> str:
    """``infer_provider`` with the template's default for unknown names."""
    return infer_provider(model) or _TEMPLATE_DEFAULT_PROVIDER


def providers_for_stages(*models) -> list:
    """Providers a two-tier install actually uses, in registry order.

    A stage with no model override keeps the template's provider, so that one
    counts too — the install needs a key for every provider in this list, not
    just for the overridden stages.
    """
    used = {
        _provider_for_model(m) if m else _TEMPLATE_DEFAULT_PROVIDER
        for m in models
    }
    return [p for p in _BACKEND_REGISTRY if p in used]


def uses_zai(*models) -> bool:
    """True if any supplied (non-empty) model resolves to the z.ai provider.

    Retained for callers that only ask about z.ai; prefer
    ``providers_for_stages`` for "which secrets does this install need".
    """
    return any(m and _provider_for_model(m) == "zai" for m in models)


def _unknown_stage_models(*models) -> list:
    """Model overrides whose backend can't be inferred from the name."""
    return [m for m in models if m and infer_provider(m) is None]


def _require_anchor(text: str, anchor: str, what: str) -> None:
    if anchor not in text:
        raise click.ClickException(
            f"{what} template on {_OUTRIDER_TEMPLATE_REPO}@{_OUTRIDER_TEMPLATE_REF} "
            f"no longer contains the expected anchor ({anchor!r}); template format "
            f"may have changed. CLI needs an update (or drop the model override)."
        )


def _apply_drafter_model(text: str, model: str) -> str:
    """Rewrite the drafter's model (and, for a proxied backend, its auth + base
    URL)."""
    import re
    _require_anchor(text, "ANTHROPIC_MODEL:", "drafter")
    text = re.sub(r"(?m)^(\s*)ANTHROPIC_MODEL:.*$", rf"\g<1>ANTHROPIC_MODEL: {model}", text)
    reg = _BACKEND_REGISTRY[_provider_for_model(model)]
    if reg["base_url"]:
        # Proxied backends need Bearer auth; ANTHROPIC_API_KEY and
        # ANTHROPIC_AUTH_TOKEN are mutually exclusive, so swap the env var
        # rather than adding one.
        _require_anchor(text, _DRAFTER_ANTHROPIC_ENV, "drafter")
        _require_anchor(text, _DRAFTER_PUBLISH_ANCHOR, "drafter")
        text = text.replace(
            _DRAFTER_ANTHROPIC_ENV,
            "ANTHROPIC_AUTH_TOKEN: ${{ secrets.%s }}" % reg["secret_env"],
        )
        text = text.replace(
            _DRAFTER_PUBLISH_ANCHOR,
            f"{_DRAFTER_PUBLISH_ANCHOR}\n          model-base-url: {reg['base_url']}",
            1,
        )
    return text


def _apply_refiner_gap_model(text: str, model: str) -> str:
    """Rewrite the refiner's gap-analysis LLM call (model, and for a proxied
    backend the endpoint + Bearer auth + that backend's step env)."""
    import re
    _require_anchor(text, '"model":', "refiner")
    text = re.sub(r'"model": "[^"]*"', f'"model": "{model}"', text, count=1)
    reg = _BACKEND_REGISTRY[_provider_for_model(model)]
    if reg["base_url"]:
        secret = reg["secret_env"]
        _require_anchor(text, _GAP_ANTHROPIC_URL, "refiner")
        _require_anchor(text, _GAP_ANTHROPIC_AUTH, "refiner")
        _require_anchor(text, _GAP_ENV_ANCHOR, "refiner")
        text = text.replace(
            _GAP_ANTHROPIC_URL, f'"{reg["base_url"]}/v1/messages"',
        )
        text = text.replace(
            _GAP_ANTHROPIC_AUTH,
            '"Authorization": f"Bearer {os.environ[\'%s\']}",' % secret,
        )
        text = text.replace(
            _GAP_ENV_ANCHOR,
            f"{_GAP_ENV_ANCHOR}\n          {secret}: ${{{{ secrets.{secret} }}}}",
            1,
        )
    return text


def _apply_refine_dispatch_model(text: str, model: str) -> str:
    """Rewrite the model + provider the refiner dispatches for the final run.
    The installed runner already maps provider=zai → z.ai base URL, so only the
    two dispatch flags need changing here."""
    import re
    _require_anchor(text, "-f model=", "refiner")
    _require_anchor(text, "-f provider=", "refiner")
    text = re.sub(r"-f model=\S+", f"-f model={model}", text, count=1)
    text = re.sub(r"-f provider=\S+", f"-f provider={_provider_for_model(model)}", text, count=1)
    return text


def _fetch_outrider_template(path: str) -> str:
    """Fetch a workflow-template file from remyxai/outrider@v1 via `gh api`.

    Raises ClickException if the fetch fails — the caller should treat this
    as a hard error (missing template = we can't install the two-tier setup).
    """
    import base64
    try:
        payload = _gh_api_json([
            f"repos/{_OUTRIDER_TEMPLATE_REPO}/contents/{path}?ref={_OUTRIDER_TEMPLATE_REF}",
        ])
    except Exception as e:
        raise click.ClickException(
            f"could not fetch {path} from {_OUTRIDER_TEMPLATE_REPO}@{_OUTRIDER_TEMPLATE_REF} "
            f"(gh api failed): {e}"
        )
    content_b64 = payload.get("content", "")
    if not content_b64:
        raise click.ClickException(
            f"{path} on {_OUTRIDER_TEMPLATE_REPO}@{_OUTRIDER_TEMPLATE_REF} is empty."
        )
    return base64.b64decode(content_b64).decode()


def _render_drafter_workflow(interest_id: str, model: Optional[str] = None) -> str:
    """Drafter template — fetched live from remyxai/outrider@v1 with the
    self-test interest-id rewritten to the customer's and the local action
    reference (``uses: ./``) rewritten to the published action ref.

    ``model`` (optional) retunes the drafter's model; a GLM model also switches
    it to z.ai Bearer auth. Omitted → the template's single-provider default.
    """
    raw = _fetch_outrider_template(_DRAFTER_TEMPLATE_PATH)
    if _OUTRIDER_SELF_INTEREST_ID not in raw:
        raise click.ClickException(
            f"drafter template on {_OUTRIDER_TEMPLATE_REPO}@{_OUTRIDER_TEMPLATE_REF} "
            f"no longer contains the expected self-interest-id placeholder "
            f"({_OUTRIDER_SELF_INTEREST_ID}); template format may have changed. "
            f"CLI needs an update."
        )
    if _LOCAL_ACTION_USES not in raw:
        raise click.ClickException(
            f"drafter template on {_OUTRIDER_TEMPLATE_REPO}@{_OUTRIDER_TEMPLATE_REF} "
            f"no longer references the action via '{_LOCAL_ACTION_USES}'; template "
            f"format may have changed. CLI needs an update."
        )
    out = (
        raw.replace(_OUTRIDER_SELF_INTEREST_ID, interest_id)
           .replace(_LOCAL_ACTION_USES, _PUBLISHED_ACTION_USES)
    )
    if model:
        out = _apply_drafter_model(out, model)
    return out


def _render_refiner_workflow(
    gap_model: Optional[str] = None, refine_model: Optional[str] = None,
) -> str:
    """Refiner template — fetched live from remyxai/outrider@v1.

    The refiner has no interest-id substitution point; it dispatches
    outrider.yml (via workflow_dispatch) which reads its own interest-id.

    ``gap_model`` (optional) retunes the gap-analysis LLM call; ``refine_model``
    (optional) retunes the model/provider the refiner dispatches for the final
    refinement run. Both omitted → the template's single-provider defaults.
    """
    out = _fetch_outrider_template(_REFINER_TEMPLATE_PATH)
    if gap_model:
        out = _apply_refiner_gap_model(out, gap_model)
    if refine_model:
        out = _apply_refine_dispatch_model(out, refine_model)
    return out


# ─── main handler ──────────────────────────────────────────────────────────

def handle_outrider_setup_local(
    repo, interest_id, auto_interest, mode,
    anthropic_key, skip_confirm, dry_run, no_cron=False, no_cocoindex=False,
    two_tier=False,
    drafter_model=None, refiner_model=None, refine_model=None, zai_key=None,
    backend="anthropic",
    agent="",
):
    """Self-provision Outrider with the user's own gh token (no Remyx App).

    When ``two_tier=True``, installs the drafter + refiner companions
    (`outrider-daily.yml`, `outrider-weekly-refine.yml`) alongside the
    manual-dispatch `outrider.yml` — the recommended default for repos
    where continuous exploration + weekly promotion is wanted. Templates
    are fetched from ``remyxai/outrider@v1`` at install time so template
    updates propagate to new installs without a CLI release. See
    ``remyxai/outrider`` docs/customization.md §5 for design details.

    ``--two-tier`` currently opts in — it's a strict superset of the
    legacy single-file install, and existing installs are unaffected.

    ``backend`` selects which model backend the single-file setup routes
    at by default; the supported set is
    :data:`TWO_TIER_BACKEND_CHOICES`, bounded by what the template can
    render rather than by every provider the action knows. ``agent``
    selects the coding-agent CLI on the other axis. Only the selected
    backend's secret is prompted + written; users who want per-dispatch
    switching add the other secrets with
    ``remyxai outrider set-provider-secret``. ``backend`` is scoped to
    the single-file path — two-tier setups route per-stage via
    ``--drafter-model`` etc. and reject a non-anthropic ``--backend`` as
    ambiguous.
    """
    import os

    if interest_id and auto_interest:
        raise click.UsageError(
            "--interest and --auto-interest are mutually exclusive."
        )

    if backend not in _BACKEND_REGISTRY:
        raise click.UsageError(
            f"unknown --backend {backend!r}; must be one of: "
            f"{', '.join(sorted(_BACKEND_REGISTRY))}"
        )
    if two_tier and backend != "anthropic":
        raise click.UsageError(
            "--backend is scoped to the single-file setup; --two-tier "
            "ignores it. Use --drafter-model / --refiner-model / "
            "--refine-model to route two-tier stages at non-Anthropic "
            "backends."
        )

    # Per-stage model overrides only apply to the two-tier drafter/refiner.
    if (drafter_model or refiner_model or refine_model) and not two_tier:
        raise click.UsageError(
            "--drafter-model / --refiner-model / --refine-model require --two-tier."
        )
    # Every backend the two-tier stages actually route at — a stage with no
    # override keeps the template's Anthropic default, so that counts too.
    stage_providers = (
        providers_for_stages(drafter_model, refiner_model, refine_model)
        if two_tier else []
    )
    unknown_models = _unknown_stage_models(
        drafter_model, refiner_model, refine_model,
    )
    # Split by how sure we are. A namespaced id is *certainly* unroutable
    # here, so it fails; a merely unrecognized one might be fine, so it
    # warns. Same rule the agent/provider check uses: only a durable fact
    # gets to hard-fail.
    gateway_models = [m for m in unknown_models if is_gateway_model(m)]
    unrecognized = [m for m in unknown_models if not is_gateway_model(m)]
    if gateway_models:
        raise click.UsageError(
            f"a two-tier stage cannot use a gateway model id: "
            f"{', '.join(gateway_models)}. Each stage rewrites an "
            f"Anthropic-Messages workflow in place and routes at one "
            f"vendor's endpoint, so a `<vendor>/<model>` id — OpenRouter's "
            f"`z-ai/glm-5.3`, R-CLI's `<provider>/<model>` — has no endpoint "
            f"to resolve to. Name a direct provider's model instead ("
            + ", ".join(
                f"{c['default_model']}" for c in _BACKEND_REGISTRY.values()
                if c.get("default_model")
            )
            + "), or install single-file and set `provider` per dispatch."
        )
    if unrecognized:
        click.secho(
            f"⚠ can't tell which backend these models belong to: "
            f"{', '.join(unrecognized)}. Treating them as "
            f"{_TEMPLATE_DEFAULT_PROVIDER} — if that's wrong the stage will "
            f"fail auth on its first run. Known prefixes: "
            + "; ".join(
                f"{p}→{'/'.join(c['model_prefixes'])}"
                for p, c in _BACKEND_REGISTRY.items()
            ),
            fg="yellow",
        )
    need_zai = "zai" in stage_providers

    # 1. REMYX key (set as a repo secret + used to resolve the interest)
    remyx_key = os.environ.get("REMYXAI_API_KEY") or click.prompt(
        "REMYXAI_API_KEY (from engine.remyx.ai Settings)", hide_input=True
    )
    if not remyx_key.strip():
        raise click.ClickException("REMYXAI_API_KEY is required.")

    # 2. Backend secret resolution.
    #
    # Two-tier needs a key for EVERY backend its stages route at — the drafter,
    # the refiner's gap analysis, and the run the refiner dispatches can each
    # sit at a different one. Collecting only Anthropic (+ z.ai when a GLM model
    # appeared) is how `--drafter-model kimi-k3` produced a repo with no
    # MOONSHOT_API_KEY that reported a clean install and then failed auth.
    #
    # Single-file path resolves ONE backend's secret — the selected
    # `backend` — from (in order) the legacy CLI flag if it matches the
    # backend name, the registry-declared env var, or an interactive
    # prompt. `moonshot` has no legacy flag and is env-or-prompt.
    stage_secrets = {}
    if two_tier:
        # Legacy per-vendor flags stay honored as a key source.
        preset = {"anthropic": anthropic_key, "zai": zai_key}
        for provider in stage_providers:
            reg = _BACKEND_REGISTRY[provider]
            env_name = reg["secret_env"]
            value = preset.get(provider) or os.environ.get(env_name)
            if provider == "zai" and not value:
                value = os.environ.get("Z_AI_KEY")     # historical alias
            if not value and dry_run:
                # A dry run changes nothing, so don't make the operator type
                # two or three hidden secrets to see the plan.
                stage_secrets[env_name] = None
                continue
            if not value:
                why = (
                    "the template default"
                    if provider == _TEMPLATE_DEFAULT_PROVIDER
                    else f"a {reg['display_name']} model was selected"
                )
                value = click.prompt(
                    f"{env_name} ({reg['display_name']}) — {why}",
                    hide_input=True,
                )
            if not (value or "").strip():
                raise click.ClickException(
                    f"{env_name} is required: a two-tier stage routes at "
                    f"{reg['display_name']}."
                )
            stage_secrets[env_name] = value
            if provider == "anthropic":
                anthropic_key = value
            elif provider == "zai":
                zai_key = value
        # The install's primary secret (first in registry order) keeps the
        # historical variable names the summary + rollback path read.
        backend_secret_env = next(iter(stage_secrets))
        backend_secret_value = stage_secrets[backend_secret_env]
    else:
        reg = _BACKEND_REGISTRY[backend]
        backend_secret_env = reg["secret_env"]
        display = reg["display_name"]
        legacy_flag_value = {"anthropic": anthropic_key, "zai": zai_key}.get(backend)
        backend_secret_value = (
            legacy_flag_value
            or os.environ.get(backend_secret_env)
        )
        if not backend_secret_value and dry_run:
            backend_secret_value = None          # see the two-tier note above
        elif not backend_secret_value:
            backend_secret_value = click.prompt(
                f"{backend_secret_env} ({display})", hide_input=True
            )
        if backend_secret_value is not None and not backend_secret_value.strip():
            raise click.ClickException(
                f"{backend_secret_env} is required for --backend {backend}."
            )
        # Populate historical variables so downstream references remain
        # consistent (they're only used in the two-tier path today, so this
        # is defensive but harmless).
        if backend == "anthropic":
            anthropic_key = backend_secret_value
        elif backend == "zai":
            zai_key = backend_secret_value

    # 3. Repo
    resolved_repo = _normalize_repo(repo) if repo else _detect_github_repo_from_cwd()
    if not resolved_repo:
        raise click.ClickException(
            "No GitHub repo specified or detected. Pass --repo owner/name."
        )
    repo_url = f"https://github.com/{resolved_repo}"

    # 4. Plan
    click.echo("")
    click.echo("Plan (no Remyx GitHub App — uses your gh credentials):")
    click.echo(f"  - Repo:      {resolved_repo}")
    click.echo(f"  - Mode:      {mode} (auto = open + merge PR + dispatch; review = open PR only)")
    if two_tier:
        secrets_line = ", ".join(
            ["REMYX_API_KEY"] + [
                name if value is not None else f"{name} (will prompt)"
                for name, value in stage_secrets.items()
            ]
        )
    else:
        secrets_line = f"REMYX_API_KEY, {backend_secret_env}"
        if backend != "anthropic":
            click.echo(f"  - Backend:   {backend} ({_BACKEND_REGISTRY[backend]['display_name']})")
    click.echo(f"  - Secrets:   {secrets_line}")
    click.echo("  - PR auth:   enable the repo 'Actions can create PRs' setting "
               "(PRs by github-actions[bot])")
    if two_tier:
        drafter_desc = f"model={drafter_model} ({_provider_for_model(drafter_model)})" if drafter_model else "Haiku 4.5"
        refiner_desc = f"gap={refiner_model} ({_provider_for_model(refiner_model)})" if refiner_model else "Sonnet gap-analysis"
        refine_desc = f"model={refine_model} ({_provider_for_model(refine_model)})" if refine_model else "dispatches Opus"
        click.echo(f"  - Writes:    {WORKFLOW_PATH} (manual dispatch, no cron)")
        click.echo(f"               {DRAFTER_WORKFLOW_PATH} (daily drafter, {drafter_desc}, publish=branch)")
        click.echo(f"               {REFINER_WORKFLOW_PATH} (weekly refiner, {refiner_desc}, {refine_desc})")
        click.echo("               (three files on a branch → one PR — templates fetched live from remyxai/outrider@v1)")
        click.secho("  - Note:      forks don't run scheduled workflows — the daily/weekly "
                    "crons need an external dispatcher on a fork (workflow_dispatch is unaffected).",
                    fg="yellow")
    else:
        click.echo(f"  - Writes:    {WORKFLOW_PATH} on a branch + opens a PR")
    click.echo("")

    if dry_run:
        if two_tier:
            click.echo("--- rendered outrider.yml (workflow_dispatch only) ---")
            click.echo(_render_local_workflow(
                "<interest-id>", no_cron=True, no_cocoindex=no_cocoindex,
                agent=agent,
            ))
            click.echo("\n--- rendered outrider-daily.yml (drafter) ---")
            click.echo(_render_drafter_workflow("<interest-id>", model=drafter_model))
            click.echo("\n--- rendered outrider-weekly-refine.yml (refiner) ---")
            click.echo(_render_refiner_workflow(
                gap_model=refiner_model, refine_model=refine_model,
            ))
        else:
            click.echo("--- rendered workflow ---")
            click.echo(_render_local_workflow(
                "<interest-id>", no_cron=no_cron, no_cocoindex=no_cocoindex,
                backend=backend, agent=agent,
            ))
        click.secho("dry-run: no changes made.", fg="yellow")
        return

    # 5. gh preconditions
    if not _gh_available():
        raise click.ClickException(
            "`gh` (GitHub CLI) is not installed. See https://cli.github.com."
        )
    if not _gh_authenticated():
        raise click.ClickException(
            "`gh` cannot authenticate. Run `gh auth login` or set a valid "
            "$GITHUB_TOKEN with repo + workflow scopes, then re-run."
        )

    if not skip_confirm:
        click.confirm("Proceed?", abort=True, default=False)

    # 6. Resolve interest (engine call — the interest lives server-side)
    resolved_interest = _resolve_interest_id(
        interest_id, auto_interest, resolved_repo, repo_url, remyx_key
    )

    default_branch = _gh_default_branch(resolved_repo)
    branch_name = "install-outrider"
    if _gh_branch_exists(resolved_repo, branch_name):
        raise click.ClickException(
            f"branch {branch_name!r} already exists on {resolved_repo}. "
            f"Delete it or merge/close the existing setup PR, then re-run."
        )

    # 7. Execute — reversible first (branch, file, PR), secrets last; rollback
    pr_number = None
    branch_created = False
    try:
        base_sha = _gh_get_branch_sha(resolved_repo, default_branch)
        _gh_create_branch(resolved_repo, branch_name, base_sha)
        branch_created = True
        click.echo(f"✓ Created branch {branch_name}")

        # Under --two-tier: outrider.yml is manual-dispatch only (no cron —
        # scheduled runs come from outrider-daily.yml + outrider-weekly-refine.yml).
        # Otherwise: legacy single-file install with whatever cron the caller wants.
        workflow = _render_local_workflow(
            resolved_interest,
            no_cron=(no_cron or two_tier),
            no_cocoindex=no_cocoindex,
            backend=backend,
            agent=agent,
        )
        _gh_put_file(resolved_repo, branch_name, WORKFLOW_PATH, workflow,
                     "Install Outrider (self-provisioned via remyxai CLI)")
        click.echo(f"✓ Wrote {WORKFLOW_PATH}")

        if two_tier:
            drafter_yml = _render_drafter_workflow(resolved_interest, model=drafter_model)
            _gh_put_file(
                resolved_repo, branch_name, DRAFTER_WORKFLOW_PATH, drafter_yml,
                "Install Outrider two-tier drafter (self-provisioned)",
            )
            click.echo(f"✓ Wrote {DRAFTER_WORKFLOW_PATH}")

            refiner_yml = _render_refiner_workflow(
                gap_model=refiner_model, refine_model=refine_model,
            )
            _gh_put_file(
                resolved_repo, branch_name, REFINER_WORKFLOW_PATH, refiner_yml,
                "Install Outrider two-tier refiner (self-provisioned)",
            )
            click.echo(f"✓ Wrote {REFINER_WORKFLOW_PATH}")

        body = (
            f"Installs [Outrider](https://github.com/remyxai/outrider) "
            f"(self-provisioned, no Remyx GitHub App).\n\n"
            f"Research interest: `{resolved_interest}`\n\n"
            f"Generated by `remyxai outrider setup-local`."
        )
        if two_tier:
            body += (
                "\n\n**Two-tier setup** — installs a daily drafter "
                "(`outrider-daily.yml`) and a weekly refiner "
                "(`outrider-weekly-refine.yml`) alongside the manual-dispatch "
                "runner.\n\n"
                "> **Note — forks:** GitHub disables/deprioritizes `schedule:` "
                "triggers on forked repos, so the drafter's daily cron and the "
                "refiner's weekly cron will not self-run on a fork. "
                "`workflow_dispatch` (manual / `gh workflow run`) works either "
                "way; drive the cadence from an external dispatcher if this repo "
                "is a fork."
            )
        pr_url, pr_number = _gh_open_pr(
            resolved_repo, branch_name, default_branch,
            PR_TITLE_TWO_TIER if two_tier else PR_TITLE, body,
            draft=(mode != "auto"),
        )
        click.echo(f"✓ Opened PR: {pr_url}")

        # Let the running Action open PRs with the built-in GITHUB_TOKEN.
        _gh_enable_pr_creation(resolved_repo)
        click.echo("✓ Enabled Actions PR creation on the repo")

        # Secrets LAST (closest to success; least cleanup risk). Two-tier pushes
        # one per backend its stages route at — a stage whose secret is missing
        # fails auth in under a second on its first run.
        _gh_set_secret(resolved_repo, "REMYX_API_KEY", remyx_key)
        click.echo("✓ Set REMYX_API_KEY")
        for env_name, value in (stage_secrets or
                                {backend_secret_env: backend_secret_value}).items():
            _gh_set_secret(resolved_repo, env_name, value)
            click.echo(f"✓ Set {env_name}")
    except Exception as e:
        if pr_number is not None:
            click.echo(f"  ↩ rolling back: closing PR #{pr_number}", err=True)
            _gh_close_pr(resolved_repo, pr_number)
        if branch_created:
            click.echo(f"  ↩ rolling back: deleting branch {branch_name}", err=True)
            _gh_delete_branch(resolved_repo, branch_name)
        if isinstance(e, click.ClickException):
            raise
        raise click.ClickException(f"setup-local failed: {e}")

    # 8. auto mode — merge + dispatch
    merged = False
    if mode == "auto":
        merged = _gh_merge_pr(resolved_repo, pr_number)
        if merged:
            click.echo("✓ Merged the setup PR")
            if _gh_dispatch(resolved_repo, default_branch):
                click.echo("✓ Dispatched the first run")
        else:
            click.secho(
                "  Could not auto-merge (branch protection?). The PR is open — "
                "merge it to activate Outrider.", fg="yellow",
            )

    # 9. Report
    click.echo("")
    click.secho("✓ Outrider set up (no App).", fg="green", bold=True)
    click.echo(f"  PR:       {pr_url}")
    if mode == "auto" and merged:
        click.echo("  Status:   active — a recommendation PR will appear shortly.")
    else:
        click.echo("  Next:     merge the PR to activate Outrider.")
    click.echo(f"  Manual:   gh workflow run {WORKFLOW_FILENAME} --repo {resolved_repo}")
