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


# ─── workflow rendering (inline; no Remyx App / bot-token step) ─────────────

_COCOINDEX_STEPS_BLOCK = """      # Attach cocoindex-code as a Claude Code skill so the Outrider agent
      # can ground-truth call-site claims via AST-based semantic code search
      # instead of speculating from paper metadata. Recommended default;
      # pass --no-cocoindex to `remyxai outrider setup-local` to omit.
      - name: Install cocoindex-code as a Claude Code skill
        run: |
          git clone --depth 1 https://github.com/cocoindex-io/cocoindex-code /tmp/cocoindex-code
          pipx install 'cocoindex-code[full]'
          mkdir -p ~/.claude/skills/
          ln -sfn /tmp/cocoindex-code ~/.claude/skills/cocoindex-code
      - name: Write ENVIRONMENTS.md for Outrider
        run: |
          cat > "$GITHUB_WORKSPACE/ENVIRONMENTS.md" <<'EOF'
          ---
          type: Workflow Environment
          title: cocoindex-code AST search available
          description: cocoindex-code AST-based semantic code search is pre-installed as a Claude Code skill.
          resource: https://github.com/cocoindex-io/cocoindex-code
          tags: [outrider, environment, cocoindex-code, ast-search]
          ---

          # Environment: cocoindex-code AST search

          ## Available tools

          - **`ccc` CLI** (AST-based semantic code search across the cloned repo).
            Prefer over reading entire large files when locating functions,
            classes, or specific code patterns. Multi-language via tree-sitter.

          ## Suggested use during implementation

          - For "find the function that does X" queries, invoke the semantic-search
            skill rather than Read/Grep on speculation.
          - For files > 500 LOC where you only need one function, prefer AST search
            + targeted Read over reading the whole file.
          EOF
"""


def _render_local_workflow(
    interest_id: str,
    no_cron: bool = False,
    no_cocoindex: bool = False,
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
    cocoindex_steps = "" if no_cocoindex else _COCOINDEX_STEPS_BLOCK
    return f"""name: Outrider

# Generated by `remyxai outrider setup-local` (no Remyx GitHub App).
# Weekly scout: queries engine.remyx.ai for a paper recommendation against
# this repo's ResearchInterest, then opens a draft PR wiring it in.
#   https://github.com/remyxai/outrider

on:
{schedule_block}  workflow_dispatch:
    inputs:
      provider:
        description: 'Which model provider to route Claude Code at. anthropic = default api.anthropic.com; zai = z.ai (GLM family).'
        type: choice
        required: false
        default: 'anthropic'
        options:
          - anthropic
          - zai
      model:
        description: 'Specific model name to request from the provider (e.g. claude-opus-4-7, glm-5.2, glm-4.6). Empty = let the provider pick its default.'
        required: false
        default: ''
      search-method:
        description: 'Optional free-text method query (e.g. "riemannian preconditioning LoRA"). Runs an engine search and implements the top hit. Use for exploratory dispatches.'
        required: false
        default: ''
      pin-arxiv:
        description: 'Optional exact arxiv_id (e.g. 2402.02347v3). Bypasses selection and implements this specific paper. Use for reproducible re-runs.'
        required: false
        default: ''
      # The five inputs below let outrider-weekly-refine.yml (the refiner)
      # dispatch this runner: it pins the picked draft branch (start-from-ref),
      # pipes a gap analysis in (lead-content), turns on staged synthesis, and
      # selects mode/publish. Defaults match the action's own, so scheduled and
      # manual runs are unchanged.
      mode:
        description: 'Run mode. recommend (default) runs the full scout→implement flow; the refiner dispatches recommend against a pinned paper + start-from-ref.'
        required: false
        default: 'recommend'
      publish:
        description: 'pr (default) opens a PR/Issue; branch produces a fork branch without opening one (drafter behavior).'
        required: false
        default: 'pr'
      start-from-ref:
        description: 'Optional base branch to build on top of (the refiner passes the picked drafter branch here). Empty = start from the default branch.'
        required: false
        default: ''
      lead-content:
        description: 'Optional inline markdown (e.g. a gap analysis) fed to the agent as leading context. Used by the refiner dispatch.'
        required: false
        default: ''
      staged-synthesis:
        description: 'Enable the multi-pass staged-synthesis flow (the refiner sets true). Empty/false = single-pass.'
        required: false
        default: 'false'
      claude-timeout:
        description: 'Wall-clock seconds for the Claude Code agent calls (preflight + implementation). Raise for very large monorepos; lower to cap cost.'
        required: false
        default: '900'

jobs:
  recommend:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    permissions:
      contents: write
      pull-requests: write
      issues: write
    env:
      REMYX_API_KEY: ${{{{ secrets.REMYX_API_KEY }}}}
      # ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / ANTHROPIC_MODEL are
      # written to $GITHUB_ENV by the "Configure provider auth" step
      # below (auth env vars are mutually exclusive — non-Anthropic
      # providers require Bearer auth via AUTH_TOKEN and reject the
      # x-api-key path; ANTHROPIC_MODEL is optional and only set
      # when the workflow_dispatch input is non-empty).
    steps:
      # Per-dispatch provider + model routing. Default cron runs hit
      # Anthropic (inputs.provider defaults to 'anthropic'); dispatch
      # with `provider=zai` to route one run at z.ai's endpoint.
      - name: Configure provider auth
        shell: bash
        env:
          ANTHROPIC_API_KEY_SECRET: ${{{{ secrets.ANTHROPIC_API_KEY }}}}
          ZAI_API_KEY_SECRET: ${{{{ secrets.ZAI_API_KEY }}}}
          MODEL_INPUT: ${{{{ inputs.model }}}}
        run: |
          if [ "${{{{ inputs.provider }}}}" = "zai" ]; then
            echo "ANTHROPIC_AUTH_TOKEN=$ZAI_API_KEY_SECRET" >> "$GITHUB_ENV"
          else
            echo "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY_SECRET" >> "$GITHUB_ENV"
          fi
          if [ -n "$MODEL_INPUT" ]; then
            echo "ANTHROPIC_MODEL=$MODEL_INPUT" >> "$GITHUB_ENV"
          fi
{cocoindex_steps}      - uses: remyxai/outrider@v1
        with:
          interest-id: {interest_id}
          # Minimum days between recommendation PRs on this repo. '0' lets
          # every scheduled or manually-triggered run open a PR; raise it
          # (e.g. '7') to cap how often Outrider posts.
          rate-limit-days: '0'
          # Forwarded from the workflow_dispatch inputs above so
          # `remyxai outrider trigger` (or a manual gh-workflow-run)
          # can pin a paper, extend the implementation timeout, or
          # route at an alternate provider per dispatch. Empty on
          # scheduled runs — the action uses its own defaults.
          search-method: ${{{{ inputs.search-method }}}}
          pin-arxiv: ${{{{ inputs.pin-arxiv }}}}
          claude-timeout: ${{{{ inputs.claude-timeout }}}}
          # Forwarded so outrider-weekly-refine.yml can dispatch a refinement
          # run (mode + start-from-ref + lead-content + staged-synthesis) and
          # so the drafter/manual runs can pick pr vs branch publishing.
          mode: ${{{{ inputs.mode }}}}
          publish: ${{{{ inputs.publish }}}}
          start-from-ref: ${{{{ inputs.start-from-ref }}}}
          lead-content: ${{{{ inputs.lead-content }}}}
          staged-synthesis: ${{{{ inputs.staged-synthesis }}}}
          # Maps provider name → base URL. Adding more providers here
          # extends the table (Bedrock / Vertex / on-prem); leave
          # empty on the default Anthropic path.
          model-base-url: ${{{{ inputs.provider == 'zai' && 'https://api.z.ai/api/anthropic' || '' }}}}
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


def _provider_for_model(model: str) -> str:
    """Infer the backend provider from a model name. GLM models route at z.ai;
    everything else (claude-*) uses the default Anthropic API."""
    return "zai" if model.lower().startswith("glm") else "anthropic"


def uses_zai(*models) -> bool:
    """True if any supplied (non-empty) model resolves to the z.ai provider —
    i.e. the install needs a ZAI_API_KEY secret."""
    return any(m and _provider_for_model(m) == "zai" for m in models)


def _require_anchor(text: str, anchor: str, what: str) -> None:
    if anchor not in text:
        raise click.ClickException(
            f"{what} template on {_OUTRIDER_TEMPLATE_REPO}@{_OUTRIDER_TEMPLATE_REF} "
            f"no longer contains the expected anchor ({anchor!r}); template format "
            f"may have changed. CLI needs an update (or drop the model override)."
        )


def _apply_drafter_model(text: str, model: str) -> str:
    """Rewrite the drafter's model (and, for GLM, its provider auth + base URL)."""
    import re
    _require_anchor(text, "ANTHROPIC_MODEL:", "drafter")
    text = re.sub(r"(?m)^(\s*)ANTHROPIC_MODEL:.*$", rf"\g<1>ANTHROPIC_MODEL: {model}", text)
    if _provider_for_model(model) == "zai":
        # z.ai needs Bearer auth; ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN are
        # mutually exclusive, so swap the env var rather than adding one.
        _require_anchor(text, _DRAFTER_ANTHROPIC_ENV, "drafter")
        _require_anchor(text, _DRAFTER_PUBLISH_ANCHOR, "drafter")
        text = text.replace(_DRAFTER_ANTHROPIC_ENV, _DRAFTER_ZAI_ENV)
        text = text.replace(
            _DRAFTER_PUBLISH_ANCHOR,
            f"{_DRAFTER_PUBLISH_ANCHOR}\n          model-base-url: {_ZAI_BASE_URL}",
            1,
        )
    return text


def _apply_refiner_gap_model(text: str, model: str) -> str:
    """Rewrite the refiner's gap-analysis LLM call (model, and for GLM the
    endpoint + Bearer auth + ZAI_API_KEY step env)."""
    import re
    _require_anchor(text, '"model":', "refiner")
    text = re.sub(r'"model": "[^"]*"', f'"model": "{model}"', text, count=1)
    if _provider_for_model(model) == "zai":
        _require_anchor(text, _GAP_ANTHROPIC_URL, "refiner")
        _require_anchor(text, _GAP_ANTHROPIC_AUTH, "refiner")
        _require_anchor(text, _GAP_ENV_ANCHOR, "refiner")
        text = text.replace(_GAP_ANTHROPIC_URL, _GAP_ZAI_URL)
        text = text.replace(_GAP_ANTHROPIC_AUTH, _GAP_ZAI_AUTH)
        text = text.replace(
            _GAP_ENV_ANCHOR,
            f"{_GAP_ENV_ANCHOR}\n          ZAI_API_KEY: ${{{{ secrets.ZAI_API_KEY }}}}",
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
    """
    import os

    if interest_id and auto_interest:
        raise click.UsageError(
            "--interest and --auto-interest are mutually exclusive."
        )

    # Per-stage model overrides only apply to the two-tier drafter/refiner.
    if (drafter_model or refiner_model or refine_model) and not two_tier:
        raise click.UsageError(
            "--drafter-model / --refiner-model / --refine-model require --two-tier."
        )
    need_zai = uses_zai(drafter_model, refiner_model, refine_model)

    # 1. REMYX key (set as a repo secret + used to resolve the interest)
    remyx_key = os.environ.get("REMYXAI_API_KEY") or click.prompt(
        "REMYXAI_API_KEY (from engine.remyx.ai Settings)", hide_input=True
    )
    if not remyx_key.strip():
        raise click.ClickException("REMYXAI_API_KEY is required.")

    # 2. Anthropic key (set as a repo secret; the engine isn't involved here)
    anthropic_key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        anthropic_key = click.prompt(
            "ANTHROPIC_API_KEY (from console.anthropic.com)", hide_input=True
        )
    if not anthropic_key.strip():
        raise click.ClickException("ANTHROPIC_API_KEY is required for the workflow.")

    # 2b. z.ai key — only when a stage was routed at GLM (set as a repo secret).
    if need_zai:
        zai_key = zai_key or os.environ.get("ZAI_API_KEY") or os.environ.get("Z_AI_KEY")
        if not zai_key:
            zai_key = click.prompt(
                "ZAI_API_KEY (z.ai GLM Coding Plan) — a GLM model was selected",
                hide_input=True,
            )
        if not zai_key.strip():
            raise click.ClickException(
                "ZAI_API_KEY is required when any stage uses a GLM model."
            )

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
    secrets_line = "REMYX_API_KEY, ANTHROPIC_API_KEY" + (", ZAI_API_KEY" if need_zai else "")
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

        # Secrets LAST (closest to success; least cleanup risk).
        _gh_set_secret(resolved_repo, "REMYX_API_KEY", remyx_key)
        click.echo("✓ Set REMYX_API_KEY")
        _gh_set_secret(resolved_repo, "ANTHROPIC_API_KEY", anthropic_key)
        click.echo("✓ Set ANTHROPIC_API_KEY")
        if need_zai:
            _gh_set_secret(resolved_repo, "ZAI_API_KEY", zai_key)
            click.echo("✓ Set ZAI_API_KEY")
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
