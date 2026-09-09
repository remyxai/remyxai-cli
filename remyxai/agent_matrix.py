"""Queries over the action's agent/provider compatibility matrix.

The matrix is generated in ``remyxai/outrider`` and vendored here by
``scripts/sync_agent_matrix.py``. This module is the only thing that should
read :data:`remyxai._agent_matrix.MATRIX`.

Two axes, not one. ``agent`` picks the coding-agent CLI that does the work;
``provider`` picks the model behind it. An agent speaks exactly one API family
and a provider serves one or more, so a pair is valid when they share a family
— it is a join, not a matrix of enumerated combinations. Everything here falls
out of that.

**On strictness.** This CLI ships independently of the action, which moves on a
``@v1`` tag, so the vendored matrix can be older than the action a user
actually runs. Validation is therefore split:

* An **error** only for something durably true — a pair where both sides are
  known and they demonstrably do not share an API family. Which family a CLI
  speaks is a property of the CLI, not a policy that changes release to
  release, so this cannot become wrong as the matrix ages.
* A **warning** for anything else worth saying: an agent or provider this
  build has never heard of (the installed action may well know it), a pair
  that routes but has not been verified end-to-end, or one that carries an
  operational precondition.

Erring the other way — rejecting on an unrecognized value — would mean every
newly added provider looked broken until the user upgraded the CLI.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from remyxai._agent_matrix import MATRIX

#: The agent the action runs when ``agent`` is unset. Empty means Claude Code
#: and always will — it is the pinned backwards-compatibility guarantee that
#: keeps every pre-existing workflow on the path it has today, so this is a
#: fact about the contract rather than a default that might be retuned.
DEFAULT_AGENT = "claude"

#: Severity levels a check can return.
ERROR = "error"
WARN = "warn"


class Problem(tuple):
    """A single finding: ``(level, message)``.

    A tuple subclass so callers can unpack it or compare it in tests without
    caring that it is a class.
    """

    __slots__ = ()

    def __new__(cls, level: str, message: str) -> "Problem":
        return super().__new__(cls, (level, message))

    @property
    def level(self) -> str:
        return self[0]

    @property
    def message(self) -> str:
        return self[1]

    @property
    def is_error(self) -> bool:
        return self[0] == ERROR


# ─── raw lookups ───────────────────────────────────────────────────────────

def known_agents() -> List[str]:
    return sorted(MATRIX["agents"])


def known_providers() -> List[str]:
    return sorted(MATRIX["providers"])


def agent_info(name: str) -> Optional[Dict]:
    return MATRIX["agents"].get(resolve_agent(name))


def provider_info(provider: str) -> Optional[Dict]:
    return MATRIX["providers"].get(provider)


def resolve_agent(name: str) -> str:
    """Empty means the default agent, matching the action."""
    return (name or "").strip() or DEFAULT_AGENT


def agent_display_name(name: str) -> str:
    info = agent_info(name)
    return info["display_name"] if info else resolve_agent(name)


def provider_display_name(provider: str) -> str:
    info = provider_info(provider)
    return info["display_name"] if info else provider


def is_native_router(name: str) -> bool:
    """True when the agent resolves provider ids itself, server-side.

    R-CLI is one: it takes ``<provider>/<model>`` and looks the pair up against
    its own catalogue, so this CLI cannot pre-validate the provider id and must
    not pretend to.
    """
    info = agent_info(name)
    return bool(info) and info["api_family"] == "native-router"


def agents_speaking(family: str) -> List[str]:
    return sorted(
        name for name, info in MATRIX["agents"].items()
        if info["api_family"] == family
    )


def agents_serving(provider: str) -> Tuple[List[str], List[str]]:
    """Agents that could reach this provider, as ``(direct, routed)``.

    Powers the suggestion in a rejection: the action says *"Use agent=claude
    for this provider"*, and this is the same join that produces it.

    The split matters. A *direct* speaker talks to the provider's own endpoint
    with the provider's own key, so it is a drop-in answer to "this pair does
    not work". A *native router* can reach almost anything, but only through
    its own account and credential — proposing it first would answer "OpenAI
    does not serve anthropic-messages" with "so go sign up for something
    else", which is not the fix the user wants.
    """
    info = provider_info(provider)
    if info is None:
        return [], []
    direct = set()
    for family in info["families"]:
        direct.update(agents_speaking(family))
    routed = {n for n in MATRIX["agents"] if is_native_router(n)}
    return sorted(direct - routed), sorted(routed - direct)


# ─── what the caller has to set ────────────────────────────────────────────

def secret_env(agent: str, provider: str) -> Optional[str]:
    """The env var the caller must put a key in for this pair.

    For a normal pair that is the provider's conventional name
    (``ZAI_API_KEY``); for a native router, or a provider that supplies its own
    endpoint, it is the agent's own credential.
    """
    a_info = agent_info(agent)
    if a_info is None:
        return None
    if is_native_router(agent):
        return a_info["key_env"]
    p_info = provider_info(provider)
    if p_info is None:
        return a_info["key_env"] if not provider else None
    return p_info["secret_env"] or a_info["key_env"]


def default_model(agent: str, provider: str) -> str:
    """The model this pair uses when the caller names none ('' if any)."""
    a_info = agent_info(agent)
    p_info = provider_info(provider)
    if a_info is None or p_info is None:
        return ""
    return p_info["default_model"].get(a_info["api_family"], "")


def endpoint(agent: str, provider: str) -> str:
    """The base URL this pair routes at ('' = the vendor's own default)."""
    a_info = agent_info(agent)
    p_info = provider_info(provider)
    if a_info is None or p_info is None:
        return ""
    return p_info["families"].get(a_info["api_family"], "")


def home_provider(agent: str) -> str:
    """The provider an agent reaches at its vendor's *own* endpoint.

    Derived, not listed: exactly one provider serves each API family with an
    empty base URL, which is what "this is that family's own vendor" means —
    `anthropic` for anthropic-messages, `openai` for openai-responses. So it
    is the natural default provider for an agent, and adding a family or a
    vendor needs no edit here.

    This exists because `--backend` used to default to `anthropic` whatever
    the agent was, so `--agent codex` alone was rejected: picking an agent
    forced you to also know which provider pairs with it. A native router
    has no family to match, so it keeps whatever default the caller has —
    it can reach anything its own catalogue lists.
    """
    info = agent_info(agent)
    if info is None or is_native_router(agent):
        return ""
    family = info["api_family"]
    for provider in known_providers():
        p_info = provider_info(provider)
        if p_info["caller_supplied_endpoint"]:
            continue
        if p_info["families"].get(family, None) == "":
            return provider
    return ""


# ─── validation ────────────────────────────────────────────────────────────

def check_pair(agent: str, provider: str, model: str = "") -> List[Problem]:
    """Findings for one ``(agent, provider, model)`` selection, worst first.

    ``model`` is what the caller named, if anything. It is needed because the
    "this provider has no default model" warning is only true when nobody
    named one — without it the most ordinary configuration there is
    (``--provider anthropic --model claude-opus-4-8``) would warn about a
    default it never uses.

    An empty list means "nothing to say". See the module docstring for why an
    unrecognized value warns rather than errors.
    """
    problems: List[Problem] = []
    agent = resolve_agent(agent)
    provider = (provider or "").strip()

    a_info = MATRIX["agents"].get(agent)
    if a_info is None:
        problems.append(Problem(WARN, (
            f"this CLI does not know agent={agent!r} (it knows: "
            f"{', '.join(known_agents())}). Passing it through — the action "
            f"installed on the repo may be newer than this CLI. Refresh with "
            f"`pip install -U remyxai` if it turns out to be a typo."
        )))
        return problems

    if not provider:
        # No provider named: the action passes through without touching auth,
        # which is the oldest supported shape and always valid.
        return problems

    p_info = MATRIX["providers"].get(provider)
    if p_info is None:
        problems.append(Problem(WARN, (
            f"this CLI does not know provider={provider!r} (it knows: "
            f"{', '.join(known_providers())}). Passing it through — the "
            f"action installed on the repo may be newer than this CLI."
        )))
        return problems

    if is_native_router(agent):
        # R-CLI resolves provider ids against its own catalogue, which is
        # large, changes without notice, and uses different names (it has no
        # `zai` — z.ai models are reached through `openrouter` there). A local
        # allowlist would go stale, so the action preflights the live
        # catalogue instead. Say so rather than validating something we can't.
        problems.append(Problem(WARN, (
            f"{agent_display_name(agent)} resolves provider names against its "
            f"own catalogue, which does not match the names above — the run "
            f"checks yours against the live catalogue at startup and names the "
            f"closest matches if it misses."
        )))
        return problems

    family = a_info["api_family"]
    if family not in p_info["families"]:
        direct, routed = agents_serving(provider)
        direct = [n for n in direct if n != agent]
        routed = [n for n in routed if n != agent]
        if direct:
            fix = f" Use --agent {' or '.join(direct)} for this provider."
        elif routed:
            fix = (
                f" Only --agent {' or '.join(routed)} can reach it, through "
                f"its own account and credential."
            )
        else:
            fix = ""
        problems.append(Problem(ERROR, (
            f"agent={agent} speaks {family}, which "
            f"{p_info['display_name']} does not serve.{fix}"
        )))
        return problems

    if family not in p_info["verified"] and not p_info["caller_supplied_endpoint"]:
        problems.append(Problem(WARN, (
            f"agent={agent} + provider={provider} is unverified: "
            f"{p_info['display_name']}'s {family} support has not been "
            f"confirmed end-to-end. If the run fails with an unexpected 4xx, "
            f"the endpoint likely speaks a different protocol."
        )))
    elif p_info.get("verification_caveat"):
        # Published in the artifact so this warning uses the action's own
        # wording rather than a copy that could drift.
        problems.append(Problem(WARN, (
            f"provider={provider} works but "
            f"{p_info['verification_caveat']}."
        )))

    if not (model or "").strip() and not default_model(agent, provider):
        problems.append(Problem(WARN, (
            f"{p_info['display_name']} has no default model for {family}, so "
            f"{agent_display_name(agent)} will send its own default model id "
            f"— which this provider may not recognise. Pass --model with an "
            f"id {p_info['display_name']} lists."
        )))

    return problems


def first_error(problems: List[Problem]) -> Optional[Problem]:
    for problem in problems:
        if problem.is_error:
            return problem
    return None


def pair_rows() -> List[Tuple[str, str, str, str, bool]]:
    """The published pairs, for ``--help``-style listings.

    Rows are ``(agent, provider, secret, default_model, verified)`` with the
    native-router placeholder left as the artifact renders it.
    """
    return [
        (
            row["agent"], row["provider"], row["secret"],
            row["default_model"], bool(row["verified"]),
        )
        for row in MATRIX["pairs"]
    ]
