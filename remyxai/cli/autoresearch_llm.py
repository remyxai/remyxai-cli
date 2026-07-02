"""
Structured LLM calls for the autoresearch loop.

Two calls per cycle:
- ``propose_hypothesis`` — pick one paper from the ranker's top-N given trace history
- ``decide_from_artifact`` — MERGE/ITERATE/REJECT on the dispatched artifact

Both use tool-use / structured-output to force JSON conformance to a schema.
Provider-agnostic — routes at Anthropic by default, z.ai when ``provider="zai"``.
"""
import json
import os
from typing import Any, Dict, List, Optional

import requests


DEFAULT_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
ZAI_ENDPOINT = "https://api.z.ai/api/anthropic/v1/messages"


class LLMError(RuntimeError):
    pass


def _endpoint_and_auth(provider: str) -> tuple:
    if provider == "zai":
        key = os.environ.get("ZAI_API_KEY")
        if not key:
            raise LLMError("ZAI_API_KEY is not set")
        return ZAI_ENDPOINT, {"Authorization": f"Bearer {key}"}
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise LLMError("ANTHROPIC_API_KEY is not set")
    return ANTHROPIC_ENDPOINT, {"x-api-key": key, "anthropic-version": "2023-06-01"}


def _call_tool(
    system: str,
    user: str,
    tool_name: str,
    tool_schema: Dict[str, Any],
    model: str,
    provider: str,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """Force a single tool-use call and return the parsed JSON input."""
    endpoint, auth = _endpoint_and_auth(provider)
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "tools": [{
            "name": tool_name,
            "description": f"Emit the {tool_name} decision.",
            "input_schema": tool_schema,
        }],
        "tool_choice": {"type": "tool", "name": tool_name},
    }
    r = requests.post(endpoint, json=body, headers={"content-type": "application/json", **auth}, timeout=120)
    if r.status_code != 200:
        raise LLMError(f"LLM call failed HTTP {r.status_code}: {r.text[:400]}")
    payload = r.json()
    for block in payload.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == tool_name:
            return block.get("input", {})
    raise LLMError(f"LLM did not emit a {tool_name} tool_use block. Payload: {payload!r}")


HYPOTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "arxiv_id": {"type": "string", "description": "arxiv id from the candidate list (e.g. 2402.02347v3), or empty if using search-method"},
        "dispatch_mode": {"type": "string", "enum": ["pin-arxiv", "search-method"], "description": "pin-arxiv for exact paper, search-method for method-family query"},
        "search_query": {"type": "string", "description": "Free-text search query if dispatch_mode is search-method; empty otherwise"},
        "rationale": {"type": "string", "description": "One sentence: why this pick given trace history + target repo shape"},
        "expected_terminal": {"type": "string", "description": "Predicted terminal state (pr_opened_draft, issue_opened_preflight, issue_opened, skipped_*)"},
    },
    "required": ["dispatch_mode", "rationale", "expected_terminal"],
}


DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["MERGE", "ITERATE", "REJECT"]},
        "rationale": {"type": "string", "description": "Format: hypothesis: {stated} — observed: {measurement} — conclusion: {decision}"},
        "failure_mode": {"type": "string", "description": "If REJECT: short taxonomy label (prior-art-collision, architecture-mismatch, off-domain, fabrication, license-blocked, other). Empty otherwise."},
        "iterate_request": {"type": "string", "description": "If ITERATE: specific refinement to request from remyx-ai[bot]. Empty otherwise."},
    },
    "required": ["decision", "rationale"],
}


HYPOTHESIS_SYSTEM = """You are the hypothesis stage of an autoresearch loop that evaluates whether published papers can be productively integrated into a target production repository.

Your job: pick ONE paper from the candidate list to dispatch this cycle. Bias toward papers that:
- Have not appeared in prior cycles' trace (dedup by arxiv_id)
- Do not repeat a failure mode already observed 2+ times
- Match the target repo's architectural shape (training paper on inference-only repo = mismatch; and vice versa)

Emit ONE decision via the propose_hypothesis tool. Do NOT explain outside the tool call."""


DECISION_SYSTEM = """You are the decision stage of an autoresearch loop.

Your job: read the dispatched artifact (PR or Issue body) and return MERGE, ITERATE, or REJECT with a scientific-method rationale.

Categories:
- MERGE: PR with real code, fidelity clean, no regressions, eval improved (if eval provided). Only for PRs.
- ITERATE: promising but incomplete signal; the paper is worth another cycle with a refinement request.
- REJECT: eval regressed, fidelity flagged fabrication, or the artifact demonstrates the method cannot land cleanly in this target (architecture mismatch, prior art, off-domain).

Format the rationale as: hypothesis: {stated} — observed: {measurement} — conclusion: {decision}

If REJECT, classify the failure_mode with a short taxonomy label so the loop can aggregate patterns.

Emit ONE decision via the emit_decision tool. Do NOT explain outside the tool call."""


def propose_hypothesis(
    interest_description: str,
    candidates: List[Dict[str, Any]],
    trace_history: List[Dict[str, Any]],
    target_repo: str,
    target_readme_snippet: str,
    model: str = DEFAULT_MODEL,
    provider: str = "anthropic",
) -> Dict[str, Any]:
    trace_summary = json.dumps([
        {"cycle_n": t.get("cycle_n"), "arxiv_id": t.get("arxiv_id"),
         "decision": t.get("decision"), "failure_mode": t.get("failure_mode"),
         "rationale": t.get("rationale")}
        for t in trace_history
    ], indent=2)
    candidate_summary = json.dumps([
        {"arxiv_id": c.get("arxiv_id"), "title": c.get("title"),
         "abstract": (c.get("abstract") or "")[:400],
         "relevance": c.get("relevance"), "github_url": c.get("github_url"),
         "license_class": c.get("license_class")}
        for c in candidates
    ], indent=2)

    user = f"""Target repo: {target_repo}

Interest description:
{interest_description}

Target README snippet:
{target_readme_snippet[:2000]}

Candidates (ranker's top-N):
{candidate_summary}

Prior cycles' trace (dedup + failure-mode signal):
{trace_summary if trace_history else "(no prior cycles)"}

Pick ONE paper to dispatch."""

    return _call_tool(
        system=HYPOTHESIS_SYSTEM,
        user=user,
        tool_name="propose_hypothesis",
        tool_schema=HYPOTHESIS_SCHEMA,
        model=model,
        provider=provider,
    )


def decide_from_artifact(
    hypothesis: Dict[str, Any],
    artifact_type: str,
    artifact_body: str,
    terminal_status: str,
    labels: List[str],
    eval_output: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    provider: str = "anthropic",
) -> Dict[str, Any]:
    user = f"""Hypothesis from this cycle:
{json.dumps(hypothesis, indent=2)}

Artifact type: {artifact_type}
Terminal status: {terminal_status}
Labels: {labels}

Artifact body:
{artifact_body[:6000]}

Eval output:
{(eval_output or "(no eval run; LLM-judge on artifact body only)")[:2000]}

Decide."""

    return _call_tool(
        system=DECISION_SYSTEM,
        user=user,
        tool_name="emit_decision",
        tool_schema=DECISION_SCHEMA,
        model=model,
        provider=provider,
        max_tokens=1500,
    )
