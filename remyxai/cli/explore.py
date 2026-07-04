"""
Explore loop orchestrator.

Repeatedly proposes papers from the ranker's top-N, dispatches Outrider,
reads the resulting artifact, decides MERGE/ITERATE/REJECT with rationale,
and appends to a per-target trace.

Safety-by-design:
- Never writes code to the target repo (only dispatches Outrider + optionally
  posts a comment on the resulting artifact).
- Hard budget + cycle caps stop the loop cleanly.
- Preflight verdicts are respected as hard REJECTs.
- Trace is append-only; interest revision is a human decision.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from remyxai.cli.explore_llm import (
    decide_from_artifact,
    propose_hypothesis,
    LLMError,
)
from remyxai.cli.explore_report import render_report
from remyxai.cli.outrider_actions import (
    WORKFLOW_FILENAME,
    _gh_dispatch_outrider,
    _gh_default_branch,
    _normalize_repo,
    _outrider_workflow_exists,
)
from remyxai.api.recommendations import list_recommended


def _read_interest_from_workflow(repo: str) -> Optional[str]:
    """Read the interest-id declared in the target's outrider.yml. This is
    the production source-of-truth for a target's interest — the workflow file
    is what actually runs. Falling back to ad-hoc engine lookups would drift."""
    r = subprocess.run(
        ["gh", "api", f"/repos/{repo}/contents/.github/workflows/{WORKFLOW_FILENAME}"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    try:
        import base64
        payload = json.loads(r.stdout)
        yml = base64.b64decode(payload["content"]).decode("utf-8", errors="replace")
    except Exception:
        return None
    import re
    m = re.search(r"interest-id:\s*['\"]?([0-9a-f-]{36})['\"]?", yml)
    return m.group(1) if m else None


TRACE_DIRNAME = ".remyx-autoresearch"
TRACE_FILENAME = "trace.jsonl"
REPORT_FILENAME = "report.md"
POLL_INTERVAL_S = 60
POLL_MAX_S = 3600


def _trace_paths(cwd: Optional[Path] = None) -> tuple:
    root = (cwd or Path.cwd()) / TRACE_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root / TRACE_FILENAME, root / REPORT_FILENAME


def _load_trace(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _append_trace(path: Path, entry: Dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _fetch_readme_snippet(repo: str) -> str:
    """Best-effort README fetch via gh (no local clone)."""
    for filename in ("README.md", "readme.md", "README.rst"):
        r = subprocess.run(
            ["gh", "api", f"/repos/{repo}/contents/{filename}"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            continue
        try:
            payload = json.loads(r.stdout)
            import base64
            return base64.b64decode(payload["content"]).decode("utf-8", errors="replace")
        except Exception:
            continue
    return ""


def _fetch_citation_set(repo: str) -> set:
    """Prior-art check via gh code search on the target repo."""
    r = subprocess.run(
        ["gh", "search", "code", "--repo", repo, "--limit", "50", "arXiv"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return set()
    import re
    ids = set()
    for m in re.finditer(r"(\d{4}\.\d{4,5})(?:v\d+)?", r.stdout):
        ids.add(m.group(1))
    return ids


def _extract_candidates(recs_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten ranker's envelope into hypothesis-stage-friendly candidates.

    Envelope shape (as of 2026-07): top-level has `title`, `url`, `resource_id`,
    `relevance_score`; `resource` (nested) has `arxiv_id`, `abstract`,
    `github_url`, `license_class`, etc.
    """
    out = []
    for env in recs_payload.get("papers", []):
        rsrc = env.get("resource") or {}
        out.append({
            "arxiv_id": env.get("resource_id") or rsrc.get("arxiv_id") or "",
            "title": env.get("title") or "",
            "abstract": rsrc.get("abstract") or "",
            "github_url": rsrc.get("github_url") or "",
            "license_class": rsrc.get("license_class") or "",
            "relevance": env.get("relevance_score") or 0.0,
            "suggested_experiment": env.get("suggested_experiment") or "",
        })
    return out


def _latest_run_url(repo: str, since_ts: float) -> Optional[str]:
    """Poll gh for the newest Outrider run started after ``since_ts``."""
    for _ in range(20):
        r = subprocess.run(
            ["gh", "run", "list", "--repo", repo, "--workflow", WORKFLOW_FILENAME,
             "--limit", "1", "--json", "databaseId,startedAt,url"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            try:
                runs = json.loads(r.stdout)
                if runs:
                    run = runs[0]
                    return run.get("url"), run.get("databaseId")
            except json.JSONDecodeError:
                pass
        time.sleep(3)
    return None, None


def _poll_terminal(repo: str, run_id: int) -> Dict[str, Any]:
    """Poll a run until completed. Returns the final json blob."""
    deadline = time.time() + POLL_MAX_S
    while time.time() < deadline:
        r = subprocess.run(
            ["gh", "run", "view", str(run_id), "--repo", repo,
             "--json", "status,conclusion,jobs"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            try:
                payload = json.loads(r.stdout)
                if payload.get("status") == "completed":
                    return payload
            except json.JSONDecodeError:
                pass
        time.sleep(POLL_INTERVAL_S)
    return {"status": "timeout", "conclusion": "timeout"}


def _extract_terminal_signals(repo: str, run_id: int) -> Dict[str, Any]:
    """Read the run log for terminal status keys the action emits."""
    r = subprocess.run(
        ["gh", "run", "view", str(run_id), "--repo", repo, "--log"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return {}
    log = r.stdout
    signals = {}
    import re
    for key in ("status", "arxiv_id", "issue_url", "pr_url", "preflight_decision"):
        m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', log)
        if m:
            signals[key] = m.group(1)
    return signals


def _fetch_artifact_body(url: str) -> tuple:
    """Read issue or PR body via gh. Returns (kind, body, labels)."""
    if "/issues/" in url:
        n = url.rstrip("/").rsplit("/", 1)[-1]
        repo_slug = url.split("github.com/")[1].split("/issues")[0]
        r = subprocess.run(
            ["gh", "issue", "view", n, "--repo", repo_slug,
             "--json", "body,labels"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            return "issue", payload.get("body", ""), [l["name"] for l in payload.get("labels", [])]
    elif "/pull/" in url:
        n = url.rstrip("/").rsplit("/", 1)[-1]
        repo_slug = url.split("github.com/")[1].split("/pull")[0]
        r = subprocess.run(
            ["gh", "pr", "view", n, "--repo", repo_slug,
             "--json", "body,labels"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            payload = json.loads(r.stdout)
            return "pr", payload.get("body", ""), [l["name"] for l in payload.get("labels", [])]
    return "unknown", "", []


def _post_decision_comment(url: str, decision: Dict[str, Any], hypothesis: Dict[str, Any]) -> None:
    verdict = decision.get("decision", "REJECT")
    body = (
        f"@remyx-ai[bot] **{verdict}** — explore loop decision.\n\n"
        f"**Hypothesis (outer agent):** {hypothesis.get('rationale', '')}\n\n"
        f"**Rationale:** {decision.get('rationale', '')}\n"
    )
    if verdict == "LEAD" and decision.get("lead_content"):
        body += (
            f"\n**Research LEAD (architecturally-viable experiment surfaced by preflight):**\n"
            f"> {decision['lead_content']}\n\n"
            f"The paper itself doesn't fit, but the experiment above uses existing target modules — "
            f"this is a labeled engineering lead worth queueing, distinct from a REJECT.\n"
        )
    if decision.get("failure_mode"):
        body += f"\n**Failure mode:** `{decision['failure_mode']}`\n"
    if decision.get("iterate_request"):
        body += f"\n**Refinement request:** {decision['iterate_request']}\n"

    if "/issues/" in url:
        n = url.rstrip("/").rsplit("/", 1)[-1]
        repo_slug = url.split("github.com/")[1].split("/issues")[0]
        subprocess.run(["gh", "issue", "comment", n, "--repo", repo_slug, "--body", body], check=False)
    elif "/pull/" in url:
        n = url.rstrip("/").rsplit("/", 1)[-1]
        repo_slug = url.split("github.com/")[1].split("/pull")[0]
        subprocess.run(["gh", "pr", "comment", n, "--repo", repo_slug, "--body", body], check=False)


def _fetch_interest_context(interest_id: str, api_key: Optional[str]) -> str:
    """Best-effort read of the interest description."""
    from remyxai.api.interests import get_interest
    try:
        interest = get_interest(interest_id, api_key=api_key)
        return interest.get("context") or interest.get("description") or interest.get("name") or ""
    except Exception:
        return ""


def handle_explore(
    repo: Optional[str],
    interest_id: Optional[str],
    cycles: int,
    budget_usd: float,
    provider: str,
    model: str,
    dry_run: bool,
    api_key: Optional[str],
    no_comment: bool,
) -> None:
    repo = _normalize_repo(repo) if repo else None
    if not repo:
        click.echo("--repo is required", err=True)
        sys.exit(1)

    if not _outrider_workflow_exists(repo):
        click.echo(f"No Outrider workflow found in {repo}. Run `remyxai outrider init` first.", err=True)
        sys.exit(1)

    if interest_id == "auto" or not interest_id:
        interest_id = _read_interest_from_workflow(repo)
        if not interest_id:
            click.echo(
                f"Could not read interest-id from {repo}'s outrider.yml. "
                f"Pass --interest <uuid> explicitly.", err=True,
            )
            sys.exit(1)

    trace_path, report_path = _trace_paths()
    trace = _load_trace(trace_path)

    click.echo(f"Explore loop on {repo}")
    click.echo(f"  Interest: {interest_id}")
    click.echo(f"  Cycles: {cycles}  ·  Budget: ${budget_usd:.0f}  ·  Prior trace: {len(trace)} cycles")
    click.echo(f"  Trace: {trace_path}")
    click.echo("")

    interest_ctx = _fetch_interest_context(interest_id, api_key)
    readme = _fetch_readme_snippet(repo)
    citations = _fetch_citation_set(repo)
    click.echo(f"Target citation set: {len(citations)} arxiv ids referenced in-tree")

    branch = _gh_default_branch(repo) or "main"
    starting_cycle = len(trace) + 1

    CONSECUTIVE_SKIP_LIMIT = 2  # honor the LLM's "pool exhausted" signal
    for i in range(cycles):
        cycle_n = starting_cycle + i
        total_cost = sum(t.get("cost_estimate_usd", 0.0) for t in trace)
        if total_cost >= budget_usd:
            click.echo(f"Budget ${budget_usd:.2f} reached (${total_cost:.2f} spent). Stopping.")
            break
        # Early-terminate when the LLM has signaled pool-exhausted twice in a row.
        # SKIPs are cheap ($0.02 each) but pointless once the LLM has read the
        # trace and confirmed no candidate fits — the ranker's picks aren't
        # changing between cycles within a batch run.
        recent_skips = [t.get("decision") == "SKIP" for t in trace[-CONSECUTIVE_SKIP_LIMIT:]]
        if len(recent_skips) >= CONSECUTIVE_SKIP_LIMIT and all(recent_skips):
            click.echo(f"{CONSECUTIVE_SKIP_LIMIT} consecutive SKIPs — candidate pool exhausted. Stopping.")
            click.echo("  → Consider refining the interest context, waiting for new papers, or switching target.")
            break

        click.echo(f"\n─── Cycle {cycle_n} ───")

        # Fetch fresh candidates every cycle (ranker may update)
        try:
            recs = list_recommended(interest_id=interest_id, limit=10, api_key=api_key)
        except Exception as e:
            click.echo(f"  Ranker fetch failed: {e}", err=True)
            break

        candidates = _extract_candidates(recs)
        # Filter prior-art (deterministic) — drop candidates whose arxiv_id is
        # cited in-tree OR whose id already appeared in trace (dedup).
        seen_ids = {(t.get("arxiv_id") or "").split("v")[0] for t in trace if t.get("arxiv_id")}
        candidates = [
            c for c in candidates
            if c["arxiv_id"] and c["arxiv_id"].split("v")[0] not in citations
            and c["arxiv_id"].split("v")[0] not in seen_ids
        ]
        if not candidates:
            click.echo("  All ranker candidates are already-cited or trace-duplicates. Stopping.")
            break
        click.echo(f"  Candidates (post prior-art filter): {len(candidates)}")

        # Hypothesis stage
        try:
            hypothesis = propose_hypothesis(
                interest_description=interest_ctx,
                candidates=candidates,
                trace_history=trace,
                target_repo=repo,
                target_readme_snippet=readme,
                model=model,
                provider=provider,
            )
        except LLMError as e:
            click.echo(f"  Hypothesis LLM call failed: {e}", err=True)
            break

        dispatch_mode = hypothesis.get("dispatch_mode", "")
        paper_requires = hypothesis.get("paper_requires", "")
        target_has_capability = hypothesis.get("target_has_capability")
        click.echo(f"  Hypothesis: {hypothesis.get('arxiv_id') or hypothesis.get('search_query') or '(skip)'} — {hypothesis.get('rationale')}")
        if paper_requires:
            click.echo(f"  Paper requires: {paper_requires}  ·  Target has: {target_has_capability}")
        click.echo(f"  Expected terminal: {hypothesis.get('expected_terminal') or '(n/a)'}")

        entry = {
            "cycle_n": cycle_n,
            "target_repo": repo,
            "arxiv_id": hypothesis.get("arxiv_id"),
            "dispatch_mode": dispatch_mode,
            "search_method_query": hypothesis.get("search_query"),
            "hypothesis": hypothesis.get("rationale"),
            "paper_requires": paper_requires,
            "target_has_capability": target_has_capability,
            "expected_terminal": hypothesis.get("expected_terminal"),
            "provider": provider,
            "model": model,
            "cost_estimate_usd": 0.02,  # hypothesis LLM call only
        }

        # Architectural-fit guard — the hypothesis LLM asked to skip when no
        # candidate fits. Honor it: don't spend on dispatches the LLM already
        # knows are doomed. This is the primary cost-saving improvement.
        if dispatch_mode == "skip":
            click.echo(f"  SKIP — hypothesis LLM found no architecturally-viable candidate")
            entry["decision"] = "SKIP"
            entry["rationale"] = hypothesis.get("rationale") or "no architecturally viable candidate"
            entry["refinement_suggestions"] = hypothesis.get("refinement_suggestions") or []
            for s in entry["refinement_suggestions"]:
                click.echo(f"    → try: \"{s.get('query', '')}\" — {s.get('why', '')[:100]}")
            _append_trace(trace_path, entry)
            trace.append(entry)
            continue

        if dry_run:
            click.echo("  --dry-run: skipping dispatch")
            entry["decision"] = "DRY_RUN"
            _append_trace(trace_path, entry)
            trace.append(entry)
            continue

        # Dispatch
        inputs = {
            "pin-arxiv": hypothesis.get("arxiv_id", "") if hypothesis.get("dispatch_mode") == "pin-arxiv" else "",
            "search-method": hypothesis.get("search_query", "") if hypothesis.get("dispatch_mode") == "search-method" else "",
            "provider": provider,
            "model": model,
        }
        ok, err = _gh_dispatch_outrider(repo, branch, inputs)
        if not ok:
            click.echo(f"  Dispatch failed: {err}", err=True)
            break

        # Locate the run we just triggered
        time.sleep(4)
        url, run_id = _latest_run_url(repo, time.time() - 60)
        if not run_id:
            click.echo("  Could not locate dispatched run URL", err=True)
            break
        entry["dispatch_run_url"] = url
        click.echo(f"  Run: {url}")

        # Poll to terminal
        terminal_payload = _poll_terminal(repo, run_id)
        entry["run_status"] = terminal_payload.get("status")
        entry["run_conclusion"] = terminal_payload.get("conclusion")
        click.echo(f"  Terminal: {terminal_payload.get('status')} / {terminal_payload.get('conclusion')}")

        # Read terminal signals + artifact
        signals = _extract_terminal_signals(repo, run_id)
        entry["terminal_status"] = signals.get("status")
        entry["preflight_decision"] = signals.get("preflight_decision")
        artifact_url = signals.get("issue_url") or signals.get("pr_url")
        entry["artifact_url"] = artifact_url

        if not artifact_url:
            # Distinguish workflow-level failures (claude_failed, timeouts,
            # provider API errors) from clean-run-no-artifact bailouts
            # (skipped_by_selection_verification etc). The former shouldn't
            # count against the loop's REJECT rate — it's an infra glitch,
            # not a considered rejection of the paper.
            if terminal_payload.get("conclusion") == "failure" or signals.get("status") == "claude_failed":
                click.echo("  Workflow failed (claude_failed / provider glitch) — INFRA_FAIL, not a paper-fit signal")
                entry["decision"] = "INFRA_FAIL"
                entry["rationale"] = "hypothesis: dispatched — observed: workflow-level failure (provider glitch, quota, or timeout) — conclusion: infra_fail (retry-eligible, not a REJECT)"
                entry["failure_mode"] = "provider-side-failure"
                entry["cost_estimate_usd"] += 0.5  # partial dispatch, not full
            else:
                click.echo("  No artifact URL — run terminated without opening PR/Issue")
                entry["decision"] = "REJECT"
                entry["rationale"] = "hypothesis: dispatched — observed: no artifact opened — conclusion: reject (no evidence to evaluate)"
                entry["failure_mode"] = "no-artifact"
                entry["cost_estimate_usd"] += 5.0  # rough dispatch estimate
            _append_trace(trace_path, entry)
            trace.append(entry)
            continue

        kind, body, labels = _fetch_artifact_body(artifact_url)
        entry["artifact_type"] = kind
        entry["labels"] = labels

        # Decision stage
        try:
            decision = decide_from_artifact(
                hypothesis=hypothesis,
                artifact_type=kind,
                artifact_body=body,
                terminal_status=signals.get("status", ""),
                labels=labels,
                eval_output=None,
                model=model,
                provider=provider,
            )
        except LLMError as e:
            click.echo(f"  Decision LLM call failed: {e}", err=True)
            break

        entry["decision"] = decision.get("decision")
        entry["rationale"] = decision.get("rationale")
        entry["failure_mode"] = decision.get("failure_mode")
        entry["iterate_request"] = decision.get("iterate_request")
        entry["lead_content"] = decision.get("lead_content")
        entry["cost_estimate_usd"] += 0.05 + (5.0 if kind == "pr" else 0.5)  # LLM + dispatch cost estimate

        click.echo(f"  Decision: {decision.get('decision')}")
        click.echo(f"  Rationale: {decision.get('rationale', '')[:200]}")
        if decision.get("decision") == "LEAD" and decision.get("lead_content"):
            click.echo(f"  Lead: {decision.get('lead_content', '')[:200]}")

        if not no_comment:
            _post_decision_comment(artifact_url, decision, hypothesis)

        _append_trace(trace_path, entry)
        trace.append(entry)

    # Render report
    report = render_report(trace, repo, budget_usd)
    report_path.write_text(report)
    click.echo(f"\n{'=' * 60}")
    click.echo(f"Report: {report_path}")
    total_cost = sum(t.get("cost_estimate_usd", 0.0) for t in trace)
    click.echo(f"Total cycles: {len(trace)}  ·  Total cost: ${total_cost:.2f}")
