"""
Markdown report generator for the autoresearch loop.

Renders ``.remyx-autoresearch/report.md`` from the appended trace.jsonl.
Kept as inline strings (no Jinja) to avoid a new dep.
"""
from collections import Counter
from typing import Any, Dict, List


def render_report(trace: List[Dict[str, Any]], target_repo: str, budget_usd: float) -> str:
    if not trace:
        return f"# Autoresearch report — {target_repo}\n\nNo cycles ran.\n"

    total_cost = sum(t.get("cost_estimate_usd", 0.0) for t in trace)
    decisions = Counter(t.get("decision", "UNKNOWN") for t in trace)
    failure_modes = Counter(t.get("failure_mode", "") for t in trace if t.get("failure_mode"))

    merge_eligible = [t for t in trace if t.get("decision") == "MERGE"]
    iterate = [t for t in trace if t.get("decision") == "ITERATE"]
    preflight_leads = [
        t for t in trace
        if t.get("side_lead") or (t.get("artifact_type") == "issue" and "suggested experiment" in (t.get("artifact_body_snippet") or "").lower())
    ]

    lines = []
    lines.append(f"# Autoresearch report — {target_repo}")
    lines.append("")
    lines.append(f"**Cycles:** {len(trace)}  ·  **Total cost:** ${total_cost:.2f}  ·  **Budget:** ${budget_usd:.0f}")
    lines.append("")
    lines.append("## Decisions")
    lines.append("")
    for k in ("MERGE", "ITERATE", "REJECT"):
        lines.append(f"- **{k}**: {decisions.get(k, 0)}")
    lines.append("")

    lines.append("## Cycle log")
    lines.append("")
    lines.append("| # | Mode | Paper | Terminal | Decision | Rationale |")
    lines.append("|---|---|---|---|---|---|")
    for t in trace:
        mode = t.get("dispatch_mode") or ("search-method" if t.get("search_method_query") else "pin-arxiv")
        paper = t.get("arxiv_id") or "?"
        terminal = t.get("terminal_status") or "?"
        decision = t.get("decision") or "?"
        rationale = (t.get("rationale") or "").replace("|", "\\|")[:180]
        lines.append(f"| {t.get('cycle_n', '?')} | {mode} | {paper} | {terminal} | {decision} | {rationale} |")
    lines.append("")

    if merge_eligible:
        lines.append("## MERGE-eligible artifacts")
        lines.append("")
        for t in merge_eligible:
            lines.append(f"- Cycle {t['cycle_n']} — [{t.get('arxiv_id')}]({t.get('artifact_url')}) — {t.get('rationale', '')[:200]}")
        lines.append("")

    if iterate:
        lines.append("## ITERATE requests")
        lines.append("")
        for t in iterate:
            lines.append(f"- Cycle {t['cycle_n']} — [{t.get('arxiv_id')}]({t.get('artifact_url')}) — refinement: {t.get('iterate_request', '')[:250]}")
        lines.append("")

    if preflight_leads:
        lines.append("## Preflight-suggested experiments (leads)")
        lines.append("")
        lines.append("Even when a paper is REJECTed, preflight's routing rationale may surface a related experiment that DOES fit the target's architecture. These are candidate directions worth pursuing in future cycles.")
        lines.append("")
        for t in preflight_leads:
            lead = t.get("side_lead") or "(see Issue body)"
            lines.append(f"- Cycle {t['cycle_n']} — [{t.get('artifact_url', 'artifact')}] — {lead[:400]}")
        lines.append("")

    if failure_modes:
        lines.append("## Failure-mode taxonomy")
        lines.append("")
        for mode, count in failure_modes.most_common():
            lines.append(f"- **{mode}**: {count} cycle(s)")
        lines.append("")

    lines.append("## Interest calibration signals")
    lines.append("")
    # Surface patterns without mutating anything
    if failure_modes:
        top_mode, top_count = failure_modes.most_common(1)[0]
        if top_count >= 2:
            lines.append(f"- **{top_mode}** appeared in {top_count} cycles. Consider refining the interest context to steer away from this pattern.")
    if decisions.get("REJECT", 0) == len(trace) and len(trace) >= 3:
        lines.append(f"- All {len(trace)} cycles REJECTed. The ranker's current picks may not fit this target's architecture; interest context or target selection likely needs review.")
    if not any([failure_modes, merge_eligible, iterate]):
        lines.append("- (No calibration signal to surface.)")
    lines.append("")

    return "\n".join(lines)
