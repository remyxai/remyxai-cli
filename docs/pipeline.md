---
type: reference
description: How Remyx + Outrider automates the paper-adoption pipeline — discovery, implementation, validation.
tags: [outrider, pipeline, discovery, implementation, validation]
---

# Pipeline: discovery → implementation → validation

Systematizing paper adoption across a team is usually a coordination problem, not a technical one — someone spots the paper, someone else evaluates fit, someone else implements, someone else reviews. Remyx + Outrider collapses that chain into a single automated pipeline that your team gates rather than staffs.

## Discovery

Every day, Remyx ranks the latest arXiv against your Research Interests (extracted from your codebase, curated per-team, or written by hand):

```bash
remyxai papers digest                             # grouped by interest
remyxai papers list --interest "My Project" -n 5  # flat view, top 5
```

When Outrider fires on a repo, it runs an **audit pass** that agentically explores the codebase, refines the recommendation pool via targeted searches for under-represented themes, and picks the paper most directly implementable against real call sites. Nothing gets drafted against imagined structure.

## Implementation

Once a paper is selected, Outrider clones the target repo and invokes Claude Code with a scoped brief: implement the paper's core contribution as a draft PR, wired into an existing call site, honoring your repo's contribution conventions (extracted from recent merged PRs).

If the paper can't be cleanly scaffolded — no natural integration point, or too large a scope for one PR — Outrider opens a design-discussion Issue instead. The routing decision is **measurement-based**, not aspirational: the coding agent must find a real call site before drafting a PR.

## Validation

Before a PR lands as ready-for-review, three passes run automatically:

- **Fidelity audit** — clones the paper's reference implementation and diffs the draft against it, flagging any invented algorithms, wrong schemas, or missing components the paper actually requires. This is what catches plausibility-optimized fabrications before they reach a human reviewer.
- **Convention pass** — folds the PR body and code layout to your target repo's contribution conventions (PR-body scaffold, AI-disclosure, docstring style, test co-location patterns), extracted from recent merged PRs.
- **Test gate** — runs lint + targeted tests on the touched files; drops the Draft state only if lint passes and tests execute cleanly.

Each pass writes its findings to the run's Actions Summary — you inspect one panel to see the drafting, verification, and refinement decisions for every recommendation.

## Where teams sit in the loop

You gate rather than staff:

- **Configure** — set up an interest that reflects what your team tracks (via `interests from-repo` for codebase-anchored, or `interests create` for hand-curated)
- **Review** — Outrider files Draft PRs and design Issues; you review, request changes, or merge
- **Steer** — dispatch ad-hoc runs on specific papers via `--pin-arxiv` or `--search-method` when you have a specific target in mind (see [method-targeted-runs.md](method-targeted-runs.md))

The automation catches the discovery gap (papers you'd have missed) and the fabrication gap (implementations that look plausible but aren't). Your review time stays focused on architecture, scope, and merge-readiness.
