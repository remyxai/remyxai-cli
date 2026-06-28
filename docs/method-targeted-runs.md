---
type: howto
description: Discover a relevant paper or method, then dispatch a targeted Outrider run that implements it.
tags: [outrider, cli, pin-method, pin-arxiv, workflow_dispatch]
---

# Method-targeted Outrider runs

Once [Outrider](https://github.com/remyxai/outrider) is installed on a repo, you can dispatch a one-shot run targeting a specific paper or method — useful for kicking off an ad-hoc PR/Issue without waiting for the next scheduled run.

The flow is two steps: **discover** an arxiv id you want implemented, then **trigger** the action with that id pinned.


## 1. Discover

### From the engine's recommendations for an interest

```bash
remyxai papers list --interest <uuid> --period week -n 10
```

Lists the engine's top recommendations for a Research Interest over the lookback window. Each entry shows the title, arxiv URL, and a relevance score. Note the arxiv id (e.g. `2410.20305v2`) of the candidate you want to implement.

`remyxai papers digest` groups recommendations by interest if you want a cross-cutting view.

### From a free-text search of the catalog

```bash
remyxai search query "knowledge distillation" -n 5
```

Searches the engine's research-asset catalog for matches against a method or topic. Useful when you have a method in mind but don't yet know the canonical paper.

`remyxai search info <arxiv-id>` returns the full asset details (license, code repo, abstract) if you want to vet a candidate before triggering.


## 2. Trigger

```bash
# Pin to a specific paper by arxiv id
remyxai outrider trigger --repo owner/name --pin-method 2410.20305v2

# Or describe the method — the action resolves it to the top arxiv hit
remyxai outrider trigger --repo owner/name --pin-method "knowledge distillation"
```

`--pin-method` accepts either a literal arxiv id (`NNNN.NNNNN[vN]`) or a free-text method query.

- When the input matches the arxiv-id shape, the action uses **direct asset lookup** — the paper doesn't need to already be in the repo's candidate pool.
- When the input is free text, the action runs the engine's search and pins to the top hit.

In both cases the LLM selection pass is bypassed and the resolved paper goes straight to the implementation phase (Phase A audit → Phase B convention pass → Phase C test gate).

### Pre-flight

The command refuses to dispatch on repos that haven't been initialized with Outrider. If Outrider isn't installed yet, you'll see:

```
Outrider is not installed on owner/name. Install it first:
  remyxai outrider init --repo owner/name
```

### --pin-arxiv (legacy)

`--pin-arxiv` is the older form: it pins to an arxiv id but requires the paper to already be present in the repo's candidate pool. `--pin-method` is a superset (it works on arxiv ids outside the pool, via direct asset lookup), so prefer it for new uses.

The two flags are mutually exclusive — setting both is a usage error.

### Other options

| Flag | Default | What it does |
|---|---|---|
| `--repo owner/name` | cwd's git remote | Target repo |
| `--interest <uuid>` | the workflow's configured interest | Override the Research Interest for this run |
| `--ref <branch>` | the repo's default branch | The git ref to dispatch against |
