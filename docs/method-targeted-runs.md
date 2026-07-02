---
type: howto
description: Discover a relevant paper or method, then dispatch a targeted Outrider run that implements it.
tags: [outrider, cli, search-method, pin-arxiv, workflow_dispatch]
---

# Method-targeted Outrider runs

Once [Outrider](https://github.com/remyxai/outrider) is installed on a repo, you can dispatch a one-shot run targeting a specific paper or method — useful for kicking off an ad-hoc PR/Issue without waiting for the next scheduled run.

Three modes of specificity, in ascending order of override:

1. **Default (no pin)** — Remyx ranks candidates from the interest-scoped pool + Outrider's audit augments via agentic refine-queries; Claude Code picks the best implementation from the ranked pool.
2. **`--search-method`** — overrides the ranked pool with an engine search on your query; implements the top hit.
3. **`--pin-arxiv`** — implements the exact arxiv paper; bypasses ranking entirely.

The flow is two steps: **discover** what you want implemented, then **trigger** the action.


## 1. Discover

### From the engine's recommendations for an interest

```bash
remyxai papers list --interest <uuid> --period week -n 10
```

Lists the engine's top recommendations for a Research Interest over the lookback window. Each entry shows the title, arxiv URL, and a relevance score. Note the arxiv id (e.g. `2402.02347v3`) of the candidate you want to implement.

`remyxai papers digest` groups recommendations by interest if you want a cross-cutting view.

### From a free-text search of the catalog

```bash
remyxai search query "riemannian preconditioning LoRA" -n 5
```

Searches the engine's research-asset catalog for matches against a method or topic. Useful when you have a method in mind but don't yet know the canonical paper.

`remyxai search info <arxiv-id>` returns the full asset details (license, code repo, abstract) if you want to vet a candidate before triggering.


## 2. Trigger

### `--pin-arxiv` — exact paper (recommended for known arxiv ids)

```bash
remyxai outrider trigger --repo owner/name --pin-arxiv 2402.02347v3
```

Implements exactly this paper. Bypasses the ranker's pool entirely — Remyx fetches the paper directly from its asset catalog and forwards it to Claude Code for implementation. Works even if the paper isn't in the repo's interest-scoped candidate pool.

Use for:
- Reproducible re-runs (same paper, same repo, deterministic paper input)
- Retries after a timeout or backend failure
- Cases where you've already identified the paper via `remyxai search info` or elsewhere

### `--search-method` — free-text method query

```bash
remyxai outrider trigger --repo owner/name \
  --search-method "riemannian preconditioning LoRA optimizer"
```

Runs an engine search over the paper catalog and implements the top hit. Use for exploratory dispatches when you know the method family but haven't pinned down a specific paper.

The top-hit semantic means: **whichever arxiv paper the engine returns first for your query gets implemented**. Not deterministic across search index updates. If reproducibility matters, use `--pin-arxiv` with the resolved arxiv id instead.

The two flags are mutually exclusive — setting both is a usage error.

### Both paths bypass selection

In both `--pin-arxiv` and `--search-method` cases, the LLM selection pass is skipped and the resolved paper goes straight to preflight → implementation → refinement chain (fidelity audit → convention pass → test gate).


### Pre-flight

The command refuses to dispatch on repos that haven't been initialized with Outrider. If Outrider isn't installed yet, you'll see:

```
Outrider is not installed on owner/name. Install it first:
  remyxai outrider init --repo owner/name
```


### Other options

| Flag | Default | What it does |
|---|---|---|
| `--repo owner/name` | cwd's git remote | Target repo |
| `--interest <uuid>` | the workflow's configured interest | Override the Research Interest for this run |
| `--ref <branch>` | the repo's default branch | The git ref to dispatch against |
| `--provider <name>` | the workflow's default (`anthropic`) | Route Claude Code at a specific model provider for this dispatch (`anthropic` or `zai`). See [Provider + model routing](#provider--model-routing) below |
| `--model <name>` | (provider default) | Specific model to request from the provider (e.g. `claude-opus-4-7`, `glm-5.2`, `glm-4.6`). Forwarded as `ANTHROPIC_MODEL` env. Empty = the provider picks |
| `--claude-timeout <seconds>` | the action's 900s default | Wall-clock ceiling for the Claude Code agent calls on this dispatch (preflight + implementation share the budget). Raise for very large monorepos |


## Provider + model routing

`--provider` selects the API endpoint (the company); `--model` picks the specific model from that provider. `setup-local`-generated workflows declare both as workflow_dispatch inputs and include a `Configure provider auth` step that picks the right auth env var + sets `ANTHROPIC_MODEL` per dispatch.

The only setup besides `outrider setup-local` is putting the alternate provider's API key in the repo's secrets:

```bash
# 1. Drop the z.ai key into the repo's secrets (safely).
remyxai outrider set-provider-secret \
  --repo owner/name --provider zai --key-from ~/zai-key

# 2. Route this run at z.ai's GLM-5.2; scheduled cron runs continue
#    to use the workflow's default (Anthropic).
remyxai outrider trigger --repo owner/name --pin-arxiv 2402.02347v3 \
  --provider zai --model glm-5.2

# Or compare against an older z.ai model on the same paper/repo:
remyxai outrider trigger --repo owner/name --pin-arxiv 2402.02347v3 \
  --provider zai --model glm-4.6

# Pin to a specific Anthropic model (e.g. for an A/B against Sonnet):
remyxai outrider trigger --repo owner/name --pin-arxiv 2402.02347v3 \
  --provider anthropic --model claude-sonnet-4-6
```

When `--provider` is omitted, the workflow's own default applies — so customers who only ever route at Anthropic don't need to think about it. When `--model` is omitted, the provider picks its current default model. See the action's [`docs/backends.md`](https://github.com/remyxai/outrider/blob/main/docs/backends.md) for the underlying auth-header matrix and the per-provider rate table that drives cost telemetry.


## Setting provider secrets safely: `set-provider-secret`

`gh secret set` has a well-known footgun: with `--body -` and disconnected stdin, it stores the literal `-` character as the secret value and the workflow then sees `Authorization: Bearer -` on every call (HTTP 401, hours of debugging). `outrider set-provider-secret` wraps it with the pitfalls handled:

```bash
remyxai outrider set-provider-secret \
  --repo owner/name --provider zai --key-from ~/zai-key
```

What it does:

- Reads the key from a file (never argv, never literal `--body -` with disconnected stdin)
- Strips a single trailing newline (the common shape from `printf '%s\n' "$KEY" > FILE`)
- Refuses to set the literal `"-"` value
- Refuses empty values
- Warns on suspiciously-short keys (`< 16` chars)
- Pipes the value via stdin to `gh secret set` — the secret never appears in process argv or shell logs

Maps `--provider` to the secret name the workflow's `Configure provider auth` step reads:

| `--provider` | Secret name set |
|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` |
| `zai` | `ZAI_API_KEY` |

Future providers extend this mapping (Bedrock → `AWS_BEARER_TOKEN_BEDROCK`, etc.) as the rate table grows.


## Long-running repos: `--claude-timeout`

The action's `claude-timeout` input ceilings the wall-clock budget on the implementation call. Default is 900s, which is comfortable for typical repos. Very large monorepos (50K+ files) and non-Anthropic backends (which tend to run 1.1–2.7× slower) can exceed it.

```bash
# Bump to 1200s for a sprawling monorepo on a slower backend.
remyxai outrider trigger \
  --repo owner/name --pin-arxiv 2402.02347v3 \
  --provider zai --claude-timeout 1200
```

The CLI rejects values below 60 seconds at the command boundary. The action itself parses the input as an integer; non-integer values fail fast at the workflow's `INPUT_CLAUDE_TIMEOUT` parser.


## Team-scale patterns

### Bulk-install across an org's repos

Install Outrider once across every active repo, each with an auto-extracted interest:

```bash
remyxai outrider init --bulk-repos repos.tsv --mode review
```

`repos.tsv` is a two-column file (`owner/name<TAB>interest-uuid-or-empty`); rows with an empty interest column trigger `--auto-interest`. See [install-paths.md](install-paths.md) for the full bulk flow.

### Coordinate a specific paper across multiple frameworks

Same paper, different codebases, one dispatch loop:

```bash
for repo in your-org/framework-a your-org/framework-b your-org/framework-c; do
  remyxai outrider trigger --repo "$repo" --pin-arxiv 2402.02347v3
done
```

Fidelity's cross-fork consistency check catches drift: if the same paper's core algorithm ends up implemented three different ways across three drafts, that's a signal about the codebase, not the paper.

### Retry a run under a different provider

If a run timed out under GLM (which tends to run 1.1–2.7× slower than Anthropic on the same paper), retry it under Anthropic:

```bash
# First attempt hit the wall-clock ceiling under GLM
remyxai outrider trigger --repo owner/name --pin-arxiv 2402.02347v3 \
  --provider zai --claude-timeout 1500

# Retry under Anthropic at the default timeout
remyxai outrider trigger --repo owner/name --pin-arxiv 2402.02347v3 \
  --provider anthropic
```

`--pin-arxiv` guarantees reproducibility across retries — the second attempt implements the same paper the first one targeted, so the comparison isolates the provider/backend variable.
