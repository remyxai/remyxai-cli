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
remyxai outrider trigger --repo owner/name --pin-method 2410.20305v2 \
  --provider zai --model glm-5.2

# Or compare against an older z.ai model on the same paper/repo:
remyxai outrider trigger --repo owner/name --pin-method 2410.20305v2 \
  --provider zai --model glm-4.6

# Pin to a specific Anthropic model (e.g. for an A/B against Sonnet):
remyxai outrider trigger --repo owner/name --pin-method 2410.20305v2 \
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
  --repo owner/name --pin-method 2410.20305v2 \
  --backend glm --claude-timeout 1200
```

The CLI rejects values below 60 seconds at the command boundary. The action itself parses the input as an integer; non-integer values fail fast at the workflow's `INPUT_CLAUDE_TIMEOUT` parser.
