---
type: reference
description: Three ways to create a Research Interest — free-form context, from a repo, or from a project.
tags: [interests, cli, research-profile]
---

# Research Interests

A Research Interest is a natural-language description of what your team tracks. The recommendation pipeline matches new arXiv papers (and other sources) to it. Every create command kicks off a first recommendation pass automatically — add `--wait` to block until picks are ready, or `--no-refresh` to skip.

## From free-form context

```bash
remyxai interests create \
  --name "LLM Efficiency" \
  --context "Quantization, speculative decoding, KV cache compression"
```

`--context` also accepts a HuggingFace or GitHub URL, which the server expands into context.

## From a GitHub repo

`remyxai interests from-repo <github-url>` analyzes the repo, generates a profile of the project (themes, architecture, history), and uses it as the interest's context:

```bash
remyxai interests from-repo https://github.com/your-org/your-repo \
  --name "My Project" --daily-count 3 --automate review --wait
```

- `--automate {none|review|auto}` — paper-PR automation on the repo:
  - `none` (default) — just create the interest
  - `review` — open a setup PR to review
  - `auto` — set it up automatically

For the full repo setup (bulk-install, credentials, secrets), see [install-paths.md](install-paths.md).

## From a project

`remyxai interests from-project <name-or-uuid>` builds the interest's context from a Remyx project's experiments:

```bash
# Track all experiments on the project
remyxai interests from-project "Spatial VQA" --wait

# Or curate a subset, and pin the context
remyxai interests from-project <uuid> -e "baseline-run" -e "dpo-v2" --no-auto-update
```

- `-e/--include-experiment` — track a specific experiment (repeatable); omit to track all
- `--no-auto-update` — pin the context instead of refreshing as new experiments land

## Managing interests

```bash
remyxai interests list                       # show all your interests
remyxai interests get <name-or-uuid>         # show one interest's details
remyxai interests update <id> --context ...  # edit fields
remyxai interests toggle <id>                # flip active/inactive
remyxai interests delete <id>                # remove
```

An inactive interest is excluded from the daily digest until toggled back on. Deleting an interest also removes all its recommendations.

## What the context body looks like

Interests are prompt-anchor artifacts — the recommendation pipeline consumes the context as natural-language framing for the ranker. Good interest contexts:

- Name specific method families ("Quantization, speculative decoding, KV cache compression"), not broad domains ("LLMs")
- Include the codebase's shape when relevant ("This is a training library; algorithm-add papers implementing new optimizers are the primary target")
- Explicitly scope out adjacent-but-off-domain areas ("Not interested in general-purpose agent frameworks; only LoRA / PEFT optimizer methods")

`interests from-repo` produces a reasonable context automatically for most repos. Hand-editing via `interests update --context ...` afterward tightens the ranker's picks when the auto-extract is too broad or catches the wrong theme.
