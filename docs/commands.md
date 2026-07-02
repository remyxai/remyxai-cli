---
type: reference
description: Full command reference for the Remyx AI CLI.
tags: [cli, reference, commands]
---

# Command reference

Run any command with `--help` for full flag listings and examples — that's the authoritative source.

## Outrider

| Command | What it does |
|---|---|
| `remyxai outrider init` | Install Outrider on a repo via the Remyx App |
| `remyxai outrider setup-local` | Install Outrider via your own `gh` (no Remyx App) |
| `remyxai outrider trigger` | Dispatch a one-shot run (`--search-method` / `--pin-arxiv` / `--provider` / `--model` / `--claude-timeout`) |
| `remyxai outrider set-provider-secret` | Set a per-provider API-key secret on a repo, safely |

See also: [method-targeted-runs.md](method-targeted-runs.md) for `outrider trigger` in depth, [install-paths.md](install-paths.md) for `init` vs `setup-local` and bulk-install.

## Papers

| Command | What it does |
|---|---|
| `remyxai papers digest` | Recommendations grouped by Research Interest |
| `remyxai papers list` | Recommendations flat view (filter by interest, period, source type) |
| `remyxai papers refresh [--wait]` | Trigger a fresh ranking |
| `remyxai papers refresh-status <task_id>` | Poll a refresh task |

## Interests

| Command | What it does |
|---|---|
| `remyxai interests list` | List your Research Interests |
| `remyxai interests get <name-or-uuid>` | Show one interest |
| `remyxai interests create` | Create an interest from free-form context |
| `remyxai interests from-repo <github-url>` | Create an interest from a GitHub repo profile |
| `remyxai interests from-project <name-or-uuid>` | Create an interest from a project's experiments |
| `remyxai interests update <id>` | Edit name / context / daily count / active state |
| `remyxai interests toggle <id>` | Flip active/inactive |
| `remyxai interests delete <id>` | Remove an interest |

See also: [research-interests.md](research-interests.md) for the three interest-creation flavors in depth.

## Search

| Command | What it does |
|---|---|
| `remyxai search query <text>` | Search the engine's research-asset catalog |
| `remyxai search list` | List recently added research assets (papers + Docker images) |
| `remyxai search info <arxiv-id>` | Asset details |
