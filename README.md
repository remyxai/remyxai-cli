# Remyx AI CLI

Install [Outrider](https://github.com/remyxai/outrider) on a repo, and Remyx handles the loop end-to-end: **discovers** newly-published arXiv methods matching your team's work, **implements** them as draft PRs, and **validates** each against the paper's reference before it lands.

## Install

```bash
pip install remyxai
export REMYXAI_API_KEY=<your-key>   # from engine.remyx.ai/account
```

## Quickstart

Install Outrider on a repo — server-side, no local git touched:

```bash
remyxai outrider init --repo your-org/your-repo --auto-interest
```

Trigger a run — three modes of specificity:

```bash
# 1. default — Remyx picks from the ranked pool; Outrider's audit augments via agentic search
remyxai outrider trigger --repo your-org/your-repo

# 2. --search-method — override the pool with a free-text query, implement the top hit
remyxai outrider trigger --repo your-org/your-repo \
  --search-method "riemannian preconditioning LoRA optimizer"

# 3. --pin-arxiv — exact paper, bypasses selection entirely
remyxai outrider trigger --repo your-org/your-repo --pin-arxiv 2402.02347v3
```

## Documentation

| Topic | Doc |
|---|---|
| Pipeline: discovery → implementation → validation | [docs/pipeline.md](docs/pipeline.md) |
| Method-targeted runs + team-scale patterns | [docs/method-targeted-runs.md](docs/method-targeted-runs.md) |
| Research Interests: three ways to create one | [docs/research-interests.md](docs/research-interests.md) |
| Install paths, credentials, bulk-install | [docs/install-paths.md](docs/install-paths.md) |

## Command reference

Run any command with `--help` for full flag listings and examples.

| Command | What it does |
|---|---|
| `remyxai outrider init` | Install Outrider on a repo via the Remyx App |
| `remyxai outrider setup-local` | Install Outrider via your own `gh` (no Remyx App) |
| `remyxai outrider trigger` | Dispatch a one-shot run (`--search-method` / `--pin-arxiv` / `--provider` / `--model` / `--claude-timeout`) |
| `remyxai outrider set-provider-secret` | Set a per-provider API-key secret on a repo, safely |
| `remyxai papers digest` | Recommendations grouped by Research Interest |
| `remyxai papers list` | Recommendations flat view (filter by interest, period, source type) |
| `remyxai papers refresh [--wait]` | Trigger a fresh ranking |
| `remyxai interests list` | List your Research Interests |
| `remyxai interests get <name-or-uuid>` | Show one interest |
| `remyxai interests create` | Create an interest from free-form context |
| `remyxai interests from-repo <github-url>` | Create an interest from a GitHub repo profile |
| `remyxai interests from-project <name-or-uuid>` | Create an interest from a project's experiments |
| `remyxai interests update <id>` | Edit name / context / daily count / active state |
| `remyxai interests toggle <id>` | Flip active/inactive |
| `remyxai search query <text>` | Search the engine's research-asset catalog |
| `remyxai search info <arxiv-id>` | Asset details |

## Development

```bash
git clone https://github.com/remyxai/remyxai-cli
cd remyxai-cli
pip install -e .
pytest tests/
```

## Links

- [Outrider](https://github.com/remyxai/outrider) — the GitHub Action this CLI installs
- [engine.remyx.ai](https://engine.remyx.ai) — web app, account settings, API key
- [Issues](https://github.com/remyxai/remyxai-cli/issues) — bug reports and feature requests
