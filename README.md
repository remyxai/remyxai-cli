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
| Full command reference | [docs/commands.md](docs/commands.md) |

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
