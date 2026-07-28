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

By default this installs the **two-tier** setup: a daily **drafter** that
explores candidates as fork branches, plus a weekly **refiner** that promotes
the strongest draft into a ready-for-review PR (crons stay off — drive runs
with `remyxai outrider trigger` or your own dispatcher). The Remyx GitHub App
authors everything (`remyx-ai[bot]`).

### Model providers

Any provider works, all on equal footing — pick whichever you've connected:

| Provider | `--provider` value | Example models |
|---|---|---|
| Claude Code (Anthropic) | `anthropic` | `claude-opus-4-8` |
| Z.ai | `zai` | `glm-5.2` |
| Moonshot AI | `moonshot` | `kimi-k3` |

Unset, each tier follows your **connected** provider — no vendor is assumed.
One provider covers both tiers; tune per tier, or opt out of two-tier entirely:

```bash
# Both tiers on one provider (any of the three)
remyxai outrider init --repo your-org/your-repo --auto-interest --provider moonshot

# Mix providers per tier (cheap drafter + capable refiner)
remyxai outrider init --repo your-org/your-repo --auto-interest \
  --drafter-provider zai --drafter-model glm-5.2 \
  --refiner-provider anthropic

# Plain single-file workflow instead of two-tier
remyxai outrider init --repo your-org/your-repo --auto-interest --single-tier
```

| Flag | Applies to | Default |
|---|---|---|
| `--provider` / `--model` | both tiers | your connected provider / each tier's default |
| `--drafter-provider` / `--drafter-model` | daily drafter only | shared provider / tier default |
| `--refiner-provider` / `--refiner-model` | weekly refiner only | shared provider / tier default |
| `--single-tier` | — | off (two-tier is the default) |

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
