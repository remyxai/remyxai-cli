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
| `--force` | — | off (re-provision a repo that's already installed) |

### Provider keys

Every tier needs its provider's API key in the repo, or the first run fails
auth in under a second. `init` handles both halves:

- every provider you've connected at engine.remyx.ai/integrations is pushed
  server-side by the engine — one per tier, so mixing providers needs nothing
  local;
- a tier pointed at a provider you **haven't** connected is pushed by `init`
  from this shell — `ANTHROPIC_API_KEY`, `ZAI_API_KEY`, or `MOONSHOT_API_KEY`
  (via `gh`).

A tier with no key either way stops the command **before** anything is
provisioned; `--dry-run` reports the same routing. Setting a key on a repo
after the fact:

```bash
remyxai outrider set-provider-secret --repo owner/name \
  --provider moonshot --key-from ~/moonshot-key
```

Changing the tier config of a repo that's already installed needs `--force` —
plain re-runs report "already enabled" and leave the workflow files alone:

```bash
remyxai outrider init --repo your-org/your-repo --interest <uuid> \
  --refiner-provider moonshot --refiner-model kimi-k3 --force
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

Refinement runs build on work that already exists — yesterday's drafter
branch, or a third party's stalled PR branch — with a written gap analysis as
leading context:

```bash
remyxai outrider trigger --repo your-org/your-repo \
  --start-from-ref outrider/deim-draft \
  --lead-content-file gap-analysis.md \
  --staged-synthesis --fidelity-policy advisory
```

`--start-from-ref` is what the agent builds **on**; `--ref` only picks which
branch the workflow file itself comes from. When firing several dispatches at
one repo, add `--wait-for-slot`: the workflow's concurrency group holds exactly
one pending run, so back-to-back dispatches otherwise cancel each other.

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
