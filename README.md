# Remyx AI command-line client

CLI for the Remyx AI platform. Install [Outrider](https://github.com/remyxai/outrider) on a repo, manage Research Interests, browse recommended papers, and search for research assets — all from the terminal.

## Install

```bash
pip install remyxai
```

## Authenticate

Get an API key from [engine.remyx.ai/account](https://engine.remyx.ai/account) and export it:

```bash
export REMYXAI_API_KEY=<your-key>
```

All commands read this environment variable.

## Quickstart

**See today's paper recommendations**

```bash
remyxai papers digest
```

Shows recommended papers grouped by Research Interest. Use `remyxai papers list --interest <name-or-uuid>` for the flat view.

**Create a Research Interest**

```bash
# Free-form context
remyxai interests create \
  --name "LLM Efficiency" \
  --context "Quantization, speculative decoding, KV cache compression"

# From a GitHub repo, or from one of your projects
remyxai interests from-repo https://github.com/your-org/your-repo
remyxai interests from-project "Spatial VQA"
```

The pipeline matches new arXiv papers to your interests daily. See [Research Interests](#research-interests) for all three ways to create one.

**Install Outrider on a repo**

```bash
remyxai outrider init --repo owner/name --auto-interest
```

Sets up [Outrider](https://github.com/remyxai/outrider) on the target repo, server-side — your local git is never touched. See [Outrider](#outrider) for what it configures and the credentials it needs.

## Research Interests

A Research Interest is a named description of what to track; the recommendation pipeline matches new arXiv papers (and other sources) to it. Every create command kicks off a first recommendation pass automatically so the interest is populated right away — add `--wait` to block until the picks are ready, or `--no-refresh` to skip.

### From free-form context

```bash
remyxai interests create \
  --name "LLM Efficiency" \
  --context "Quantization, speculative decoding, KV cache compression"
```

`--context` also accepts a HuggingFace or GitHub URL, which the server expands into context.

### From a GitHub repo

`remyxai interests from-repo <github-url>` analyzes the repo, generates a profile of the project (themes, architecture, history), and uses it as the interest's context:

```bash
remyxai interests from-repo https://github.com/your-org/your-repo \
  --name "My Project" --daily-count 3 --automate review --wait
```

- `--automate {none|review|auto}` — paper-PR automation on the repo: `none` (default; just create the interest), `review` (open a setup PR to review), or `auto` (set it up automatically). For the full repo setup, see [Outrider](#outrider).

### From a project

`remyxai interests from-project <name-or-uuid>` builds the interest's context from a project's experiments:

```bash
# Track all experiments on the project
remyxai interests from-project "Spatial VQA" --wait

# Or curate a subset, and pin the context
remyxai interests from-project <uuid> -e "baseline-run" -e "dpo-v2" --no-auto-update
```

- `-e/--include-experiment` — track a specific experiment (repeatable); omit to track all.
- `--no-auto-update` — pin the context instead of refreshing as new experiments land.

## Outrider

[Outrider](https://github.com/remyxai/outrider) is a GitHub Action that opens pull requests integrating relevant new papers into a repo. `remyxai outrider init` sets it up for you, server-side — your local git is never touched.

```bash
remyxai outrider init --repo owner/name --auto-interest
```

This uses the **Remyx GitHub App** (`remyx-ai[bot]`) to:

- Create a Research Interest from the repo (`--auto-interest`), or use an existing one (`--interest <uuid>`)
- Write the Outrider workflow to the target repo and set its required Actions secrets
- Open a bot-authored setup PR — and, in `auto` mode, merge it and fire the first run

### Modes

Set with `--mode` (default `auto`):

- `auto` — provision, merge the setup PR, and start the first run
- `review` — provision and open the setup PR for you to review and merge

### Trigger a one-shot run

Once Outrider is installed, `remyxai outrider trigger` dispatches an ad-hoc run targeting a specific paper or method — see [docs/method-targeted-runs.md](docs/method-targeted-runs.md) for the discover → trigger workflow.

### Credentials

**You set up two things, once:**

1. Your `REMYXAI_API_KEY`, exported in your shell (see [Authenticate](#authenticate)) — this authorizes the CLI.
2. A model provider connected on the [Integrations page](https://engine.remyx.ai/integrations) — Claude Code today, more providers coming soon. The Action needs this to call the model.

**Remyx handles the rest.** You never create repo secrets by hand. During provisioning the Remyx GitHub App sets two Actions secrets on the target repo for you:

| Repo secret | What it is |
|---|---|
| `REMYX_API_KEY` | A scoped automation key Remyx mints just for this repo — *not* your personal `REMYXAI_API_KEY`, and revocable on its own. |
| `ANTHROPIC_API_KEY` (or your provider's key) | Copied from the provider you connected on the Integrations page, so the Action can call the model at runtime. |

> **No provider connected yet?** As a one-time fallback, pass `--anthropic-key` (or set `ANTHROPIC_API_KEY`) and the CLI connects it for you. Otherwise no model key ever goes on the command line.

If the Remyx GitHub App isn't installed on the target repo yet, the command surfaces the install link.

For the no-App `setup-local` path, the `--no-cron` switch, and `--bulk-repos` onboarding across many repos at once, see [docs/install-paths.md](docs/install-paths.md).

## Command reference

Run any command with `--help` for full flag listings and examples.

| Command | What it does |
|---|---|
| `remyxai papers digest` | Recommendations grouped by Research Interest |
| `remyxai papers list` | Recommendations flat view (filter by interest, period, source type) |
| `remyxai papers refresh [--wait]` | Trigger a fresh ranking |
| `remyxai papers refresh-status <task_id>` | Poll a refresh task |
| `remyxai interests list` | List your Research Interests |
| `remyxai interests get <name-or-uuid>` | Show one interest |
| `remyxai interests create` | Create an interest from free-form context |
| `remyxai interests from-repo <github-url>` | Create an interest from a GitHub repo profile |
| `remyxai interests from-project <name-or-uuid>` | Create an interest from a project's experiments |
| `remyxai interests update <id>` | Edit name / context / daily count / active state |
| `remyxai interests toggle <id>` | Flip active/inactive |
| `remyxai interests delete <id>` | Remove an interest |
| `remyxai outrider init` | Install Outrider on a GitHub repo via the Remyx App |
| `remyxai outrider setup-local` | Install Outrider via your own `gh` (no Remyx App) |
| `remyxai outrider trigger` | Dispatch a one-shot Outrider run; supports `--search-method` / `--pin-arxiv` / `--provider` / `--model` / `--claude-timeout` |
| `remyxai outrider set-provider-secret` | Set a per-provider API-key secret on a repo, safely (avoids the `gh secret set --body -` truncation trap) |
| `remyxai search query <text>` | Search the engine's research-asset catalog |
| `remyxai search list` | List recently added research assets (papers + Docker images) |
| `remyxai search info <arxiv-id>` | Asset details |

## Development

```bash
git clone https://github.com/remyxai/remyxai-cli
cd remyxai-cli
pip install -e .

# Run tests
pytest tests/
```

Releases: tag a `v*` release on GitHub and the `publish.yml` workflow builds + uploads to PyPI via Trusted Publishing (no API token stored).

## Links

- [Outrider](https://github.com/remyxai/outrider) — the GitHub Action this CLI installs
- [engine.remyx.ai](https://engine.remyx.ai) — web app, account settings, API key
- [Issues](https://github.com/remyxai/remyxai-cli/issues) — bug reports and feature requests
```
