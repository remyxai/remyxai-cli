# Remyx AI command-line client

CLI for the Remyx AI platform. Install [Outrider](https://github.com/remyxai/outrider) on a repo, manage Research Interests, browse Gemini-ranked paper recommendations from GitRank, and search for research assets — all from the terminal.

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

# Or seed from a GitHub repo (auto-extracts context)
remyxai interests create --context "https://github.com/your-org/your-repo"
```

The recommendation pipeline matches new arXiv papers to your interests daily. First run takes 40-120s to populate the pool; subsequent runs are instant.

**Install Outrider on a repo**

```bash
remyxai outrider init --repo owner/name --auto-interest
```

Drives the Remyx engine to install [Outrider](https://github.com/remyxai/outrider) on the target repo via the Remyx GitHub App: writes the workflow, sets the repo secrets, and opens a bot-authored setup PR. Your local git isn't touched. Requires `REMYXAI_API_KEY` and an Anthropic key for Claude Code (`--anthropic-key` or `ANTHROPIC_API_KEY`).

If the Remyx GitHub App isn't installed on the target repo yet, the command surfaces the install link.

## Command reference

Run any command with `--help` for full flag listings and examples.

| Command | What it does |
|---|---|
| `remyxai papers digest` | Recommendations grouped by Research Interest |
| `remyxai papers list` | Recommendations flat view (filter by interest, period, source type) |
| `remyxai papers refresh [--wait]` | Trigger a fresh Gemini re-ranking |
| `remyxai papers refresh-status <task_id>` | Poll a refresh task |
| `remyxai interests list` | List your Research Interests |
| `remyxai interests get <name-or-uuid>` | Show one interest |
| `remyxai interests create` | Create a new interest |
| `remyxai interests update <id>` | Edit name / context / daily count / active state |
| `remyxai interests toggle <id>` | Flip active/inactive |
| `remyxai interests delete <id>` | Remove an interest |
| `remyxai outrider init` | Install Outrider on a GitHub repo via the Remyx App |
| `remyxai search list` | List recently added research assets (papers + Docker images) |
| `remyxai search info <arxiv-id>` | Asset details |
| `remyxai list-models` | List available trained models on your account |
| `remyxai summarize-model <name>` | Show a model's summary |
| `remyxai deploy-model <name> <up\|down>` | Bring a containerized deployment up or down |
| `remyxai dataset <action> [name]` | Manage datasets (`list`, `download`, `delete`) |

## Outrider install — what happens

`remyxai outrider init` calls the Remyx engine, which uses the **Remyx GitHub App** (`remyx-ai[bot]`) to:

- Write the Outrider workflow to the target repo
- Set the workflow's required Actions secrets
- Open a bot-authored setup PR (and merge it automatically in `--mode auto`)
- Fire the first Outrider run

Your local git is not touched. The only credential you provide is your `REMYXAI_API_KEY`, which authorizes the engine to act on your behalf through the App — it is not copied into the target repo's secrets. Anthropic and GitHub credentials needed at workflow runtime are configured by the App.

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
