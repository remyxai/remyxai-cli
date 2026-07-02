# Remyx AI CLI

Automate paper-to-PR delivery for the teams building on top of your code. Install [Outrider](https://github.com/remyxai/outrider) on a GitHub repo, and Remyx handles the loop end-to-end: **discovers** newly-published arXiv methods matching your team's work, **implements** them as draft PRs wired into your existing call sites, and **validates** each draft against the paper's reference implementation before it lands.

The CLI is your control surface for the whole loop.

## Install

```bash
pip install remyxai
export REMYXAI_API_KEY=<your-key>   # from engine.remyx.ai/account
```

## Quickstart

**Install Outrider on your repo.** One command; Remyx handles the workflow, secrets, and first run server-side (no local git touched):

```bash
remyxai outrider init --repo your-org/your-repo --auto-interest
```

`--auto-interest` extracts a Research Interest from your repo's commit history + README. It's what Outrider matches new arXiv papers against.

**Three ways to trigger a run after init.** Each is a different level of specificity — pick by how much you want to override Remyx's own ranking:

```bash
# 1. Default: let Remyx pick the best implementable paper from your interest's
#    ranked pool. Outrider's audit augments via agentic refine-queries;
#    Claude Code selects the top match against your codebase's actual call sites.
remyxai outrider trigger --repo your-org/your-repo

# 2. --search-method: override the ranker with a free-text query.
#    Remyx searches its catalog and Outrider implements the top hit.
#    Use when you know the method family but not the specific paper.
remyxai outrider trigger --repo your-org/your-repo \
  --search-method "riemannian preconditioning LoRA optimizer"

# 3. --pin-arxiv: implement THIS specific paper. Bypasses ranking entirely.
#    Use for reproducible re-runs, retries, or when you already know the arxiv id.
remyxai outrider trigger --repo your-org/your-repo --pin-arxiv 2402.02347v3
```

## How the pipeline scales

Systematizing paper adoption across a team is usually a coordination problem, not a technical one — someone spots the paper, someone else evaluates fit, someone else implements, someone else reviews. Remyx + Outrider collapses that chain into a single automated pipeline that your team gates rather than staffs.

**Discovery.** Every day, Remyx ranks the latest arXiv against your Research Interests (extracted from your codebase, curated per-team, or written by hand). See what surfaced:

```bash
remyxai papers digest                             # grouped by interest
remyxai papers list --interest "My Project" -n 5  # flat view, top 5
```

For an interest, Outrider runs an audit pass that agentically explores your codebase, refines the recommendation pool via targeted searches for under-represented themes, and picks the paper most directly implementable against real call sites. Nothing gets drafted against imagined structure.

**Implementation.** Once a paper is selected, Outrider clones the target repo and invokes Claude Code with a scoped brief: implement the paper's core contribution as a draft PR, wired into an existing call site, honoring your repo's contribution conventions (extracted from recent merged PRs). If the paper can't be cleanly scaffolded — no natural integration point, or too large a scope for one PR — Outrider opens a design-discussion Issue instead. The routing decision is measurement-based, not aspirational.

**Validation.** Before a PR lands as ready-for-review, three passes run automatically:

- **Fidelity audit** — clones the paper's reference implementation and diffs the draft against it, flagging any invented algorithms, wrong schemas, or missing components the paper actually requires. This is what catches plausibility-optimized fabrications before they reach a human reviewer.
- **Convention pass** — folds the PR body and code layout to your target repo's contribution conventions (PR-body scaffold, AI-disclosure, docstring style, test co-location patterns), extracted from recent merged PRs.
- **Test gate** — runs lint + targeted tests on the touched files; drops the Draft state only if lint passes and tests execute cleanly.

Each pass writes its findings to the run's Actions Summary — you inspect one panel to see the drafting, verification, and refinement decisions for every recommendation.

## Team-scale patterns

**Bulk-install across an org's repos.** Install Outrider once across every active repo, each with an auto-extracted interest:

```bash
remyxai outrider init --bulk-repos repos.tsv --mode review
```

`repos.tsv` is a two-column file (`owner/name<TAB>interest-uuid-or-empty`); rows with an empty interest column trigger `--auto-interest`. See [docs/install-paths.md](docs/install-paths.md) for the full bulk flow.

**Coordinate a specific paper across multiple frameworks.** Same paper, different codebases, one dispatch loop:

```bash
for repo in your-org/framework-a your-org/framework-b your-org/framework-c; do
  remyxai outrider trigger --repo "$repo" --pin-arxiv 2402.02347v3
done
```

Fidelity's cross-fork consistency check catches drift: if the same paper's core algorithm ends up implemented three different ways across three drafts, that's a signal about the codebase, not the paper.

**Route the model provider per dispatch.** Different backends for different cost/quality tradeoffs:

```bash
remyxai outrider trigger --repo your-org/your-repo \
  --pin-arxiv 2402.02347v3 --provider zai --model glm-5.2
```

See [`docs/method-targeted-runs.md`](docs/method-targeted-runs.md) for the full discover-then-trigger workflow (including `remyxai search query` to surface candidate methods before pinning).

## Research Interests

An interest is a natural-language description of what your team tracks. Three ways to create one:

```bash
# From free-form context (or a HuggingFace/GitHub URL, expanded server-side)
remyxai interests create --name "LLM Efficiency" \
  --context "Quantization, speculative decoding, KV cache compression"

# From a GitHub repo — extracts themes, architecture, and commit history
remyxai interests from-repo https://github.com/your-org/your-repo --wait

# From a project's experiments (all, or a curated subset via -e/--include-experiment)
remyxai interests from-project "Spatial VQA" --wait
```

Every create command kicks off a first recommendation pass automatically. Use `--wait` to block until picks are ready.

## Credentials

You set up two things once:

1. **`REMYXAI_API_KEY`** in your shell (from [engine.remyx.ai/account](https://engine.remyx.ai/account)) — authorizes the CLI.
2. **A model provider** connected on the [Integrations page](https://engine.remyx.ai/integrations) — Claude Code today, more providers coming soon. Outrider calls the model at runtime with this.

During provisioning, the Remyx GitHub App creates the target repo's secrets for you (a scoped `REMYX_API_KEY` and the provider's key) — no manual `gh secret set` steps. If the App isn't installed on the target yet, the CLI surfaces the install link.

For the no-App `setup-local` path (uses your own `gh`), see [docs/install-paths.md](docs/install-paths.md).

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
