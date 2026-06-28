---
type: howto
description: Two paths to install Outrider on a repo, when to use each, and how to onboard many repos at once.
tags: [outrider, install, github-app, setup-local, bulk-repos]
---

# Installing Outrider on a repo

There are two CLI paths to install Outrider on a target repo, plus a bulk path for onboarding many repos in one shot.


## `outrider init` — the default

Drives the Remyx engine to register the **Remyx GitHub App** (`remyx-ai[bot]`) on the target, write the workflow, set the repo's Actions secrets, and open a bot-authored setup PR. In `auto` mode it also merges the PR and fires the first run.

```bash
remyxai outrider init --repo owner/name --auto-interest
```

The App is the load-bearing piece. At runtime, the workflow asks the engine to mint a short-lived `remyx-ai[bot]` token so the action's PRs and Issues are authored by the bot — not `github-actions[bot]`. **Without the App installed on the target, that token-mint silently returns empty and the action falls back to the workflow's built-in `GITHUB_TOKEN`** — the run still succeeds but the artifacts are anonymous and skip the convention-pass enrichments that key off the bot author. Always prefer this path when the org can grant the App.

If the App isn't installed yet, the command surfaces the install link — accept it once and the engine handles the rest.


## `outrider setup-local` — when the App can't be granted

For enterprises that can't grant a third-party App yet (pending security review, restricted org policy):

```bash
remyxai outrider setup-local --repo owner/name --auto-interest
```

Uses your own authenticated `gh` CLI to set the repo secrets, write the workflow, and (in `auto` mode) merge the setup PR. No Remyx App, nothing new to security-review. The only Remyx dependency is the `REMYX_API_KEY` the workflow uses at runtime.

The action then opens its PRs / Issues with the repo's built-in `GITHUB_TOKEN` (authored by `github-actions[bot]` rather than `remyx-ai[bot]`).

### `--no-cron`

By default the workflow ships with a scheduled cron at 14:00 UTC Mondays. For trials, cost control, or any "I'll dispatch manually" use case:

```bash
remyxai outrider setup-local --repo owner/name --interest <uuid> --no-cron
```

The schedule block is rendered commented-out (not removed entirely), so re-enabling later means uncommenting three lines — no need to re-run setup-local.

Engine-side `outrider init --no-cron` is not yet supported; for now, prefer `setup-local --no-cron` if you need that knob.


## Don't write the workflow file manually

Committing `.github/workflows/outrider.yml` by hand (without one of the two CLI paths above) skips the App-install / secrets-set steps. The action will deploy and run, but:

- The bot-token mint silently fails → PRs are `github-actions[bot]`-authored
- The `REMYX_API_KEY` / `ANTHROPIC_API_KEY` Actions secrets must be set manually

If you've already done this and your runs are producing 0 artifacts, run `remyxai outrider init --repo owner/name --interest <uuid>` after the fact — it's idempotent for the secrets-set step and surfaces the App install link if needed.


## `--bulk-repos` — many repos in one shot

Onboarding a portfolio of forks or installing across a team's repos:

```bash
remyxai outrider init --bulk-repos repos.tsv --mode review --yes
```

`repos.tsv` is a tab-separated mapping of repo → ResearchInterest UUID, one per line. Blank lines and `#`-prefixed comments are allowed:

```
# wave-1 forks
smellslikeml/agents	649828cc-11ec-4ce5-9549-303d1da6f1ce
smellslikeml/helicone	734a06d2-84c0-4b3f-bb5f-d0ab476b36a9

# wave-2
smellslikeml/promptfoo	3ede9f1b-ac4b-4468-9718-7314acd85fce
```

Per-repo errors are captured and reported at the end — one failure does not abort the remaining rows. Use `--pace SECONDS` to tune the inter-repo pacing (default 3s).

`--bulk-repos` is also supported on `setup-local` with the same TSV format.

`--bulk-repos` is mutually exclusive with `--repo` / `--interest` / `--auto-interest` (the TSV is the source of truth for the per-row pair).
