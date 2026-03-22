---
name: remyx
description: >
  Fetches today's recommendations from the Remyx AI CLI using
  `remyxai papers digest`, formats a Slack digest grouped by Research Interest,
  and posts it to #research. Handles multiple resource types (arXiv papers,
  GitHub repos) via a source_type discriminator. Triggered manually or by the
  9 AM PT weekday cron.
metadata:
  openclaw:
    requires:
      bins:
        - remyxai
      env:
        - REMYXAI_API_KEY
---

# Remyx Daily Recommendations Skill

## Purpose

Calls `remyxai papers digest` to pull today's top recommendations across all
Research Interest profiles, formats a Slack digest, and posts to #research.

The digest is source-agnostic: it currently contains arXiv papers, and will
include GitHub repos and other resource types as GitRank expands. The skill
handles each type correctly without needing to be updated.

---

## Slack channel

The #research channel ID is `C0ANWMAE88G`.

When posting to Slack, always use the channel ID `C0ANWMAE88G` directly via
the message tool. Do NOT use `sessions.resolve` or attempt to look up the
channel by name — this will fail. The correct target is `channel:C0ANWMAE88G`.

---

## Steps when triggered

### Step 1 — Fetch digest

```bash
remyxai papers digest --period today --limit 5 --format json
```

Response shape:
```json
{
  "date": "2026-03-21",
  "period": "today",
  "source_types": ["arxiv_paper"],
  "interests": [
    {
      "id": "<uuid>",
      "name": "RAG & Retrieval",
      "count": 3,
      "recommendations": [
        {
          "recommendation_id": "<uuid>",
          "source_type": "arxiv_paper",
          "resource_id": "2403.01234",
          "title": "Adaptive Retrieval for...",
          "url": "https://arxiv.org/abs/2403.01234",
          "relevance_score": 0.87,
          "reasoning": "Directly addresses...",
          "suggested_experiment": "Run BEIR benchmark...",
          "interest_name": "RAG & Retrieval",
          "resource": {
            "arxiv_id": "2403.01234",
            "authors": ["Wang", "Li", "Chen"],
            "abstract_summary": "...",
            "has_docker": true,
            "docker_image": "remyxai/240301234:latest",
            "github_url": "https://github.com/..."
          }
        }
      ],
      "count": 3
    }
  ],
  "total_papers": 8
}
```

**If `total_papers == 0`:** report "no new recommendations today" and offer to run:
```bash
# Refresh all interests
remyxai papers refresh --wait

# Refresh one interest only
remyxai papers refresh --interest-id <uuid> --wait
```
Then re-fetch and post.

**On error:** "401" → check `REMYXAI_API_KEY`; network timeout → retry once.

### Step 2 — Format the Slack message

Key off `source_type` to format each item. Always use top-level `title` and
`url` — they are safe to render for any source type.

**For `source_type: "arxiv_paper"`:**
```
{rank}. *<{url}|{title}>*
   _{authors, truncated to 3 + "et al."}_
   {reasoning[:160] or abstract_summary}
   🐳 {docker_image}  |  ⑂ <{github_url}|GitHub>   ← only if present
```

**For `source_type: "github_repo"` (future):**
```
{rank}. *<{url}|{title}>*
   {language} · ★ {stars}
   {reasoning[:160]}
```

**For unknown source types:** render title, url, and reasoning only — never
fail because of an unrecognised type.

Full digest structure:
```
🔬 *Remyx Daily Recommendations* — March 21

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

*RAG & Retrieval*  _3 items_

1. *<url|title>*
   ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_Total: 8 · Powered by Remyx AI · ExperimentOps_
```

Keep under 40,000 chars; trim lowest-scored items first if needed.

### Step 3 — Post or preview

- "Show me today's recommendations" → display in chat only
- "Post today's recommendations" / "post to #research" → post to Slack channel C0ANWMAE88G

### Step 4 — Post to Slack

Send the formatted digest to Slack channel `C0ANWMAE88G` (#research) using the
message tool directly. Do NOT use `sessions.resolve` — always use the channel
ID `C0ANWMAE88G` as the target.

On Slack failure: display the digest in chat and report the error.

---

## Other commands

| Intent | Command |
|---|---|
| Trigger fresh recommendations | `remyxai papers refresh --wait` |
| Papers only | `remyxai papers list --source-type arxiv_paper --period today` |
| GitHub repos only (future) | `remyxai papers list --source-type github_repo` |
| One interest only | `remyxai papers list --interest-id <uuid> --period today` |
| Poll a refresh task | `remyxai papers refresh-status <task_id>` |
| Last week's digest | `remyxai papers digest --period week` |

---

## Cron setup (say this to your agent once)

```
Set up the Remyx daily recommendations cron to post to #research every weekday at 9 AM Pacific.
```

The agent will run:
```bash
openclaw cron add \
  --name "Remyx Daily Recommendations" \
  --cron "0 9 * * 1-5" \
  --tz "America/Los_Angeles" \
  --session isolated \
  --message "Fetch today's Remyx recommendations and post the digest to #research on Slack." \
  --announce \
  --channel slack \
  --to "channel:C0ANWMAE88G" \
  --light-context
```
