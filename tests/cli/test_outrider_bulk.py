"""Tests for the bulk-repos + no-cron CLI improvements — REMYX-147.

Covers:
- TSV parser (happy path, bad rows surfaced with line numbers, blanks /
  comments skipped, empty file rejected)
- _run_bulk loop (per-row error capture, continue-on-error, pacing)
- --no-cron renders the schedule block commented-out (not absent)
- --bulk-repos / --repo mutex enforced at the click layer
"""
import textwrap
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from remyxai.cli import outrider_actions, outrider_local
from remyxai.cli.commands import cli


# ─── _parse_bulk_repos_tsv ─────────────────────────────────────────────────


def _write_tsv(tmp_path, content):
    p = tmp_path / "repos.tsv"
    p.write_text(content)
    return str(p)


def test_parse_tsv_happy(tmp_path):
    path = _write_tsv(tmp_path, textwrap.dedent("""\
        # comment row, skipped
        owner/repo-a\t00000000-0000-0000-0000-000000000001

        owner/repo-b\t00000000-0000-0000-0000-000000000002
    """))
    rows = outrider_actions._parse_bulk_repos_tsv(path)
    assert rows == [
        ("owner/repo-a", "00000000-0000-0000-0000-000000000001"),
        ("owner/repo-b", "00000000-0000-0000-0000-000000000002"),
    ]


def test_parse_tsv_missing_file_raises():
    with pytest.raises(click.UsageError, match="not found"):
        outrider_actions._parse_bulk_repos_tsv("/no/such/path.tsv")


def test_parse_tsv_empty_file_raises(tmp_path):
    path = _write_tsv(tmp_path, "")
    with pytest.raises(click.UsageError, match="no installable rows"):
        outrider_actions._parse_bulk_repos_tsv(path)


def test_parse_tsv_bad_rows_surfaces_line_numbers(tmp_path):
    path = _write_tsv(tmp_path, textwrap.dedent("""\
        owner/good\t00000000-0000-0000-0000-000000000001
        not a valid repo\t00000000-0000-0000-0000-000000000002
        owner/bad-uuid\tnot-a-uuid
        owner/repo-c only one column
    """))
    with pytest.raises(click.UsageError) as exc:
        outrider_actions._parse_bulk_repos_tsv(path)
    msg = exc.value.message
    # Each error names the line number so the user can fix-then-retry.
    assert "line 2" in msg and "not a valid GitHub repo" in msg
    assert "line 3" in msg and "not a valid UUID" in msg
    assert "line 4" in msg and "2 tab-separated columns" in msg


def test_parse_tsv_normalizes_github_url(tmp_path):
    path = _write_tsv(tmp_path, textwrap.dedent("""\
        https://github.com/owner/url-form\t00000000-0000-0000-0000-000000000001
    """))
    rows = outrider_actions._parse_bulk_repos_tsv(path)
    assert rows == [("owner/url-form", "00000000-0000-0000-0000-000000000001")]


# ─── _run_bulk ─────────────────────────────────────────────────────────────


def test_run_bulk_calls_handler_per_row_with_common_kwargs():
    captured = []

    def fake_handler(**kwargs):
        captured.append(kwargs)

    rows = [("owner/a", "uuid-a"), ("owner/b", "uuid-b")]
    results = outrider_actions._run_bulk(
        fake_handler, rows,
        common_kwargs={"mode": "review", "skip_confirm": True},
        pace_s=0, echo=lambda *a, **k: None,
    )
    assert results == [("owner/a", "ok"), ("owner/b", "ok")]
    assert captured == [
        {"repo": "owner/a", "interest_id": "uuid-a",
         "mode": "review", "skip_confirm": True},
        {"repo": "owner/b", "interest_id": "uuid-b",
         "mode": "review", "skip_confirm": True},
    ]


def test_run_bulk_continues_after_per_row_failure():
    calls = []

    def fake_handler(**kwargs):
        calls.append(kwargs["repo"])
        if kwargs["repo"] == "owner/b":
            raise click.ClickException("simulated failure")

    rows = [("owner/a", "u"), ("owner/b", "u"), ("owner/c", "u")]
    results = outrider_actions._run_bulk(
        fake_handler, rows,
        common_kwargs={}, pace_s=0, echo=lambda *a, **k: None,
    )
    # All three were attempted (the loop didn't abort).
    assert calls == ["owner/a", "owner/b", "owner/c"]
    assert [r[0] for r in results] == ["owner/a", "owner/b", "owner/c"]
    assert results[0][1] == "ok"
    assert "simulated failure" in results[1][1]
    assert results[2][1] == "ok"


def test_run_bulk_paces_between_rows(monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    def fake_handler(**kwargs):
        pass

    rows = [("a/a", "u"), ("b/b", "u"), ("c/c", "u")]
    outrider_actions._run_bulk(
        fake_handler, rows, common_kwargs={},
        pace_s=5, echo=lambda *a, **k: None,
    )
    # N-1 sleeps for N rows (no sleep after the last).
    assert sleeps == [5, 5]


# ─── _render_local_workflow no_cron ────────────────────────────────────────


def test_render_local_workflow_default_has_active_cron():
    yaml = outrider_local._render_local_workflow("uuid-x")
    assert "  schedule:\n    - cron:" in yaml
    assert "# schedule:" not in yaml


def test_render_local_workflow_no_cron_comments_out_schedule():
    yaml = outrider_local._render_local_workflow("uuid-x", no_cron=True)
    # Cron is commented (uncomment-to-enable), not removed entirely.
    assert "# schedule:" in yaml
    assert "#   - cron:" in yaml
    # The uncommented form must not be present.
    assert "\n  schedule:\n" not in yaml
    # workflow_dispatch trigger is still active.
    assert "workflow_dispatch:" in yaml


# ─── CLI mutex enforcement ─────────────────────────────────────────────────


def test_init_bulk_repos_mutex_with_repo(tmp_path):
    path = _write_tsv(tmp_path, "owner/a\t00000000-0000-0000-0000-000000000001\n")
    runner = CliRunner()
    result = runner.invoke(cli, [
        "outrider", "init",
        "--bulk-repos", path, "--repo", "owner/b",
    ])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()


def test_setup_local_bulk_repos_mutex_with_interest(tmp_path):
    path = _write_tsv(tmp_path, "owner/a\t00000000-0000-0000-0000-000000000001\n")
    runner = CliRunner()
    result = runner.invoke(cli, [
        "outrider", "setup-local",
        "--bulk-repos", path,
        "--interest", "00000000-0000-0000-0000-000000000002",
    ])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()


# ─── CLI integration: --bulk-repos drives the loop ─────────────────────────


def test_init_bulk_repos_invokes_handler_per_row(tmp_path):
    path = _write_tsv(tmp_path, textwrap.dedent("""\
        owner/repo-a\t00000000-0000-0000-0000-000000000001
        owner/repo-b\t00000000-0000-0000-0000-000000000002
    """))
    captured = []

    def fake_init(**kwargs):
        captured.append((kwargs["repo"], kwargs["interest_id"], kwargs["mode"]))

    runner = CliRunner()
    with patch("remyxai.cli.commands.handle_outrider_init", side_effect=fake_init), \
         patch("time.sleep"):
        result = runner.invoke(cli, [
            "outrider", "init",
            "--bulk-repos", path,
            "--mode", "review",
            "--yes",
        ])
    assert result.exit_code == 0, result.output
    assert captured == [
        ("owner/repo-a", "00000000-0000-0000-0000-000000000001", "review"),
        ("owner/repo-b", "00000000-0000-0000-0000-000000000002", "review"),
    ]
    assert "summary: 2/2 ok" in result.output


def test_setup_local_no_cron_threads_through(tmp_path):
    """--no-cron must reach the handler so the rendered workflow disables cron."""
    captured = {}

    def fake_local(**kwargs):
        captured.update(kwargs)

    runner = CliRunner()
    with patch("remyxai.cli.commands.handle_outrider_setup_local",
               side_effect=fake_local):
        result = runner.invoke(cli, [
            "outrider", "setup-local",
            "--repo", "owner/name",
            "--interest", "00000000-0000-0000-0000-000000000001",
            "--no-cron",
            "--dry-run",
            "--yes",
        ])
    assert result.exit_code == 0, result.output
    assert captured.get("no_cron") is True
