"""Tests for the `interests from-repo` and `interests from-project` commands.

These exercise the CLI wiring + action handlers with the API client layer
mocked, so no network calls are made. The handlers import the API
functions into the ``interest_actions`` namespace, so we patch them there.
"""
from contextlib import contextmanager
from unittest.mock import patch

from click.testing import CliRunner

from remyxai.cli import interest_actions
from remyxai.cli.commands import cli


# ─── shared fixtures-as-constants ──────────────────────────────────────────

ANALYSIS_RESULT = {
    "full_name": "remyxai/outrider",
    "source_repo_url": "https://github.com/remyxai/outrider",
    "source_repo_metadata": {"stars": 7},
    "report_markdown": "# Research Interest Profile: outrider\n\nblah blah",
    "repo_analysis": {"themes": ["agents"]},
    "report_generated_at": "2026-06-10T03:41:19.699514",
}

CREATED_FROM_REPO = {
    "id": "29ca03e7-454d-446c-9941-32c96c53d95d",
    "name": "outrider",
    "daily_count": 3,
    "is_active": True,
    "source_repo_url": "https://github.com/remyxai/outrider",
    "history_extraction_task_id": "df9c4bd8-2d52-4917-9112-5de66718111e",
}

REFRESH_RESPONSE = {
    "tasks": [{"task_id": "refresh-1", "interest_name": "outrider",
               "status": "pending"}],
}


@contextmanager
def _refresh_patch():
    """Patch the recommendations refresh trigger (called by default after
    every create). Yields the mock so callers can assert on it."""
    with patch.object(interest_actions, "trigger_recommendations_refresh",
                      return_value=REFRESH_RESPONSE) as m:
        yield m


@contextmanager
def _repo_mocks(create_return=None):
    """Patch the three repo-flow API calls + the refresh trigger."""
    create_return = create_return or CREATED_FROM_REPO
    with patch.object(interest_actions, "analyze_repo",
                      return_value={"task_id": "task-123"}) as analyze, \
         patch.object(interest_actions, "poll_analyze_repo",
                      return_value={"status": "completed", "progress": 100,
                                    "result": ANALYSIS_RESULT}) as poll, \
         patch.object(interest_actions, "create_interest",
                      return_value=create_return) as create, \
         _refresh_patch() as refresh:
        yield analyze, poll, create, refresh


# ─── from-repo ──────────────────────────────────────────────────────────────

def test_from_repo_default_no_provisioning_and_refreshes():
    with _repo_mocks() as (analyze, poll, create, refresh):
        result = CliRunner().invoke(
            cli, ["interests", "from-repo",
                  "https://github.com/remyxai/outrider"],
        )

    assert result.exit_code == 0, result.output
    analyze.assert_called_once_with(
        "https://github.com/remyxai/outrider", api_key=None
    )

    kwargs = create.call_args.kwargs
    # Repo fields forwarded so the server triggers history extraction.
    assert kwargs["source_repo_url"] == "https://github.com/remyxai/outrider"
    assert kwargs["generated_report"] == ANALYSIS_RESULT["report_markdown"]
    assert kwargs["repo_analysis"] == ANALYSIS_RESULT["repo_analysis"]
    # "Not now" → no provisioning kwargs at all.
    assert "provision_action" not in kwargs
    assert "extraction dispatched" in result.output
    assert "paper-PR automation: not now" in result.output
    # Recommendations refresh kicked off for the new interest by default.
    refresh.assert_called_once()
    assert refresh.call_args.kwargs["interest_id"] == CREATED_FROM_REPO["id"]
    assert "recommendations refresh started" in result.output


def test_from_repo_no_refresh_skips_trigger():
    with _repo_mocks() as (analyze, poll, create, refresh):
        result = CliRunner().invoke(
            cli, ["interests", "from-repo",
                  "https://github.com/remyxai/outrider", "--no-refresh"],
        )
    assert result.exit_code == 0, result.output
    refresh.assert_not_called()


def test_from_repo_automate_auto():
    with _repo_mocks(
        create_return={**CREATED_FROM_REPO, "provision_task_id": "prov-1"},
    ) as (analyze, poll, create, refresh):
        result = CliRunner().invoke(
            cli, ["interests", "from-repo",
                  "https://github.com/remyxai/outrider", "--automate", "auto"],
        )

    assert result.exit_code == 0, result.output
    kwargs = create.call_args.kwargs
    assert kwargs["provision_action"] is True
    assert kwargs["provision_auto_merge"] is True
    assert kwargs["provision_repo_url"] == "https://github.com/remyxai/outrider"


def test_from_repo_automate_review_no_auto_merge():
    with _repo_mocks(
        create_return={**CREATED_FROM_REPO, "provision_task_id": "prov-2"},
    ) as (analyze, poll, create, refresh):
        result = CliRunner().invoke(
            cli, ["interests", "from-repo",
                  "https://github.com/remyxai/outrider", "--automate", "review"],
        )

    assert result.exit_code == 0, result.output
    kwargs = create.call_args.kwargs
    assert kwargs["provision_action"] is True
    assert kwargs["provision_auto_merge"] is False


def test_from_repo_analysis_failure_exits_nonzero():
    analyze = patch.object(interest_actions, "analyze_repo",
                           return_value={"task_id": "task-x"})
    poll = patch.object(interest_actions, "poll_analyze_repo",
                        return_value={"status": "failed", "error": "boom"})
    with analyze, poll:
        result = CliRunner().invoke(
            cli, ["interests", "from-repo", "https://github.com/x/y"],
        )
    assert result.exit_code != 0
    assert "failed" in result.output.lower()


def test_from_repo_custom_name_overrides_repo_name():
    with _repo_mocks() as (analyze, poll, create, refresh):
        CliRunner().invoke(
            cli, ["interests", "from-repo", "https://github.com/remyxai/outrider",
                  "--name", "My Interest", "--daily-count", "5"],
        )
    kwargs = create.call_args.kwargs
    assert kwargs["name"] == "My Interest"
    assert kwargs["daily_count"] == 5


# ─── from-project ─────────────────────────────────────────────────────────

PROJECTS = [
    {"id": "11111111-1111-1111-1111-111111111111", "name": "Spatial VQA"},
    {"id": "22222222-2222-2222-2222-222222222222", "name": "RAG"},
]

CREATED_FROM_PROJECT = {
    "id": "33333333-3333-3333-3333-333333333333",
    "name": "Spatial VQA recs",
    "daily_count": 2,
    "is_active": True,
    "build_task_id": "build-9",
}


def test_from_project_resolves_name_and_sets_project_mode():
    lp = patch.object(interest_actions, "list_projects", return_value=PROJECTS)
    create = patch.object(interest_actions, "create_interest",
                          return_value=CREATED_FROM_PROJECT)
    with lp, create as c, _refresh_patch() as refresh:
        result = CliRunner().invoke(
            cli, ["interests", "from-project", "Spatial VQA"],
        )

    assert result.exit_code == 0, result.output
    kwargs = c.call_args.kwargs
    assert kwargs["project_id"] == PROJECTS[0]["id"]
    assert kwargs["mode"] == "project_structured"
    assert kwargs["included_experiment_ids"] is None
    assert "building recommendation context" in result.output
    assert "all experiments" in result.output
    # Refresh kicked off for the project interest.
    refresh.assert_called_once()
    assert refresh.call_args.kwargs["interest_id"] == CREATED_FROM_PROJECT["id"]


def test_from_project_wait_sequences_build_then_refresh():
    """--wait should poll the build task before the recs are usable."""
    lp = patch.object(interest_actions, "list_projects", return_value=PROJECTS)
    create = patch.object(interest_actions, "create_interest",
                          return_value=CREATED_FROM_PROJECT)
    # poll_refresh_task is used for both the build-task await and the
    # refresh-task wait; return completed for both.
    poll = patch.object(interest_actions, "poll_refresh_task",
                        return_value={"status": "completed",
                                      "result": {"count": 2}})
    with lp, create, _refresh_patch(), poll as p:
        result = CliRunner().invoke(
            cli, ["interests", "from-project", "Spatial VQA", "--wait"],
        )
    assert result.exit_code == 0, result.output
    # Build task (build-9) polled first, then the refresh task (refresh-1).
    polled = [call.args[0] for call in p.call_args_list]
    assert "build-9" in polled
    assert "refresh-1" in polled
    assert "recommendation(s) ready" in result.output


def test_from_project_unknown_name_exits():
    lp = patch.object(interest_actions, "list_projects", return_value=PROJECTS)
    with lp:
        result = CliRunner().invoke(
            cli, ["interests", "from-project", "Nonexistent"],
        )
    assert result.exit_code != 0
    assert "No project found" in result.output


def test_from_project_curated_experiments_by_name():
    experiments = [
        {"id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "name": "baseline-run"},
        {"id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "name": "dpo-v2"},
    ]
    lp = patch.object(interest_actions, "list_projects", return_value=PROJECTS)
    le = patch.object(interest_actions, "list_experiments",
                      return_value=experiments)
    create = patch.object(interest_actions, "create_interest",
                          return_value=CREATED_FROM_PROJECT)
    with lp, le, create as c, _refresh_patch():
        result = CliRunner().invoke(
            cli, ["interests", "from-project", "Spatial VQA",
                  "-e", "baseline-run", "-e", "dpo-v2", "--no-auto-update"],
        )

    assert result.exit_code == 0, result.output
    kwargs = c.call_args.kwargs
    assert kwargs["included_experiment_ids"] == [
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    ]
    assert kwargs["auto_update_from_experiments"] is False
    assert "2 selected experiment" in result.output


# ─── plain create ────────────────────────────────────────────────────────

def test_create_refreshes_by_default():
    created = {"id": "44444444-4444-4444-4444-444444444444",
               "name": "RAG", "daily_count": 2, "is_active": True}
    create = patch.object(interest_actions, "create_interest",
                          return_value=created)
    with create, _refresh_patch() as refresh:
        result = CliRunner().invoke(
            cli, ["interests", "create", "--name", "RAG",
                  "--context", "retrieval augmented generation"],
        )
    assert result.exit_code == 0, result.output
    refresh.assert_called_once()
    assert refresh.call_args.kwargs["interest_id"] == created["id"]


def test_create_no_refresh_shows_manual_hint():
    created = {"id": "44444444-4444-4444-4444-444444444444",
               "name": "RAG", "daily_count": 2, "is_active": True}
    create = patch.object(interest_actions, "create_interest",
                          return_value=created)
    with create, _refresh_patch() as refresh:
        result = CliRunner().invoke(
            cli, ["interests", "create", "--name", "RAG",
                  "--context", "retrieval augmented generation", "--no-refresh"],
        )
    assert result.exit_code == 0, result.output
    refresh.assert_not_called()
    assert "papers refresh" in result.output
