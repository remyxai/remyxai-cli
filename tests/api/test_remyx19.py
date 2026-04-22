"""
Unit tests for the REMYX-19 CLI client additions:
  - remyxai.api.experiments  (list, get, start_validation_run, get_validation_run)
  - remyxai.api.projects     (list, get, configure_eval_template, set_decision_policy)
  - remyxai.api.interests    (repo-analysis helpers + repo-field kwargs on
                              create/update_interest)

Mirrors the style of tests/api/test_recommendations.py:
  - Signature checks (api_key=None on every public function)
  - Request-shape checks (path, method, body, params)
  - Response-parsing checks
"""
from __future__ import annotations

import inspect
import os
from unittest.mock import Mock, patch

import pytest


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def ensure_env_key():
    original = os.environ.get("REMYXAI_API_KEY")
    os.environ["REMYXAI_API_KEY"] = "test-key-for-fixtures"
    yield
    if original:
        os.environ["REMYXAI_API_KEY"] = original
    else:
        os.environ.pop("REMYXAI_API_KEY", None)


def _ok(body: dict, status: int = 200) -> Mock:
    m = Mock()
    m.status_code = status
    m.json.return_value = body
    m.raise_for_status = Mock()
    return m


# ═════════════════════════════════════════════════════════════════════════════
# remyxai.api.experiments
# ═════════════════════════════════════════════════════════════════════════════


class TestExperimentsSignatures:
    FUNCS = [
        "list_experiments",
        "get_experiment",
        "start_validation_run",
        "get_validation_run",
    ]

    def test_all_accept_api_key(self):
        from remyxai.api import experiments as m
        for fname in self.FUNCS:
            fn = getattr(m, fname)
            sig = inspect.signature(fn)
            assert "api_key" in sig.parameters, f"{fname} missing api_key param"
            assert sig.parameters["api_key"].default is None


class TestListExperiments:
    @patch("remyxai.api.experiments.requests.get")
    def test_no_filters(self, mock_get):
        from remyxai.api.experiments import list_experiments

        mock_get.return_value = _ok({"experiments": [{"id": "e1"}], "count": 1})
        result = list_experiments()

        assert result == [{"id": "e1"}]
        args, kwargs = mock_get.call_args
        assert args[0].endswith("/experiments")
        assert kwargs["params"] == {"limit": 20}

    @patch("remyxai.api.experiments.requests.get")
    def test_with_filters(self, mock_get):
        from remyxai.api.experiments import list_experiments

        mock_get.return_value = _ok({"experiments": []})
        list_experiments(
            project_id="p1",
            status="validating",
            initiative="Support AI",
            limit=50,
        )

        params = mock_get.call_args.kwargs["params"]
        assert params["project_id"] == "p1"
        assert params["status"] == "validating"
        assert params["initiative"] == "Support AI"
        assert params["limit"] == 50


class TestStartValidationRun:
    @patch("remyxai.api.experiments.requests.post")
    def test_minimal_body(self, mock_post):
        from remyxai.api.experiments import start_validation_run

        mock_post.return_value = _ok(
            {"run": {"id": "r1", "status": "building_envs"}, "status": "building_envs"},
            status=202,
        )
        variants = [
            {"name": "baseline", "commit_sha": "9f8a"},
            {"name": "feature", "commit_sha": "c4e5"},
        ]
        start_validation_run(
            experiment_id="exp-1",
            template_id="tpl-1",
            github_url="https://github.com/owner/repo",
            variants=variants,
        )

        args, kwargs = mock_post.call_args
        assert args[0].endswith("/eval-env/runs")
        assert kwargs["json"]["experiment_id"] == "exp-1"
        assert kwargs["json"]["template_id"] == "tpl-1"
        assert kwargs["json"]["github_url"] == "https://github.com/owner/repo"
        assert kwargs["json"]["variants"] == variants
        assert kwargs["json"]["seeds"] == 1
        # pr_number / pr_url omitted when not provided
        assert "pr_number" not in kwargs["json"]
        assert "pr_url" not in kwargs["json"]

    @patch("remyxai.api.experiments.requests.post")
    def test_with_pr_lineage(self, mock_post):
        from remyxai.api.experiments import start_validation_run

        mock_post.return_value = _ok({"run": {"id": "r2"}}, status=202)
        start_validation_run(
            experiment_id="exp-1",
            template_id="tpl-1",
            github_url="https://github.com/owner/repo",
            variants=[{"name": "baseline", "commit_sha": "abc"}],
            pr_number=42,
            pr_url="https://github.com/owner/repo/pull/42",
        )

        body = mock_post.call_args.kwargs["json"]
        assert body["pr_number"] == 42
        assert body["pr_url"].endswith("/pull/42")


# ═════════════════════════════════════════════════════════════════════════════
# remyxai.api.projects
# ═════════════════════════════════════════════════════════════════════════════


class TestProjectsSignatures:
    FUNCS = [
        "list_projects",
        "get_project",
        "configure_eval_template",
        "set_decision_policy",
    ]

    def test_all_accept_api_key(self):
        from remyxai.api import projects as m
        for fname in self.FUNCS:
            fn = getattr(m, fname)
            sig = inspect.signature(fn)
            assert "api_key" in sig.parameters, f"{fname} missing api_key param"


class TestListProjects:
    @patch("remyxai.api.projects.requests.get")
    def test_omits_team_id_when_unset(self, mock_get):
        from remyxai.api.projects import list_projects

        mock_get.return_value = _ok({"projects": [{"id": "p1"}], "count": 1})
        result = list_projects()

        assert result == [{"id": "p1"}]
        assert mock_get.call_args.kwargs["params"] == {}

    @patch("remyxai.api.projects.requests.get")
    def test_passes_team_id(self, mock_get):
        from remyxai.api.projects import list_projects

        mock_get.return_value = _ok({"projects": []})
        list_projects(team_id="t1")

        assert mock_get.call_args.kwargs["params"] == {"team_id": "t1"}


class TestConfigureEvalTemplate:
    @patch("remyxai.api.projects.requests.post")
    def test_posts_template_body_to_dedicated_endpoint(self, mock_post):
        from remyxai.api.projects import configure_eval_template

        mock_post.return_value = _ok({
            "project_id": "p1",
            "template_name": "default",
            "template": {"provider": "modal"},
        })
        template = {"provider": "modal", "entry_point": "python run.py"}
        configure_eval_template(
            project_id="p1",
            template_name="default",
            template=template,
        )

        args, kwargs = mock_post.call_args
        assert args[0].endswith("/projects/p1/eval-templates/default")
        assert kwargs["json"] == template


class TestSetDecisionPolicy:
    @patch("remyxai.api.projects.requests.post")
    def test_posts_policy_body_to_dedicated_endpoint(self, mock_post):
        from remyxai.api.projects import set_decision_policy

        mock_post.return_value = _ok({
            "project_id": "p1",
            "policy_name": "default",
            "policy": {"ship": {"all": []}},
        })
        policy = {"ship": {"all": [{"metric": "delta", "gte": 0.03}]}}
        set_decision_policy(
            project_id="p1",
            policy_name="default",
            policy=policy,
        )

        args, kwargs = mock_post.call_args
        assert args[0].endswith("/projects/p1/decision-policies/default")
        assert kwargs["json"] == policy


# ═════════════════════════════════════════════════════════════════════════════
# remyxai.api.interests — REMYX-28 additions via REMYX-19
# ═════════════════════════════════════════════════════════════════════════════


class TestRepoInterestHelpers:
    @patch("remyxai.api.interests.requests.post")
    def test_analyze_repo_posts_repo_url(self, mock_post):
        from remyxai.api.interests import analyze_repo

        mock_post.return_value = _ok(
            {"task_id": "tk-1", "status_url": "/api/v1.0/interests/analyze-repo/tk-1"},
            status=202,
        )
        result = analyze_repo("https://github.com/owner/repo")

        args, kwargs = mock_post.call_args
        assert args[0].endswith("/interests/analyze-repo")
        assert kwargs["json"] == {"repo_url": "https://github.com/owner/repo"}
        assert result["task_id"] == "tk-1"

    @patch("remyxai.api.interests.requests.get")
    def test_poll_repo_analysis(self, mock_get):
        from remyxai.api.interests import poll_repo_analysis

        mock_get.return_value = _ok({"status": "running", "message": "Fetching README"})
        poll_repo_analysis("tk-1")

        assert mock_get.call_args.args[0].endswith(
            "/interests/analyze-repo/tk-1"
        )

    @patch("remyxai.api.interests.requests.post")
    def test_regenerate_interest_empty_body(self, mock_post):
        from remyxai.api.interests import regenerate_interest

        mock_post.return_value = _ok({"task_id": "tk-2"}, status=202)
        regenerate_interest("i1")

        args, kwargs = mock_post.call_args
        assert args[0].endswith("/interests/i1/regenerate")
        assert kwargs["json"] == {}

    @patch("remyxai.api.interests.requests.post")
    def test_regenerate_interest_with_override(self, mock_post):
        from remyxai.api.interests import regenerate_interest

        mock_post.return_value = _ok({"task_id": "tk-3"}, status=202)
        regenerate_interest("i1", repo_url="https://github.com/other/repo")

        body = mock_post.call_args.kwargs["json"]
        assert body == {"repo_url": "https://github.com/other/repo"}

    @patch("remyxai.api.interests.requests.get")
    def test_list_github_repos_disconnected(self, mock_get):
        from remyxai.api.interests import list_github_repos

        mock_get.return_value = _ok({"connected": False, "repos": []})
        result = list_github_repos()

        assert mock_get.call_args.args[0].endswith("/interests/github/repos")
        assert result == {"connected": False, "repos": []}


class TestCreateInterestRepoFields:
    @patch("remyxai.api.interests.requests.post")
    def test_omits_repo_fields_when_not_provided(self, mock_post):
        from remyxai.api.interests import create_interest

        mock_post.return_value = _ok({"id": "i1"}, status=201)
        create_interest(name="Manual", context="plain text")

        body = mock_post.call_args.kwargs["json"]
        for field in (
            "source_repo_url", "source_repo_metadata",
            "generated_report", "repo_analysis",
        ):
            assert field not in body

    @patch("remyxai.api.interests.requests.post")
    def test_passes_repo_fields_when_provided(self, mock_post):
        from remyxai.api.interests import create_interest

        mock_post.return_value = _ok({"id": "i2"}, status=201)
        create_interest(
            name="From repo",
            context="# Report",
            source_repo_url="https://github.com/owner/repo",
            source_repo_metadata={"stars": 42},
            generated_report="# Report",
            repo_analysis={"stages": []},
        )

        body = mock_post.call_args.kwargs["json"]
        assert body["source_repo_url"] == "https://github.com/owner/repo"
        assert body["source_repo_metadata"] == {"stars": 42}
        assert body["generated_report"] == "# Report"
        assert body["repo_analysis"] == {"stages": []}


class TestUpdateInterestRepoFields:
    @patch("remyxai.api.interests.requests.put")
    def test_passes_repo_fields_on_update(self, mock_put):
        from remyxai.api.interests import update_interest

        mock_put.return_value = _ok({"id": "i1", "pool_invalidated": 3})
        update_interest(
            interest_id="i1",
            generated_report="# refreshed",
            repo_analysis={"stages": ["a"]},
        )

        body = mock_put.call_args.kwargs["json"]
        assert body["generated_report"] == "# refreshed"
        assert body["repo_analysis"] == {"stages": ["a"]}
        # Fields not passed stay out of the body
        assert "name" not in body
        assert "source_repo_url" not in body
