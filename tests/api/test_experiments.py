"""
Unit tests for remyxai.api.experiments.

Covers list, get, start_validation_run (via the REMYX-24 eval-env pipeline),
and get_validation_run.

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
# Signatures
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


# ═════════════════════════════════════════════════════════════════════════════
# list_experiments
# ═════════════════════════════════════════════════════════════════════════════


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


# ═════════════════════════════════════════════════════════════════════════════
# start_validation_run
# ═════════════════════════════════════════════════════════════════════════════


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
