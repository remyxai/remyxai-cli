"""
Unit tests for remyxai.api.projects.

Covers list, get, configure_eval_template, and set_decision_policy — the
latter two back the `remyxai projects configure-eval` and `set-policy`
commands added for the CLI expansion.

Mirrors the style of tests/api/test_recommendations.py.
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


# ═════════════════════════════════════════════════════════════════════════════
# list_projects
# ═════════════════════════════════════════════════════════════════════════════


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


# ═════════════════════════════════════════════════════════════════════════════
# configure_eval_template
# ═════════════════════════════════════════════════════════════════════════════


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


# ═════════════════════════════════════════════════════════════════════════════
# set_decision_policy
# ═════════════════════════════════════════════════════════════════════════════


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
