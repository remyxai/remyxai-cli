"""
Unit tests for the Outrider client surfaces:
  - remyxai.api.github_app    (install-url, installation status)
  - remyxai.api.integrations  (provider status, credential connect)
  - remyxai.api.interests      (provision-action POST/poll/status)

Mirrors tests/api/test_recommendations.py: mock `requests`, check the URL /
body / params the client builds and that responses are unwrapped correctly.
"""
from __future__ import annotations

import os
from unittest.mock import Mock, patch

import pytest

from remyxai.api import github_app, integrations
from remyxai.api import interests as interests_api


@pytest.fixture(autouse=True)
def ensure_env_key():
    original = os.environ.get("REMYXAI_API_KEY")
    os.environ["REMYXAI_API_KEY"] = "test-key"
    yield
    if original:
        os.environ["REMYXAI_API_KEY"] = original
    else:
        os.environ.pop("REMYXAI_API_KEY", None)


def _resp(body, status=200):
    m = Mock()
    m.status_code = status
    m.json.return_value = body
    m.raise_for_status = Mock()
    return m


# ─── github_app ─────────────────────────────────────────────────────────────

def test_get_app_install_url_returns_payload():
    with patch("remyxai.api.github_app.requests.get",
               return_value=_resp({"install_url": "https://github.com/apps/remyx-ai/installations/new",
                                    "state": "abc", "app_slug": "remyx-ai", "configured": True})) as g:
        out = github_app.get_app_install_url(api_key="k")
    assert out["app_slug"] == "remyx-ai"
    assert g.call_args[0][0].endswith("/github/app/install-url")


def test_get_app_install_url_handles_503_unconfigured():
    """503 returns the body instead of raising, so the CLI can message cleanly."""
    with patch("remyxai.api.github_app.requests.get",
               return_value=_resp({"configured": False, "error": "not configured"}, status=503)):
        out = github_app.get_app_install_url(api_key="k")
    assert out["configured"] is False


def test_is_app_installed_true_and_passes_repo_param():
    with patch("remyxai.api.github_app.requests.get",
               return_value=_resp({"installed": True, "repo": "o/r"})) as g:
        assert github_app.is_app_installed("o/r", api_key="k") is True
    assert g.call_args.kwargs["params"] == {"repo": "o/r"}


def test_is_app_installed_false():
    with patch("remyxai.api.github_app.requests.get",
               return_value=_resp({"installed": False, "repo": "o/r"})):
        assert github_app.is_app_installed("o/r", api_key="k") is False


# ─── integrations ─────────────────────────────────────────────────────────────

def test_get_integration_status_unwraps_connected():
    with patch("remyxai.api.integrations.requests.get",
               return_value=_resp({"connected": True})) as g:
        out = integrations.get_integration_status("claude_code", api_key="k")
    assert out["connected"] is True
    assert g.call_args[0][0].endswith("/integrations/claude_code/status")


def test_connect_credential_posts_api_key_body():
    with patch("remyxai.api.integrations.requests.post",
               return_value=_resp({"connection": {"provider": "claude_code"}}, status=201)) as p:
        integrations.connect_credential("claude_code", {"api_key": "sk-ant-x"}, api_key="k")
    assert p.call_args[0][0].endswith("/integrations/claude_code/connect")
    assert p.call_args.kwargs["json"] == {"api_key": "sk-ant-x"}


# ─── interests: provision-action ──────────────────────────────────────────────

def test_provision_action_auto_merge_true_body():
    with patch("remyxai.api.interests.requests.post",
               return_value=_resp({"task_id": "t1", "status_url": "/.../t1"}, status=202)) as p:
        out = interests_api.provision_action(
            "iid", repo_url="https://github.com/o/r", auto_merge=True, api_key="k"
        )
    assert out["task_id"] == "t1"
    body = p.call_args.kwargs["json"]
    assert body["auto_merge"] is True
    assert body["repo_url"] == "https://github.com/o/r"
    assert p.call_args[0][0].endswith("/interests/iid/provision-action")


def test_provision_action_review_mode_auto_merge_false():
    with patch("remyxai.api.interests.requests.post",
               return_value=_resp({"task_id": "t2"}, status=202)) as p:
        interests_api.provision_action("iid", repo_url="https://github.com/o/r",
                                       auto_merge=False, api_key="k")
    assert p.call_args.kwargs["json"]["auto_merge"] is False


def test_provision_action_omits_optional_fields_when_none():
    with patch("remyxai.api.interests.requests.post",
               return_value=_resp({"task_id": "t3"}, status=202)) as p:
        interests_api.provision_action("iid", repo_url="https://github.com/o/r", api_key="k")
    body = p.call_args.kwargs["json"]
    assert "branch" not in body and "workflow_filename" not in body


def test_poll_provision_action_returns_task_dict():
    task = {"status": "completed", "result": {"pr_url": "https://github.com/o/r/pull/1"}}
    with patch("remyxai.api.interests.requests.get", return_value=_resp(task)) as g:
        out = interests_api.poll_provision_action("iid", "t1", api_key="k")
    assert out["result"]["pr_url"].endswith("/pull/1")
    assert g.call_args[0][0].endswith("/interests/iid/provision-action/t1")


def test_get_provision_status_not_provisioned():
    with patch("remyxai.api.interests.requests.get",
               return_value=_resp({"provisioned": False})):
        out = interests_api.get_provision_status("iid", api_key="k")
    assert out["provisioned"] is False
