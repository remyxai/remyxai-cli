"""
Unit tests for:
  - remyxai.api.recommendations  (GET digest, GET list, POST refresh, GET poll)
  - remyxai.api.interests         (list, get, create, update, delete, toggle)

Mirrors the style of tests/api/test_lazy_api_key.py:
  - Signature checks (api_key=None on every public function)
  - Header-threading checks (explicit key ends up in Authorization header)
  - Backwards-compatibility check (no api_key= still works)
  - Response-parsing checks (return values are correctly unwrapped)
"""
from __future__ import annotations

import inspect
import os
import time
from unittest.mock import Mock, patch

import pytest


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def ensure_env_key():
    """Keep REMYXAI_API_KEY set for module-level HEADERS throughout."""
    original = os.environ.get("REMYXAI_API_KEY")
    os.environ["REMYXAI_API_KEY"] = "test-key-for-fixtures"
    yield
    if original:
        os.environ["REMYXAI_API_KEY"] = original
    else:
        os.environ.pop("REMYXAI_API_KEY", None)


def _ok_get(body: dict) -> Mock:
    m = Mock()
    m.status_code = 200
    m.json.return_value = body
    m.raise_for_status = Mock()
    return m


def _ok_post(body: dict) -> Mock:
    m = Mock()
    m.status_code = 201
    m.json.return_value = body
    m.raise_for_status = Mock()
    return m


# ═════════════════════════════════════════════════════════════════════════════
# remyxai.api.recommendations
# ═════════════════════════════════════════════════════════════════════════════

class TestRecommendationsSignatures:
    """Every public function must accept api_key=None."""

    FUNCS = [
        "get_recommendations_digest",
        "list_recommended",
        "trigger_recommendations_refresh",
        "poll_refresh_task",
    ]

    def test_all_have_api_key_param(self):
        import remyxai.api.recommendations as mod

        for name in self.FUNCS:
            fn = getattr(mod, name)
            sig = inspect.signature(fn)
            assert "api_key" in sig.parameters, (
                f"remyxai.api.recommendations.{name} missing api_key parameter"
            )
            assert sig.parameters["api_key"].default is None, (
                f"remyxai.api.recommendations.{name} api_key should default to None"
            )


class TestRecommendationsHeaderThreading:
    """Explicit api_key must reach the Authorization header."""

    @patch("remyxai.api.recommendations.requests.get")
    def test_get_recommendations_digest_explicit_key(self, mock_get):
        mock_get.return_value = _ok_get({
            "date": "2026-03-21", "period": "today",
            "interests": [], "total_papers": 0, "source_types": [],
        })
        from remyxai.api.recommendations import get_recommendations_digest
        get_recommendations_digest(api_key="digest-key-123")
        assert mock_get.call_args[1]["headers"]["Authorization"] == "Bearer digest-key-123"

    @patch("remyxai.api.recommendations.requests.get")
    def test_list_recommended_explicit_key(self, mock_get):
        mock_get.return_value = _ok_get({"papers": [], "count": 0, "period": "all"})
        from remyxai.api.recommendations import list_recommended
        list_recommended(api_key="list-key-456")
        assert mock_get.call_args[1]["headers"]["Authorization"] == "Bearer list-key-456"

    @patch("remyxai.api.recommendations.requests.post")
    def test_trigger_refresh_explicit_key(self, mock_post):
        mock_post.return_value = _ok_get({"tasks": []})
        from remyxai.api.recommendations import trigger_recommendations_refresh
        trigger_recommendations_refresh(api_key="refresh-key-789")
        assert mock_post.call_args[1]["headers"]["Authorization"] == "Bearer refresh-key-789"

    @patch("remyxai.api.recommendations.requests.get")
    def test_poll_refresh_task_explicit_key(self, mock_get):
        mock_get.return_value = _ok_get({
            "task_id": "abc", "status": "completed", "progress": 100,
            "message": "done", "result": {"count": 2},
        })
        from remyxai.api.recommendations import poll_refresh_task
        poll_refresh_task("abc-task-id", api_key="poll-key-000")
        assert mock_get.call_args[1]["headers"]["Authorization"] == "Bearer poll-key-000"

    @patch("remyxai.api.recommendations.requests.get")
    def test_no_explicit_key_falls_back_to_env(self, mock_get):
        """
        When no api_key is passed, the module-level HEADERS (built at import
        time from whatever REMYXAI_API_KEY was set then) should be used.
        We verify the Authorization header is a non-empty Bearer token rather
        than checking a specific value — HEADERS is frozen at import time so
        the autouse fixture's os.environ change doesn't affect it.
        """
        mock_get.return_value = _ok_get({"papers": [], "count": 0, "period": "all"})
        from remyxai.api.recommendations import list_recommended
        list_recommended()
        auth = mock_get.call_args[1]["headers"]["Authorization"]
        assert auth.startswith("Bearer ")
        assert len(auth) > len("Bearer ")  # something is actually there


class TestRecommendationsResponseParsing:
    """Verify correct fields are forwarded from the response."""

    @patch("remyxai.api.recommendations.requests.get")
    def test_digest_returns_full_response(self, mock_get):
        payload = {
            "date": "2026-03-21",
            "period": "today",
            "interests": [{"id": "abc", "name": "RAG", "count": 2, "recommendations": []}],
            "total_papers": 2,
            "source_types": ["arxiv_paper"],
        }
        mock_get.return_value = _ok_get(payload)
        from remyxai.api.recommendations import get_recommendations_digest
        result = get_recommendations_digest(period="today", limit=5)
        assert result["total_papers"] == 2
        assert result["interests"][0]["name"] == "RAG"

    @patch("remyxai.api.recommendations.requests.post")
    def test_trigger_refresh_returns_tasks(self, mock_post):
        payload = {"tasks": [
            {"task_id": "t1", "interest_id": "i1", "interest_name": "RAG", "status": "pending"}
        ]}
        mock_post.return_value = _ok_get(payload)
        from remyxai.api.recommendations import trigger_recommendations_refresh
        result = trigger_recommendations_refresh(interest_id="i1")
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["status"] == "pending"

    @patch("remyxai.api.recommendations.requests.get")
    def test_poll_returns_status_fields(self, mock_get):
        payload = {
            "task_id": "t1", "status": "running",
            "progress": 60, "message": "AI ranking 47 papers...",
        }
        mock_get.return_value = _ok_get(payload)
        from remyxai.api.recommendations import poll_refresh_task
        result = poll_refresh_task("t1")
        assert result["status"] == "running"
        assert result["progress"] == 60

    @patch("remyxai.api.recommendations.requests.get")
    def test_list_recommended_passes_filters(self, mock_get):
        mock_get.return_value = _ok_get({"papers": [], "count": 0, "period": "week"})
        from remyxai.api.recommendations import list_recommended
        list_recommended(
            interest_id="i1", limit=10, period="week", source_type="arxiv_paper"
        )
        params = mock_get.call_args[1]["params"]
        assert params["interest_id"] == "i1"
        assert params["period"] == "week"
        assert params["limit"] == 10
        assert params["source_type"] == "arxiv_paper"

    @patch("remyxai.api.recommendations.requests.get")
    def test_digest_limit_clamped_to_10(self, mock_get):
        mock_get.return_value = _ok_get(
            {"date": "2026-03-21", "period": "all", "interests": [], "total_papers": 0}
        )
        from remyxai.api.recommendations import get_recommendations_digest
        get_recommendations_digest(limit=999)
        params = mock_get.call_args[1]["params"]
        assert params["limit"] == 10

    @patch("remyxai.api.recommendations.requests.get")
    def test_list_limit_clamped_to_50(self, mock_get):
        mock_get.return_value = _ok_get({"papers": [], "count": 0, "period": "all"})
        from remyxai.api.recommendations import list_recommended
        list_recommended(limit=999)
        params = mock_get.call_args[1]["params"]
        assert params["limit"] == 50


# ═════════════════════════════════════════════════════════════════════════════
# remyxai.api.interests
# ═════════════════════════════════════════════════════════════════════════════

class TestInterestsSignatures:
    """Every public function must accept api_key=None."""

    FUNCS = [
        "list_interests",
        "get_interest",
        "create_interest",
        "update_interest",
        "delete_interest",
        "toggle_interest",
    ]

    def test_all_have_api_key_param(self):
        import remyxai.api.interests as mod

        for name in self.FUNCS:
            fn = getattr(mod, name)
            sig = inspect.signature(fn)
            assert "api_key" in sig.parameters, (
                f"remyxai.api.interests.{name} missing api_key parameter"
            )
            assert sig.parameters["api_key"].default is None, (
                f"remyxai.api.interests.{name} api_key should default to None"
            )


class TestInterestsHeaderThreading:
    """Explicit api_key must reach the Authorization header."""

    @patch("remyxai.api.interests.requests.get")
    def test_list_interests_explicit_key(self, mock_get):
        mock_get.return_value = _ok_get({"interests": [], "count": 0})
        from remyxai.api.interests import list_interests
        list_interests(api_key="list-int-key")
        assert mock_get.call_args[1]["headers"]["Authorization"] == "Bearer list-int-key"

    @patch("remyxai.api.interests.requests.get")
    def test_get_interest_explicit_key(self, mock_get):
        mock_get.return_value = _ok_get({"id": "abc", "name": "RAG"})
        from remyxai.api.interests import get_interest
        get_interest("abc", api_key="get-int-key")
        assert mock_get.call_args[1]["headers"]["Authorization"] == "Bearer get-int-key"

    @patch("remyxai.api.interests.requests.post")
    def test_create_interest_explicit_key(self, mock_post):
        mock_post.return_value = _ok_post({
            "id": "new-id", "name": "RAG", "context": "...",
            "daily_count": 2, "is_active": True,
        })
        from remyxai.api.interests import create_interest
        create_interest("RAG", "context text", api_key="create-int-key")
        assert mock_post.call_args[1]["headers"]["Authorization"] == "Bearer create-int-key"

    @patch("remyxai.api.interests.requests.put")
    def test_update_interest_explicit_key(self, mock_put):
        mock_put.return_value = _ok_get({"id": "abc", "name": "RAG Updated"})
        from remyxai.api.interests import update_interest
        update_interest("abc", name="RAG Updated", api_key="update-int-key")
        assert mock_put.call_args[1]["headers"]["Authorization"] == "Bearer update-int-key"

    @patch("remyxai.api.interests.requests.delete")
    def test_delete_interest_explicit_key(self, mock_delete):
        mock_delete.return_value = _ok_get({"message": "Interest deleted", "id": "abc"})
        from remyxai.api.interests import delete_interest
        delete_interest("abc", api_key="delete-int-key")
        assert mock_delete.call_args[1]["headers"]["Authorization"] == "Bearer delete-int-key"

    @patch("remyxai.api.interests.requests.post")
    def test_toggle_interest_explicit_key(self, mock_post):
        mock_post.return_value = _ok_get({"id": "abc", "is_active": False})
        from remyxai.api.interests import toggle_interest
        toggle_interest("abc", api_key="toggle-int-key")
        assert mock_post.call_args[1]["headers"]["Authorization"] == "Bearer toggle-int-key"

    @patch("remyxai.api.interests.requests.get")
    def test_no_explicit_key_falls_back_to_env(self, mock_get):
        """
        Same as above — HEADERS is frozen at import time, so we just verify
        a Bearer token is present rather than checking a specific value.
        """
        mock_get.return_value = _ok_get({"interests": [], "count": 0})
        from remyxai.api.interests import list_interests
        list_interests()
        auth = mock_get.call_args[1]["headers"]["Authorization"]
        assert auth.startswith("Bearer ")
        assert len(auth) > len("Bearer ")  # something is actually there


class TestInterestsResponseParsing:
    """Verify response bodies are correctly unwrapped."""

    @patch("remyxai.api.interests.requests.get")
    def test_list_interests_returns_list(self, mock_get):
        mock_get.return_value = _ok_get({
            "interests": [
                {"id": "a", "name": "RAG", "daily_count": 2, "is_active": True},
                {"id": "b", "name": "LLM Efficiency", "daily_count": 3, "is_active": False},
            ],
            "count": 2,
        })
        from remyxai.api.interests import list_interests
        result = list_interests()
        assert len(result) == 2
        assert result[0]["name"] == "RAG"
        assert result[1]["is_active"] is False

    @patch("remyxai.api.interests.requests.post")
    def test_create_sends_correct_payload(self, mock_post):
        mock_post.return_value = _ok_post({
            "id": "new-id", "name": "My Interest",
            "context": "LLMs and efficiency", "daily_count": 3, "is_active": True,
        })
        from remyxai.api.interests import create_interest
        result = create_interest(
            name="My Interest",
            context="LLMs and efficiency",
            daily_count=3,
            is_active=True,
        )
        sent = mock_post.call_args[1]["json"]
        assert sent["name"] == "My Interest"
        assert sent["context"] == "LLMs and efficiency"
        assert sent["daily_count"] == 3
        assert sent["is_active"] is True
        assert result["id"] == "new-id"

    @patch("remyxai.api.interests.requests.put")
    def test_update_sends_only_provided_fields(self, mock_put):
        mock_put.return_value = _ok_get({"id": "abc", "daily_count": 5})
        from remyxai.api.interests import update_interest
        update_interest("abc", daily_count=5)
        sent = mock_put.call_args[1]["json"]
        assert sent == {"daily_count": 5}
        assert "name" not in sent
        assert "context" not in sent

    @patch("remyxai.api.interests.requests.put")
    def test_update_omits_none_fields(self, mock_put):
        mock_put.return_value = _ok_get({"id": "abc", "name": "New Name"})
        from remyxai.api.interests import update_interest
        update_interest("abc", name="New Name", context=None, daily_count=None)
        sent = mock_put.call_args[1]["json"]
        assert "name" in sent
        assert "context" not in sent
        assert "daily_count" not in sent

    @patch("remyxai.api.interests.requests.delete")
    def test_delete_calls_correct_url(self, mock_delete):
        mock_delete.return_value = _ok_get({"message": "Interest deleted", "id": "target-id"})
        from remyxai.api.interests import delete_interest
        delete_interest("target-id")
        url = mock_delete.call_args[0][0]
        assert url.endswith("/interests/target-id")

    @patch("remyxai.api.interests.requests.post")
    def test_toggle_calls_correct_url(self, mock_post):
        mock_post.return_value = _ok_get({"id": "abc", "is_active": True})
        from remyxai.api.interests import toggle_interest
        toggle_interest("abc")
        url = mock_post.call_args[0][0]
        assert url.endswith("/interests/abc/toggle")


# ═════════════════════════════════════════════════════════════════════════════
# Backwards compatibility
# ═════════════════════════════════════════════════════════════════════════════

class TestBackwardsCompatibility:
    """New modules don't break anything; old calling conventions still work."""

    def test_recommendations_module_importable(self):
        import remyxai.api.recommendations  # noqa: F401

    def test_interests_module_importable(self):
        import remyxai.api.interests  # noqa: F401

    @patch("remyxai.api.recommendations.requests.get")
    def test_digest_no_api_key_arg_works(self, mock_get):
        """Original calling convention — no api_key kwarg."""
        mock_get.return_value = _ok_get(
            {"date": "2026-03-21", "period": "today",
             "interests": [], "total_papers": 0}
        )
        from remyxai.api.recommendations import get_recommendations_digest
        result = get_recommendations_digest()
        assert "interests" in result

    @patch("remyxai.api.interests.requests.get")
    def test_list_interests_no_api_key_arg_works(self, mock_get):
        mock_get.return_value = _ok_get({"interests": [], "count": 0})
        from remyxai.api.interests import list_interests
        result = list_interests()
        assert isinstance(result, list)


# ═════════════════════════════════════════════════════════════════════════════
# remyxai.api.interests — repo-sourced flow (REMYX-28 via REMYX-19)
# ═════════════════════════════════════════════════════════════════════════════


class TestRepoInterestHelpers:
    @patch("remyxai.api.interests.requests.post")
    def test_analyze_repo_posts_repo_url(self, mock_post):
        from remyxai.api.interests import analyze_repo

        m = Mock()
        m.status_code = 202
        m.json.return_value = {
            "task_id": "tk-1",
            "status_url": "/api/v1.0/interests/analyze-repo/tk-1",
        }
        m.raise_for_status = Mock()
        mock_post.return_value = m

        result = analyze_repo("https://github.com/owner/repo")

        args, kwargs = mock_post.call_args
        assert args[0].endswith("/interests/analyze-repo")
        assert kwargs["json"] == {"repo_url": "https://github.com/owner/repo"}
        assert result["task_id"] == "tk-1"

    @patch("remyxai.api.interests.requests.get")
    def test_poll_repo_analysis(self, mock_get):
        from remyxai.api.interests import poll_repo_analysis

        mock_get.return_value = _ok_get(
            {"status": "running", "message": "Fetching README"}
        )
        poll_repo_analysis("tk-1")

        assert mock_get.call_args.args[0].endswith(
            "/interests/analyze-repo/tk-1"
        )

    @patch("remyxai.api.interests.requests.post")
    def test_regenerate_interest_empty_body(self, mock_post):
        from remyxai.api.interests import regenerate_interest

        m = Mock()
        m.status_code = 202
        m.json.return_value = {"task_id": "tk-2"}
        m.raise_for_status = Mock()
        mock_post.return_value = m

        regenerate_interest("i1")

        args, kwargs = mock_post.call_args
        assert args[0].endswith("/interests/i1/regenerate")
        assert kwargs["json"] == {}

    @patch("remyxai.api.interests.requests.post")
    def test_regenerate_interest_with_override(self, mock_post):
        from remyxai.api.interests import regenerate_interest

        m = Mock()
        m.status_code = 202
        m.json.return_value = {"task_id": "tk-3"}
        m.raise_for_status = Mock()
        mock_post.return_value = m

        regenerate_interest("i1", repo_url="https://github.com/other/repo")

        body = mock_post.call_args.kwargs["json"]
        assert body == {"repo_url": "https://github.com/other/repo"}

    @patch("remyxai.api.interests.requests.get")
    def test_list_github_repos_disconnected(self, mock_get):
        from remyxai.api.interests import list_github_repos

        mock_get.return_value = _ok_get({"connected": False, "repos": []})
        result = list_github_repos()

        assert mock_get.call_args.args[0].endswith("/interests/github/repos")
        assert result == {"connected": False, "repos": []}


class TestCreateInterestRepoFields:
    @patch("remyxai.api.interests.requests.post")
    def test_omits_repo_fields_when_not_provided(self, mock_post):
        from remyxai.api.interests import create_interest

        mock_post.return_value = _ok_post({"id": "i1"})
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

        mock_post.return_value = _ok_post({"id": "i2"})
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

        mock_put.return_value = _ok_get({"id": "i1", "pool_invalidated": 3})
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
