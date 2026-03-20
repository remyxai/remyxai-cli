"""
Tests for lazy API key resolution.

Verifies that:
- Import succeeds without REMYXAI_API_KEY set
- get_api_key() resolves explicit key > env var > raises
- get_headers() builds correct auth headers
- api_key= threads through all API modules
- SearchClient(api_key=...) works for per-request keys
- Backwards compatibility: no api_key= uses module-level HEADERS
"""
import os
import inspect
import pytest
from unittest.mock import patch, Mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def ensure_env_key():
    """Ensure REMYXAI_API_KEY is set for module-level HEADERS during tests."""
    original = os.environ.get("REMYXAI_API_KEY")
    os.environ["REMYXAI_API_KEY"] = "test-key-for-fixtures"
    yield
    if original:
        os.environ["REMYXAI_API_KEY"] = original
    else:
        os.environ.pop("REMYXAI_API_KEY", None)


@pytest.fixture
def mock_post_response():
    """Mock successful POST response."""
    mock = Mock()
    mock.status_code = 200
    mock.json.return_value = {"assets": [], "total": 0}
    return mock


@pytest.fixture
def mock_get_response():
    """Mock successful GET response."""
    mock = Mock()
    mock.status_code = 200
    mock.json.return_value = {"message": []}
    return mock


# ---------------------------------------------------------------------------
# Tests: remyxai.api.__init__ (get_api_key, get_headers)
# ---------------------------------------------------------------------------


class TestGetApiKey:
    """Tests for get_api_key()."""

    def test_explicit_key_wins(self):
        from remyxai.api import get_api_key

        assert get_api_key("explicit-key") == "explicit-key"

    def test_falls_back_to_env_var(self):
        from remyxai.api import get_api_key

        os.environ["REMYXAI_API_KEY"] = "env-key-abc"
        assert get_api_key() == "env-key-abc"

    def test_explicit_key_overrides_env_var(self):
        from remyxai.api import get_api_key

        os.environ["REMYXAI_API_KEY"] = "env-key"
        assert get_api_key("explicit-key") == "explicit-key"

    def test_raises_when_no_key(self):
        from remyxai.api import get_api_key

        os.environ.pop("REMYXAI_API_KEY", None)
        with pytest.raises(ValueError, match="REMYXAI_API_KEY not found"):
            get_api_key()


class TestGetHeaders:
    """Tests for get_headers()."""

    def test_builds_headers_with_explicit_key(self):
        from remyxai.api import get_headers

        headers = get_headers("my-key")
        assert headers["Authorization"] == "Bearer my-key"
        assert headers["Content-Type"] == "application/json"

    def test_builds_headers_from_env_var(self):
        from remyxai.api import get_headers

        os.environ["REMYXAI_API_KEY"] = "env-key-xyz"
        headers = get_headers()
        assert headers["Authorization"] == "Bearer env-key-xyz"

    def test_raises_when_no_key(self):
        from remyxai.api import get_headers

        os.environ.pop("REMYXAI_API_KEY", None)
        with pytest.raises(ValueError):
            get_headers()


# ---------------------------------------------------------------------------
# Tests: api_key threads through all API modules
# ---------------------------------------------------------------------------


class TestApiKeySignatures:
    """Verify every public API function accepts api_key=None."""

    def test_search_module(self):
        from remyxai.api import search

        funcs = [search.search_assets, search.get_asset, search.list_assets, search.get_stats]
        self._assert_all_have_api_key(funcs)

    def test_datasets_module(self):
        from remyxai.api import datasets

        funcs = [datasets.list_datasets, datasets.download_dataset, datasets.delete_dataset]
        self._assert_all_have_api_key(funcs)

    def test_models_module(self):
        from remyxai.api import models

        funcs = [models.list_models, models.get_model_summary, models.delete_model, models.download_model]
        self._assert_all_have_api_key(funcs)

    def test_tasks_module(self):
        from remyxai.api import tasks

        funcs = [
            tasks.run_myxmatch, tasks.run_benchmark, tasks.get_job_status,
            tasks.train_classifier, tasks.train_detector, tasks.train_generator,
            tasks.run_datacomposer,
        ]
        self._assert_all_have_api_key(funcs)

    def test_evaluations_module(self):
        from remyxai.api import evaluations

        funcs = [evaluations.list_evaluations, evaluations.download_evaluation, evaluations.delete_evaluation]
        self._assert_all_have_api_key(funcs)

    def test_deployment_module(self):
        from remyxai.api import deployment

        funcs = [deployment.download_deployment_package, deployment.deploy_model]
        self._assert_all_have_api_key(funcs)

    def test_user_module(self):
        from remyxai.api import user

        funcs = [user.get_user_profile, user.get_user_credits]
        self._assert_all_have_api_key(funcs)

    def test_myxboard_module(self):
        from remyxai.api import myxboard

        funcs = [
            myxboard.store_myxboard, myxboard.list_myxboards,
            myxboard.update_myxboard, myxboard.delete_myxboard,
            myxboard.download_myxboard,
        ]
        self._assert_all_have_api_key(funcs)

    @staticmethod
    def _assert_all_have_api_key(funcs):
        for fn in funcs:
            sig = inspect.signature(fn)
            assert "api_key" in sig.parameters, (
                f"{fn.__module__}.{fn.__name__} missing api_key parameter"
            )
            param = sig.parameters["api_key"]
            assert param.default is None, (
                f"{fn.__module__}.{fn.__name__} api_key should default to None"
            )


# ---------------------------------------------------------------------------
# Tests: api_key actually threads through to requests calls
# ---------------------------------------------------------------------------


class TestApiKeyThreading:
    """Verify explicit api_key ends up in the Authorization header."""

    @patch("remyxai.api.search.requests.post")
    def test_search_assets_uses_explicit_key(self, mock_post, mock_post_response):
        from remyxai.api.search import search_assets

        mock_post.return_value = mock_post_response
        search_assets("test query", api_key="explicit-search-key")

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer explicit-search-key"

    @patch("remyxai.api.search.requests.post")
    def test_search_assets_without_key_uses_default(self, mock_post, mock_post_response):
        from remyxai.api.search import search_assets

        mock_post.return_value = mock_post_response
        search_assets("test query")

        call_kwargs = mock_post.call_args[1]
        assert "Bearer" in call_kwargs["headers"]["Authorization"]

    @patch("remyxai.api.datasets.requests.get")
    def test_list_datasets_uses_explicit_key(self, mock_get, mock_get_response):
        from remyxai.api.datasets import list_datasets

        mock_get.return_value = mock_get_response
        list_datasets(api_key="dataset-key-123")

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer dataset-key-123"

    @patch("remyxai.api.models.requests.get")
    def test_list_models_uses_explicit_key(self, mock_get, mock_get_response):
        from remyxai.api.models import list_models

        mock_get_response.json.return_value = ["model_1"]
        mock_get.return_value = mock_get_response
        list_models(api_key="models-key-456")

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer models-key-456"

    @patch("remyxai.api.user.requests.get")
    def test_get_user_profile_uses_explicit_key(self, mock_get, mock_get_response):
        from remyxai.api.user import get_user_profile

        mock_get_response.json.return_value = {"name": "test"}
        mock_get.return_value = mock_get_response
        get_user_profile(api_key="user-key-789")

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer user-key-789"

    @patch("remyxai.api.myxboard.requests.get")
    def test_list_myxboards_uses_explicit_key(self, mock_get, mock_get_response):
        from remyxai.api.myxboard import list_myxboards

        mock_get.return_value = mock_get_response
        list_myxboards(api_key="myxboard-key")

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer myxboard-key"


# ---------------------------------------------------------------------------
# Tests: SearchClient
# ---------------------------------------------------------------------------


class TestSearchClient:
    """Tests for SearchClient with api_key support."""

    def test_init_without_key(self):
        from remyxai.client.search import SearchClient

        client = SearchClient()
        assert client.api_key is None

    def test_init_with_key(self):
        from remyxai.client.search import SearchClient

        client = SearchClient(api_key="user-key")
        assert client.api_key == "user-key"

    @patch("remyxai.api.search.requests.post")
    def test_search_threads_key(self, mock_post, mock_post_response):
        from remyxai.client.search import SearchClient

        mock_post.return_value = mock_post_response
        client = SearchClient(api_key="client-key-abc")
        client.search("test query")

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer client-key-abc"

    @patch("remyxai.api.search.requests.post")
    def test_search_without_key_uses_env(self, mock_post, mock_post_response):
        from remyxai.client.search import SearchClient

        mock_post.return_value = mock_post_response
        client = SearchClient()
        client.search("test query")

        call_kwargs = mock_post.call_args[1]
        assert "Bearer" in call_kwargs["headers"]["Authorization"]

    @patch("remyxai.api.search.requests.get")
    def test_get_threads_key(self, mock_get):
        from remyxai.client.search import SearchClient

        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "arxiv_id": "2010.11929v2", "title": "Test", "has_docker": True
        }
        mock_get.return_value = mock_resp

        client = SearchClient(api_key="get-key")
        client.get("2010.11929v2")

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer get-key"

    @patch("remyxai.api.search.requests.get")
    def test_stats_threads_key(self, mock_get, mock_get_response):
        from remyxai.client.search import SearchClient

        mock_get_response.json.return_value = {"total_assets": 100}
        mock_get.return_value = mock_get_response

        client = SearchClient(api_key="stats-key")
        client.stats()

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer stats-key"


# ---------------------------------------------------------------------------
# Tests: Backwards compatibility
# ---------------------------------------------------------------------------


class TestBackwardsCompatibility:
    """Ensure nothing breaks for existing callers."""

    def test_module_exports_unchanged(self):
        """All original module-level names still exist."""
        import remyxai.api as api

        assert hasattr(api, "BASE_URL")
        assert hasattr(api, "HEADERS")
        assert hasattr(api, "REMYXAI_API_KEY")
        assert hasattr(api, "log_api_response")

    def test_headers_is_dict(self):
        import remyxai.api as api

        assert isinstance(api.HEADERS, dict)
        assert "Authorization" in api.HEADERS
        assert "Content-Type" in api.HEADERS

    def test_base_url_unchanged(self):
        import remyxai.api as api

        assert api.BASE_URL == "https://engine.remyx.ai/api/v1.0"

    @patch("remyxai.api.search.requests.post")
    def test_search_assets_works_without_api_key_arg(self, mock_post, mock_post_response):
        """Original calling convention still works."""
        from remyxai.api.search import search_assets

        mock_post.return_value = mock_post_response
        # This is how all existing code calls it — no api_key arg
        results = search_assets("test", max_results=5, has_docker=True)

        assert results["total"] == 0
        mock_post.assert_called_once()

    @patch("remyxai.api.models.requests.get")
    def test_list_models_works_without_api_key_arg(self, mock_get, mock_get_response):
        from remyxai.api.models import list_models

        mock_get_response.json.return_value = ["m1", "m2"]
        mock_get.return_value = mock_get_response

        result = list_models()
        assert result == ["m1", "m2"]

    @patch("remyxai.api.tasks.requests.post")
    def test_train_classifier_works_without_api_key_arg(self, mock_post):
        from remyxai.api.tasks import train_classifier

        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"task_id": "123"}

        result = train_classifier("model", ["a", "b"], "3")
        assert result["task_id"] == "123"
