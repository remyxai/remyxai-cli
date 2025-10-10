"""
Tests for search API module
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from remyxai.api.search import (
    Asset,
    search_assets,
    get_asset,
    list_assets,
    get_stats,
)


@pytest.fixture
def mock_response():
    """Create a mock response object."""
    mock = Mock()
    mock.status_code = 200
    return mock


@pytest.fixture
def sample_asset_data():
    """Sample asset data for testing."""
    return {
        "arxiv_id": "2010.11929v2",
        "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "abstract": "While the Transformer architecture has become the de-facto standard...",
        "authors": ["Alexey Dosovitskiy", "Lucas Beyer"],
        "categories": ["cs.CV", "cs.LG"],
        "url": "https://arxiv.org/abs/2010.11929",
        "published_at": "2020-10-22T17:58:08Z",
        "has_docker": True,
        "docker_image": "remyxai/201011929v2:latest",
        "github_url": "https://github.com/google-research/vision_transformer",
        "docker_build_status": "completed",
        "working_directory": "/app",
        "environment_vars": ["HF_TOKEN", "OPENAI_API_KEY"],
        "reasoning": "Vision Transformer is a foundational paper in computer vision",
        "quickstart_hint": "python train.py --demo",
    }


class TestAsset:
    """Tests for Asset dataclass."""
    
    def test_asset_creation(self, sample_asset_data):
        """Test creating an Asset from data."""
        asset = Asset.from_dict(sample_asset_data)
        
        assert asset.arxiv_id == "2010.11929v2"
        assert asset.title == sample_asset_data["title"]
        assert asset.has_docker is True
        assert asset.docker_image == "remyxai/201011929v2:latest"
    
    def test_asset_display_name(self, sample_asset_data):
        """Test asset display name property."""
        asset = Asset.from_dict(sample_asset_data)
        display_name = asset.display_name
        
        assert "2010.11929v2" in display_name
        assert len(display_name) <= 100  # Should be truncated
    
    def test_asset_to_dict(self, sample_asset_data):
        """Test converting asset to dictionary."""
        asset = Asset.from_dict(sample_asset_data)
        asset_dict = asset.to_dict()
        
        assert asset_dict["arxiv_id"] == "2010.11929v2"
        assert asset_dict["has_docker"] is True
    
    def test_asset_without_docker(self):
        """Test asset without Docker image."""
        data = {
            "arxiv_id": "2105.14424v1",
            "title": "Test Paper",
            "has_docker": False,
            "docker_image": None,
        }
        asset = Asset.from_dict(data)
        
        assert asset.has_docker is False
        assert asset.docker_image is None


class TestSearchAssets:
    """Tests for search_assets function."""
    
    @patch('remyxai.api.search.requests.post')
    def test_search_assets_success(self, mock_post, mock_response, sample_asset_data):
        """Test successful asset search."""
        mock_response.json.return_value = {
            "papers": [sample_asset_data],
            "total": 1,
            "query": "vision transformers",
            "strategy": "semantic"
        }
        mock_post.return_value = mock_response
        
        results = search_assets("vision transformers", max_results=10)
        
        assert results["total"] == 1
        assert len(results["assets"]) == 1
        assert results["assets"][0].arxiv_id == "2010.11929v2"
        assert results["strategy"] == "semantic"
        
        # Verify request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "/search/assets" in call_args[0][0]
    
    @patch('remyxai.api.search.requests.post')
    def test_search_with_docker_filter(self, mock_post, mock_response, sample_asset_data):
        """Test searching with Docker filter."""
        mock_response.json.return_value = {
            "papers": [sample_asset_data],
            "total": 1,
            "query": "data synthesis",
            "strategy": "traditional"
        }
        mock_post.return_value = mock_response
        
        results = search_assets("data synthesis", has_docker=True)
        
        assert results["total"] == 1
        assert all(asset.has_docker for asset in results["assets"])
        
        # Verify has_docker parameter was sent
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["has_docker"] is True
    
    @patch('remyxai.api.search.requests.post')
    def test_search_without_docker_filter(self, mock_post, mock_response):
        """Test searching for assets without Docker."""
        asset_without_docker = {
            "arxiv_id": "2105.14424v1",
            "title": "Test Paper",
            "has_docker": False,
            "docker_image": None,
        }
        mock_response.json.return_value = {
            "papers": [asset_without_docker],
            "total": 1,
            "query": "test",
            "strategy": "traditional"
        }
        mock_post.return_value = mock_response
        
        results = search_assets("test", has_docker=False)
        
        assert results["total"] == 1
        assert not results["assets"][0].has_docker
    
    @patch('remyxai.api.search.requests.post')
    def test_search_with_categories(self, mock_post, mock_response, sample_asset_data):
        """Test searching with category filter."""
        mock_response.json.return_value = {
            "papers": [sample_asset_data],
            "total": 1,
            "query": "machine learning",
            "strategy": "traditional"
        }
        mock_post.return_value = mock_response
        
        results = search_assets(
            "machine learning",
            categories=["cs.LG", "cs.AI"]
        )
        
        assert results["total"] == 1
        
        # Verify categories were sent
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["categories"] == ["cs.LG", "cs.AI"]
    
    def test_search_empty_query(self):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            search_assets("")
    
    @patch('remyxai.api.search.requests.post')
    def test_search_api_error(self, mock_post):
        """Test handling of API errors."""
        mock_post.return_value.status_code = 500
        mock_post.return_value.raise_for_status.side_effect = Exception("Server error")
        
        with pytest.raises(Exception):
            search_assets("test query")


class TestGetAsset:
    """Tests for get_asset function."""
    
    @patch('remyxai.api.search.requests.get')
    def test_get_asset_success(self, mock_get, mock_response, sample_asset_data):
        """Test successfully retrieving an asset."""
        mock_response.json.return_value = sample_asset_data
        mock_get.return_value = mock_response
        
        asset = get_asset("2010.11929v2")
        
        assert asset is not None
        assert asset.arxiv_id == "2010.11929v2"
        assert asset.has_docker is True
        
        # Verify correct endpoint was called
        mock_get.assert_called_once()
        call_args = mock_get.call_args[0][0]
        assert "/search/assets/2010.11929v2" in call_args
    
    @patch('remyxai.api.search.requests.get')
    def test_get_asset_not_found(self, mock_get):
        """Test retrieving non-existent asset."""
        mock_get.return_value.status_code = 404
        
        asset = get_asset("invalid_id")
        
        assert asset is None
    
    @patch('remyxai.api.search.requests.get')
    def test_get_asset_without_docker(self, mock_get, mock_response):
        """Test retrieving asset without Docker image."""
        asset_data = {
            "arxiv_id": "2105.14424v1",
            "title": "Test Paper",
            "has_docker": False,
            "docker_image": None,
        }
        mock_response.json.return_value = asset_data
        mock_get.return_value = mock_response
        
        asset = get_asset("2105.14424v1")
        
        assert asset is not None
        assert asset.has_docker is False


class TestListAssets:
    """Tests for list_assets function."""
    
    @patch('remyxai.api.search.requests.get')
    def test_list_assets_success(self, mock_get, mock_response, sample_asset_data):
        """Test successfully listing assets."""
        mock_response.json.return_value = {
            "papers": [sample_asset_data, sample_asset_data],
            "total": 2,
            "limit": 20,
            "offset": 0
        }
        mock_get.return_value = mock_response
        
        results = list_assets(limit=20, offset=0)
        
        assert results["total"] == 2
        assert len(results["assets"]) == 2
        assert results["limit"] == 20
        assert results["offset"] == 0
    
    @patch('remyxai.api.search.requests.get')
    def test_list_with_docker_filter(self, mock_get, mock_response, sample_asset_data):
        """Test listing only assets with Docker."""
        mock_response.json.return_value = {
            "papers": [sample_asset_data],
            "total": 1,
            "limit": 20,
            "offset": 0
        }
        mock_get.return_value = mock_response
        
        results = list_assets(has_docker=True)
        
        assert results["total"] == 1
        assert all(asset.has_docker for asset in results["assets"])
        
        # Verify has_docker parameter was sent
        call_kwargs = mock_get.call_args[1]
        assert "has_docker=true" in str(call_kwargs.get("params", {})).lower()
    
    @patch('remyxai.api.search.requests.get')
    def test_list_with_categories(self, mock_get, mock_response, sample_asset_data):
        """Test listing with category filter."""
        mock_response.json.return_value = {
            "papers": [sample_asset_data],
            "total": 1,
            "limit": 20,
            "offset": 0
        }
        mock_get.return_value = mock_response
        
        results = list_assets(categories=["cs.CV"])
        
        assert results["total"] == 1
        
        # Verify categories were sent
        call_kwargs = mock_get.call_args[1]
        assert "category" in call_kwargs.get("params", {})
    
    @patch('remyxai.api.search.requests.get')
    def test_list_pagination(self, mock_get, mock_response, sample_asset_data):
        """Test pagination with offset."""
        mock_response.json.return_value = {
            "papers": [sample_asset_data],
            "total": 100,
            "limit": 20,
            "offset": 20
        }
        mock_get.return_value = mock_response
        
        results = list_assets(limit=20, offset=20)
        
        assert results["offset"] == 20
        assert results["total"] == 100


class TestGetStats:
    """Tests for get_stats function."""
    
    @patch('remyxai.api.search.requests.get')
    def test_get_stats_success(self, mock_get, mock_response):
        """Test successfully retrieving statistics."""
        stats_data = {
            "total_assets": 1000,
            "assets_with_docker": 500,
            "assets_without_docker": 500,
            "recent_additions": 50,
            "categories": {
                "cs.LG": 300,
                "cs.CV": 250,
                "cs.AI": 200
            }
        }
        mock_response.json.return_value = stats_data
        mock_get.return_value = mock_response
        
        stats = get_stats()
        
        assert stats["total_assets"] == 1000
        assert stats["assets_with_docker"] == 500
        assert stats["assets_without_docker"] == 500
        assert stats["recent_additions"] == 50
        assert "cs.LG" in stats["categories"]



class TestIntegration:
    """Integration tests for search functionality."""
    
    @patch('remyxai.api.search.requests.post')
    @patch('remyxai.api.search.requests.get')
    def test_search_and_get_workflow(
        self, mock_get, mock_post, mock_response, sample_asset_data
    ):
        """Test typical workflow: search then get details."""
        # Mock search response
        search_response = Mock()
        search_response.status_code = 200
        search_response.json.return_value = {
            "papers": [sample_asset_data],
            "total": 1,
            "query": "vision transformers",
            "strategy": "semantic"
        }
        mock_post.return_value = search_response
        
        # Mock get response
        get_response = Mock()
        get_response.status_code = 200
        get_response.json.return_value = sample_asset_data
        mock_get.return_value = get_response
        
        # Search for assets
        search_results = search_assets("vision transformers")
        assert len(search_results["assets"]) > 0
        
        # Get details for first asset
        first_asset_id = search_results["assets"][0].arxiv_id
        asset_details = get_asset(first_asset_id)
        
        assert asset_details.arxiv_id == first_asset_id
        assert asset_details.has_docker is True
