# tests/api/test_papers.py
"""
Tests for papers API client
"""
import pytest
from unittest.mock import Mock, patch
from remyxai.api.papers import (
    Paper,
    search_papers,
    get_paper,
    list_papers,
    get_stats,
    check_health
)


@pytest.fixture
def mock_response():
    """Create a mock response object."""
    response = Mock()
    response.status_code = 200
    return response


def test_paper_dataclass():
    """Test Paper dataclass creation."""
    paper = Paper(
        arxiv_id="2010.11929v2",
        title="Test Paper",
        docker_image="remyxai/test:latest"
    )
    
    assert paper.arxiv_id == "2010.11929v2"
    assert paper.title == "Test Paper"
    assert paper.working_directory == "/app"


def test_paper_from_dict():
    """Test creating Paper from dictionary."""
    data = {
        "arxiv_id": "2010.11929v2",
        "title": "Test Paper",
        "docker_image": "remyxai/test:latest",
        "extra_field": "should be ignored"
    }
    
    paper = Paper.from_dict(data)
    
    assert paper.arxiv_id == "2010.11929v2"
    assert paper.title == "Test Paper"
    assert not hasattr(paper, "extra_field")


@patch('remyxai.api.papers.requests.post')
def test_search_papers(mock_post, mock_response):
    """Test search_papers function."""
    mock_response.json.return_value = {
        "papers": [{
            "arxiv_id": "2010.11929v2",
            "title": "Test Paper",
            "docker_image": "remyxai/test:latest"
        }],
        "total": 1,
        "query": "test",
        "strategy": "semantic"
    }
    mock_post.return_value = mock_response
    
    results = search_papers("test", max_results=10)
    
    assert len(results['papers']) == 1
    assert results['total'] == 1
    assert results['strategy'] == "semantic"
    assert isinstance(results['papers'][0], Paper)


@patch('remyxai.api.papers.requests.get')
def test_get_paper_found(mock_get, mock_response):
    """Test get_paper when paper exists."""
    mock_response.json.return_value = {
        "arxiv_id": "2010.11929v2",
        "title": "Test Paper",
        "docker_image": "remyxai/test:latest"
    }
    mock_get.return_value = mock_response
    
    paper = get_paper("2010.11929v2")
    
    assert paper is not None
    assert paper.arxiv_id == "2010.11929v2"
    assert isinstance(paper, Paper)


@patch('remyxai.api.papers.requests.get')
def test_get_paper_not_found(mock_get):
    """Test get_paper when paper doesn't exist."""
    response = Mock()
    response.status_code = 404
    mock_get.return_value = response
    
    paper = get_paper("9999.99999")
    
    assert paper is None


@patch('remyxai.api.papers.requests.get')
def test_list_papers(mock_get, mock_response):
    """Test list_papers function."""
    mock_response.json.return_value = {
        "papers": [
            {
                "arxiv_id": "2010.11929v2",
                "title": "Paper 1",
                "docker_image": "remyxai/test1:latest"
            },
            {
                "arxiv_id": "2006.11239v2",
                "title": "Paper 2",
                "docker_image": "remyxai/test2:latest"
            }
        ],
        "total": 2,
        "limit": 20,
        "offset": 0
    }
    mock_get.return_value = mock_response
    
    results = list_papers(limit=20, offset=0)
    
    assert len(results['papers']) == 2
    assert results['total'] == 2
    assert all(isinstance(p, Paper) for p in results['papers'])


@patch('remyxai.api.papers.requests.get')
def test_get_stats(mock_get, mock_response):
    """Test get_stats function."""
    mock_response.json.return_value = {
        "total_papers": 1234,
        "papers_with_docker": 567,
        "recent_additions": 45,
        "categories": {"cs.LG": 234, "cs.CV": 189}
    }
    mock_get.return_value = mock_response
    
    stats = get_stats()
    
    assert stats['total_papers'] == 1234
    assert stats['papers_with_docker'] == 567
    assert "cs.LG" in stats['categories']


@patch('remyxai.api.papers.requests.get')
def test_check_health(mock_get, mock_response):
    """Test check_health function."""
    mock_response.json.return_value = {
        "status": "healthy",
        "version": "1.0",
        "features": {
            "llm_search": True,
            "docker_images": True
        }
    }
    mock_get.return_value = mock_response
    
    health = check_health()
    
    assert health['status'] == "healthy"
    assert health['features']['llm_search'] is True
