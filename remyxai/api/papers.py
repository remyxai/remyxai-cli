"""
Papers API client for accessing Remyx containerized research papers.

Used by:
- remyxai CLI commands
- External integrations
"""
import logging
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field
from . import BASE_URL, HEADERS, log_api_response

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """
    Represents a containerized research paper from Remyx.
    
    Attributes:
        arxiv_id: arXiv paper identifier (e.g., "2010.11929v2")
        title: Paper title
        docker_image: Full Docker image name (e.g., "remyxai/201011929v2:latest")
        abstract: Paper abstract (may be truncated)
        authors: List of author names
        categories: arXiv categories (e.g., ["cs.CV", "cs.LG"])
        url: Canonical arXiv URL
        published_at: Publication date (ISO format string)
        github_url: GitHub repository URL if available
        has_docker: Whether Docker image is available
        working_directory: Working directory inside container (default: /app)
        environment_vars: Required environment variables
        docker_build_status: Build status (completed, building, failed, etc.)
        reasoning: Why this paper was containerized (from Remyx)
        quickstart_hint: Suggested command to run
        dockerfile_s3_uri: S3 path to Dockerfile
        dockerfile_text: Dockerfile content if available
    """
    
    arxiv_id: str
    title: str
    docker_image: str
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    url: Optional[str] = None
    published_at: Optional[str] = None
    github_url: Optional[str] = None
    has_docker: bool = True
    working_directory: str = "/app"
    environment_vars: List[str] = field(default_factory=lambda: ["HF_TOKEN", "OPENAI_API_KEY"])
    docker_build_status: Optional[str] = None
    reasoning: Optional[str] = None
    quickstart_hint: Optional[str] = None
    dockerfile_s3_uri: Optional[str] = None
    dockerfile_text: Optional[str] = None
    
    @property
    def display_name(self) -> str:
        """Short display name for UI."""
        return f"{self.arxiv_id}: {self.title[:60]}..."
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Paper':
        """Create Paper from dictionary."""
        # Only use fields that exist in the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


def search_papers(
    query: str,
    max_results: int = 10,
    has_docker: bool = None,
    categories: Optional[List[str]] = None,
    use_llm: bool = True
) -> Dict[str, any]:
    """
    Search Remyx catalog for research papers.

    Args:
        query: Natural language search query (e.g., "vision transformers")
        max_results: Maximum number of results to return (max 50)
        has_docker: Filter by Docker availability:
                    - True: Only papers with Docker images
                    - False: Only papers without Docker images
                    - None: All papers (default)
        categories: Filter by arXiv categories (e.g., ["cs.CV", "cs.LG"])
        use_llm: Use LLM-enhanced search if available

    Returns:
        Dictionary containing:
        - papers: List of Paper objects
        - total: Total number of results
        - query: Original query string
        - strategy: Search strategy used ("semantic" or "traditional")

    Raises:
        requests.HTTPError: If API request fails
        ValueError: If query is empty

    Example:
        >>> # Search all papers
        >>> results = search_papers("diffusion models", max_results=5)
        >>>
        >>> # Search only papers with Docker
        >>> results = search_papers("diffusion models", has_docker=True)
        >>>
        >>> # Search only papers without Docker
        >>> results = search_papers("diffusion models", has_docker=False)
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    url = f"{BASE_URL}/papers/search"

    payload = {
        "query": query.strip(),
        "max_results": min(max_results, 50),
        "use_llm": use_llm
    }

    if has_docker is not None:
        payload["has_docker"] = has_docker

    if categories:
        payload["categories"] = categories

    logging.info(f"POST request to {url}")
    logging.debug(f"Payload: {payload}")

    response = requests.post(url, json=payload, headers=HEADERS, timeout=30)

    log_api_response(response)

    if response.status_code == 200:
        data = response.json()

        papers = [Paper.from_dict(p) for p in data.get("papers", [])]

        return {
            "papers": papers,
            "total": data.get("total", len(papers)),
            "query": data.get("query", query),
            "strategy": data.get("strategy", "traditional")
        }
    else:
        logging.error(f"Failed to search papers: {response.status_code}")
        response.raise_for_status()


def get_paper(arxiv_id: str) -> Optional[Paper]:
    """
    Get detailed information about a specific paper.
    
    Args:
        arxiv_id: arXiv identifier (e.g., "2010.11929v2")
    
    Returns:
        Paper object or None if not found
    
    Raises:
        requests.HTTPError: If API request fails (except 404)
    
    Example:
        >>> paper = get_paper("2010.11929v2")
        >>> if paper:
        ...     print(paper.title)
    """
    url = f"{BASE_URL}/papers/{arxiv_id}"
    
    logging.info(f"GET request to {url}")
    
    response = requests.get(url, headers=HEADERS, timeout=30)
    
    if response.status_code == 404:
        logging.warning(f"Paper {arxiv_id} not found")
        return None
    
    log_api_response(response)
    
    if response.status_code == 200:
        return Paper.from_dict(response.json())
    else:
        logging.error(f"Failed to get paper: {response.status_code}")
        response.raise_for_status()


def list_papers(
    limit: int = 20,
    offset: int = 0,
    categories: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    List recently containerized papers.
    
    Args:
        limit: Number of papers to return (max 100)
        offset: Pagination offset
        categories: Filter by arXiv categories
    
    Returns:
        Dictionary containing:
        - papers: List of Paper objects
        - total: Total number of papers matching criteria
        - limit: Limit used
        - offset: Offset used
    
    Example:
        >>> results = list_papers(limit=10)
        >>> for paper in results['papers']:
        ...     print(paper.arxiv_id)
    """
    url = f"{BASE_URL}/papers/list"
    
    params = {
        "limit": min(limit, 100),
        "offset": offset
    }
    
    if categories:
        for cat in categories:
            params.setdefault("category", []).append(cat)
    
    logging.info(f"GET request to {url}")
    logging.debug(f"Params: {params}")
    
    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    
    log_api_response(response)
    
    if response.status_code == 200:
        data = response.json()
        
        papers = [Paper.from_dict(p) for p in data.get("papers", [])]
        
        return {
            "papers": papers,
            "total": data.get("total", 0),
            "limit": data.get("limit", limit),
            "offset": data.get("offset", offset)
        }
    else:
        logging.error(f"Failed to list papers: {response.status_code}")
        response.raise_for_status()


def get_stats() -> Dict[str, any]:
    """
    Get statistics about available papers.
    
    Returns:
        Dictionary containing statistics
    
    Example:
        >>> stats = get_stats()
        >>> print(f"Total papers: {stats['total_papers']}")
    """
    url = f"{BASE_URL}/papers/stats"
    
    logging.info(f"GET request to {url}")
    
    response = requests.get(url, headers=HEADERS, timeout=30)
    
    log_api_response(response)
    
    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Failed to get stats: {response.status_code}")
        response.raise_for_status()


def check_health() -> Dict[str, any]:
    """
    Check Remyx papers API health and features.
    
    Returns:
        Dictionary with status and available features
    
    Example:
        >>> health = check_health()
        >>> if health["status"] == "healthy":
        ...     print("API is ready!")
    """
    # Use integration health endpoint (no auth required)
    url = f"{BASE_URL.replace('/api/v1.0', '')}/api/v1.0/integration/health"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
