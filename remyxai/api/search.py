"""
Search API client for accessing Remyx research assets (papers + Docker images).

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
class Asset:
    """
    Represents a research asset (paper with optional Docker image).
    """
    
    arxiv_id: str
    title: str
    docker_image: str = None
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    url: Optional[str] = None
    published_at: Optional[str] = None
    github_url: Optional[str] = None
    has_docker: bool = False
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
    def from_dict(cls, data: Dict) -> 'Asset':
        """Create Asset from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


def search_assets(
    query: str,
    max_results: int = 10,
    has_docker: bool = None,
    categories: Optional[List[str]] = None,
    use_llm: bool = True
) -> Dict[str, any]:
    """
    Search Remyx catalog for research assets.

    Args:
        query: Natural language search query
        max_results: Maximum number of results (max 50)
        has_docker: Filter by Docker availability (True/False/None)
        categories: Filter by arXiv categories
        use_llm: Use LLM-enhanced search if available

    Returns:
        Dictionary containing:
        - assets: List of Asset objects
        - total: Total number of results
        - query: Original query string
        - strategy: Search strategy used
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    url = f"{BASE_URL}/search/assets"

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
        assets = [Asset.from_dict(p) for p in data.get("assets", [])]

        return {
            "assets": assets,
            "total": data.get("total", len(assets)),
            "query": data.get("query", query),
            "strategy": data.get("strategy", "traditional")
        }
    else:
        logging.error(f"Failed to search assets: {response.status_code}")
        response.raise_for_status()


def get_asset(arxiv_id: str) -> Optional[Asset]:
    """Get detailed information about a specific asset."""
    url = f"{BASE_URL}/search/assets/{arxiv_id}"
    
    logging.info(f"GET request to {url}")
    
    response = requests.get(url, headers=HEADERS, timeout=30)
    
    if response.status_code == 404:
        logging.warning(f"Asset {arxiv_id} not found")
        return None
    
    log_api_response(response)
    
    if response.status_code == 200:
        return Asset.from_dict(response.json())
    else:
        logging.error(f"Failed to get asset: {response.status_code}")
        response.raise_for_status()


def list_assets(
    limit: int = 20,
    offset: int = 0,
    categories: Optional[List[str]] = None,
    has_docker: Optional[bool] = None
) -> Dict[str, any]:
    """List recently added research assets."""
    url = f"{BASE_URL}/search/assets/list"
    
    params = {
        "limit": min(limit, 100),
        "offset": offset
    }
    
    if has_docker is not None:
        params["has_docker"] = str(has_docker).lower()
    
    if categories:
        for cat in categories:
            params.setdefault("category", []).append(cat)
    
    logging.info(f"GET request to {url}")
    logging.debug(f"Params: {params}")
    
    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    
    log_api_response(response)
    
    if response.status_code == 200:
        data = response.json()
        assets = [Asset.from_dict(p) for p in data.get("assets", [])]
        
        return {
            "assets": assets,
            "total": data.get("total", 0),
            "limit": data.get("limit", limit),
            "offset": data.get("offset", offset)
        }
    else:
        logging.error(f"Failed to list assets: {response.status_code}")
        response.raise_for_status()


def get_stats() -> Dict[str, any]:
    """Get statistics about available research assets."""
    url = f"{BASE_URL}/search/stats"
    
    logging.info(f"GET request to {url}")
    
    response = requests.get(url, headers=HEADERS, timeout=30)
    
    log_api_response(response)
    
    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Failed to get stats: {response.status_code}")
        response.raise_for_status()

