"""
Python client for search API

Thin wrapper around remyxai.api.search for convenience.
"""
from typing import List, Optional

from remyxai.api.search import (
    Asset,
    search_assets as api_search_assets,
    get_asset as api_get_asset,
    list_assets as api_list_assets,
    get_stats as api_get_stats
)


class SearchClient:
    """
    Client for interacting with Remyx search API.
    
    Example:
        >>> from remyxai.client.search import SearchClient
        >>> 
        >>> client = SearchClient()
        >>> assets = client.search("data synthesis")
        >>> 
        >>> for asset in assets:
        ...     print(f"{asset.arxiv_id}: {asset.title}")
    """
    
    def __init__(self):
        """Initialize search client."""
        pass
    
    def search(
        self,
        query: str,
        max_results: int = 10,
        categories: Optional[List[str]] = None,
        has_docker: Optional[bool] = None,
        use_llm: bool = True
    ) -> List[Asset]:
        """Search for assets."""
        results = api_search_assets(
            query=query,
            max_results=max_results,
            categories=categories,
            has_docker=has_docker,
            use_llm=use_llm
        )
        return results['assets']
    
    def get(self, arxiv_id: str) -> Optional[Asset]:
        """Get a specific asset by arXiv ID."""
        return api_get_asset(arxiv_id)
    
    def list(
        self,
        limit: int = 20,
        offset: int = 0,
        categories: Optional[List[str]] = None,
        has_docker: Optional[bool] = None
    ) -> List[Asset]:
        """List recent assets."""
        results = api_list_assets(
            limit=limit,
            offset=offset,
            categories=categories,
            has_docker=has_docker
        )
        return results['assets']
    
    def stats(self) -> dict:
        """Get statistics about available assets."""
        return api_get_stats()
