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
    get_stats as api_get_stats,
)


class SearchClient:
    """
    Client for interacting with Remyx search API.

    Examples:

        # Using environment variable (default — works with AG2 and CLI)
        >>> from remyxai.client.search import SearchClient
        >>>
        >>> client = SearchClient()
        >>> assets = client.search("data synthesis")

        # Using explicit API key (e.g. in a web app or HF Space)
        >>> client = SearchClient(api_key="your_key_here")
        >>> assets = client.search("data synthesis", has_docker=True)
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize search client.

        Args:
            api_key: Optional explicit API key. If not provided, the
                     REMYXAI_API_KEY environment variable is used.
        """
        self.api_key = api_key

    def search(
        self,
        query: str,
        max_results: int = 10,
        categories: Optional[List[str]] = None,
        has_docker: Optional[bool] = None,
        use_llm: bool = True,
    ) -> List[Asset]:
        """Search for assets."""
        results = api_search_assets(
            query=query,
            max_results=max_results,
            categories=categories,
            has_docker=has_docker,
            use_llm=use_llm,
            api_key=self.api_key,
        )
        return results["assets"]

    def get(self, arxiv_id: str) -> Optional[Asset]:
        """Get a specific asset by arXiv ID."""
        return api_get_asset(arxiv_id, api_key=self.api_key)

    def list(
        self,
        limit: int = 20,
        offset: int = 0,
        categories: Optional[List[str]] = None,
        has_docker: Optional[bool] = None,
    ) -> List[Asset]:
        """List recent assets."""
        results = api_list_assets(
            limit=limit,
            offset=offset,
            categories=categories,
            has_docker=has_docker,
            api_key=self.api_key,
        )
        return results["assets"]

    def stats(self) -> dict:
        """Get statistics about available assets."""
        return api_get_stats(api_key=self.api_key)
