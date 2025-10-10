"""
Python client for papers API

This is a thin wrapper around remyxai.api.papers for convenience.
"""
import os
import logging
from typing import List, Optional

# Import from api module
from remyxai.api.papers import (
    Paper,
    search_papers as api_search_papers,
    get_paper as api_get_paper,
    list_papers as api_list_papers,
    get_stats as api_get_stats
)


class PapersClient:
    """
    Client for interacting with Remyx papers API.
    
    Used internally by AG2 RemyxCodeExecutor.
    Can also be used standalone for paper discovery.
    
    Example:
        >>> from remyxai.client.papers import PapersClient
        >>> 
        >>> client = PapersClient()
        >>> papers = client.search("vision transformers")
        >>> 
        >>> for paper in papers:
        ...     print(f"{paper.arxiv_id}: {paper.title}")
    """
    
    def __init__(self):
        """
        Initialize papers client.
        
        Uses REMYXAI_API_KEY from environment.
        """
        # API key validation happens in remyxai.api.__init__
        pass
    

    def search(
        self,
        query: str,
        max_results: int = 10,
        categories: Optional[List[str]] = None,
        has_docker: Optional[bool] = None,  # NEW: Add has_docker parameter
        use_llm: bool = True
    ) -> List[Paper]:
        """
        Search for papers.

        Args:
            query: Search query
            max_results: Maximum results
            categories: Filter by categories
            has_docker: Filter by Docker availability (True/False/None for all)
            use_llm: Use LLM-enhanced search

        Returns:
            List of Paper objects

        Example:
            >>> client = PapersClient()
            >>>
            >>> # Search all papers
            >>> papers = client.search("transformers")
            >>>
            >>> # Search only papers with Docker
            >>> papers = client.search("transformers", has_docker=True)
            >>>
            >>> # Search only papers without Docker
            >>> papers = client.search("transformers", has_docker=False)
        """
        results = api_search_papers(
            query=query,
            max_results=max_results,
            categories=categories,
            has_docker=has_docker,
            use_llm=use_llm
        )
        return results['papers']
    
    def get(self, arxiv_id: str) -> Optional[Paper]:
        """
        Get a specific paper by arXiv ID.
        
        Args:
            arxiv_id: arXiv identifier
        
        Returns:
            Paper object or None if not found
        """
        return api_get_paper(arxiv_id)
    
    def list(
        self,
        limit: int = 20,
        offset: int = 0,
        categories: Optional[List[str]] = None
    ) -> List[Paper]:
        """
        List recent papers.
        
        Args:
            limit: Number of papers
            offset: Pagination offset
            categories: Filter by categories
        
        Returns:
            List of Paper objects
        """
        results = api_list_papers(
            limit=limit,
            offset=offset,
            categories=categories
        )
        return results['papers']
