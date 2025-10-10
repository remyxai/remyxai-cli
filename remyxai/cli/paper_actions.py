"""
CLI actions for managing papers
"""
import logging
import json
from typing import Optional, List
from remyxai.api.papers import (
    search_papers,
    get_paper,
    list_papers,
    get_stats,
    check_health,
)

logger = logging.getLogger(__name__)



def handle_paper_search(
    query: str, 
    max_results: int = 10, 
    categories: Optional[List[str]] = None,
    has_docker: Optional[bool] = None  # NEW: Add has_docker parameter
):
    """Handle paper search action."""
    print(f"\n🔍 Searching for: '{query}'")
    
    # Show filter status
    filter_info = []
    if has_docker is True:
        filter_info.append("with Docker only")
    elif has_docker is False:
        filter_info.append("without Docker only")
    else:
        filter_info.append("all papers")
    
    if categories:
        filter_info.append(f"categories: {', '.join(categories)}")
    
    print(f"   Filters: {', '.join(filter_info)}")
    print("=" * 80)
    
    try:
        results = search_papers(
            query=query,
            max_results=max_results,
            categories=categories,
            has_docker=has_docker,  # Pass through the filter
            use_llm=True
        )
        
        papers = results['papers']
        strategy = results.get('strategy', 'traditional')
        
        if not papers:
            print("No papers found.")
            return
        
        print(f"\nFound {results['total']} papers (strategy: {strategy}):\n")
        
        for i, paper in enumerate(papers, 1):
            # Show Docker status icon
            docker_icon = "🐳" if paper.has_docker else "📄"
            
            print(f"{i}. {docker_icon} {paper.title}")
            print(f"   arXiv: {paper.arxiv_id}")
            print(f"   Categories: {', '.join(paper.categories[:3])}")
            
            if paper.has_docker:
                print(f"   Docker: {paper.docker_image}")
            else:
                print(f"   Docker: Not available")
            
            if paper.github_url:
                print(f"   GitHub: {paper.github_url}")
            if paper.quickstart_hint:
                print(f"   💡 Hint: {paper.quickstart_hint}")
            print()
            
    except Exception as e:
        print(f"❌ Search failed: {e}")
        logger.error(f"Search error: {e}", exc_info=True)

def handle_paper_info(arxiv_id: str, output_format: str = "text"):
    """Handle paper info action."""
    try:
        paper = get_paper(arxiv_id)
        
        if not paper:
            print(f"❌ Paper {arxiv_id} not found.")
            return
        
        if output_format == "json":
            # Output as JSON for programmatic use (e.g., AG2)
            print(json.dumps(paper.to_dict(), indent=2))
        else:
            # Human-readable format
            print("\n" + "=" * 80)
            print(f"Paper: {paper.title}")
            print("=" * 80)
            print(f"\narXiv ID: {paper.arxiv_id}")
            print(f"Authors: {', '.join(paper.authors[:3])}")
            if len(paper.authors) > 3:
                print(f"         ... and {len(paper.authors) - 3} more")
            print(f"Categories: {', '.join(paper.categories)}")
            if paper.published_at:
                print(f"Published: {paper.published_at[:10]}")
            
            print(f"\nAbstract:\n{paper.abstract}\n")
            
            print(f"Docker Image: {paper.docker_image}")
            print(f"Build Status: {paper.docker_build_status}")
            print(f"Working Directory: {paper.working_directory}")
            print(f"Required Env Vars: {', '.join(paper.environment_vars)}")
            
            if paper.url:
                print(f"\nArXiv URL: {paper.url}")
            if paper.github_url:
                print(f"GitHub: {paper.github_url}")
            if paper.dockerfile_s3_uri:
                print(f"Dockerfile: {paper.dockerfile_s3_uri}")
            
            if paper.reasoning:
                print(f"\n📝 Context:\n{paper.reasoning}")
            
            if paper.quickstart_hint:
                print(f"\n💡 Quickstart Hint:\n{paper.quickstart_hint}")
            
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Get paper error: {e}", exc_info=True)


def handle_paper_list(limit: int = 20, offset: int = 0, categories: Optional[List[str]] = None):
    """Handle paper list action."""
    print("\n📚 Recently Containerized Papers")
    print("=" * 80)
    
    try:
        results = list_papers(limit=limit, offset=offset, categories=categories)
        
        papers = results['papers']
        total = results['total']
        
        if not papers:
            print("No papers found.")
            return
        
        print(f"\nShowing {len(papers)} of {total} papers (offset: {offset}):\n")
        
        for i, paper in enumerate(papers, 1):
            idx = offset + i
            print(f"{idx}. [{paper.arxiv_id}] {paper.title}")
            print(f"    Docker: {paper.docker_image}")
            if paper.github_url:
                print(f"    GitHub: {paper.github_url}")
            print()
        
        # Pagination hint
        if offset + len(papers) < total:
            print(f"💡 To see more: remyxai papers list --limit {limit} --offset {offset + limit}")
            
    except Exception as e:
        print(f"❌ List failed: {e}")
        logger.error(f"List papers error: {e}", exc_info=True)


def handle_paper_stats():
    """Handle paper stats action."""
    print("\n📊 Remyx Papers Statistics")
    print("=" * 80)
    
    try:
        stats = get_stats()
        
        print(f"\nTotal Papers: {stats.get('total_papers', 0)}")
        print(f"With Docker: {stats.get('papers_with_docker', 0)}")
        print(f"Recent Additions (7 days): {stats.get('recent_additions', 0)}")
        
        categories = stats.get('categories', {})
        if categories:
            print(f"\nTop Categories:")
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cat}: {count}")
        
        print()
        
    except Exception as e:
        print(f"❌ Stats failed: {e}")
        logger.error(f"Stats error: {e}", exc_info=True)

