# remyxai/cli/search_actions.py
"""
CLI actions for searching and managing research assets
"""
import logging
import json
import sys
from typing import Optional, List
from remyxai.api.search import (
    search_assets,
    get_asset,
    list_assets,
    get_stats,
)

logger = logging.getLogger(__name__)


def handle_search(
    query: str, 
    max_results: int = 10, 
    categories: Optional[List[str]] = None,
    has_docker: Optional[bool] = None,
    output_format: str = "text"
):
    """Handle asset search action."""
    as_json = output_format == "json"

    if not as_json:
        print(f"\n🔍 Searching for: '{query}'")

        # Show filter status
        filter_info = []
        if has_docker is True:
            filter_info.append("with Docker only")
        elif has_docker is False:
            filter_info.append("without Docker only")
        else:
            filter_info.append("all assets")

        if categories:
            filter_info.append(f"categories: {', '.join(categories)}")

        print(f"   Filters: {', '.join(filter_info)}")
        print("=" * 80)
    
    try:
        results = search_assets(
            query=query,
            max_results=max_results,
            categories=categories,
            has_docker=has_docker,
            use_llm=True
        )
        
        assets = results['assets']
        strategy = results.get('strategy', 'traditional')

        if as_json:
            # Envelope mirrors the API response so scripts get the result
            # metadata, not just the hits. An empty result set is still valid
            # JSON — no tips, no chrome on stdout.
            print(json.dumps({
                "query": results.get("query", query),
                "strategy": strategy,
                "total": results.get("total", len(assets)),
                "assets": [a.to_dict() for a in assets],
            }, indent=2))
            return

        if not assets:
            print("\nNo assets found.")
            if has_docker is True:
                print("\n💡 Tip: Try searching without --docker flag to see all assets")
            return
        
        print(f"\nFound {results['total']} assets (strategy: {strategy}):\n")
        
        for i, asset in enumerate(assets, 1):
            # Show Docker status icon
            docker_icon = "🐳" if asset.has_docker else "📄"
            
            print(f"{i}. {docker_icon} {asset.title}")
            print(f"   arXiv: {asset.arxiv_id}")
            
            if asset.categories:
                print(f"   Categories: {', '.join(asset.categories[:3])}")
            
            if asset.has_docker:
                print(f"   Docker: {asset.docker_image}")
                if asset.quickstart_hint:
                    print(f"   💡 Hint: {asset.quickstart_hint}")
            else:
                print(f"   Docker: Not available")
            
            if asset.github_url:
                print(f"   GitHub: {asset.github_url}")
            
            print()
        
        # Add helpful tips based on results
        if has_docker is None:
            docker_count = sum(1 for a in assets if a.has_docker)
            if docker_count > 0:
                print(f"💡 {docker_count} of these assets have Docker images")
                print("   Use --docker flag to see only assets with Docker")
            
    except Exception as e:
        # In JSON mode a failure must not land on stdout — a consumer piping to
        # jq would otherwise parse an error banner as data — and it must exit
        # non-zero so the caller notices.
        if as_json:
            print(f"Search failed: {e}", file=sys.stderr)
            logger.error(f"Search error: {e}", exc_info=True)
            sys.exit(1)
        print(f"❌ Search failed: {e}")
        logger.error(f"Search error: {e}", exc_info=True)


def handle_info(arxiv_id: str, output_format: str = "text"):
    """Handle asset info action."""
    as_json = output_format == "json"
    try:
        asset = get_asset(arxiv_id)
        
        if not asset:
            # A missing asset is a failure a script has to notice, so in JSON
            # mode it exits non-zero with a clean stdout rather than printing a
            # tip a consumer would try to parse.
            if as_json:
                print(f"Asset {arxiv_id} not found in Remyx catalog or arXiv.",
                      file=sys.stderr)
                sys.exit(1)
            print(f"❌ Asset {arxiv_id} not found in Remyx catalog or arXiv.")
            print("\n💡 Tip: Use 'remyxai search query' to find assets")
            return
        
        if output_format == "json":
            # Output as JSON for programmatic use
            print(json.dumps(asset.to_dict(), indent=2))
        else:
            # Human-readable format
            docker_icon = "🐳" if asset.has_docker else "📄"
            
            print("\n" + "=" * 80)
            print(f"{docker_icon} Asset: {asset.title}")
            print("=" * 80)
            print(f"\narXiv ID: {asset.arxiv_id}")
            
            if asset.authors:
                print(f"Authors: {', '.join(asset.authors[:3])}")
                if len(asset.authors) > 3:
                    print(f"         ... and {len(asset.authors) - 3} more")
            
            if asset.categories:
                print(f"Categories: {', '.join(asset.categories)}")
            
            if asset.published_at:
                print(f"Published: {asset.published_at[:10]}")
            
            if asset.abstract:
                print(f"\nAbstract:\n{asset.abstract}\n")
            
            # Docker information
            if asset.has_docker:
                print(f"✅ Docker Image: {asset.docker_image}")
                if asset.docker_build_status:
                    print(f"   Build Status: {asset.docker_build_status}")
                print(f"   Working Directory: {asset.working_directory}")
                if asset.environment_vars:
                    print(f"   Required Env Vars: {', '.join(asset.environment_vars)}")
            else:
                print("❌ Docker Image: Not available")
                print("\n   This asset doesn't have a containerized implementation yet.")
                print("   Search for assets with Docker: remyxai search query 'your query' --docker")
            
            if asset.url:
                print(f"\nArXiv URL: {asset.url}")
            if asset.github_url:
                print(f"GitHub: {asset.github_url}")
            if asset.dockerfile_s3_uri:
                print(f"Dockerfile: {asset.dockerfile_s3_uri}")
            
            if asset.reasoning:
                print(f"\n📝 Context:\n{asset.reasoning}")
            
            if asset.quickstart_hint:
                print(f"\n💡 Quickstart Hint:\n{asset.quickstart_hint}")
            
            # AG2 usage hint if Docker available
            if asset.has_docker:
                print(f"\n🤖 AG2 Integration:")
                print(f"   pip install remyxai-ag2")
                print(f"   from remyxai_ag2 import RemyxCodeExecutor")
                print(f"   executor = RemyxCodeExecutor(arxiv_id='{asset.arxiv_id}')")
            
            print()
            
    except Exception as e:
        if as_json:
            print(f"Failed to get asset {arxiv_id}: {e}", file=sys.stderr)
            logger.error(f"Get asset error: {e}", exc_info=True)
            sys.exit(1)
        print(f"❌ Error: {e}")
        logger.error(f"Get asset error: {e}", exc_info=True)


def handle_list(
    limit: int = 20, 
    offset: int = 0, 
    categories: Optional[List[str]] = None,
    has_docker: Optional[bool] = None,
    output_format: str = "text"
):
    """Handle asset list action."""
    as_json = output_format == "json"

    if not as_json:
        print("\n📚 Recently Added Research Assets")

        # Show filter status
        filter_info = []
        if has_docker is True:
            filter_info.append("with Docker only")
        elif has_docker is False:
            filter_info.append("without Docker only")

        if categories:
            filter_info.append(f"categories: {', '.join(categories)}")

        if filter_info:
            print(f"   Filters: {', '.join(filter_info)}")

        print("=" * 80)
    
    try:
        results = list_assets(
            limit=limit, 
            offset=offset, 
            categories=categories,
            has_docker=has_docker
        )
        
        assets = results['assets']
        total = results['total']

        if as_json:
            # limit/offset/total ride along so a script can page without
            # re-deriving them from the request.
            print(json.dumps({
                "total": total,
                "limit": results.get("limit", limit),
                "offset": results.get("offset", offset),
                "assets": [a.to_dict() for a in assets],
            }, indent=2))
            return

        if not assets:
            print("\nNo assets found.")
            if has_docker is True:
                print("\n💡 Tip: Try without --docker flag to see all assets")
            return
        
        print(f"\nShowing {len(assets)} of {total} assets (offset: {offset}):\n")
        
        for i, asset in enumerate(assets, 1):
            idx = offset + i
            docker_icon = "🐳" if asset.has_docker else "📄"
            
            print(f"{idx}. {docker_icon} [{asset.arxiv_id}] {asset.title}")
            
            if asset.has_docker:
                print(f"    Docker: {asset.docker_image}")
            else:
                print(f"    Docker: Not available")
            
            if asset.github_url:
                print(f"    GitHub: {asset.github_url}")
            print()
        
        # Pagination hint
        if offset + len(assets) < total:
            print(f"💡 To see more: remyxai search list --limit {limit} --offset {offset + limit}")
        
        # Summary stats
        docker_count = sum(1 for a in assets if a.has_docker)
        if has_docker is None and docker_count > 0:
            print(f"\n📊 {docker_count} of {len(assets)} assets have Docker images")
            print("   Use --docker flag to see only containerized assets")
            
    except Exception as e:
        if as_json:
            print(f"List failed: {e}", file=sys.stderr)
            logger.error(f"List assets error: {e}", exc_info=True)
            sys.exit(1)
        print(f"❌ List failed: {e}")
        logger.error(f"List assets error: {e}", exc_info=True)


def handle_stats(output_format: str = "text"):
    """Handle asset stats action."""
    as_json = output_format == "json"

    if not as_json:
        print("\n📊 Remyx Research Assets Statistics")
        print("=" * 80)
    
    try:
        stats = get_stats()

        if as_json:
            # The endpoint's payload is already flat and JSON-able; pass it
            # through rather than reshaping, so new fields surface for free.
            print(json.dumps(stats, indent=2))
            return
        
        # The endpoint returns assets_*, not papers_* — reading the wrong keys
        # printed 0 for every count regardless of the catalog size (REMYX-286).
        total = stats.get('total_assets', 0)
        with_docker = stats.get('assets_with_docker', 0)
        without_docker = stats.get('assets_without_docker', 0)
        recent = stats.get('recent_additions', 0)
        
        print(f"\nTotal Assets: {total}")
        print(f"  With Docker: {with_docker} ({(with_docker/total*100):.1f}%)" if total > 0 else "  With Docker: 0")
        print(f"  Without Docker: {without_docker} ({(without_docker/total*100):.1f}%)" if total > 0 else "  Without Docker: 0")
        print(f"\nRecent Additions (7 days): {recent}")
        
        categories = stats.get('categories', {})
        if categories:
            print(f"\nTop Categories:")
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {cat}: {count}")
        
        print("\n💡 Tips:")
        print("  • Search for assets: remyxai search query 'your query'")
        print("  • Find containerized assets: remyxai search query 'query' --docker")
        print("  • Get asset details: remyxai search info <arxiv_id>")
        print()
        
    except Exception as e:
        if as_json:
            print(f"Stats failed: {e}", file=sys.stderr)
            logger.error(f"Stats error: {e}", exc_info=True)
            sys.exit(1)
        print(f"❌ Stats failed: {e}")
        logger.error(f"Stats error: {e}", exc_info=True)
