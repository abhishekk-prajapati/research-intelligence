import re
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from backend.arxiv_client import ArxivClient
from backend.semantic_scholar_client import SemanticScholarClient

class MultiSourceFanOutAgent:
    """
    Multi-Source Fan-Out Agent.
    Reaches out live to multiple paper databases (local database, arXiv, Semantic Scholar),
    combines all search results concurrently, and deduplicates based on paper title and IDs.
    """

    @classmethod
    def _normalize_title(cls, title: str) -> str:
        """Strip punctuation and whitespace for clean title matching."""
        return re.sub(r'[^a-zA-Z0-9]', '', title.lower())

    @classmethod
    def execute_fanout_search(cls, query: str, local_papers: List[Dict[str, Any]], limit_per_source: int = 15) -> List[Dict[str, Any]]:
        """
        Executes parallel multi-source search across local DB, arXiv API, and Semantic Scholar API.
        """
        def fetch_arxiv():
            results = ArxivClient.search_papers(query, limit=limit_per_source)
            for item in results:
                item["source"] = "arXiv"
            return results

        def fetch_semantic_scholar():
            results = SemanticScholarClient.search_papers(query, limit=limit_per_source)
            return results

        # Run external requests concurrently using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_arxiv = executor.submit(fetch_arxiv)
            future_s2 = executor.submit(fetch_semantic_scholar)

            arxiv_results = future_arxiv.result()
            s2_results = future_s2.result()

        # Mark local DB papers
        local_results = []
        for p in local_papers:
            item = dict(p)
            item["source"] = "Local Index"
            local_results.append(item)

        # Merge and deduplicate
        combined = local_results + arxiv_results + s2_results
        deduped = []
        seen_titles = set()
        seen_ids = set()

        for paper in combined:
            paper_id = paper.get("id")
            raw_title = paper.get("title", "")
            norm_title = cls._normalize_title(raw_title)

            if not norm_title or norm_title in seen_titles or paper_id in seen_ids:
                continue

            seen_titles.add(norm_title)
            if paper_id:
                seen_ids.add(paper_id)

            deduped.append(paper)

        return deduped
