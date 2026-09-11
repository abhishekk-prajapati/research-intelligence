import requests
from datetime import datetime

class SemanticScholarClient:
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    @classmethod
    def search_papers(cls, query: str, limit: int = 20) -> list:
        """
        Query Semantic Scholar REST API and return structured paper metadata.
        """
        params = {
            "query": query,
            "limit": limit,
            "fields": "paperId,title,abstract,authors,year,publicationDate,externalIds,openAccessPdf"
        }
        headers = {
            "User-Agent": "ResearchIntelligencePlatform/1.0"
        }
        
        try:
            response = requests.get(cls.BASE_URL, params=params, headers=headers, timeout=12)
            if response.status_code != 200:
                print(f"Semantic Scholar API returned status code {response.status_code}")
                return []
            
            data = response.json()
            papers_raw = data.get("data", [])
            return cls._parse_results(papers_raw)
        except Exception as e:
            print(f"Error querying Semantic Scholar API for '{query}': {e}")
            return []

    @classmethod
    def _parse_results(cls, items: list) -> list:
        results = []
        for item in items:
            paper_id = item.get("paperId") or item.get("externalIds", {}).get("ArXiv", "")
            title = item.get("title", "Untitled")
            abstract = item.get("abstract") or "No abstract provided."
            
            # Format authors list
            authors_data = item.get("authors", [])
            authors_str = ", ".join([a.get("name", "") for a in authors_data if a.get("name")]) or "Unknown Authors"
            
            # Parse publication date
            pub_date_str = item.get("publicationDate")
            pub_date = datetime.utcnow()
            if pub_date_str:
                try:
                    pub_date = datetime.fromisoformat(pub_date_str)
                except ValueError:
                    pass
            elif item.get("year"):
                try:
                    pub_date = datetime(int(item.get("year")), 1, 1)
                except ValueError:
                    pass

            # PDF link
            pdf_info = item.get("openAccessPdf") or {}
            pdf_link = pdf_info.get("url", f"https://www.semanticscholar.org/paper/{paper_id}")

            results.append({
                "id": f"s2:{paper_id[:12]}" if paper_id else f"s2:{hash(title)}",
                "title": title,
                "abstract": abstract,
                "authors": authors_str,
                "published_date": pub_date,
                "primary_category": "cs.AI",
                "categories": "cs.AI, Semantic Scholar",
                "pdf_link": pdf_link,
                "source": "Semantic Scholar"
            })
        return results
