import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import re

class ArxivClient:
    BASE_URL = "http://export.arxiv.org/api/query"
    
    # Atom namespaces used by arXiv XML feed
    NAMESPACES = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }

    @classmethod
    def search_papers(cls, query: str, limit: int = 30) -> list:
        """
        Query arXiv and return structured metadata.
        """
        params = {
            "search_query": f'all:"{query}"',
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        try:
            response = requests.get(cls.BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            return cls._parse_xml(response.text)
        except Exception as e:
            print(f"Error querying arXiv for '{query}': {e}")
            return []

    @classmethod
    def _parse_xml(cls, xml_text: str) -> list:
        papers = []
        try:
            root = ET.fromstring(xml_text)
            entries = root.findall('atom:entry', cls.NAMESPACES)
            
            for entry in entries:
                # 1. Parse ID and extract the raw arXiv code
                id_uri = entry.find('atom:id', cls.NAMESPACES)
                id_uri_text = id_uri.text.strip() if id_uri is not None else ""
                
                # Extract arXiv code (e.g. 1706.03762v7 -> 1706.03762)
                id_match = re.search(r'/abs/([^v/]+)', id_uri_text)
                paper_id = id_match.group(1) if id_match else id_uri_text.split('/')[-1]
                
                # 2. Extract Title and Abstract (cleaning whitespace/newlines)
                title = entry.find('atom:title', cls.NAMESPACES)
                title_text = re.sub(r'\s+', ' ', title.text).strip() if title is not None else "Untitled"
                
                summary = entry.find('atom:summary', cls.NAMESPACES)
                abstract_text = re.sub(r'\s+', ' ', summary.text).strip() if summary is not None else ""
                
                # 3. Extract Published Date
                published = entry.find('atom:published', cls.NAMESPACES)
                published_date = datetime.utcnow()
                if published is not None:
                    try:
                        # Handle standard ISO date e.g. 2017-06-12T17:34:00Z
                        clean_date = published.text.strip().replace('Z', '')
                        published_date = datetime.fromisoformat(clean_date)
                    except ValueError:
                        pass
                
                # 4. Extract Authors
                author_elements = entry.findall('atom:author', cls.NAMESPACES)
                authors = []
                for auth_el in author_elements:
                    name_el = auth_el.find('atom:name', cls.NAMESPACES)
                    if name_el is not None:
                        authors.append(name_el.text.strip())
                
                # 5. Extract Categories
                primary_cat_el = entry.find('arxiv:primary_category', cls.NAMESPACES)
                primary_category = primary_cat_el.attrib.get('term', '') if primary_cat_el is not None else ""
                
                category_els = entry.findall('atom:category', cls.NAMESPACES)
                categories = []
                for cat_el in category_els:
                    term = cat_el.attrib.get('term', '')
                    if term:
                        categories.append(term)
                
                if not primary_category and categories:
                    primary_category = categories[0]

                # 6. Extract PDF Link
                pdf_link = f"https://arxiv.org/pdf/{paper_id}.pdf"
                link_els = entry.findall('atom:link', cls.NAMESPACES)
                for link in link_els:
                    if link.attrib.get('title') == 'pdf' or link.attrib.get('type') == 'application/pdf':
                        pdf_link = link.attrib.get('href', pdf_link)
                        break

                papers.append({
                    "id": paper_id,
                    "title": title_text,
                    "abstract": abstract_text,
                    "authors": ", ".join(authors),
                    "published_date": published_date,
                    "primary_category": primary_category,
                    "categories": ", ".join(categories),
                    "pdf_link": pdf_link
                })
        except Exception as e:
            print(f"Failed parsing XML tree: {e}")
            
        return papers
