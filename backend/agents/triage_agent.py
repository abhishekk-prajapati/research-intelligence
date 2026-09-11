import os
from typing import List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class PaperTriageScore(BaseModel):
    paper_id: str
    relevance_score: float = Field(..., description="Relevance to query from 1 to 10")
    recency_score: float = Field(..., description="Recency score from 1 to 10")
    method_solidity_score: float = Field(..., description="Methodological solidity from 1 to 10")
    overall_score: float = Field(..., description="Weighted average score")
    rationale: str = Field(..., description="Short explanation of score")

class TriageAgent:
    """
    Triage / Relevance-Ranking Agent.
    Evaluates search results on query relevance, recency, and methodological validity,
    re-ranking the candidates so users see high-signal papers at the top.
    """

    @classmethod
    def evaluate_papers(cls, query: str, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluates candidate papers using LLM / heuristic scoring pipeline and sorts them by triage score.
        """
        scored_papers = []

        # Check if LangChain OpenAI / Gemini API key is configured
        has_api_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

        if has_api_key:
            scored_papers = cls._evaluate_with_langchain(query, papers)
        else:
            # Smart analytical fallback engine when API keys are not provided
            scored_papers = cls._evaluate_heuristic(query, papers)

        # Sort by overall triage score descending
        scored_papers.sort(key=lambda x: x.get("triage", {}).get("overall_score", 0), reverse=True)
        return scored_papers

    @classmethod
    def _evaluate_heuristic(cls, query: str, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        query_terms = set(query.lower().split())
        current_year = datetime.now().year
        results = []

        for p in papers:
            title = p.get("title", "")
            abstract = p.get("abstract", "")
            text = f"{title} {abstract}".lower()

            # 1. Relevance Score based on term overlap
            matched_terms = [t for t in query_terms if t in text]
            rel_ratio = len(matched_terms) / max(len(query_terms), 1)
            relevance = min(10.0, max(2.0, rel_ratio * 10.0 + (3.0 if title.lower() in query.lower() else 0.0)))

            # 2. Recency Score
            pub_date = p.get("published_date")
            pub_year = current_year
            if isinstance(pub_date, datetime):
                pub_year = pub_date.year
            elif isinstance(pub_date, str):
                try:
                    pub_year = datetime.fromisoformat(pub_date).year
                except ValueError:
                    pub_year = current_year - 3
            
            years_old = max(0, current_year - pub_year)
            recency = max(3.0, 10.0 - (years_old * 1.5))

            # 3. Method Solidity (Detection of empirical baseline markers)
            solidity = 6.0
            solidity_keywords = ["benchmark", "experiment", "dataset", "ablation", "state-of-the-art", "sota", "evaluation", "framework"]
            matches = [k for k in solidity_keywords if k in text]
            solidity = min(10.0, 5.0 + len(matches) * 1.0)

            # Overall Weighted Composite Score
            overall = round((relevance * 0.5) + (recency * 0.25) + (solidity * 0.25), 2)

            item = dict(p)
            item["triage"] = {
                "relevance_score": round(relevance, 1),
                "recency_score": round(recency, 1),
                "method_solidity_score": round(solidity, 1),
                "overall_score": overall,
                "rationale": f"Matched terms ({', '.join(matched_terms[:3]) if matched_terms else 'semantic similarity'}); published {pub_year}; includes key method signals ({', '.join(matches[:2]) if matches else 'standard presentation'})."
            }
            results.append(item)
        return results

    @classmethod
    def _evaluate_with_langchain(cls, query: str, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # If API key available, instantiate LangChain ChatOpenAI / ChatGoogleGenerativeAI
        try:
            if os.getenv("OPENAI_API_KEY"):
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            else:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
            
            from langchain_core.prompts import ChatPromptTemplate
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert AI research triage assistant. Evaluate papers based on query relevance, recency, and methodological rigor."),
                ("user", "Query: {query}\nPaper Title: {title}\nAbstract: {abstract}\nProvide JSON with relevance_score (1-10), recency_score (1-10), method_solidity_score (1-10), overall_score (1-10), and rationale.")
            ])

            structured_llm = llm.with_structured_output(PaperTriageScore)
            chain = prompt | structured_llm

            results = []
            for p in papers:
                try:
                    score = chain.invoke({
                        "query": query,
                        "title": p.get("title", ""),
                        "abstract": p.get("abstract", "")
                    })
                    item = dict(p)
                    item["triage"] = score.dict()
                    results.append(item)
                except Exception:
                    # Fallback to heuristic for single paper failure
                    sub = cls._evaluate_heuristic(query, [p])
                    results.extend(sub)
            return results
        except Exception as e:
            print(f"LangChain triage invocation fallback: {e}")
            return cls._evaluate_heuristic(query, papers)
