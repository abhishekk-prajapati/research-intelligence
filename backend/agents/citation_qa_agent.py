import os
import re
from typing import List, Dict, Any

class CitationGroundedQAAgent:
    """
    Citation-Grounded Q&A Agent.
    Forces the AI to read candidate papers first, answer strictly from retrieved paper context,
    and attach exact inline citations [arXiv:ID] or [Paper: Title] for every claim made.
    """

    @classmethod
    def answer_question(cls, question: str, retrieved_papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesizes a citation-grounded answer based strictly on retrieved papers.
        """
        if not retrieved_papers:
            return {
                "question": question,
                "answer": "No relevant papers were found in the database to answer your question.",
                "citations": []
            }

        has_api_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))

        if has_api_key:
            return cls._answer_with_langchain(question, retrieved_papers)
        else:
            return cls._answer_heuristic(question, retrieved_papers)

    @classmethod
    def _answer_heuristic(cls, question: str, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Deterministic, rule-based citation generator for local offline mode.
        """
        citations = []
        snippets = []

        for p in papers[:5]:
            paper_id = p.get("id", "Unknown")
            title = p.get("title", "Untitled")
            abstract = p.get("abstract", "")
            pdf_link = p.get("pdf_link", "")

            citations.append({
                "id": paper_id,
                "title": title,
                "pdf_link": pdf_link
            })

            # Pick top sentences from abstract matching question terms
            q_terms = [t.lower() for t in question.split() if len(t) > 3]
            sentences = re.split(r'(?<=[.!?])\s+', abstract)
            relevant_sentences = []

            for sent in sentences:
                if any(term in sent.lower() for term in q_terms):
                    relevant_sentences.append(sent.strip())

            if not relevant_sentences and sentences:
                relevant_sentences = sentences[:2]

            summary_text = " ".join(relevant_sentences[:2])
            snippets.append(f"**[{title}]** (ID: `{paper_id}`):\n> \"{summary_text}\" [arXiv:{paper_id}]")

        answer = (
            f"Based strictly on the **{len(citations)} retrieved papers** in your research index:\n\n" +
            "\n\n".join(snippets) +
            "\n\n*Note: All statements above are directly grounded in the retrieved paper database without external memory assumptions.*"
        )

        return {
            "question": question,
            "answer": answer,
            "citations": citations
        }

    @classmethod
    def _answer_with_langchain(cls, question: str, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            if os.getenv("OPENAI_API_KEY"):
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            else:
                from langchain_google_genai import ChatGoogleGenerativeAI
                google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=google_key)

            from langchain_core.prompts import ChatPromptTemplate

            context_blocks = []
            citations = []

            for p in papers[:6]:
                pid = p.get("id", "")
                title = p.get("title", "")
                abstract = p.get("abstract", "")
                pdf_link = p.get("pdf_link", "")

                context_blocks.append(f"Paper ID: {pid}\nTitle: {title}\nAbstract: {abstract}")
                citations.append({
                    "id": pid,
                    "title": title,
                    "pdf_link": pdf_link
                })

            context_str = "\n---\n".join(context_blocks)

            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a Citation-Grounded Research AI. Your task is to answer user questions strictly using ONLY the provided paper contexts.
Rules:
1. Do NOT guess or use outside knowledge.
2. If the context does not contain the answer, say "Based on the provided papers, this information is not mentioned."
3. Every claim MUST be followed by an exact citation in brackets like [Paper ID: <id>] or [arXiv:<id>]."""),
                ("user", "Context Papers:\n{context}\n\nQuestion: {question}")
            ])

            chain = prompt | llm
            res = chain.invoke({"context": context_str, "question": question})

            return {
                "question": question,
                "answer": res.content,
                "citations": citations
            }
        except Exception as e:
            print(f"LangChain citation QA invocation fallback: {e}")
            return cls._answer_heuristic(question, papers)
