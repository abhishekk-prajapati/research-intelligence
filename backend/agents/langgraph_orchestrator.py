import os
from typing import Dict, Any, List, TypedDict, Optional
from backend.agents.fanout_agent import MultiSourceFanOutAgent
from backend.agents.triage_agent import TriageAgent
from backend.agents.extraction_agent import StructuredExtractionAgent
from backend.agents.citation_qa_agent import CitationGroundedQAAgent

# Try importing LangGraph; if not installed or during fallback, provide state graph orchestration wrapper
try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

class ResearchAgentState(TypedDict):
    task_type: str
    query: str
    question: str
    candidate_papers: List[Dict[str, Any]]
    results: Optional[List[Dict[str, Any]]]
    answer: Optional[str]
    citations: Optional[List[Dict[str, Any]]]
    comparison_matrix: Optional[List[Dict[str, Any]]]
    error: Optional[str]


class LangGraphResearchOrchestrator:
    """
    LangGraph Agent Orchestrator.
    Builds a robust, state-managed execution graph using LangGraph & LangChain components.
    Guarantees smooth execution with fail-safe boundaries so no pipeline stage gets stuck.
    """

    def __init__(self):
        if HAS_LANGGRAPH:
            self.graph = self._build_langgraph()
        else:
            self.graph = None

    def _build_langgraph(self):
        """Constructs the LangGraph StateGraph workflow."""
        builder = StateGraph(ResearchAgentState)

        # 1. Define Nodes
        builder.add_node("fanout_node", self._node_fanout)
        builder.add_node("triage_node", self._node_triage)
        builder.add_node("extraction_node", self._node_extraction)
        builder.add_node("citation_qa_node", self._node_citation_qa)

        # 2. Define Conditional Entry Router
        builder.set_conditional_entry_point(
            self._route_task,
            {
                "fanout": "fanout_node",
                "triage": "triage_node",
                "extraction": "extraction_node",
                "citation_qa": "citation_qa_node"
            }
        )

        # 3. Add Edges to END
        builder.add_edge("fanout_node", END)
        builder.add_edge("triage_node", END)
        builder.add_edge("extraction_node", END)
        builder.add_edge("citation_qa_node", END)

        return builder.compile()

    def _route_task(self, state: ResearchAgentState) -> str:
        task = state.get("task_type", "fanout")
        if task in ["fanout", "triage", "extraction", "citation_qa"]:
            return task
        return "fanout"

    # --- NODE FUNCTIONS ---
    def _node_fanout(self, state: ResearchAgentState) -> ResearchAgentState:
        query = state.get("query", "")
        papers = state.get("candidate_papers", [])
        try:
            results = MultiSourceFanOutAgent.execute_fanout_search(query, papers, limit_per_source=15)
            state["results"] = results
        except Exception as e:
            state["error"] = f"Fanout node error: {str(e)}"
            state["results"] = papers
        return state

    def _node_triage(self, state: ResearchAgentState) -> ResearchAgentState:
        query = state.get("query", "")
        papers = state.get("candidate_papers", [])
        try:
            results = TriageAgent.evaluate_papers(query, papers)
            state["results"] = results
        except Exception as e:
            state["error"] = f"Triage node error: {str(e)}"
            state["results"] = papers
        return state

    def _node_extraction(self, state: ResearchAgentState) -> ResearchAgentState:
        papers = state.get("candidate_papers", [])
        try:
            matrix = StructuredExtractionAgent.extract_paper_forms(papers)
            state["comparison_matrix"] = matrix
            state["results"] = matrix
        except Exception as e:
            state["error"] = f"Extraction node error: {str(e)}"
            state["comparison_matrix"] = []
        return state

    def _node_citation_qa(self, state: ResearchAgentState) -> ResearchAgentState:
        question = state.get("question") or state.get("query", "")
        papers = state.get("candidate_papers", [])
        try:
            qa_res = CitationGroundedQAAgent.answer_question(question, papers)
            state["answer"] = qa_res.get("answer", "")
            state["citations"] = qa_res.get("citations", [])
        except Exception as e:
            state["error"] = f"Citation QA node error: {str(e)}"
            state["answer"] = "An error occurred while synthesizing the answer."
            state["citations"] = []
        return state

    def run_agent_workflow(self, task_type: str, query: str = "", question: str = "", candidate_papers: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Public execution gateway for the LangGraph agent pipeline.
        Fail-safe wrapper ensuring no stage hangs or crashes.
        """
        initial_state: ResearchAgentState = {
            "task_type": task_type,
            "query": query,
            "question": question or query,
            "candidate_papers": candidate_papers or [],
            "results": None,
            "answer": None,
            "citations": None,
            "comparison_matrix": None,
            "error": None
        }

        try:
            if self.graph is not None:
                final_state = self.graph.invoke(initial_state)
                return final_state
            else:
                # Direct fallback execution if LangGraph graph is initializing
                if task_type == "fanout":
                    res = self._node_fanout(initial_state)
                elif task_type == "triage":
                    res = self._node_triage(initial_state)
                elif task_type == "extraction":
                    res = self._node_extraction(initial_state)
                elif task_type == "citation_qa":
                    res = self._node_citation_qa(initial_state)
                else:
                    res = initial_state
                return res
        except Exception as err:
            initial_state["error"] = f"Pipeline execution fallback triggered: {str(err)}"
            return initial_state


# Global orchestrator singleton instance
orchestrator = LangGraphResearchOrchestrator()
