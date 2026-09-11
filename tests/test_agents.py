import unittest
from datetime import datetime
from backend.agents.fanout_agent import MultiSourceFanOutAgent
from backend.agents.triage_agent import TriageAgent
from backend.agents.extraction_agent import StructuredExtractionAgent
from backend.agents.citation_qa_agent import CitationGroundedQAAgent

class TestAIAgents(unittest.TestCase):

    def setUp(self):
        self.sample_papers = [
            {
                "id": "1706.03762",
                "title": "Attention Is All You Need",
                "abstract": "We propose the Transformer, a novel neural network architecture based on self-attention mechanisms. Evaluated on WMT 2014 English-to-German translation dataset achieving 28.4 BLEU score. Outperforms recurrence and convolution models with faster training time. Stated limitation is high memory footprint for long sequence lengths.",
                "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar",
                "published_date": datetime(2017, 6, 12),
                "primary_category": "cs.CL",
                "pdf_link": "https://arxiv.org/pdf/1706.03762.pdf"
            },
            {
                "id": "2005.14165",
                "title": "Language Models are Few-Shot Learners",
                "abstract": "We demonstrate that scaling language models achieves strong few-shot performance. Evaluated on GLUE and SuperGLUE benchmarks. Uses 175B parameter autoregressive Transformer architecture. Results show SOTA performance without fine-tuning. Limitation includes high inference cost and memory requirements.",
                "authors": "Tom B. Brown, Benjamin Mann, Nick Ryder",
                "published_date": datetime(2020, 5, 28),
                "primary_category": "cs.CL",
                "pdf_link": "https://arxiv.org/pdf/2005.14165.pdf"
            }
        ]

    def test_multi_source_fanout_agent(self):
        """Test fanout search deduplication and merging."""
        results = MultiSourceFanOutAgent.execute_fanout_search("Transformer attention", self.sample_papers, limit_per_source=5)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Check source tags exist
        sources = [p.get("source") for p in results]
        self.assertIn("Local Index", sources)

    def test_triage_agent(self):
        """Test triage scoring criteria: relevance, recency, solidity."""
        scored = TriageAgent.evaluate_papers("Transformer self-attention architecture", self.sample_papers)
        self.assertEqual(len(scored), len(self.sample_papers))
        
        first = scored[0]
        self.assertIn("triage", first)
        t = first["triage"]
        self.assertGreaterEqual(t["relevance_score"], 1.0)
        self.assertGreaterEqual(t["recency_score"], 1.0)
        self.assertGreaterEqual(t["method_solidity_score"], 1.0)
        self.assertGreaterEqual(t["overall_score"], 1.0)
        self.assertIsInstance(t["rationale"], str)

    def test_structured_extraction_agent(self):
        """Test structured schema extraction into dataset, method, result, limitation."""
        extracted = StructuredExtractionAgent.extract_paper_forms(self.sample_papers)
        self.assertEqual(len(extracted), len(self.sample_papers))
        
        first = extracted[0]
        self.assertIn("dataset", first)
        self.assertIn("method", first)
        self.assertIn("result", first)
        self.assertIn("limitation", first)
        self.assertTrue(len(first["dataset"]) > 0)
        self.assertTrue(len(first["method"]) > 0)

    def test_citation_qa_agent(self):
        """Test grounded Q&A generation with inline citations."""
        res = CitationGroundedQAAgent.answer_question("What architecture do these papers use?", self.sample_papers)
        self.assertIn("answer", res)
        self.assertIn("citations", res)
        self.assertGreaterEqual(len(res["citations"]), 1)
        self.assertIn("[arXiv:", res["answer"])

    def test_langgraph_orchestrator(self):
        """Test LangGraph orchestrator state graph workflow."""
        from backend.agents.langgraph_orchestrator import orchestrator
        res = orchestrator.run_agent_workflow("fanout", query="attention mechanism", candidate_papers=self.sample_papers)
        self.assertIsNotNone(res.get("results"))
        self.assertIsNone(res.get("error"))

        res_qa = orchestrator.run_agent_workflow("citation_qa", question="What benchmarks are used?", candidate_papers=self.sample_papers)
        self.assertIsNotNone(res_qa.get("answer"))

if __name__ == "__main__":
    unittest.main()
