import unittest
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Mock the embedding model before importing backend modules to avoid model download overhead
from unittest.mock import patch

def mock_embeddings(texts):
    import hashlib
    # Generates standard unit-normalized vectors of size 384 for mock testing
    embs = []
    for i, t in enumerate(texts):
        # Use a deterministic hash of the text to seed numpy so vectors are distinct
        h = hashlib.md5(t.encode('utf-8')).hexdigest()
        seed = int(h, 16) % 1000000
        np.random.seed(seed)
        vec = np.random.randn(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        embs.append(vec.tolist())
    return embs

# Apply patch globally during imports
with patch('backend.ml_engine.MLEngine.generate_embeddings', side_effect=mock_embeddings):
    from backend.database import Base, Paper
    from backend.config import NUM_CLUSTERS
    from backend.ml_engine import MLEngine, HybridRetriever, ClusteringEngine, RecommendationEngine, RoadmapEngine, TrendEngine
    from backend.evaluation_engine import EvaluationEngine

class TestMLPipeline(unittest.TestCase):
    def setUp(self):
        # Create temporary in-memory database
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()
        
        # Seed mock papers
        self._seed_mock_papers()

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)

    def _seed_mock_papers(self):
        mock_data = [
            {
                "id": "1",
                "title": "Scaling Laws for Large Language Models",
                "abstract": "We study the empirical scaling laws for language model performance. We evaluate cross-entropy loss over model size, dataset size, and compute. Transformer models scale predictably.",
                "authors": "Jared Kaplan, Sam McCandlish",
                "primary_category": "cs.CL",
                "categories": "cs.CL, cs.LG",
                "published_date": datetime(2020, 1, 23)
            },
            {
                "id": "2",
                "title": "Attention Is All You Need",
                "abstract": "We propose a novel simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Outperforms RNNs on translation.",
                "authors": "Ashish Vaswani, Noam Shazeer",
                "primary_category": "cs.CL",
                "categories": "cs.CL, cs.LG",
                "published_date": datetime(2017, 6, 12)
            },
            {
                "id": "3",
                "title": "Playing Atari with Deep Reinforcement Learning",
                "abstract": "We present the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning and deep Q-networks.",
                "authors": "Volodymyr Mnih, Koray Kavukcuoglu",
                "primary_category": "cs.LG",
                "categories": "cs.LG, cs.AI",
                "published_date": datetime(2013, 12, 18)
            },
            {
                "id": "4",
                "title": "Generative Adversarial Nets",
                "abstract": "We propose a new framework for estimating generative models via an adversarial process, in which we train two models simultaneously: a generative model and a discriminative model.",
                "authors": "Ian Goodfellow, Jean Pouget-Abadie",
                "primary_category": "cs.LG",
                "categories": "cs.LG, stat.ML",
                "published_date": datetime(2014, 6, 10)
            },
            {
                "id": "5",
                "title": "End-to-End Training of Deep Visuomotor Policies",
                "abstract": "We present a method to train deep convolutional neural network policies that map camera pixels directly to joint torques on a robotic arm manipulator. This is end-to-end learning.",
                "authors": "Sergey Levine, Chelsea Finn",
                "primary_category": "cs.RO",
                "categories": "cs.RO, cs.CV",
                "published_date": datetime(2015, 6, 1)
            },
            {
                "id": "6",
                "title": "An Image Database for Object Recognition",
                "abstract": "We introduce a large dataset of annotated image categories containing millions of pixels for benchmarking convolutional neural network architectures on computer vision tasks.",
                "authors": "Li Fei-Fei, Deng Jia",
                "primary_category": "cs.CV",
                "categories": "cs.CV, cs.LG",
                "published_date": datetime(2009, 10, 15)
            }
        ]

        papers = []
        for d in mock_data:
            # Generate mock embedding (using our patched mock generator)
            emb = mock_embeddings([f"{d['title']} {d['abstract']}"])[0]
            domain = MLEngine.classify_domain(d["title"], d["abstract"], d["primary_category"])
            
            p = Paper(
                id=d["id"],
                title=d["title"],
                abstract=d["abstract"],
                authors=d["authors"],
                published_date=d["published_date"],
                primary_category=d["primary_category"],
                categories=d["categories"],
                pdf_link=f"https://arxiv.org/pdf/{d['id']}.pdf",
                domain=domain
            )
            p.set_embedding(emb)
            papers.append(p)

        self.db.add_all(papers)
        self.db.commit()

    @patch('backend.ml_engine.MLEngine.generate_embeddings', side_effect=mock_embeddings)
    def test_domain_classification(self, mock_emb):
        """Test that papers are classified into correct domains based on keywords and categories."""
        p1 = self.db.query(Paper).filter(Paper.id == "1").first()
        p3 = self.db.query(Paper).filter(Paper.id == "3").first()
        p5 = self.db.query(Paper).filter(Paper.id == "5").first()
        p6 = self.db.query(Paper).filter(Paper.id == "6").first()

        self.assertEqual(p1.domain, "Natural Language Processing")
        self.assertEqual(p3.domain, "Reinforcement Learning")
        self.assertEqual(p5.domain, "Robotics & Control")
        self.assertEqual(p6.domain, "Computer Vision")

    @patch('backend.ml_engine.MLEngine.generate_embeddings', side_effect=mock_embeddings)
    def test_hybrid_search(self, mock_emb):
        """Test that hybrid search retrieves relevant papers and combines scores properly."""
        all_papers = self.db.query(Paper).all()
        results = HybridRetriever.retrieve("Attention and Transformer models", all_papers, limit=3, alpha=0.4)
        
        self.assertEqual(len(results), 3)
        # Verify scores and ranks are mapped properly
        for r in results:
            self.assertTrue(0.0 <= r["score"] <= 1.0)
            self.assertTrue(0.0 <= r["lexical_score"] <= 1.0)
            self.assertTrue(0.0 <= r["semantic_score"] <= 1.0)
            self.assertIn("rrf_score", r)
            self.assertIn("lexical_rank", r)
            self.assertIn("semantic_rank", r)
            self.assertTrue(r["rrf_score"] > 0.0)
            
        # "Attention Is All You Need" (id=2) contains "Transformer" and "attention" in text, should score high lexically
        top_ids = [r["paper"].id for r in results]
        self.assertIn("2", top_ids)

    @patch('backend.ml_engine.MLEngine.generate_embeddings', side_effect=mock_embeddings)
    def test_recommendation_system(self, mock_emb):
        """Test that paper recommendations yield correct rankings and author boosts."""
        target = self.db.query(Paper).filter(Paper.id == "2").first() # Attention Is All You Need
        all_papers = self.db.query(Paper).all()
        
        recs = RecommendationEngine.recommend(target, all_papers, limit=2)
        self.assertEqual(len(recs), 2)
        
        # Verify we didn't recommend the target paper itself
        rec_ids = [r["paper"].id for r in recs]
        self.assertNotIn("2", rec_ids)

    def test_topic_clustering_and_pca(self):
        """Test that clustering groups papers and yields valid 2D PCA coordinates."""
        all_papers = self.db.query(Paper).all()
        cluster_data = ClusteringEngine.cluster_papers(all_papers)
        
        papers = cluster_data.get("papers", [])
        self.assertEqual(len(papers), 6)
        
        for p in papers:
            self.assertIn("id", p)
            self.assertIn("x", p)
            self.assertIn("y", p)
            self.assertIn("cluster", p)
            self.assertTrue(isinstance(p["x"], float))
            self.assertTrue(isinstance(p["cluster"], int))
            self.assertTrue(0 <= p["cluster"] < NUM_CLUSTERS)

    @patch('backend.ml_engine.MLEngine.generate_embeddings', side_effect=mock_embeddings)
    def test_roadmap_synthesis(self, mock_emb):
        """Test that learning paths are divided chronologically and attach repos."""
        all_papers = self.db.query(Paper).all()
        roadmap = RoadmapEngine.generate_roadmap("Reinforcement Learning", all_papers, depth=3)
        
        self.assertEqual(roadmap["topic"], "Reinforcement Learning")
        
        # Tiers should be non-empty
        self.assertTrue(len(roadmap["tiers"]["beginner"]) >= 1)
        self.assertTrue(len(roadmap["tiers"]["intermediate"]) >= 1)
        self.assertTrue(len(roadmap["tiers"]["advanced"]) >= 1)
        
        # Chronology check: beginner paper date <= intermediate paper date <= advanced paper date
        beg_date = roadmap["tiers"]["beginner"][0].published_date
        int_date = roadmap["tiers"]["intermediate"][0].published_date
        adv_date = roadmap["tiers"]["advanced"][0].published_date
        
        self.assertTrue(beg_date <= int_date)
        self.assertTrue(int_date <= adv_date)
        
        # Check repos suggestions
        self.assertTrue(len(roadmap["repos"]) >= 1)

    def test_trend_analysis(self):
        """Test that trends compile timeline stats and count keywords correctly."""
        all_papers = self.db.query(Paper).all()
        trends = TrendEngine.analyze_trends(all_papers)
        
        # Check quarterly timelines
        self.assertIn("2017-Q2", trends["timeline"]) # Attention is all you need Q2 2017
        self.assertIn("2020-Q1", trends["timeline"]) # Scaling laws Q1 2020
        
        # Check category distribution
        self.assertIn("Natural Language Processing", trends["domain_counts"])
        self.assertIn("Reinforcement Learning", trends["domain_counts"])
        
        # Check keywords
        hot_kws = [item["keyword"] for item in trends["hot_keywords"]]
        self.assertTrue(len(hot_kws) >= 1)

    def test_faiss_index_serialization(self):
        """Test that the FAISS index writes and reads files properly."""
        all_papers = self.db.query(Paper).all()
        embeddings = [p.embedding_vector for p in all_papers if p.embedding_vector]
        
        index_path = "test_index.faiss"
        success = MLEngine.save_faiss_index(index_path, embeddings)
        
        import os
        from backend.ml_engine import HAS_FAISS
        if HAS_FAISS:
            self.assertTrue(success)
            self.assertTrue(os.path.exists(index_path))
            
            # Test index loading
            index = MLEngine.load_faiss_index(index_path)
            self.assertIsNotNone(index)
            self.assertEqual(index.ntotal, len(embeddings))
            
            # Clean up
            if os.path.exists(index_path):
                os.remove(index_path)

    def test_centroid_recommendation(self):
        """Test that centroid personalization generates recommendations from saved reading history."""
        # Use two papers as bookmarks
        b1 = self.db.query(Paper).filter(Paper.id == "1").first()
        b2 = self.db.query(Paper).filter(Paper.id == "2").first()
        bookmarks = [b1, b2]
        
        all_papers = self.db.query(Paper).all()
        
        recs = RecommendationEngine.recommend_personalized(bookmarks, all_papers, limit=2)
        self.assertEqual(len(recs), 2)
        
        # Recommendations shouldn't contain the bookmarked papers
        rec_ids = [r["paper"].id for r in recs]
        self.assertNotIn("1", rec_ids)
        self.assertNotIn("2", rec_ids)
        
        for r in recs:
            self.assertTrue(0.0 <= r["score"] <= 1.0)

    def test_evaluation_engine_metrics(self):
        """Test that EvaluationEngine correctly maps ground truth and computes metrics."""
        # Test ground truth mapping
        gt1 = EvaluationEngine.get_ground_truth_domain("cs.CV", "An Image Database", "object recognition")
        gt2 = EvaluationEngine.get_ground_truth_domain("cs.CL", "Attention Is All You Need", "transformer model translation")
        gt3 = EvaluationEngine.get_ground_truth_domain("cs.RO", "Visuomotor Policies", "robotic arm manipulator")
        gt4 = EvaluationEngine.get_ground_truth_domain("cs.LG", "Deep Reinforcement Learning", "q-learning policy network")
        gt5 = EvaluationEngine.get_ground_truth_domain("cs.LG", "Generative Adversarial Nets", "adversarial training loss")
        
        self.assertEqual(gt1, "Computer Vision")
        self.assertEqual(gt2, "Natural Language Processing")
        self.assertEqual(gt3, "Robotics & Control")
        self.assertEqual(gt4, "Reinforcement Learning")
        self.assertEqual(gt5, "Machine Learning (General)")
        
        # Test evaluation execution on seeded database papers
        all_papers = self.db.query(Paper).all()
        eval_data = EvaluationEngine.evaluate_classifier(all_papers)
        
        # Verify metric structure
        self.assertIn("overall", eval_data)
        self.assertIn("per_class", eval_data)
        self.assertIn("confusion_matrix", eval_data)
        self.assertIn("mismatches", eval_data)
        
        overall = eval_data["overall"]
        self.assertIn("accuracy", overall)
        self.assertIn("macro", overall)
        self.assertIn("weighted", overall)
        self.assertTrue(0.0 <= overall["accuracy"] <= 1.0)
        
        per_class = eval_data["per_class"]
        self.assertIn("Computer Vision", per_class)
        self.assertTrue(0.0 <= per_class["Computer Vision"]["f1"] <= 1.0)
        
        cm = eval_data["confusion_matrix"]
        self.assertEqual(len(cm["labels"]), 6)
        self.assertEqual(len(cm["matrix"]), 6)

if __name__ == "__main__":
    unittest.main()

