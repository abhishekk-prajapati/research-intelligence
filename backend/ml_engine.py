import numpy as np
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from backend.config import EMBEDDING_MODEL_NAME, NUM_CLUSTERS
from backend.database import Paper

# Check if FAISS is available, otherwise fall back to pure numpy matrix calculations
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

class MLEngine:
    _model = None

    @classmethod
    def get_embedding_model(cls):
        """Lazy load SentenceTransformers model to reduce startup time."""
        if cls._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading SentenceTransformer: {EMBEDDING_MODEL_NAME}...")
            cls._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return cls._model

    @classmethod
    def generate_embeddings(cls, texts: list) -> list:
        """Generate dense embedding vectors for a list of texts."""
        if not texts:
            return []
        model = cls.get_embedding_model()
        embeddings = model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    @classmethod
    def save_faiss_index(cls, index_path: str, embeddings: list) -> bool:
        """Serializes and saves a FAISS vector index to a file on disk."""
        if not HAS_FAISS or not embeddings:
            return False
        try:
            emb_arr = np.array(embeddings, dtype=np.float32)
            # Normalize vectors for Cosine Similarity (Inner Product)
            norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0 # avoid divide by zero
            emb_arr = emb_arr / norms
            
            d = emb_arr.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(emb_arr)
            faiss.write_index(index, index_path)
            print(f"Successfully serialized and saved FAISS index to: {index_path}")
            return True
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
            return False

    @classmethod
    def load_faiss_index(cls, index_path: str):
        """Loads a FAISS index binary file from disk."""
        if not HAS_FAISS:
            return None
        if not os.path.exists(index_path):
            return None
        try:
            index = faiss.read_index(index_path)
            print(f"Successfully loaded serialized FAISS index from: {index_path}")
            return index
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return None

    @classmethod
    def classify_domain(cls, title: str, abstract: str, primary_cat: str) -> str:
        """
        Extends the "Research Domain Classifier" rules to group papers
        into core AI fields.
        """
        text = f"{title} {abstract}".lower()
        cat = primary_cat.lower()

        # Rules based on arXiv categories
        if any(c in cat for c in ["cs.cv", "cs.gr"]):
            return "Computer Vision"
        elif any(c in cat for c in ["cs.cl", "stat.ml"]) and any(w in text for w in ["llm", "language", "nlp", "translation"]):
            return "Natural Language Processing"
        elif any(c in cat for c in ["cs.lg", "stat.ml"]):
            if any(w in text for w in ["reinforcement", "rl", "policy", "q-learning", "dqn", "agent"]):
                return "Reinforcement Learning"
            return "Machine Learning (General)"
        elif any(c in cat for c in ["cs.ro", "cs.sy"]):
            return "Robotics & Control"
        elif "quant-ph" in cat:
            return "Quantum Computing"
        
        # Text heuristic fallback if categories don't match
        if any(w in text for w in ["image", "object detection", "segmentation", "pixels", "vision"]):
            return "Computer Vision"
        elif any(w in text for w in ["text", "natural language", "translation", "llm", "gpt", "token"]):
            return "Natural Language Processing"
        elif any(w in text for w in ["reinforcement", "policy", "markov", "q-learning", "reward"]):
            return "Reinforcement Learning"
        elif any(w in text for w in ["robot", "manipulator", "kinematics", "uav"]):
            return "Robotics & Control"
            
        return "Machine Learning (General)"


class HybridRetriever:
    @staticmethod
    def retrieve(query: str, db_papers: list, limit: int = 10, k: int = 60, alpha: float = 0.4) -> list:
        """
        Retrieves papers using Reciprocal Rank Fusion (RRF) between lexical rank and semantic rank.
        RRF_Score(d) = 1 / (k + rank_lexical(d)) + 1 / (k + rank_semantic(d))
        Also returns individual scores and ranks for diagnostics.
        """
        if not db_papers:
            return []

        # 1. Compute Lexical Scores using scikit-learn's TF-IDF Vectorizer
        documents = [f"{p.title} {p.abstract}" for p in db_papers]
        vectorizer = TfidfVectorizer(stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform(documents)
            query_tfidf = vectorizer.transform([query])
            lexical_scores = (tfidf_matrix * query_tfidf.T).toarray().flatten()
        except Exception as e:
            print(f"Error computing lexical scores: {e}")
            lexical_scores = np.zeros(len(db_papers))

        # 2. Compute Semantic Scores (Vector search)
        query_emb = MLEngine.generate_embeddings([query])[0]
        semantic_scores = np.zeros(len(db_papers))
        
        # Fetch valid embeddings
        papers_with_embs = []
        valid_indices = []
        for i, p in enumerate(db_papers):
            vec = p.embedding_vector
            if vec:
                papers_with_embs.append(vec)
                valid_indices.append(i)

        if papers_with_embs:
            papers_with_embs = np.array(papers_with_embs, dtype=np.float32)
            query_emb = np.array(query_emb, dtype=np.float32).reshape(1, -1)

            # Normalise vectors for Cosine Similarity
            papers_with_embs = papers_with_embs / np.linalg.norm(papers_with_embs, axis=1, keepdims=True)
            query_emb = query_emb / np.linalg.norm(query_emb)

            # Attempt loading serialized FAISS index from disk for search query, fallback to numpy
            index_path = "research_platform.faiss"
            faiss_index = MLEngine.load_faiss_index(index_path)

            if HAS_FAISS and faiss_index is not None and faiss_index.ntotal == len(papers_with_embs):
                scores, indices = faiss_index.search(query_emb, len(papers_with_embs))
                for score, idx in zip(scores[0], indices[0]):
                    real_idx = valid_indices[idx]
                    semantic_scores[real_idx] = float(score)
            elif HAS_FAISS:
                # FAISS flat index fallback
                d = papers_with_embs.shape[1]
                index = faiss.IndexFlatIP(d)
                index.add(papers_with_embs)
                scores, indices = index.search(query_emb, len(papers_with_embs))
                for score, idx in zip(scores[0], indices[0]):
                    real_idx = valid_indices[idx]
                    semantic_scores[real_idx] = float(score)
            else:
                # Pure numpy fallback
                scores = np.dot(papers_with_embs, query_emb.T).flatten()
                for score, idx in zip(scores, valid_indices):
                    semantic_scores[idx] = float(score)

        # Normalize semantic scores to [0, 1] for diagnostics and combined scoring
        sem_min = semantic_scores.min()
        sem_max = semantic_scores.max()
        if sem_max - sem_min > 0:
            semantic_scores = (semantic_scores - sem_min) / (sem_max - sem_min + 1e-8)

        # 3. Calculate Ranks
        # Sort indices by lexical scores descending to assign ranks
        lexical_sorted_indices = np.argsort(-lexical_scores)
        lexical_ranks = np.zeros(len(db_papers))
        current_rank = 1
        for i, idx in enumerate(lexical_sorted_indices):
            if i > 0 and lexical_scores[idx] < lexical_scores[lexical_sorted_indices[i - 1]]:
                current_rank = i + 1
            lexical_ranks[idx] = current_rank

        # Sort indices by semantic scores descending to assign ranks
        semantic_sorted_indices = np.argsort(-semantic_scores)
        semantic_ranks = np.zeros(len(db_papers))
        current_rank = 1
        for i, idx in enumerate(semantic_sorted_indices):
            if i > 0 and semantic_scores[idx] < semantic_scores[semantic_sorted_indices[i - 1]]:
                current_rank = i + 1
            semantic_ranks[idx] = current_rank

        # 4. Combine scores using Reciprocal Rank Fusion (RRF)
        scored_papers = []
        for idx, paper in enumerate(db_papers):
            lex_rank = lexical_ranks[idx]
            sem_rank = semantic_ranks[idx]
            
            # RRF Formula
            rrf_score = 1.0 / (k + lex_rank) + 1.0 / (k + sem_rank)
            
            # Simple combined score for legacy compatibility
            max_lex = lexical_scores.max() if lexical_scores.max() > 0 else 1.0
            linear_score = alpha * (lexical_scores[idx] / max_lex) + (1.0 - alpha) * semantic_scores[idx]

            scored_papers.append({
                "paper": paper,
                "lexical_score": float(lexical_scores[idx]),
                "semantic_score": float(semantic_scores[idx]),
                "lexical_rank": int(lex_rank),
                "semantic_rank": int(sem_rank),
                "rrf_score": float(rrf_score),
                "score": float(linear_score)
            })

        # Sort by RRF score descending
        scored_papers.sort(key=lambda x: x["rrf_score"], reverse=True)
        return scored_papers[:limit]


class ClusteringEngine:
    @staticmethod
    def cluster_papers(db_papers: list) -> dict:
        """
        Groups all papers into K-Means clusters and reduces coordinates
        to a 2D PCA representation for visualizations.
        """
        if not db_papers:
            return {"coords": [], "labels": []}

        # Filter papers with valid embeddings
        valid_papers = [p for p in db_papers if p.embedding_vector]
        if len(valid_papers) < NUM_CLUSTERS:
            # Not enough papers to form standard clusters
            return {
                "papers": [{"id": p.id, "title": p.title, "x": 0.0, "y": 0.0, "cluster": 0, "domain": p.domain} for p in valid_papers]
            }

        embeddings = np.array([p.embedding_vector for p in valid_papers], dtype=np.float32)

        # 1. Fit K-Means
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # 2. Fit PCA (Dimension Reduction to 2D)
        pca = PCA(n_components=2, random_state=42)
        reduced_coords = pca.fit_transform(embeddings)

        results = []
        for i, paper in enumerate(valid_papers):
            results.append({
                "id": paper.id,
                "title": paper.title,
                "x": float(reduced_coords[i, 0]),
                "y": float(reduced_coords[i, 1]),
                "cluster": int(cluster_labels[i]),
                "domain": paper.domain
            })

        return {"papers": results}


class RecommendationEngine:
    @staticmethod
    def recommend(target_paper: Paper, db_papers: list, limit: int = 5) -> list:
        """
        Generates paper recommendation lists based on target paper embedding similarity
        plus co-authorship and category overlaps.
        """
        if not db_papers or not target_paper.embedding_vector:
            return []

        target_vec = np.array(target_paper.embedding_vector, dtype=np.float32)
        target_vec = target_vec / np.linalg.norm(target_vec)

        recs = []
        for p in db_papers:
            if p.id == target_paper.id or not p.embedding_vector:
                continue

            # Calculate vector cosine similarity
            p_vec = np.array(p.embedding_vector, dtype=np.float32)
            p_vec = p_vec / np.linalg.norm(p_vec)
            cosine_sim = float(np.dot(target_vec, p_vec))

            # Meta booster scores
            author_boost = 0.0
            target_authors = set(target_paper.author_list)
            p_authors = set(p.author_list)
            shared_authors = target_authors.intersection(p_authors)
            if shared_authors:
                author_boost = 0.15

            category_boost = 0.0
            if p.primary_category == target_paper.primary_category:
                category_boost = 0.10

            final_score = min(cosine_sim + author_boost + category_boost, 1.0)
            
            recs.append({
                "paper": p,
                "score": final_score,
                "cosine_sim": cosine_sim,
                "shared_authors": list(shared_authors)
            })

        # Sort by final score descending
        recs.sort(key=lambda x: x["score"], reverse=True)
        return recs[:limit]

    @staticmethod
    def recommend_personalized(bookmarked_papers: list, db_papers: list, limit: int = 5) -> list:
        """
        Calculates User Profile Vector (centroid of bookmark embeddings)
        and queries candidate papers to yield top-N personalized items.
        """
        if not db_papers or not bookmarked_papers:
            return []

        bookmarked_ids = set(p.id for p in bookmarked_papers)
        
        # Collect valid embeddings from bookmarked papers
        bookmarked_vecs = [np.array(p.embedding_vector, dtype=np.float32) for p in bookmarked_papers if p.embedding_vector]
        if not bookmarked_vecs:
            return []

        # Compute centroid (mean vector of reading history)
        centroid = np.mean(bookmarked_vecs, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        recs = []
        for p in db_papers:
            if p.id in bookmarked_ids or not p.embedding_vector:
                continue

            # Calculate cosine similarity with centroid
            p_vec = np.array(p.embedding_vector, dtype=np.float32)
            p_norm = np.linalg.norm(p_vec)
            if p_norm > 0:
                p_vec = p_vec / p_norm
            
            similarity = float(np.dot(centroid, p_vec))

            recs.append({
                "paper": p,
                "score": similarity
            })

        # Sort by score descending
        recs.sort(key=lambda x: x["score"], reverse=True)
        return recs[:limit]


class RoadmapEngine:
    @staticmethod
    def generate_roadmap(topic: str, db_papers: list, depth: int = 6) -> dict:
        """
        Takes hybrid search results for a topic, partitions them chronologically
        into Beginner, Intermediate, and Advanced papers, and suggests Github repos.
        """
        # Search papers relevant to topic
        results = HybridRetriever.retrieve(topic, db_papers, limit=depth * 2, alpha=0.3)
        if not results:
            return {"topic": topic, "tiers": {"beginner": [], "intermediate": [], "advanced": []}, "repos": []}

        # Extract papers from search structures
        topic_papers = [r["paper"] for r in results]
        
        # Sort chronologically (oldest to newest) to represent learning pathways
        topic_papers.sort(key=lambda x: x.published_date)
        
        # Cap at requested depth
        selected = topic_papers[:depth]

        # Partition into three tiers
        tier_size = len(selected) // 3
        if tier_size == 0:
            tier_size = 1

        beginner = selected[:tier_size]
        intermediate = selected[tier_size:tier_size * 2]
        advanced = selected[tier_size * 2:]

        # Map typical open-source GitHub repositories based on domains
        domain_counts = {}
        for p in selected:
            domain_counts[p.domain] = domain_counts.get(p.domain, 0) + 1
        
        dominant_domain = max(domain_counts, key=domain_counts.get) if domain_counts else "Machine Learning (General)"

        github_suggestions = {
            "Natural Language Processing": [
                {"name": "huggingface/transformers", "desc": "State-of-the-art Machine Learning for PyTorch, TensorFlow, and JAX."},
                {"name": "Instruction-Tuning-Papers", "desc": "A curated list of instruction-tuning papers and open-source models."}
            ],
            "Computer Vision": [
                {"name": "open-mmlab/mmdetection", "desc": "OpenMMLab modern object detection toolbox and benchmark."},
                {"name": "ultralytics/yolov8", "desc": "YOLOv8 real-time object detection, segmentation, and classification."}
            ],
            "Reinforcement Learning": [
                {"name": "openai/spinningup", "desc": "An educational resource designed to let anyone learn deep RL."},
                {"name": "stable-baselines3", "desc": "Reliable implementations of reinforcement learning algorithms in PyTorch."}
            ],
            "Robotics & Control": [
                {"name": "ros-planning/navigation2", "desc": "The ROS 2 Navigation System planner framework."},
                {"name": "leggedrobotics/ocs2", "desc": "Optimal Control for Switched Systems toolbox."}
            ]
        }.get(dominant_domain, [
            {"name": "scikit-learn/scikit-learn", "desc": "Machine learning in Python wrapper for standard classifiers and clustering."},
            {"name": "karpathy/micrograd", "desc": "A tiny scalar-valued autograd engine with a small PyTorch-like API."}
        ])

        return {
            "topic": topic,
            "domain": dominant_domain,
            "tiers": {
                "beginner": beginner,
                "intermediate": intermediate,
                "advanced": advanced
            },
            "repos": github_suggestions
        }


class TrendEngine:
    @staticmethod
    def analyze_trends(db_papers: list) -> dict:
        """
        Analyzes quarterly publication volumes, counts top NLP keywords,
        and lists active author publishing patterns.
        """
        if not db_papers:
            return {"timeline": {}, "domain_counts": {}, "hot_keywords": []}

        # 1. Timeline quarterly volumes
        timeline = {}
        domain_counts = {}
        stop_words = {"this", "that", "with", "from", "have", "were", "been", "their", "which", "also", 
                      "more", "some", "these", "paper", "novel", "proposed", "results", "model", "method",
                      "using", "based", "approach", "framework", "algorithm", "learning", "data", "analysis"}
        word_counts = {}

        for p in db_papers:
            # Quarter binning (e.g. 2024-Q1)
            pub_date = p.published_date
            quarter = f"{pub_date.year}-Q{(pub_date.month - 1) // 3 + 1}"
            timeline[quarter] = timeline.get(quarter, 0) + 1

            # Domain share counts
            domain_counts[p.domain] = domain_counts.get(p.domain, 0) + 1

            # Keyword processing
            words = [w.lower() for w in re_split_words(f"{p.title} {p.abstract}")]
            unique_words = set(w for w in words if len(w) > 4 and w not in stop_words)
            for w in unique_words:
                word_counts[w] = word_counts.get(w, 0) + 1

        # Sort quarters chronologically
        sorted_quarters = dict(sorted(timeline.items()))

        # Sort hot keywords
        hot_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        hot_keywords_list = [{"keyword": k, "count": c} for k, c in hot_keywords]

        return {
            "timeline": sorted_quarters,
            "domain_counts": domain_counts,
            "hot_keywords": hot_keywords_list
        }

def re_split_words(text: str) -> list:
    """Helper regex splitter for words."""
    import re
    return re.findall(r'\b[a-zA-Z]{4,}\b', text)
