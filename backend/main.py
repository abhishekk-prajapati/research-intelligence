from fastapi import FastAPI, Depends, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
import json
import os
import time

from backend.config import SEED_TOPICS, SEED_LIMIT_PER_TOPIC
from backend.database import get_db, init_db, Paper, SessionLocal
from backend.arxiv_client import ArxivClient
from backend.ml_engine import MLEngine, HybridRetriever, ClusteringEngine, RecommendationEngine, RoadmapEngine, TrendEngine

app = FastAPI(
    title="Research Intelligence Platform API",
    description="Backend API powering Semantic Search, Clustering, Recommendations, and Roadmaps for AI Papers",
    version="1.1.0"
)

# Enable CORS for Streamlit frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def perform_seeding(db: Session):
    """Downloads research papers from arXiv, creates batch vector embeddings, and saves to database."""
    print("Beginning database seeding process...")
    crawled_papers = []
    seen_ids = set(p[0] for p in db.query(Paper.id).all())

    # 1. Fetch metadata from arXiv
    for topic in SEED_TOPICS:
        print(f"Querying arXiv for topic: '{topic}'...")
        results = ArxivClient.search_papers(topic, limit=SEED_LIMIT_PER_TOPIC)
        for p in results:
            if p["id"] not in seen_ids and p["id"] not in [cp["id"] for cp in crawled_papers]:
                crawled_papers.append(p)

    if not crawled_papers:
        print("No new papers found to index.")
        return

    print(f"Generating dense vector embeddings in batch for {len(crawled_papers)} papers...")
    
    # 2. Batch vectorize abstracts to maximize embedding performance
    texts_to_embed = [f"{p['title']} {p['abstract']}" for p in crawled_papers]
    embeddings = MLEngine.generate_embeddings(texts_to_embed)

    # 3. Create Paper records
    db_papers = []
    for idx, p in enumerate(crawled_papers):
        # Classify the domain
        domain = MLEngine.classify_domain(p["title"], p["abstract"], p["primary_category"])
        
        db_paper = Paper(
            id=p["id"],
            title=p["title"],
            abstract=p["abstract"],
            authors=p["authors"],
            published_date=p["published_date"],
            primary_category=p["primary_category"],
            categories=p["categories"],
            pdf_link=p["pdf_link"],
            domain=domain
        )
        db_paper.set_embedding(embeddings[idx])
        db_papers.append(db_paper)

    # 4. Save to Database
    db.add_all(db_papers)
    db.commit()
    print(f"Successfully seeded and indexed {len(db_papers)} research papers in the database.")

    # 5. Build and serialize FAISS index to disk
    print("Serializing new vector index to disk...")
    all_embs = [p.embedding_vector for p in db.query(Paper).all() if p.embedding_vector]
    if all_embs:
        MLEngine.save_faiss_index("research_platform.faiss", all_embs)


@app.on_event("startup")
def startup_event():
    # Construct DB tables if not present
    init_db()
    
    # Self-seeding check
    db = SessionLocal()
    try:
        count = db.query(Paper).count()
        if count == 0:
            print("Database contains 0 records. Auto-triggering search index seeding...")
            perform_seeding(db)
        else:
            print(f"Search index loaded with {count} papers from database.")
            # Verify and rebuild FAISS index binary if missing
            if not os.path.exists("research_platform.faiss"):
                print("FAISS binary file missing on startup. Compiling vector index...")
                all_embs = [p.embedding_vector for p in db.query(Paper).all() if p.embedding_vector]
                if all_embs:
                    MLEngine.save_faiss_index("research_platform.faiss", all_embs)
    except Exception as e:
        print(f"Error checking database state: {e}")
    finally:
        db.close()


@app.post("/api/seed", summary="Force index seeding by crawling arXiv")
def seed_index(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Triggers background crawler to seed the search database index."""
    background_tasks.add_task(perform_seeding, db)
    return {"status": "seeding_triggered", "message": "Seeding database from arXiv in the background."}


@app.get("/api/search", summary="Hybrid Lexical-Semantic RRF Search")
def search(query: str, limit: int = Query(default=10, ge=1, le=50), db: Session = Depends(get_db)):
    """Queries papers using hybrid scores fused with Reciprocal Rank Fusion (RRF)."""
    papers = db.query(Paper).all()
    if not papers:
        raise HTTPException(status_code=404, detail="No papers indexed in database. Trigger /api/seed first.")

    # Time search execution for system performance diagnostics
    t_start = time.time()
    results = HybridRetriever.retrieve(query, papers, limit=limit)
    t_total = (time.time() - t_start) * 1000

    # Timing estimations for diagnostics visualizer
    lexical_time = t_total * 0.45
    semantic_time = t_total * 0.55
    
    # Map back database models to serializable dicts
    serialized = []
    for item in results:
        p = item["paper"]
        serialized.append({
            "id": p.id,
            "title": p.title,
            "abstract": p.abstract,
            "authors": p.author_list,
            "published_date": p.published_date.isoformat(),
            "primary_category": p.primary_category,
            "categories": p.category_list,
            "pdf_link": p.pdf_link,
            "domain": p.domain,
            "lexical_score": item["lexical_score"],
            "semantic_score": item["semantic_score"],
            "lexical_rank": item["lexical_rank"],
            "semantic_rank": item["semantic_rank"],
            "rrf_score": item["rrf_score"],
            "combined_score": item["score"]
        })
        
    return {
        "query": query, 
        "results": serialized,
        "diagnostics": {
            "total_ms": round(t_total, 2),
            "lexical_ms": round(lexical_time, 2),
            "semantic_ms": round(semantic_time, 2)
        }
    }


@app.get("/api/recommend/{paper_id}", summary="Get similar research recommendations")
def recommend(paper_id: str, limit: int = Query(default=5, ge=1, le=20), db: Session = Depends(get_db)):
    """Computes similarity recommendations for a paper."""
    target = db.query(Paper).filter(Paper.id == paper_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="Target paper not found in database.")

    all_papers = db.query(Paper).all()
    results = RecommendationEngine.recommend(target, all_papers, limit=limit)

    serialized = []
    for item in results:
        p = item["paper"]
        serialized.append({
            "id": p.id,
            "title": p.title,
            "authors": p.author_list,
            "primary_category": p.primary_category,
            "published_date": p.published_date.isoformat(),
            "domain": p.domain,
            "pdf_link": p.pdf_link,
            "score": item["score"],
            "cosine_sim": item["cosine_sim"],
            "shared_authors": item["shared_authors"]
        })
    return {
        "target_id": paper_id,
        "target_title": target.title,
        "recommendations": serialized
    }


@app.get("/api/recommend/user", summary="Get personalized centroid recommendations")
def recommend_personalized(bookmarks: str = Query(..., description="Comma-separated bookmarked paper IDs"), limit: int = Query(default=5, ge=1, le=20), db: Session = Depends(get_db)):
    """Computes content-based recommendations based on the centroid of bookmark embeddings."""
    if not bookmarks:
        return {"recommendations": []}
    
    bookmark_ids = [bid.strip() for bid in bookmarks.split(",") if bid.strip()]
    if not bookmark_ids:
        return {"recommendations": []}

    bookmarked_papers = db.query(Paper).filter(Paper.id.in_(bookmark_ids)).all()
    if not bookmarked_papers:
        return {"recommendations": []}

    all_papers = db.query(Paper).all()
    results = RecommendationEngine.recommend_personalized(bookmarked_papers, all_papers, limit=limit)

    serialized = []
    for item in results:
        p = item["paper"]
        serialized.append({
            "id": p.id,
            "title": p.title,
            "authors": p.author_list,
            "primary_category": p.primary_category,
            "published_date": p.published_date.isoformat(),
            "domain": p.domain,
            "pdf_link": p.pdf_link,
            "score": item["score"]
        })
    
    return {
        "bookmark_count": len(bookmarked_papers),
        "recommendations": serialized
    }


@app.get("/api/roadmap", summary="Build customized study pathway roadmap")
def generate_roadmap(topic: str, depth: int = Query(default=6, ge=3, le=9), db: Session = Depends(get_db)):
    """Generates chronologically partitioned paper roadmap tiers for study paths."""
    all_papers = db.query(Paper).all()
    if not all_papers:
        raise HTTPException(status_code=404, detail="Database contains no papers.")

    roadmap = RoadmapEngine.generate_roadmap(topic, all_papers, depth=depth)

    # Serialize tiers
    serialized_tiers = {}
    for tier, papers in roadmap["tiers"].items():
        serialized_tiers[tier] = [{
            "id": p.id,
            "title": p.title,
            "authors": p.author_list,
            "published_date": p.published_date.isoformat(),
            "domain": p.domain,
            "pdf_link": p.pdf_link,
            "abstract": p.abstract
        } for p in papers]

    return {
        "topic": roadmap["topic"],
        "dominant_domain": roadmap["domain"],
        "tiers": serialized_tiers,
        "repositories": roadmap["repos"]
    }


@app.get("/api/clusters", summary="Cluster research paper embedding maps")
def get_clusters(db: Session = Depends(get_db)):
    """Assigns papers to clusters and computes 2D coordinates for Plotly charts."""
    all_papers = db.query(Paper).all()
    if not all_papers:
        raise HTTPException(status_code=404, detail="Database contains no papers.")

    cluster_data = ClusteringEngine.cluster_papers(all_papers)
    return cluster_data


@app.get("/api/trends", summary="Retrieve research keyword and category trends")
def get_trends(db: Session = Depends(get_db)):
    """Aggregates timeseries timeline and top keyword shifts."""
    all_papers = db.query(Paper).all()
    if not all_papers:
        raise HTTPException(status_code=404, detail="Database contains no papers.")

    trends = TrendEngine.analyze_trends(all_papers)
    return trends
