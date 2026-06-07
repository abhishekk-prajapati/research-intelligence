import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Page Configurations
st.set_page_config(
    page_title="Research Intelligence Platform",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend service address
BACKEND_URL = "http://localhost:8000"

st.sidebar.markdown("# 🌌 Research Intelligence")
st.sidebar.markdown("### Scholarly AI Discovery Engine")
st.sidebar.markdown("---")

# Initialize Session States for Personalization
if "bookmarks" not in st.session_state:
    st.session_state["bookmarks"] = []

# Health check & status helper
db_connected = True
db_papers_count = 0
try:
    # Use trends endpoint as a proxy to check if backend is running and has papers
    resp = requests.get(f"{BACKEND_URL}/api/trends", timeout=2)
    if resp.status_code == 200:
        db_connected = True
        # Get count from clusters
        clust_resp = requests.get(f"{BACKEND_URL}/api/clusters", timeout=2)
        if clust_resp.status_code == 200:
            db_papers_count = len(clust_resp.json().get("papers", []))
except requests.exceptions.ConnectionError:
    db_connected = False

# Sidebar Widgets
if db_connected:
    st.sidebar.success("Backend Service: Connected")
    st.sidebar.metric("Indexed AI Papers", db_papers_count)
    if st.sidebar.button("Re-crawl & Index arXiv"):
        with st.sidebar.spinner("Crawling new papers in background..."):
            try:
                seed_resp = requests.post(f"{BACKEND_URL}/api/seed")
                if seed_resp.status_code == 200:
                    st.sidebar.info("Crawl triggered! Refresh in a few seconds.")
            except Exception as e:
                st.sidebar.error(f"Error seeding: {e}")
else:
    st.sidebar.error("Backend Service: Disconnected")
    st.sidebar.warning("Please start the FastAPI server at `localhost:8000` to proceed.")

st.sidebar.markdown("""
---
**Advanced ML Architecture:**
- **Hybrid Fusion**: Reciprocal Rank Fusion (RRF)
- **Vector Serialization**: FAISS FlatIP Disk Index
- **Information Retrieval**: BM25 / TF-IDF + Dense Embeddings
- **Personalization**: User Centroid Embedding Profiles
- **Topic Mapping**: K-Means Clustering + PCA Coordinates
""")

st.title("🌌 Research Intelligence Platform")
st.caption("A premium machine learning portfolio dashboard for semantic search, recommendation, and topic clustering over academic AI literature.")

if not db_connected:
    st.warning("⚠️ FastAPI Backend is offline. Please launch the API server using uvicorn or docker-compose to explore the platform.")
elif db_papers_count == 0:
    st.info("📂 The search index contains 0 papers. Let's seed the database index with crawled arXiv records.")
    if st.button("Trigger Initial Index Seeding"):
        with st.spinner("Seeding database..."):
            try:
                seed_resp = requests.post(f"{BACKEND_URL}/api/seed")
                st.success("Seeding started! Please reload the page in 5-10 seconds.")
            except Exception as e:
                st.error(f"Seeding failed: {e}")
else:
    # Define Tabs
    tab_search, tab_recs, tab_roadmap, tab_cluster, tab_trends = st.tabs([
        "🔍 Hybrid RRF Search",
        "🎯 Personalized Recommendations",
        "🗺️ Learning Roadmaps",
        "🔮 Topic Clustering Map",
        "📈 Trend Analysis"
    ])

    # ==========================================================================
    # TAB 1: HYBRID SEMANTIC SEARCH
    # ==========================================================================
    with tab_search:
        st.header("Hybrid RRF Search Engine")
        st.write("Perform semantic queries combined with lexical matching. Features Reciprocal Rank Fusion (RRF) to merge keyword matching and dense vector searches.")
        
        search_query = st.text_input("Search Query", "RAG performance on medical documents or clinical notes", placeholder="e.g. Diffusion models, LLM alignment, Policy Gradient...")
        
        col_k, col_limit = st.columns(2)
        with col_k:
            rrf_k = st.slider("RRF Rank Constant (k)", 10, 100, 60, 5, 
                              help="Formula constant for rank smoothing. Default is 60.")
        with col_limit:
            limit = st.slider("Results limit", 5, 30, 10, 5)

        if st.button("Run Search Query", type="primary") or search_query:
            with st.spinner("Searching and ranking results..."):
                try:
                    resp = requests.get(f"{BACKEND_URL}/api/search", params={"query": search_query, "limit": limit, "k": rrf_k})
                    if resp.status_code == 200:
                        data = resp.json()
                        results = data.get("results", [])
                        diagnostics = data.get("diagnostics", {})
                        
                        if not results:
                            st.warning("No papers matched your search query.")
                        else:
                            # 1. Latency Diagnostics Visualisation
                            st.markdown("### 📊 Retrieval Diagnostics & Rank Analysis")
                            col_diag_chart, col_diag_table = st.columns([1, 2])
                            
                            with col_diag_chart:
                                st.write("**Query Latency (ms)**")
                                diag_df = pd.DataFrame([
                                    {"Step": "Lexical (TF-IDF)", "Latency (ms)": diagnostics.get("lexical_ms", 0.0)},
                                    {"Step": "Semantic (FAISS)", "Latency (ms)": diagnostics.get("semantic_ms", 0.0)},
                                    {"Step": "Total Search", "Latency (ms)": diagnostics.get("total_ms", 0.0)}
                                ])
                                fig_latency = px.bar(
                                    diag_df,
                                    x="Step",
                                    y="Latency (ms)",
                                    color="Step",
                                    template="plotly_dark",
                                    height=250,
                                    color_discrete_map={
                                        "Lexical (TF-IDF)": "rgba(99, 102, 241, 0.7)",
                                        "Semantic (FAISS)": "rgba(6, 182, 212, 0.7)",
                                        "Total Search": "rgba(16, 185, 129, 0.7)"
                                    }
                                )
                                fig_latency.update_layout(showlegend=False)
                                st.plotly_chart(fig_latency, use_container_width=True)
                                
                            with col_diag_table:
                                st.write("**Top-5 Rank Comparitive Matrix**")
                                # Extract top papers ordered under different ranks
                                lex_sorted = sorted(results, key=lambda x: x["lexical_rank"])[:5]
                                sem_sorted = sorted(results, key=lambda x: x["semantic_rank"])[:5]
                                rrf_sorted = results[:5]
                                
                                rank_matrix = []
                                for idx in range(min(5, len(results))):
                                    rank_matrix.append({
                                        "Rank": idx + 1,
                                        "Lexical (TF-IDF)": lex_sorted[idx]["title"][:45] + "...",
                                        "Semantic (FAISS)": sem_sorted[idx]["title"][:45] + "...",
                                        "RRF Hybrid (Fused)": rrf_sorted[idx]["title"][:45] + "..."
                                    })
                                st.dataframe(pd.DataFrame(rank_matrix), use_container_width=True, hide_index=True)
                            
                            st.markdown("---")
                            st.subheader(f"Fused Search Results (Ordered by RRF Score)")
                            
                            for idx, paper in enumerate(results):
                                with st.container():
                                    st.markdown(f"### {idx + 1}. {paper['title']}")
                                    st.markdown(f"**Authors**: {', '.join(paper['authors'])} | **Published**: {paper['published_date'][:10]} | **Domain**: `{paper['domain']}`")
                                    
                                    # Score indicators
                                    col_s1, col_s2, col_s3, col_link = st.columns([1, 1, 1, 1])
                                    col_s1.metric("RRF Score", f"{paper['rrf_score']:.4f}")
                                    col_s2.caption(f"Semantic Rank: **#{paper['semantic_rank']}** (Score: `{paper['semantic_score']:.3f}`)")
                                    col_s3.caption(f"Lexical Rank: **#{paper['lexical_rank']}** (Score: `{paper['lexical_score']:.3f}`)")
                                    
                                    # Bookmark / Saved state handler
                                    is_bookmarked = paper["id"] in st.session_state["bookmarks"]
                                    btn_label = "🔖 Bookmarked" if is_bookmarked else "➕ Add to Profile"
                                    
                                    with col_link:
                                        if paper['pdf_link']:
                                            st.markdown(f"[📄 View PDF on arXiv]({paper['pdf_link']})")
                                        if st.button(btn_label, key=f"bookmark-{paper['id']}"):
                                            if paper["id"] in st.session_state["bookmarks"]:
                                                st.session_state["bookmarks"].remove(paper["id"])
                                            else:
                                                st.session_state["bookmarks"].append(paper["id"])
                                            st.rerun()
                                            
                                    st.write(paper['abstract'])
                                    
                                    # Direct recommendation linkage
                                    if st.button("Find Similar Papers", key=f"rec-btn-{paper['id']}"):
                                        st.session_state["recommend_paper_id"] = paper["id"]
                                        st.info("Switched to Recommendations tab. Please select the Recommendations tab to view.")
                                    st.markdown("---")
                except Exception as e:
                    st.error(f"Search failed: {e}")

    # ==========================================================================
    # TAB 2: RECOMMENDATIONS & PERSONALIZATION
    # ==========================================================================
    with tab_recs:
        st.header("Personalization & Recommendation Matrix")
        
        # Split screen: Left = Content-Based User Personalization Centroid, Right = Pairwise Paper Similarity
        col_pers, col_pairwise = st.columns(2)
        
        with col_pers:
            st.subheader("🎯 Centroid-Based Personalization")
            st.write("Constructs a **User Profile Embedding** by taking the centroid vector (mean vector) of all bookmarked papers in your reading history, then queries the FAISS index for matching papers.")
            
            # Retrieve bookmark objects from database
            bookmark_titles = []
            if st.session_state["bookmarks"]:
                try:
                    clust_resp = requests.get(f"{BACKEND_URL}/api/clusters")
                    if clust_resp.status_code == 200:
                        papers_list = clust_resp.json().get("papers", [])
                        papers_df = pd.DataFrame(papers_list)
                        
                        bookmarked_df = papers_df[papers_df["id"].isin(st.session_state["bookmarks"])]
                        bookmark_titles = bookmarked_df["title"].tolist()
                except Exception as e:
                    pass

            st.write(f"**Your Profile Reading History ({len(st.session_state['bookmarks'])} papers)**")
            if bookmark_titles:
                for title in bookmark_titles:
                    st.markdown(f"- *{title}*")
            else:
                st.caption("No papers bookmarked yet. Search papers and click 'Add to Profile' to build your personalization context.")

            # Multi-select input to customize bookmarks explicitly
            try:
                clust_resp = requests.get(f"{BACKEND_URL}/api/clusters")
                if clust_resp.status_code == 200:
                    papers_list = clust_resp.json().get("papers", [])
                    papers_df = pd.DataFrame(papers_list)
                    paper_options = dict(zip(papers_df["id"], papers_df["title"]))
                    
                    selected_bookmarks = st.multiselect("Edit bookmark selections", options=list(paper_options.keys()),
                                                        format_func=lambda x: paper_options[x],
                                                        default=st.session_state["bookmarks"])
                    
                    if selected_bookmarks != st.session_state["bookmarks"]:
                        st.session_state["bookmarks"] = selected_bookmarks
                        st.rerun()
            except Exception:
                pass

            if st.button("Generate Personalized Recommendations", type="primary") and st.session_state["bookmarks"]:
                with st.spinner("Calculating profile centroid embedding..."):
                    try:
                        b_params = {"bookmarks": ",".join(st.session_state["bookmarks"]), "limit": 5}
                        pers_resp = requests.get(f"{BACKEND_URL}/api/recommend/user", params=b_params)
                        if pers_resp.status_code == 200:
                            data = pers_resp.json()
                            recs = data.get("recommendations", [])
                            
                            st.write("**Recommended Papers matching your centroid profile:**")
                            if not recs:
                                st.warning("No personalized recommendations could be generated.")
                            else:
                                for idx, rec in enumerate(recs):
                                    st.markdown(f"**{idx + 1}. {rec['title']}**")
                                    st.caption(f"Centroid Cosine Match Score: `{rec['score']:.4f}` | Domain: `{rec['domain']}`")
                                    st.markdown(f"Authors: {', '.join(rec['authors'])}")
                                    if rec['pdf_link']:
                                        st.markdown(f"[📄 View PDF]({rec['pdf_link']})")
                                    st.markdown("---")
                    except Exception as e:
                        st.error(f"Failed loading personalization list: {e}")

        with col_pairwise:
            st.subheader("🔗 Pairwise Paper Similarity")
            st.write("Queries standard co-authorship overlaps and cosine distances matching a single designated paper.")

            try:
                # Load list of all papers in DB
                clust_resp = requests.get(f"{BACKEND_URL}/api/clusters")
                if clust_resp.status_code == 200:
                    papers_list = clust_resp.json().get("papers", [])
                    papers_df = pd.DataFrame(papers_list)
                    paper_options = dict(zip(papers_df["id"], papers_df["title"]))
                    
                    # Check session state for search-linked recommendations
                    default_idx = 0
                    if "recommend_paper_id" in st.session_state and st.session_state["recommend_paper_id"] in paper_options:
                        default_idx = list(paper_options.keys()).index(st.session_state["recommend_paper_id"])
                    
                    selected_paper_id = st.selectbox("Select Target Paper", options=list(paper_options.keys()), 
                                                     format_func=lambda x: paper_options[x], index=default_idx)

                    rec_limit = st.slider("Recommendations count", 2, 10, 5)

                    if selected_paper_id:
                        with st.spinner("Calculating recommendation vector weights..."):
                            rec_resp = requests.get(f"{BACKEND_URL}/api/recommend/{selected_paper_id}", params={"limit": rec_limit})
                            if rec_resp.status_code == 200:
                                data = rec_resp.json()
                                recs = data.get("recommendations", [])
                                
                                if not recs:
                                    st.warning("No recommendations found.")
                                else:
                                    for idx, rec in enumerate(recs):
                                        st.markdown(f"**{idx + 1}. {rec['title']}**")
                                        st.caption(f"Composite Score: `{rec['score']:.3f}` (Vector Cosine: `{rec['cosine_sim']:.3f}`) | Domain: `{rec['domain']}`")
                                        st.markdown(f"Authors: {', '.join(rec['authors'])}")
                                        
                                        if rec['shared_authors']:
                                            st.markdown(f"🤝 **Shared Authors**: {', '.join(rec['shared_authors'])}")
                                        
                                        if rec['pdf_link']:
                                            st.markdown(f"[📄 View PDF]({rec['pdf_link']})")
                                        st.markdown("---")
            except Exception as e:
                st.error(f"Failed loading recommendations: {e}")

    # ==========================================================================
    # TAB 3: LEARNING ROADMAPS
    # ==========================================================================
    with tab_roadmap:
        st.header("Learning Roadmap Synthesizer")
        st.write("Enter an AI research topic to generate a tiered reading curriculum structured from foundational literature to recent works.")

        col_r1, col_r2 = st.columns([2, 1])
        with col_r1:
            roadmap_topic = st.text_input("Roadmap Topic", "Reinforcement Learning", placeholder="e.g. Transformers, Diffusion Models...")
        with col_r2:
            roadmap_depth = st.selectbox("Roadmap Depth (Number of papers)", [3, 6, 9], index=1)

        if st.button("Synthesize Roadmap", type="primary") or roadmap_topic:
            with st.spinner("Constructing tiered syllabus path..."):
                try:
                    road_resp = requests.get(f"{BACKEND_URL}/api/roadmap", params={"topic": roadmap_topic, "depth": roadmap_depth})
                    if road_resp.status_code == 200:
                        data = road_resp.json()
                        st.subheader(f"Reading Syllabus: {data['topic']}")
                        st.markdown(f"Dominant Platform Domain: **`{data['dominant_domain']}`**")
                        
                        # Render tiers
                        tiers = data.get("tiers", {})
                        for tier_name in ["beginner", "intermediate", "advanced"]:
                            papers = tiers.get(tier_name, [])
                            if papers:
                                st.markdown(f"### 📍 {tier_name.capitalize()} Tier")
                                for idx, p in enumerate(papers):
                                    with st.expander(f"Step {idx+1}: {p['title']} ({p['published_date'][:10]})"):
                                        st.markdown(f"**Authors**: {', '.join(p['authors'])}")
                                        st.markdown(f"**Abstract**: {p['abstract']}")
                                        if p['pdf_link']:
                                            st.markdown(f"[📄 View PDF on arXiv]({p['pdf_link']})")
                        
                        # Render GitHub repos
                        st.markdown("### 🛠️ Recommended GitHub Repositories")
                        repos = data.get("repositories", [])
                        for repo in repos:
                            st.markdown(f"- **[{repo['name']}](https://github.com/{repo['name']})**: {repo['desc']}")
                except Exception as e:
                    st.error(f"Roadmap failed: {e}")

    # ==========================================================================
    # TAB 4: TOPIC CLUSTERING MAP
    # ==========================================================================
    with tab_cluster:
        st.header("2D Unsupervised Topic Clustering Map")
        st.write("Interactive cluster visualization of all research papers in the database. Vector embeddings are generated via Sentence Transformers, grouped using K-Means, and projected to 2D via PCA.")

        with st.spinner("Rendering clustering diagram..."):
            try:
                clust_resp = requests.get(f"{BACKEND_URL}/api/clusters")
                if clust_resp.status_code == 200:
                    papers_list = clust_resp.json().get("papers", [])
                    if papers_list:
                        df = pd.DataFrame(papers_list)
                        df["Cluster Index"] = df["cluster"].astype(str)
                        
                        fig = px.scatter(
                            df,
                            x="x",
                            y="y",
                            color="Cluster Index",
                            symbol="domain",
                            hover_data=["title", "domain"],
                            title="Interactive PCA Projection Grid",
                            labels={"x": "PCA Component 1", "y": "PCA Component 2"},
                            template="plotly_dark",
                            height=600
                        )
                        
                        # Add clean layout glows
                        fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color="white")))
                        fig.update_layout(legend_title="Cluster Labels & Domains")
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough papers to render a 2D cluster map. Seed the database with more papers.")
            except Exception as e:
                st.error(f"Failed loading cluster coordinates: {e}")

    # ==========================================================================
    # TAB 5: TREND ANALYSIS
    # ==========================================================================
    with tab_trends:
        st.header("NLP Research Trends Explorer")
        st.write("Visualizes velocity metrics, category distributions, and NLP keyword frequencies from the indexed paper database.")

        with st.spinner("Compiling publication time-series..."):
            try:
                trends_resp = requests.get(f"{BACKEND_URL}/api/trends")
                if trends_resp.status_code == 200:
                    data = trends_resp.json()
                    
                    # 1. Timeline Velocity Chart
                    st.subheader("Publication Velocity")
                    st.write("Submission density of research papers aggregated quarterly.")
                    timeline = data.get("timeline", {})
                    if timeline:
                        time_df = pd.DataFrame(list(timeline.items()), columns=["Quarter", "Paper Count"])
                        fig_line = px.line(
                            time_df,
                            x="Quarter",
                            y="Paper Count",
                            markers=True,
                            template="plotly_dark",
                            height=350,
                            color_discrete_sequence=["#06b6d4"]
                        )
                        st.plotly_chart(fig_line, use_container_width=True)
                    
                    # 2. Columns for categories and keywords
                    col_cat, col_key = st.columns(2)
                    
                    with col_cat:
                        st.subheader("Domain Distribution")
                        st.write("Domain shares tagged by the NLP Research Classifier.")
                        domain_counts = data.get("domain_counts", {})
                        if domain_counts:
                            dom_df = pd.DataFrame(list(domain_counts.items()), columns=["Domain", "Count"])
                            fig_pie = px.pie(
                                dom_df,
                                names="Domain",
                                values="Count",
                                template="plotly_dark",
                                height=350,
                                hole=0.5
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                    with col_key:
                        st.subheader("Trending Technical Keywords")
                        st.write("Top occurring terminology tokens across paper abstracts.")
                        keywords = data.get("hot_keywords", [])
                        if keywords:
                            key_df = pd.DataFrame(keywords)
                            fig_bar = px.bar(
                                key_df,
                                x="count",
                                y="keyword",
                                orientation="h",
                                template="plotly_dark",
                                height=350,
                                color_discrete_sequence=["#6366f1"]
                            )
                            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig_bar, use_container_width=True)
            except Exception as e:
                st.error(f"Trends failed to load: {e}")
