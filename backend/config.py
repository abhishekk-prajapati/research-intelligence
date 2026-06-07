import os

# Database configurations: default to SQLite for zero-setup local deployment, supports PostgreSQL
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./research_platform.db")

# Machine Learning Configurations
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
NUM_CLUSTERS = 5

# Topics to download from arXiv on initial setup to seed the search index
SEED_TOPICS = [
    "Large Language Models",
    "Computer Vision",
    "Reinforcement Learning",
    "Robotics",
    "Graph Neural Networks"
]

# Max papers fetched per topic during initial seeding
SEED_LIMIT_PER_TOPIC = 40
