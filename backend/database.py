import json
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Text, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from backend.config import DATABASE_URL

# Setup DB connection
# SQLite needs connect_args={"check_same_thread": False}
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Paper(Base):
    __tablename__ = "papers"

    id = Column(String, primary_key=True, index=True) # arXiv ID (e.g. 1706.03762)
    title = Column(String, nullable=False)
    abstract = Column(Text, nullable=False)
    authors = Column(Text, nullable=False) # Comma-separated names
    published_date = Column(DateTime, nullable=False)
    primary_category = Column(String, nullable=False)
    categories = Column(String, nullable=False) # Comma-separated terms
    pdf_link = Column(String, nullable=True)
    domain = Column(String, nullable=True) # Classified domain
    embedding = Column(Text, nullable=True) # JSON-serialized float array

    @property
    def author_list(self):
        return [a.strip() for a in self.authors.split(",")] if self.authors else []

    @property
    def category_list(self):
        return [c.strip() for c in self.categories.split(",")] if self.categories else []

    @property
    def embedding_vector(self):
        if self.embedding:
            return json.loads(self.embedding)
        return []

    def set_embedding(self, vec):
        self.embedding = json.dumps(list(vec))

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
