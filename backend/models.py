from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json
import numpy as np

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(100), nullable=False)
    role = Column(String(20), nullable=False, default='user')

    sessions = relationship("Session", back_populates="user")
    categories = relationship("Category", back_populates="user")
    codes = relationship("Code", back_populates="user")
    ontologies = relationship("Ontology", back_populates="user")

class Category(Base):
    __tablename__ = 'categories'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    user_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="categories")
    codes = relationship("Code", back_populates="category")

class Code(Base):
    __tablename__ = 'codes'

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    category_id = Column(Integer, ForeignKey('categories.id'), nullable=True)
    session_id = Column(String(100), ForeignKey('sessions.id'))
    parent_id = Column(Integer, ForeignKey('codes.id'), nullable=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    ontology_concept_id = Column(Integer, ForeignKey('ontology_concepts.id'), nullable=True)
    embedding = Column(Text, nullable=True)  # JSON string of embedding vector
    created_at = Column(DateTime, default=datetime.utcnow)

    category = relationship("Category", back_populates="codes")
    session = relationship("Session", back_populates="session_codes")
    user = relationship("User", back_populates="codes")
    ontology_concept = relationship("OntologyConcept", back_populates="codes")
    parent = relationship("Code", remote_side=[id], back_populates="children")
    children = relationship("Code", back_populates="parent")

class Session(Base):
    __tablename__ = 'sessions'

    id = Column(String(100), primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    name = Column(String(255), nullable=True)  # Session name for user-friendly identification
    codes = Column(Text)  # JSON string - kept for backward compatibility
    data = Column(Text)   # JSON string
    questions = Column(Text)  # JSON string
    feedback = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="sessions")
    documents = relationship("Document", back_populates="session")
    session_codes = relationship("Code", back_populates="session")

class Document(Base):
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey('sessions.id'))
    type = Column(String(20), nullable=False)  # 'consolidated', 'output', 'uploaded'
    content = Column(Text, nullable=False)
    filename = Column(String(255), nullable=True)
    file_type = Column(String(50), nullable=True)  # e.g., 'pdf', 'docx', 'txt'
    timestamp = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="documents")

class Ontology(Base):
    __tablename__ = 'ontologies'

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    user_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="ontologies")
    concepts = relationship("OntologyConcept", back_populates="ontology")

class OntologyConcept(Base):
    __tablename__ = 'ontology_concepts'

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    ontology_id = Column(Integer, ForeignKey('ontologies.id'))
    parent_id = Column(Integer, ForeignKey('ontology_concepts.id'), nullable=True)
    embedding = Column(Text, nullable=True)  # JSON string of embedding vector
    created_at = Column(DateTime, default=datetime.utcnow)

    ontology = relationship("Ontology", back_populates="concepts")
    parent = relationship("OntologyConcept", remote_side=[id], back_populates="children")
    children = relationship("OntologyConcept", back_populates="parent")
    codes = relationship("Code", back_populates="ontology_concept")

# Database setup
DATABASE_URL = "sqlite:///./database.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

def create_tables():
    Base.metadata.create_all(bind=engine)