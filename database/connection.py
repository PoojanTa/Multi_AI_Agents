"""Database connection and session management"""

import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
from typing import Generator
from database.models import Base

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/dbname')

# Create engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False  # Set to True for SQL debugging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Get database session with automatic cleanup"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()

def get_db() -> Session:
    """Get database session (for dependency injection)"""
    return SessionLocal()

def init_database():
    """Initialize database with default data"""
    try:
        # Create tables
        create_tables()
        
        with get_db_session() as session:
            from database.models import Agent, SystemMetric
            from datetime import datetime
            
            # Check if agents exist
            existing_agents = session.query(Agent).count()
            
            if existing_agents == 0:
                # Create default agents
                default_agents = [
                    {
                        "name": "Research Agent",
                        "type": "research",
                        "description": "Specialized in information gathering and research",
                        "capabilities": ["web_search", "data_gathering", "fact_checking"],
                        "status": "active"
                    },
                    {
                        "name": "Analyst Agent",
                        "type": "analyst",
                        "description": "Specialized in data analysis and insights",
                        "capabilities": ["data_analysis", "pattern_recognition", "insights"],
                        "status": "active"
                    },
                    {
                        "name": "Coding Agent",
                        "type": "coding",
                        "description": "Specialized in code generation and debugging",
                        "capabilities": ["code_generation", "debugging", "testing"],
                        "status": "active"
                    },
                    {
                        "name": "Document Agent",
                        "type": "document",
                        "description": "Specialized in document processing and analysis",
                        "capabilities": ["document_processing", "text_extraction", "summarization"],
                        "status": "active"
                    }
                ]
                
                for agent_data in default_agents:
                    agent = Agent(**agent_data)
                    session.add(agent)
                
                logger.info("Default agents created")
            
            # Create initial system metrics
            initial_metrics = SystemMetric(
                active_agents=4,
                completed_tasks=0,
                failed_tasks=0,
                avg_response_time=0.0,
                memory_usage=0.0,
                cpu_usage=0.0,
                total_documents=0,
                total_chunks=0,
                search_queries=0
            )
            session.add(initial_metrics)
            
            session.commit()
            logger.info("Database initialized successfully")
            
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def check_database_connection() -> bool:
    """Check if database connection is working"""
    try:
        from sqlalchemy import text
        with get_db_session() as session:
            session.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False