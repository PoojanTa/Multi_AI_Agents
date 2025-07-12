"""Database models for AI Agent Platform"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class Agent(Base):
    """Agent model"""
    __tablename__ = 'agents'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    type = Column(String(50), nullable=False)
    description = Column(Text)
    status = Column(String(20), default='active')
    capabilities = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Performance metrics
    tasks_completed = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    avg_response_time = Column(Float, default=0.0)
    
    # Relationships
    tasks = relationship("Task", back_populates="agent")

class Task(Base):
    """Task model"""
    __tablename__ = 'tasks'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey('agents.id'))
    prompt = Column(Text, nullable=False)
    response = Column(Text)
    status = Column(String(20), default='pending')
    confidence = Column(Float)
    execution_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Context and metadata
    context = Column(JSON)
    task_metadata = Column(JSON)
    tools_used = Column(JSON)
    reasoning = Column(Text)
    
    # Relationships
    agent = relationship("Agent", back_populates="tasks")

class Document(Base):
    """Document model for RAG system"""
    __tablename__ = 'documents'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(500))
    file_size = Column(Integer)
    file_type = Column(String(50))
    
    # Content and processing
    content = Column(Text)
    chunk_count = Column(Integer, default=0)
    word_count = Column(Integer, default=0)
    keywords = Column(JSON)
    
    # Metadata
    upload_date = Column(DateTime, default=datetime.utcnow)
    processed_date = Column(DateTime)
    status = Column(String(20), default='processing')
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document")

class DocumentChunk(Base):
    """Document chunk model"""
    __tablename__ = 'document_chunks'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey('documents.id'))
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    
    # Embedding and search
    embedding = Column(JSON)  # Store as JSON array
    embedding_model = Column(String(100))
    
    # Metadata
    start_index = Column(Integer)
    end_index = Column(Integer)
    chunk_length = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")

class Workflow(Base):
    """Workflow model"""
    __tablename__ = 'workflows'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # Workflow definition
    steps = Column(JSON)  # Store workflow steps as JSON
    status = Column(String(20), default='draft')
    
    # Execution info
    created_at = Column(DateTime, default=datetime.utcnow)
    last_executed = Column(DateTime)
    execution_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    
    # Performance
    avg_execution_time = Column(Float, default=0.0)
    
    # Relationships
    executions = relationship("WorkflowExecution", back_populates="workflow")

class WorkflowExecution(Base):
    """Workflow execution model"""
    __tablename__ = 'workflow_executions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id = Column(String, ForeignKey('workflows.id'))
    
    # Execution details
    status = Column(String(20), default='running')
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    execution_time = Column(Float)
    
    # Results
    results = Column(JSON)
    error_message = Column(Text)
    
    # Step tracking
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer)
    step_results = Column(JSON)
    
    # Relationships
    workflow = relationship("Workflow", back_populates="executions")

class SystemMetric(Base):
    """System metrics model"""
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Agent metrics
    active_agents = Column(Integer, default=0)
    completed_tasks = Column(Integer, default=0)
    failed_tasks = Column(Integer, default=0)
    avg_response_time = Column(Float, default=0.0)
    
    # System metrics
    memory_usage = Column(Float, default=0.0)
    cpu_usage = Column(Float, default=0.0)
    disk_usage = Column(Float, default=0.0)
    
    # RAG metrics
    total_documents = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    search_queries = Column(Integer, default=0)

class User(Base):
    """User model"""
    __tablename__ = 'users'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    
    # Profile
    full_name = Column(String(100))
    avatar_url = Column(String(500))
    
    # Preferences
    preferences = Column(JSON)
    settings = Column(JSON)
    
    # Activity
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Usage statistics
    tasks_created = Column(Integer, default=0)
    documents_uploaded = Column(Integer, default=0)
    workflows_created = Column(Integer, default=0)