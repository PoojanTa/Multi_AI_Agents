from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime

class AgentType(str, Enum):
    RESEARCH = "research"
    ANALYST = "analyst"
    CODING = "coding"
    DOCUMENT = "document"

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentTask(BaseModel):
    id: str
    agent_type: AgentType
    prompt: str
    context: Dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    execution_time: Optional[float] = None

class AgentResponse(BaseModel):
    agent_id: str
    agent_type: AgentType
    response: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    tools_used: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WorkflowStep(BaseModel):
    id: str
    agent_type: AgentType
    prompt: str
    dependencies: List[str] = Field(default_factory=list)
    context_keys: List[str] = Field(default_factory=list)

class Workflow(BaseModel):
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    created_at: datetime = Field(default_factory=datetime.now)

class DocumentChunk(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class RAGQuery(BaseModel):
    query: str
    top_k: int = 5
    filters: Dict[str, Any] = Field(default_factory=dict)

class RAGResult(BaseModel):
    query: str
    results: List[DocumentChunk]
    context: str
    confidence_scores: List[float]

class MCPMessage(BaseModel):
    id: str
    method: str
    params: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

class SystemMetrics(BaseModel):
    active_agents: int
    completed_tasks: int
    failed_tasks: int
    avg_response_time: float
    memory_usage: float
    cpu_usage: float
    timestamp: datetime = Field(default_factory=datetime.now)
