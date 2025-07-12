"""Database service layer for AI Agent Platform"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_
from database.connection import get_db_session
from database.models import Agent, Task, Document, DocumentChunk, Workflow, WorkflowExecution, SystemMetric, User

logger = logging.getLogger(__name__)

class DatabaseService:
    """Main database service class"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    # Agent operations
    def create_agent(self, agent_data: Dict[str, Any]) -> Agent:
        """Create a new agent"""
        with get_db_session() as session:
            agent = Agent(**agent_data)
            session.add(agent)
            session.commit()
            session.refresh(agent)
            return agent
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID"""
        with get_db_session() as session:
            return session.query(Agent).filter(Agent.id == agent_id).first()
    
    def get_agents(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all agents, optionally filtered by status"""
        with get_db_session() as session:
            query = session.query(Agent)
            if status:
                query = query.filter(Agent.status == status)
            agents = query.all()
            
            # Convert to dictionaries to avoid session issues
            return [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "type": agent.type,
                    "description": agent.description,
                    "status": agent.status,
                    "capabilities": agent.capabilities,
                    "created_at": agent.created_at,
                    "updated_at": agent.updated_at,
                    "tasks_completed": agent.tasks_completed,
                    "success_rate": agent.success_rate,
                    "avg_response_time": agent.avg_response_time
                }
                for agent in agents
            ]
    
    def update_agent(self, agent_id: str, update_data: Dict[str, Any]) -> Optional[Agent]:
        """Update agent information"""
        with get_db_session() as session:
            agent = session.query(Agent).filter(Agent.id == agent_id).first()
            if agent:
                for key, value in update_data.items():
                    setattr(agent, key, value)
                agent.updated_at = datetime.utcnow()
                session.commit()
                session.refresh(agent)
            return agent
    
    def update_agent_performance(self, agent_id: str, execution_time: float, success: bool):
        """Update agent performance metrics"""
        with get_db_session() as session:
            agent = session.query(Agent).filter(Agent.id == agent_id).first()
            if agent:
                agent.tasks_completed += 1
                
                # Update average response time
                if agent.avg_response_time == 0:
                    agent.avg_response_time = execution_time
                else:
                    agent.avg_response_time = (agent.avg_response_time + execution_time) / 2
                
                # Update success rate
                if success:
                    total_tasks = agent.tasks_completed
                    successful_tasks = int(agent.success_rate * (total_tasks - 1)) + 1
                    agent.success_rate = successful_tasks / total_tasks
                else:
                    total_tasks = agent.tasks_completed
                    successful_tasks = int(agent.success_rate * (total_tasks - 1))
                    agent.success_rate = successful_tasks / total_tasks
                
                session.commit()
    
    # Task operations
    def create_task(self, task_data: Dict[str, Any]) -> Task:
        """Create a new task"""
        with get_db_session() as session:
            task = Task(**task_data)
            session.add(task)
            session.commit()
            session.refresh(task)
            return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        with get_db_session() as session:
            return session.query(Task).filter(Task.id == task_id).first()
    
    def get_tasks(self, agent_id: Optional[str] = None, status: Optional[str] = None, limit: int = 100) -> List[Task]:
        """Get tasks with optional filtering"""
        with get_db_session() as session:
            query = session.query(Task)
            if agent_id:
                query = query.filter(Task.agent_id == agent_id)
            if status:
                query = query.filter(Task.status == status)
            return query.order_by(desc(Task.created_at)).limit(limit).all()
    
    def update_task(self, task_id: str, update_data: Dict[str, Any]) -> Optional[Task]:
        """Update task information"""
        with get_db_session() as session:
            task = session.query(Task).filter(Task.id == task_id).first()
            if task:
                for key, value in update_data.items():
                    setattr(task, key, value)
                if 'status' in update_data and update_data['status'] in ['completed', 'failed']:
                    task.completed_at = datetime.utcnow()
                session.commit()
                session.refresh(task)
            return task
    
    # Document operations
    def create_document(self, document_data: Dict[str, Any]) -> Document:
        """Create a new document"""
        with get_db_session() as session:
            document = Document(**document_data)
            session.add(document)
            session.commit()
            session.refresh(document)
            return document
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID"""
        with get_db_session() as session:
            return session.query(Document).filter(Document.id == document_id).first()
    
    def get_documents(self, status: Optional[str] = None) -> List[Document]:
        """Get all documents"""
        with get_db_session() as session:
            query = session.query(Document)
            if status:
                query = query.filter(Document.status == status)
            return query.order_by(desc(Document.upload_date)).all()
    
    def delete_document(self, document_id: str) -> bool:
        """Delete document and its chunks"""
        with get_db_session() as session:
            # Delete chunks first
            session.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
            
            # Delete document
            document = session.query(Document).filter(Document.id == document_id).first()
            if document:
                session.delete(document)
                session.commit()
                return True
            return False
    
    # Document chunk operations
    def create_document_chunk(self, chunk_data: Dict[str, Any]) -> DocumentChunk:
        """Create a new document chunk"""
        with get_db_session() as session:
            chunk = DocumentChunk(**chunk_data)
            session.add(chunk)
            session.commit()
            session.refresh(chunk)
            return chunk
    
    def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document"""
        with get_db_session() as session:
            return session.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).order_by(DocumentChunk.chunk_index).all()
    
    def search_document_chunks(self, query: str, limit: int = 10) -> List[DocumentChunk]:
        """Search document chunks by content"""
        with get_db_session() as session:
            # Simple text search - in production, use full-text search or vector similarity
            return session.query(DocumentChunk).filter(
                DocumentChunk.content.ilike(f'%{query}%')
            ).limit(limit).all()
    
    # Workflow operations
    def create_workflow(self, workflow_data: Dict[str, Any]) -> Workflow:
        """Create a new workflow"""
        with get_db_session() as session:
            workflow = Workflow(**workflow_data)
            session.add(workflow)
            session.commit()
            session.refresh(workflow)
            return workflow
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID"""
        with get_db_session() as session:
            return session.query(Workflow).filter(Workflow.id == workflow_id).first()
    
    def get_workflows(self, status: Optional[str] = None) -> List[Workflow]:
        """Get all workflows"""
        with get_db_session() as session:
            query = session.query(Workflow)
            if status:
                query = query.filter(Workflow.status == status)
            return query.order_by(desc(Workflow.created_at)).all()
    
    def create_workflow_execution(self, execution_data: Dict[str, Any]) -> WorkflowExecution:
        """Create a new workflow execution"""
        with get_db_session() as session:
            execution = WorkflowExecution(**execution_data)
            session.add(execution)
            session.commit()
            session.refresh(execution)
            return execution
    
    def update_workflow_execution(self, execution_id: str, update_data: Dict[str, Any]) -> Optional[WorkflowExecution]:
        """Update workflow execution"""
        with get_db_session() as session:
            execution = session.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()
            if execution:
                for key, value in update_data.items():
                    setattr(execution, key, value)
                session.commit()
                session.refresh(execution)
            return execution
    
    # System metrics operations
    def create_system_metric(self, metric_data: Dict[str, Any]) -> SystemMetric:
        """Create a new system metric entry"""
        with get_db_session() as session:
            metric = SystemMetric(**metric_data)
            session.add(metric)
            session.commit()
            session.refresh(metric)
            return metric
    
    def get_latest_system_metrics(self) -> Optional[SystemMetric]:
        """Get the latest system metrics"""
        with get_db_session() as session:
            return session.query(SystemMetric).order_by(desc(SystemMetric.timestamp)).first()
    
    def get_system_metrics_history(self, hours: int = 24) -> List[SystemMetric]:
        """Get system metrics history"""
        with get_db_session() as session:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            return session.query(SystemMetric).filter(
                SystemMetric.timestamp >= cutoff_time
            ).order_by(SystemMetric.timestamp).all()
    
    # Analytics and reporting
    def get_agent_performance_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        with get_db_session() as session:
            agents = session.query(Agent).all()
            
            stats = {
                'total_agents': len(agents),
                'active_agents': len([a for a in agents if a.status == 'active']),
                'total_tasks': sum(a.tasks_completed for a in agents),
                'avg_response_time': sum(a.avg_response_time for a in agents) / len(agents) if agents else 0,
                'avg_success_rate': sum(a.success_rate for a in agents) / len(agents) if agents else 0,
                'agent_details': [
                    {
                        'id': a.id,
                        'name': a.name,
                        'type': a.type,
                        'status': a.status,
                        'tasks_completed': a.tasks_completed,
                        'success_rate': a.success_rate,
                        'avg_response_time': a.avg_response_time
                    }
                    for a in agents
                ]
            }
            
            return stats
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get document statistics"""
        with get_db_session() as session:
            total_docs = session.query(Document).count()
            total_chunks = session.query(DocumentChunk).count()
            
            # Get document type distribution
            doc_types = session.query(
                Document.file_type,
                func.count(Document.id).label('count')
            ).group_by(Document.file_type).all()
            
            return {
                'total_documents': total_docs,
                'total_chunks': total_chunks,
                'avg_chunks_per_doc': total_chunks / total_docs if total_docs > 0 else 0,
                'document_types': [{'type': dt[0], 'count': dt[1]} for dt in doc_types]
            }
    
    def get_task_stats(self) -> Dict[str, Any]:
        """Get task statistics"""
        with get_db_session() as session:
            total_tasks = session.query(Task).count()
            completed_tasks = session.query(Task).filter(Task.status == 'completed').count()
            failed_tasks = session.query(Task).filter(Task.status == 'failed').count()
            
            return {
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'success_rate': completed_tasks / total_tasks if total_tasks > 0 else 0
            }

# Global database service instance
db_service = DatabaseService()