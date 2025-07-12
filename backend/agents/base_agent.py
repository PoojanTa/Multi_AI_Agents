import uuid
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from abc import ABC, abstractmethod
from backend.models.schemas import AgentTask, AgentResponse, AgentType, TaskStatus
from backend.services.groq_client import GroqClient

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all AI agents"""
    
    def __init__(self, agent_type: AgentType, name: str, description: str):
        self.agent_id = str(uuid.uuid4())
        self.agent_type = agent_type
        self.name = name
        self.description = description
        self.groq_client = GroqClient()
        self.is_busy = False
        self.task_history: List[AgentTask] = []
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        
    @abstractmethod
    async def execute_task(self, task: AgentTask) -> AgentResponse:
        """Execute a task and return response"""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities"""
        pass
    
    async def process_task(self, task: AgentTask) -> AgentResponse:
        """Process a task with error handling and logging"""
        if self.is_busy:
            return AgentResponse(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                response="Agent is currently busy with another task",
                confidence=0.0,
                reasoning="Agent unavailable"
            )
        
        try:
            self.is_busy = True
            self.last_active = datetime.now()
            
            # Update task status
            task.status = TaskStatus.RUNNING
            task.updated_at = datetime.now()
            
            # Record start time
            start_time = datetime.now()
            
            # Execute the task
            response = await self.execute_task(task)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update task with results
            task.status = TaskStatus.COMPLETED
            task.result = response.response
            task.execution_time = execution_time
            task.updated_at = datetime.now()
            
            # Add to history
            self.task_history.append(task)
            
            # Update response metadata
            response.metadata.update({
                "execution_time": execution_time,
                "task_id": task.id,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"Agent {self.name} completed task {task.id} in {execution_time:.2f}s")
            
            return response
            
        except Exception as e:
            # Handle errors
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.updated_at = datetime.now()
            
            logger.error(f"Agent {self.name} failed task {task.id}: {str(e)}")
            
            return AgentResponse(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                response=f"Task failed: {str(e)}",
                confidence=0.0,
                reasoning=f"Error occurred: {str(e)}",
                metadata={"error": str(e), "task_id": task.id}
            )
        finally:
            self.is_busy = False
    
    async def generate_response(
        self,
        prompt: str,
        model: str = "mixtral-8x7b-32768",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        include_system_prompt: bool = True
    ) -> Dict[str, Any]:
        """Generate response using Groq API"""
        try:
            messages = []
            
            if include_system_prompt:
                messages.append({
                    "role": "system",
                    "content": self.get_system_prompt()
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            return await self.groq_client.generate_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "success": False,
                "error": str(e),
                "content": ""
            }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "name": self.name,
            "description": self.description,
            "capabilities": self.get_capabilities(),
            "is_busy": self.is_busy,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "task_count": len(self.task_history),
            "successful_tasks": len([t for t in self.task_history if t.status == TaskStatus.COMPLETED]),
            "failed_tasks": len([t for t in self.task_history if t.status == TaskStatus.FAILED])
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        completed_tasks = [t for t in self.task_history if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in self.task_history if t.status == TaskStatus.FAILED]
        
        if not self.task_history:
            return {
                "total_tasks": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "total_execution_time": 0.0
            }
        
        total_execution_time = sum(t.execution_time for t in completed_tasks if t.execution_time)
        avg_execution_time = total_execution_time / len(completed_tasks) if completed_tasks else 0
        
        return {
            "total_tasks": len(self.task_history),
            "completed_tasks": len(completed_tasks),
            "failed_tasks": len(failed_tasks),
            "success_rate": len(completed_tasks) / len(self.task_history) if self.task_history else 0,
            "average_execution_time": avg_execution_time,
            "total_execution_time": total_execution_time
        }
    
    async def cleanup(self):
        """Cleanup agent resources"""
        await self.groq_client.close()
        logger.info(f"Agent {self.name} cleanup completed")
