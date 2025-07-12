import asyncio
import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from backend.models.schemas import (
    AgentTask, AgentResponse, AgentType, TaskStatus, 
    Workflow, WorkflowStep
)
from backend.agents.research_agent import ResearchAgent
from backend.agents.analyst_agent import AnalystAgent
from backend.agents.coding_agent import CodingAgent
from backend.agents.document_agent import DocumentAgent
from config import Config

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """Orchestrates multiple AI agents for complex task execution"""
    
    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.agent_types: Dict[AgentType, List[str]] = {
            AgentType.RESEARCH: [],
            AgentType.ANALYST: [],
            AgentType.CODING: [],
            AgentType.DOCUMENT: []
        }
        self.active_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: List[AgentTask] = []
        self.workflows: Dict[str, Workflow] = {}
        self.is_initialized = False
        self.task_queue = asyncio.Queue()
        self.max_concurrent_tasks = Config.MAX_AGENTS
        self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
    async def initialize(self):
        """Initialize the orchestrator and create agents"""
        try:
            logger.info("Initializing Agent Orchestrator...")
            
            # Create default agents
            await self._create_default_agents()
            
            # Start task processing
            asyncio.create_task(self._process_task_queue())
            
            self.is_initialized = True
            logger.info(f"Agent Orchestrator initialized with {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize Agent Orchestrator: {e}")
            raise
    
    async def _create_default_agents(self):
        """Create default set of agents"""
        try:
            # Create research agents
            for i in range(2):
                agent = ResearchAgent()
                self.agents[agent.agent_id] = agent
                self.agent_types[AgentType.RESEARCH].append(agent.agent_id)
            
            # Create analyst agents
            for i in range(2):
                agent = AnalystAgent()
                self.agents[agent.agent_id] = agent
                self.agent_types[AgentType.ANALYST].append(agent.agent_id)
            
            # Create coding agents
            for i in range(2):
                agent = CodingAgent()
                self.agents[agent.agent_id] = agent
                self.agent_types[AgentType.CODING].append(agent.agent_id)
            
            # Create document agents
            for i in range(2):
                agent = DocumentAgent()
                self.agents[agent.agent_id] = agent
                self.agent_types[AgentType.DOCUMENT].append(agent.agent_id)
            
            logger.info("Default agents created successfully")
            
        except Exception as e:
            logger.error(f"Error creating default agents: {e}")
            raise
    
    async def _process_task_queue(self):
        """Process tasks from the queue"""
        while True:
            try:
                task = await self.task_queue.get()
                async with self.semaphore:
                    await self._execute_single_task(task)
                self.task_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing task queue: {e}")
                await asyncio.sleep(1)
    
    async def _execute_single_task(self, task: AgentTask):
        """Execute a single task"""
        try:
            # Find available agent
            agent = self._get_available_agent(task.agent_type)
            
            if not agent:
                task.status = TaskStatus.FAILED
                task.error = f"No available {task.agent_type} agent"
                task.updated_at = datetime.now()
                return
            
            # Execute task
            self.active_tasks[task.id] = task
            response = await agent.process_task(task)
            
            # Update task status
            task.status = TaskStatus.COMPLETED if response.confidence > 0.5 else TaskStatus.FAILED
            task.result = response.response
            task.updated_at = datetime.now()
            
            # Move to completed tasks
            self.completed_tasks.append(task)
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
            
            logger.info(f"Task {task.id} completed by agent {agent.name}")
            
        except Exception as e:
            logger.error(f"Error executing task {task.id}: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.updated_at = datetime.now()
    
    def _get_available_agent(self, agent_type: AgentType):
        """Get an available agent of the specified type"""
        agent_ids = self.agent_types.get(agent_type, [])
        
        for agent_id in agent_ids:
            agent = self.agents.get(agent_id)
            if agent and not agent.is_busy:
                return agent
        
        return None
    
    async def execute_task(self, task: AgentTask) -> AgentResponse:
        """Execute a single task"""
        try:
            agent = self._get_available_agent(task.agent_type)
            
            if not agent:
                return AgentResponse(
                    agent_id="orchestrator",
                    agent_type=task.agent_type,
                    response=f"No available {task.agent_type} agent",
                    confidence=0.0,
                    reasoning="No agent available"
                )
            
            # Execute task directly
            response = await agent.process_task(task)
            
            # Record task
            self.completed_tasks.append(task)
            
            return response
            
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            return AgentResponse(
                agent_id="orchestrator",
                agent_type=task.agent_type,
                response=f"Task execution failed: {str(e)}",
                confidence=0.0,
                reasoning=f"Error: {str(e)}"
            )
    
    async def execute_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """Execute a multi-step workflow"""
        try:
            logger.info(f"Starting workflow: {workflow.name}")
            
            workflow_id = workflow.id
            self.workflows[workflow_id] = workflow
            
            # Initialize workflow context
            context = {}
            results = {}
            
            # Execute steps in order, considering dependencies
            for step in workflow.steps:
                logger.info(f"Executing workflow step: {step.id}")
                
                # Wait for dependencies
                await self._wait_for_dependencies(step.dependencies, results)
                
                # Prepare step context
                step_context = {}
                for key in step.context_keys:
                    if key in context:
                        step_context[key] = context[key]
                
                # Create and execute task
                task = AgentTask(
                    id=str(uuid.uuid4()),
                    agent_type=step.agent_type,
                    prompt=step.prompt,
                    context=step_context
                )
                
                response = await self.execute_task(task)
                
                # Store result
                results[step.id] = {
                    "response": response.response,
                    "confidence": response.confidence,
                    "reasoning": response.reasoning,
                    "metadata": response.metadata
                }
                
                # Update context
                context[step.id] = response.response
                
                logger.info(f"Workflow step {step.id} completed")
            
            # Generate workflow summary
            summary = await self._generate_workflow_summary(workflow, results)
            
            logger.info(f"Workflow {workflow.name} completed successfully")
            
            return {
                "workflow_id": workflow_id,
                "workflow_name": workflow.name,
                "status": "completed",
                "results": results,
                "summary": summary,
                "total_steps": len(workflow.steps),
                "completed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            return {
                "workflow_id": workflow_id,
                "workflow_name": workflow.name,
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now().isoformat()
            }
    
    async def _wait_for_dependencies(self, dependencies: List[str], results: Dict[str, Any]):
        """Wait for workflow step dependencies to complete"""
        for dep_id in dependencies:
            while dep_id not in results:
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
    
    async def _generate_workflow_summary(self, workflow: Workflow, results: Dict[str, Any]) -> str:
        """Generate a summary of workflow execution"""
        try:
            # Create summary prompt
            summary_prompt = f"""
            Generate a comprehensive summary of this workflow execution:
            
            Workflow: {workflow.name}
            Description: {workflow.description}
            
            Step Results:
            {json.dumps(results, indent=2)}
            
            Provide:
            1. Executive Summary
            2. Key Achievements
            3. Important Insights
            4. Recommendations
            5. Next Steps
            
            Make it concise but comprehensive.
            """
            
            # Use a research agent to generate summary
            research_agent = self._get_available_agent(AgentType.RESEARCH)
            if research_agent:
                task = AgentTask(
                    id=str(uuid.uuid4()),
                    agent_type=AgentType.RESEARCH,
                    prompt=summary_prompt
                )
                
                response = await research_agent.process_task(task)
                return response.response
            else:
                return "Workflow completed successfully but summary generation unavailable"
                
        except Exception as e:
            logger.error(f"Error generating workflow summary: {e}")
            return f"Workflow completed but summary generation failed: {str(e)}"
    
    async def create_collaborative_workflow(self, objective: str, agent_types: List[AgentType]) -> Workflow:
        """Create a collaborative workflow involving multiple agent types"""
        try:
            workflow_id = str(uuid.uuid4())
            
            # Generate workflow steps based on objective and agent types
            steps = []
            
            # Research phase
            if AgentType.RESEARCH in agent_types:
                steps.append(WorkflowStep(
                    id="research_phase",
                    agent_type=AgentType.RESEARCH,
                    prompt=f"Conduct comprehensive research on: {objective}",
                    dependencies=[],
                    context_keys=[]
                ))
            
            # Analysis phase
            if AgentType.ANALYST in agent_types:
                deps = ["research_phase"] if AgentType.RESEARCH in agent_types else []
                context_keys = ["research_phase"] if AgentType.RESEARCH in agent_types else []
                
                steps.append(WorkflowStep(
                    id="analysis_phase",
                    agent_type=AgentType.ANALYST,
                    prompt=f"Analyze the information and provide insights for: {objective}",
                    dependencies=deps,
                    context_keys=context_keys
                ))
            
            # Coding phase
            if AgentType.CODING in agent_types:
                deps = []
                context_keys = []
                
                if AgentType.RESEARCH in agent_types:
                    deps.append("research_phase")
                    context_keys.append("research_phase")
                
                if AgentType.ANALYST in agent_types:
                    deps.append("analysis_phase")
                    context_keys.append("analysis_phase")
                
                steps.append(WorkflowStep(
                    id="coding_phase",
                    agent_type=AgentType.CODING,
                    prompt=f"Generate code solution for: {objective}",
                    dependencies=deps,
                    context_keys=context_keys
                ))
            
            # Documentation phase
            if AgentType.DOCUMENT in agent_types:
                deps = []
                context_keys = []
                
                for phase in ["research_phase", "analysis_phase", "coding_phase"]:
                    if any(step.id == phase for step in steps):
                        deps.append(phase)
                        context_keys.append(phase)
                
                steps.append(WorkflowStep(
                    id="documentation_phase",
                    agent_type=AgentType.DOCUMENT,
                    prompt=f"Create comprehensive documentation for: {objective}",
                    dependencies=deps,
                    context_keys=context_keys
                ))
            
            workflow = Workflow(
                id=workflow_id,
                name=f"Collaborative Workflow: {objective}",
                description=f"Multi-agent collaborative workflow for {objective}",
                steps=steps
            )
            
            return workflow
            
        except Exception as e:
            logger.error(f"Error creating collaborative workflow: {e}")
            raise
    
    def get_agent(self, agent_id: str):
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[Any]:
        """Get all agents of a specific type"""
        agent_ids = self.agent_types.get(agent_type, [])
        return [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "is_initialized": self.is_initialized,
            "total_agents": len(self.agents),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "agent_types": {
                agent_type.value: len(agent_ids) 
                for agent_type, agent_ids in self.agent_types.items()
            },
            "workflows": len(self.workflows)
        }
    
    async def cleanup(self):
        """Cleanup orchestrator resources"""
        try:
            logger.info("Cleaning up Agent Orchestrator...")
            
            # Cancel active tasks
            for task in self.active_tasks.values():
                task.status = TaskStatus.CANCELLED
                task.updated_at = datetime.now()
            
            # Cleanup agents
            for agent in self.agents.values():
                await agent.cleanup()
            
            self.is_initialized = False
            logger.info("Agent Orchestrator cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
