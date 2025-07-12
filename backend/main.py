import asyncio
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import os
import tempfile

# Import models and services
from backend.models.schemas import (
    AgentTask, AgentResponse, AgentType, TaskStatus, 
    RAGQuery, RAGResult, Workflow, SystemMetrics
)
from backend.services.orchestrator import AgentOrchestrator
from backend.services.rag_service import RAGService
from backend.services.mcp_server import MCPServer
from backend.services.groq_client import GroqClient
from config import Config

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Agent Platform API",
    description="Multi-Agent Orchestration with RAG and MCP Integration",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
orchestrator = AgentOrchestrator()
rag_service = RAGService()
mcp_server = MCPServer()
groq_client = GroqClient()

# System metrics storage
system_metrics = {
    "active_agents": 0,
    "completed_tasks": 0,
    "failed_tasks": 0,
    "avg_response_time": 0.0,
    "total_requests": 0,
    "uptime_start": datetime.now()
}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        logger.info("Starting AI Agent Platform API...")
        
        # Initialize orchestrator
        await orchestrator.initialize()
        
        # Start MCP server in background
        asyncio.create_task(mcp_server.start())
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        logger.info("Shutting down AI Agent Platform API...")
        
        # Stop MCP server
        await mcp_server.stop()
        
        # Cleanup orchestrator
        await orchestrator.cleanup()
        
        # Close Groq client
        await groq_client.close()
        
        logger.info("Shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "orchestrator": orchestrator.is_initialized,
            "rag_service": True,
            "mcp_server": mcp_server.running,
            "groq_client": True
        }
    }

# System metrics endpoint
@app.get("/metrics")
async def get_system_metrics():
    """Get system metrics"""
    uptime = datetime.now() - system_metrics["uptime_start"]
    
    return {
        **system_metrics,
        "uptime_seconds": uptime.total_seconds(),
        "agent_count": len(orchestrator.agents),
        "memory_usage": 0.0,  # Could be implemented with psutil
        "cpu_usage": 0.0,
        "timestamp": datetime.now().isoformat()
    }

# Agent management endpoints
@app.get("/agents/status")
async def get_agents_status():
    """Get status of all agents"""
    try:
        agents_info = []
        for agent in orchestrator.agents.values():
            info = agent.get_agent_info()
            metrics = agent.get_performance_metrics()
            info.update(metrics)
            agents_info.append(info)
        
        return agents_info
        
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/{agent_id}/info")
async def get_agent_info(agent_id: str):
    """Get detailed information about a specific agent"""
    try:
        agent = orchestrator.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {
            "info": agent.get_agent_info(),
            "metrics": agent.get_performance_metrics(),
            "capabilities": agent.get_capabilities()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/task")
async def create_agent_task(
    agent_type: AgentType,
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    background_tasks: BackgroundTasks = None
):
    """Create a new agent task"""
    try:
        # Create task
        task = AgentTask(
            id=str(uuid.uuid4()),
            agent_type=agent_type,
            prompt=prompt,
            context=context or {}
        )
        
        # Execute task
        start_time = datetime.now()
        response = await orchestrator.execute_task(task)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Update metrics
        system_metrics["total_requests"] += 1
        if response.confidence > 0.5:
            system_metrics["completed_tasks"] += 1
        else:
            system_metrics["failed_tasks"] += 1
        
        # Update average response time
        total_time = system_metrics["avg_response_time"] * (system_metrics["total_requests"] - 1)
        system_metrics["avg_response_time"] = (total_time + execution_time) / system_metrics["total_requests"]
        
        return {
            "task_id": task.id,
            "response": response.dict(),
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating agent task: {e}")
        system_metrics["failed_tasks"] += 1
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/workflow")
async def execute_workflow(workflow: Workflow):
    """Execute a multi-step workflow"""
    try:
        result = await orchestrator.execute_workflow(workflow)
        return result
        
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# RAG endpoints
@app.post("/rag/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the RAG system"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Add document to RAG system
        result = await rag_service.add_document(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/search")
async def search_documents(query: RAGQuery):
    """Search documents in the RAG system"""
    try:
        result = await rag_service.search_documents(query)
        return result.dict()
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/documents")
async def list_documents():
    """List all documents in the RAG system"""
    try:
        documents = await rag_service.list_documents()
        return documents
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/rag/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the RAG system"""
    try:
        result = await rag_service.delete_document(document_id)
        return result
        
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/stats")
async def get_rag_stats():
    """Get RAG system statistics"""
    try:
        stats = await rag_service.get_collection_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# MCP endpoints
@app.get("/mcp/status")
async def get_mcp_status():
    """Get MCP server status"""
    try:
        return mcp_server.get_server_status()
        
    except Exception as e:
        logger.error(f"Error getting MCP status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp/broadcast")
async def broadcast_message(message: Dict[str, Any]):
    """Broadcast message to all MCP clients"""
    try:
        await mcp_server.broadcast_message(message)
        return {"status": "success", "message": "Message broadcasted"}
        
    except Exception as e:
        logger.error(f"Error broadcasting message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Groq API endpoints
@app.post("/groq/completion")
async def groq_completion(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 1024
):
    """Direct Groq API completion"""
    try:
        result = await groq_client.generate_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return result
        
    except Exception as e:
        logger.error(f"Error with Groq completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/groq/sentiment")
async def analyze_sentiment(text: str):
    """Analyze sentiment using Groq"""
    try:
        result = await groq_client.analyze_sentiment(text)
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/groq/summarize")
async def summarize_text(text: str, max_length: int = 200):
    """Summarize text using Groq"""
    try:
        result = await groq_client.summarize_text(text, max_length)
        return {"summary": result}
        
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Specialized agent endpoints
@app.post("/agents/research")
async def research_task(prompt: str, context: Optional[Dict[str, Any]] = None):
    """Execute research task"""
    try:
        task = AgentTask(
            id=str(uuid.uuid4()),
            agent_type=AgentType.RESEARCH,
            prompt=prompt,
            context=context or {}
        )
        
        response = await orchestrator.execute_task(task)
        return response.dict()
        
    except Exception as e:
        logger.error(f"Error in research task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/analysis")
async def analysis_task(prompt: str, context: Optional[Dict[str, Any]] = None):
    """Execute analysis task"""
    try:
        task = AgentTask(
            id=str(uuid.uuid4()),
            agent_type=AgentType.ANALYST,
            prompt=prompt,
            context=context or {}
        )
        
        response = await orchestrator.execute_task(task)
        return response.dict()
        
    except Exception as e:
        logger.error(f"Error in analysis task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/coding")
async def coding_task(prompt: str, context: Optional[Dict[str, Any]] = None):
    """Execute coding task"""
    try:
        task = AgentTask(
            id=str(uuid.uuid4()),
            agent_type=AgentType.CODING,
            prompt=prompt,
            context=context or {}
        )
        
        response = await orchestrator.execute_task(task)
        return response.dict()
        
    except Exception as e:
        logger.error(f"Error in coding task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/document")
async def document_task(prompt: str, context: Optional[Dict[str, Any]] = None):
    """Execute document processing task"""
    try:
        task = AgentTask(
            id=str(uuid.uuid4()),
            agent_type=AgentType.DOCUMENT,
            prompt=prompt,
            context=context or {}
        )
        
        response = await orchestrator.execute_task(task)
        return response.dict()
        
    except Exception as e:
        logger.error(f"Error in document task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=Config.BACKEND_HOST,
        port=Config.BACKEND_PORT,
        reload=True,
        log_level=Config.LOG_LEVEL.lower()
    )
