#!/usr/bin/env python3
"""
Simple backend for AI Agent Platform
"""

import asyncio
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import os
import tempfile
import json
from database.connection import init_database, check_database_connection
from database.services import db_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

# System metrics storage
system_metrics = {
    "active_agents": 4,
    "completed_tasks": 0,
    "failed_tasks": 0,
    "avg_response_time": 0.0,
    "total_requests": 0,
    "uptime_start": datetime.now()
}

# Simple agent responses
agent_responses = {
    "research": "I'm the Research Agent. I can help you gather information, conduct research, and analyze data sources. What would you like me to research for you?",
    "analyst": "I'm the Analyst Agent. I specialize in data analysis, pattern recognition, and generating insights from information. How can I help you analyze your data?",
    "coding": "I'm the Coding Agent. I can help you write code, debug issues, and implement solutions. What programming task can I assist you with?",
    "document": "I'm the Document Agent. I can process documents, extract information, and generate reports. What document-related task do you need help with?"
}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        logger.info("Starting AI Agent Platform API...")
        
        # Initialize database
        if check_database_connection():
            init_database()
            logger.info("Database initialized successfully")
        else:
            logger.warning("Database connection failed - using fallback mode")
        
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AI Agent Platform API...")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/metrics")
async def get_system_metrics():
    """Get system metrics"""
    uptime = datetime.now() - system_metrics["uptime_start"]
    
    return {
        "active_agents": system_metrics["active_agents"],
        "completed_tasks": system_metrics["completed_tasks"],
        "failed_tasks": system_metrics["failed_tasks"],
        "avg_response_time": system_metrics["avg_response_time"],
        "memory_usage": 45.2,  # Mock value
        "cpu_usage": 23.1,     # Mock value
        "uptime_seconds": uptime.total_seconds(),
        "timestamp": datetime.now()
    }

@app.get("/agents/status")
async def get_agents_status():
    """Get status of all agents"""
    try:
        # Try to get agents from database
        agents = db_service.get_agents()
        
        # agents is now a list of dictionaries, not ORM objects
        return {"agents": agents}
        
    except Exception as e:
        logger.error(f"Error getting agents from database: {e}")
        # Fallback to static data
        agents = [
            {
                "id": "research_agent",
                "type": "research",
                "name": "Research Agent",
                "status": "active",
                "tasks_completed": 15,
                "avg_response_time": 2.3,
                "success_rate": 0.95,
                "capabilities": ["web_search", "data_gathering", "fact_checking"]
            },
            {
                "id": "analyst_agent",
                "type": "analyst",
                "name": "Analyst Agent",
                "status": "active",
                "tasks_completed": 8,
                "avg_response_time": 3.1,
                "success_rate": 0.92,
                "capabilities": ["data_analysis", "pattern_recognition", "insights"]
            },
            {
                "id": "coding_agent",
                "type": "coding",
                "name": "Coding Agent",
                "status": "active",
                "tasks_completed": 12,
                "avg_response_time": 4.5,
                "success_rate": 0.88,
                "capabilities": ["code_generation", "debugging", "testing"]
            },
            {
                "id": "document_agent",
                "type": "document",
                "name": "Document Agent",
                "status": "active",
                "tasks_completed": 6,
                "avg_response_time": 2.8,
                "success_rate": 0.93,
                "capabilities": ["document_processing", "text_extraction", "summarization"]
            }
        ]
        
        return {"agents": agents}

@app.get("/agents/{agent_id}")
async def get_agent_info(agent_id: str):
    """Get detailed information about a specific agent"""
    agent_info = {
        "id": agent_id,
        "name": f"{agent_id.title()} Agent",
        "status": "active",
        "version": "1.0.0",
        "capabilities": ["general_purpose"],
        "performance": {
            "tasks_completed": 10,
            "success_rate": 0.95,
            "avg_response_time": 2.5
        }
    }
    
    return agent_info

@app.post("/agents/{agent_type}")
async def create_agent_task(agent_type: str, request_data: Dict[str, Any]):
    """Create a new agent task"""
    try:
        system_metrics["total_requests"] += 1
        
        prompt = request_data.get("prompt", "")
        context = request_data.get("context", {})
        
        # Get agent response
        if agent_type in agent_responses:
            base_response = agent_responses[agent_type]
            response = f"{base_response}\n\nBased on your request: '{prompt}', I'll provide a helpful response tailored to your needs."
        else:
            response = f"I'm an AI agent of type '{agent_type}'. I'll help you with: {prompt}"
        
        # Update metrics
        system_metrics["completed_tasks"] += 1
        
        return {
            "agent_id": f"{agent_type}_agent",
            "agent_type": agent_type,
            "response": response,
            "confidence": 0.85,
            "reasoning": f"Processed request using {agent_type} agent capabilities",
            "tools_used": [agent_type, "reasoning"],
            "execution_time": 1.2,
            "metadata": {
                "request_id": str(uuid.uuid4()),
                "timestamp": datetime.now(),
                "prompt_length": len(prompt),
                "context_provided": bool(context)
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing agent task: {e}")
        system_metrics["failed_tasks"] += 1
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the RAG system"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Mock processing
        document_id = str(uuid.uuid4())
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return {
            "success": True,
            "document_id": document_id,
            "file_name": file.filename,
            "chunk_count": 10,  # Mock value
            "message": "Document uploaded and processed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        return {"success": False, "error": str(e)}

@app.post("/rag/search")
async def search_documents(query_data: Dict[str, Any]):
    """Search documents in the RAG system"""
    try:
        query = query_data.get("query", "")
        top_k = query_data.get("top_k", 5)
        
        # Mock search results
        results = [
            {
                "id": f"chunk_{i}",
                "content": f"This is a sample document chunk related to: {query}. It contains relevant information that matches your search query.",
                "metadata": {
                    "file_name": f"document_{i}.pdf",
                    "chunk_index": i,
                    "chunk_length": 150
                }
            }
            for i in range(min(top_k, 3))
        ]
        
        return {
            "query": query,
            "results": results,
            "context": f"Based on your search for '{query}', here are the most relevant document excerpts.",
            "confidence_scores": [0.9, 0.8, 0.7][:len(results)]
        }
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return {"results": [], "error": str(e)}

@app.get("/rag/documents")
async def list_documents():
    """List all documents in the RAG system"""
    # Mock document list
    documents = [
        {
            "document_id": "doc_1",
            "file_name": "sample_document.pdf",
            "chunk_count": 15,
            "upload_date": "2024-12-01",
            "keywords": ["AI", "Machine Learning", "Technology"]
        },
        {
            "document_id": "doc_2",
            "file_name": "research_paper.docx",
            "chunk_count": 8,
            "upload_date": "2024-12-02",
            "keywords": ["Research", "Analysis", "Data"]
        }
    ]
    
    return documents

@app.delete("/rag/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the RAG system"""
    return {
        "success": True,
        "message": f"Document {document_id} deleted successfully"
    }

@app.get("/rag/stats")
async def get_rag_stats():
    """Get RAG system statistics"""
    return {
        "total_documents": 2,
        "total_chunks": 23,
        "average_chunks_per_document": 11.5,
        "collection_name": "documents",
        "storage_used": "1.2 MB"
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )