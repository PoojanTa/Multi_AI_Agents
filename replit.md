# AI Agent Platform - System Architecture

## Overview

This is a comprehensive multi-agent AI platform built with a FastAPI backend and Streamlit frontend. The system orchestrates multiple specialized AI agents (research, analyst, coding, document processing) with RAG (Retrieval-Augmented Generation) capabilities, workflow management, and PostgreSQL database integration. The platform provides a complete enterprise-ready solution for AI agent collaboration.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes (December 2024)

✓ **Database Integration**: Added PostgreSQL database with SQLAlchemy ORM
✓ **Database Models**: Created comprehensive models for agents, tasks, documents, workflows, metrics, and users
✓ **Database Services**: Built service layer for database operations
✓ **Backend Enhancement**: Updated backend to use real database data
✓ **Fixed Dependencies**: Resolved plotly, graphviz, and other missing dependencies
✓ **Error Resolution**: Fixed frontend display errors and SQLAlchemy model conflicts

## System Architecture

### Backend Architecture
- **Framework**: FastAPI with async/await support
- **Design Pattern**: Service-oriented architecture with agent orchestration
- **API Structure**: RESTful endpoints for agent management, RAG operations, and workflow execution
- **Concurrency**: Asyncio-based with semaphore controls for task management
- **Error Handling**: Comprehensive exception handling with logging throughout

### Frontend Architecture
- **Framework**: Streamlit for rapid prototyping and interactive dashboards
- **Component Structure**: Modular components for chat interface, RAG management, agent monitoring, and workflow building
- **State Management**: Streamlit session state for maintaining user context
- **Real-time Updates**: Request-based communication with backend services

### Agent System
- **Base Agent Pattern**: Abstract base class with common functionality
- **Specialized Agents**: Research, Analyst, Coding, and Document processing agents
- **Agent Orchestration**: Centralized orchestrator managing agent lifecycle and task distribution
- **Task Queue**: Async queue system for handling concurrent agent tasks

## Key Components

### 1. Agent Orchestrator (`backend/services/orchestrator.py`)
- **Purpose**: Manages multiple AI agents and coordinates complex multi-step tasks
- **Features**: Agent lifecycle management, task queuing, workflow execution
- **Concurrency**: Semaphore-based limiting of concurrent tasks
- **Agent Types**: Research, Analyst, Coding, Document processing agents

### 2. Database Layer (`database/`)
- **Models**: SQLAlchemy models for agents, tasks, documents, workflows, metrics, users
- **Connection**: PostgreSQL connection management with session handling
- **Services**: Database service layer with CRUD operations and analytics
- **Migration**: Automatic table creation and default data initialization

### 3. RAG Service (`backend/services/rag_service.py` & `simple_rag.py`)
- **Purpose**: Handles document ingestion, vectorization, and retrieval
- **Storage**: File-based storage with JSON for simple deployment
- **Document Processing**: Support for PDF, DOCX, and text files
- **Embedding Service**: Simple hash-based embeddings for lightweight operation
- **Search**: Similarity search with configurable top-k results

### 3. Groq Client (`backend/services/groq_client.py`)
- **Purpose**: Interface to Groq's LLM API
- **Models**: Support for Mixtral, Llama 2, and Gemma models
- **Features**: Async HTTP client, error handling, retry logic
- **Configuration**: Temperature, max tokens, and streaming support

### 4. MCP Server (`backend/services/mcp_server.py`)
- **Purpose**: Model Context Protocol server implementation
- **Communication**: WebSocket-based real-time communication
- **Client Management**: Multiple client support with connection tracking
- **Integration**: Direct integration with Groq client for model access

### 5. Frontend Components
- **Chat Interface**: Interactive chat with agent selection and context management
- **RAG Interface**: Document upload, search, and management
- **Agent Monitor**: Real-time agent status and performance metrics
- **Workflow Builder**: Visual workflow creation and execution

## Data Flow

### 1. Chat Interaction Flow
1. User selects agent type and enters message in Streamlit frontend
2. Frontend sends request to FastAPI backend `/agents/{agent_type}` endpoint
3. Orchestrator routes task to appropriate agent
4. Agent processes task using Groq API
5. Response returned through backend to frontend
6. Frontend displays response with context and metadata

### 2. RAG Document Processing Flow
1. User uploads document through RAG interface
2. Backend receives file and processes with DocumentProcessor
3. Document chunked and embedded using SentenceTransformer
4. Chunks stored in ChromaDB with metadata
5. Search queries retrieve relevant chunks for context
6. Retrieved context provided to agents for enhanced responses

### 3. Workflow Execution Flow
1. User creates workflow with multiple steps and dependencies
2. Orchestrator validates workflow and creates execution plan
3. Steps executed in dependency order with context passing
4. Each step result stored and passed to dependent steps
5. Final workflow result aggregated and returned

## External Dependencies

### APIs and Services
- **Groq API**: Primary LLM service for agent responses
- **HuggingFace**: Sentence transformers for embeddings
- **ChromaDB**: Vector database for document storage
- **NLTK**: Natural language processing utilities

### Python Libraries
- **FastAPI**: Web framework for backend API
- **Streamlit**: Frontend framework for interactive dashboards
- **SQLAlchemy**: Database ORM for PostgreSQL integration
- **Psycopg2**: PostgreSQL database adapter
- **Alembic**: Database migration tool
- **Pydantic**: Data validation and serialization
- **aiohttp**: Async HTTP client for API calls
- **plotly**: Interactive visualizations
- **pandas**: Data manipulation and analysis

### File Processing
- **PyPDF2**: PDF text extraction
- **python-docx**: Word document processing
- **NLTK**: Text tokenization and processing

## Deployment Strategy

### Configuration Management
- **Environment Variables**: API keys and configuration through environment variables
- **Config Class**: Centralized configuration management in `config.py`
- **Flexible Deployment**: Support for different environments (development, production)

### Service Architecture
- **Backend Service**: FastAPI app with uvicorn ASGI server
- **Frontend Service**: Streamlit app with component-based architecture
- **MCP Service**: WebSocket server for real-time communication
- **Database**: File-based ChromaDB for vector storage

### Scalability Considerations
- **Async Processing**: Full async/await support for I/O operations
- **Connection Pooling**: Managed HTTP sessions for external API calls
- **Task Queuing**: Async queue system for handling concurrent requests
- **Resource Limits**: Configurable limits on concurrent agents and tasks

### Error Handling and Logging
- **Comprehensive Logging**: Structured logging throughout application
- **Graceful Degradation**: Fallback mechanisms for service failures
- **Error Recovery**: Retry logic for transient failures
- **Monitoring**: Built-in metrics and monitoring capabilities

The architecture is designed for extensibility, allowing easy addition of new agent types, integration with different LLM providers, and scaling to handle increased load. The separation of concerns between frontend, backend, and agent services provides flexibility for deployment and maintenance.