# AI Agent Platform - Complete Project Code

## Project Description
This is a comprehensive multi-agent AI platform built with:
- **Frontend**: Streamlit (Python) - Interactive web interface
- **Backend**: FastAPI (Python) - RESTful API server
- **Database**: PostgreSQL - Data persistence
- **Features**: Multi-agent orchestration, RAG system, workflow management

## Project Structure
```
├── app.py                          # Main Streamlit frontend application
├── simple_backend.py              # FastAPI backend server
├── config.py                      # Configuration management
├── run_backend.py                 # Backend runner script
├── replit.md                      # Project documentation
├── .streamlit/config.toml         # Streamlit configuration
├── database/                      # Database layer
│   ├── __init__.py
│   ├── models.py                  # SQLAlchemy database models
│   ├── connection.py              # Database connection management
│   └── services.py                # Database service layer
├── backend/                       # Backend services
│   ├── agents/                    # AI agent implementations
│   │   ├── base_agent.py          # Base agent class
│   │   ├── research_agent.py      # Research agent
│   │   ├── analyst_agent.py       # Analysis agent
│   │   ├── coding_agent.py        # Coding agent
│   │   └── document_agent.py      # Document processing agent
│   ├── services/                  # Backend services
│   │   ├── orchestrator.py        # Agent orchestration
│   │   ├── groq_client.py         # Groq API client
│   │   ├── rag_service.py         # RAG system
│   │   ├── simple_rag.py          # Simplified RAG implementation
│   │   └── mcp_server.py          # MCP server
│   ├── models/                    # Data models
│   │   └── schemas.py             # Pydantic schemas
│   ├── utils/                     # Utility functions
│   │   ├── document_processor.py  # Document processing
│   │   ├── embeddings.py          # Text embeddings
│   │   └── simple_embeddings.py   # Simple embeddings
│   └── main.py                    # Main backend application
├── frontend/                      # Frontend components
│   └── components/
│       ├── chat_interface.py      # Chat interface
│       ├── agent_monitor.py       # Agent monitoring
│       ├── rag_interface.py       # RAG interface
│       └── workflow_builder.py    # Workflow builder
├── rag_storage/                   # RAG document storage
├── pyproject.toml                 # Python dependencies
└── uv.lock                        # Dependency lock file
```

## Key Features

### 1. Multi-Agent System
- **Research Agent**: Information gathering and web search
- **Analyst Agent**: Data analysis and pattern recognition
- **Coding Agent**: Code generation and debugging
- **Document Agent**: Document processing and summarization

### 2. Database Integration
- **PostgreSQL**: Primary database for all data
- **SQLAlchemy ORM**: Object-relational mapping
- **Models**: Agents, tasks, documents, workflows, metrics, users
- **Services**: Database service layer with CRUD operations

### 3. RAG (Retrieval-Augmented Generation)
- **Document Upload**: Support for PDF, DOCX, TXT files
- **Text Chunking**: Intelligent document segmentation
- **Vector Storage**: Embedding-based document retrieval
- **Search**: Semantic search for relevant content

### 4. Web Interface
- **Dashboard**: System overview and metrics
- **Chat Interface**: Interact with AI agents
- **Agent Monitor**: Real-time agent status
- **RAG Interface**: Document management
- **Workflow Builder**: Create multi-step workflows

### 5. API Integration
- **Groq API**: LLM inference (Mixtral, Llama, Gemma)
- **FastAPI**: RESTful API endpoints
- **Real-time Updates**: WebSocket support
- **Error Handling**: Comprehensive error management

## How to Run

### Prerequisites
- Python 3.11+
- PostgreSQL database
- Groq API key (optional)

### Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up PostgreSQL database
4. Configure environment variables

### Starting the Application
1. **Backend**: `python simple_backend.py` (runs on port 8000)
2. **Frontend**: `streamlit run app.py --server.port 5000` (runs on port 5000)

### Environment Variables
```bash
DATABASE_URL=postgresql://username:password@localhost/dbname
GROQ_API_KEY=your_groq_api_key_here
```

## Database Schema

### Agents Table
- ID, name, type, description, status
- Performance metrics: tasks_completed, success_rate, avg_response_time
- Capabilities and configuration

### Tasks Table
- ID, agent_id, prompt, response, status
- Execution metrics: confidence, execution_time
- Context and metadata

### Documents Table
- ID, file_name, file_path, content
- Processing status and metadata
- Word count and keywords

### Workflows Table
- ID, name, description, steps
- Execution history and metrics

### System Metrics Table
- Timestamp, active_agents, completed_tasks
- System resource usage
- Performance statistics

## API Endpoints

### Health & Metrics
- `GET /health` - Health check
- `GET /metrics` - System metrics

### Agent Management
- `GET /agents/status` - Get all agents
- `GET /agents/{agent_id}` - Get agent details
- `POST /agents/{agent_type}` - Create agent task

### Document Management
- `POST /rag/upload` - Upload document
- `GET /rag/documents` - List documents
- `POST /rag/search` - Search documents
- `DELETE /rag/documents/{doc_id}` - Delete document

## Architecture Patterns

### Service Layer Pattern
- Database services for data access
- Business logic separation
- Error handling and logging

### Repository Pattern
- Data access abstraction
- ORM integration
- Transaction management

### Observer Pattern
- Real-time updates
- Event-driven architecture
- WebSocket communication

### Strategy Pattern
- Multiple agent types
- Pluggable algorithms
- Configuration-driven behavior

## Security Features
- Input validation with Pydantic
- SQL injection prevention
- Error handling without data exposure
- Environment variable configuration

## Performance Optimizations
- Async/await throughout
- Connection pooling
- Caching strategies
- Resource limiting

## Monitoring & Logging
- Comprehensive logging
- Performance metrics
- Error tracking
- System health monitoring

This platform provides a complete foundation for building AI agent applications with enterprise-grade features and scalability.