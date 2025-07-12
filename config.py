import os
from typing import Dict, Any

class Config:
    # API Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
    
    # Server Configuration
    BACKEND_HOST = "0.0.0.0"
    BACKEND_PORT = 8000
    FRONTEND_HOST = "0.0.0.0"
    FRONTEND_PORT = 5000
    
    # Database Configuration
    CHROMA_DB_PATH = "./chroma_db"
    
    # Agent Configuration
    MAX_AGENTS = 10
    AGENT_TIMEOUT = 300  # seconds
    
    # RAG Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 5
    
    # MCP Configuration
    MCP_SERVER_PORT = 8001
    
    # Logging Configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def get_groq_models(cls) -> Dict[str, Any]:
        return {
            "mixtral-8x7b-32768": {
                "name": "Mixtral 8x7B",
                "context_length": 32768,
                "best_for": "General reasoning, code generation"
            },
            "llama2-70b-4096": {
                "name": "Llama 2 70B",
                "context_length": 4096,
                "best_for": "Complex reasoning, analysis"
            },
            "gemma-7b-it": {
                "name": "Gemma 7B",
                "context_length": 8192,
                "best_for": "Fast inference, lightweight tasks"
            }
        }
    
    @classmethod
    def get_agent_types(cls) -> Dict[str, str]:
        return {
            "research": "Research Agent - Conducts web research and information gathering",
            "analyst": "Analyst Agent - Performs data analysis and insights generation",
            "coding": "Coding Agent - Generates and reviews code, debugging",
            "document": "Document Agent - Processes and analyzes documents"
        }
