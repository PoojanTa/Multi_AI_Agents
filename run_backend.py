#!/usr/bin/env python3
"""
Backend runner script for AI Agent Platform
"""

import asyncio
import logging
import sys
import os
import signal
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import uvicorn
from backend.main import app
from backend.services.mcp_server import MCPServer
from config import Config

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

class BackendRunner:
    """Backend runner with graceful shutdown"""
    
    def __init__(self):
        self.server = None
        self.mcp_server = None
        self.running = False
        
    async def start_services(self):
        """Start all backend services"""
        try:
            logger.info("Starting AI Agent Platform Backend...")
            
            # Create necessary directories
            os.makedirs(Config.CHROMA_DB_PATH, exist_ok=True)
            os.makedirs("logs", exist_ok=True)
            
            # Verify API keys
            if not Config.GROQ_API_KEY:
                logger.warning("GROQ_API_KEY not found in environment variables")
                logger.warning("Some features may not work properly")
            
            # Start FastAPI server
            config = uvicorn.Config(
                app,
                host=Config.BACKEND_HOST,
                port=Config.BACKEND_PORT,
                log_level=Config.LOG_LEVEL.lower(),
                reload=False,
                access_log=True
            )
            
            self.server = uvicorn.Server(config)
            self.running = True
            
            logger.info(f"Backend server starting on {Config.BACKEND_HOST}:{Config.BACKEND_PORT}")
            
            # Start the server
            await self.server.serve()
            
        except Exception as e:
            logger.error(f"Failed to start backend services: {e}")
            raise
    
    async def stop_services(self):
        """Stop all backend services"""
        try:
            logger.info("Stopping backend services...")
            
            self.running = False
            
            if self.server:
                self.server.should_exit = True
                await self.server.shutdown()
            
            logger.info("Backend services stopped")
            
        except Exception as e:
            logger.error(f"Error stopping services: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.stop_services())

def check_requirements():
    """Check if all required dependencies are available"""
    try:
        import fastapi
        import uvicorn
        import chromadb
        import sentence_transformers
        import groq
        import streamlit
        logger.info("All required dependencies are available")
        return True
    except ImportError as e:
        logger.error(f"Missing required dependency: {e}")
        logger.error("Please install all required packages")
        return False

def check_environment():
    """Check environment configuration"""
    logger.info("Checking environment configuration...")
    
    # Check API keys
    if Config.GROQ_API_KEY:
        logger.info("✅ GROQ_API_KEY found")
    else:
        logger.warning("⚠️  GROQ_API_KEY not found - some features may be limited")
    
    if Config.HUGGINGFACE_API_KEY:
        logger.info("✅ HUGGINGFACE_API_KEY found")
    else:
        logger.info("ℹ️  HUGGINGFACE_API_KEY not found - optional")
    
    # Check directories
    try:
        os.makedirs(Config.CHROMA_DB_PATH, exist_ok=True)
        logger.info(f"✅ Database directory: {Config.CHROMA_DB_PATH}")
    except Exception as e:
        logger.error(f"❌ Cannot create database directory: {e}")
        return False
    
    # Check ports
    logger.info(f"✅ Backend will run on {Config.BACKEND_HOST}:{Config.BACKEND_PORT}")
    logger.info(f"✅ MCP server will run on {Config.BACKEND_HOST}:{Config.MCP_SERVER_PORT}")
    
    return True

async def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("AI Agent Platform Backend")
    logger.info("=" * 60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Create backend runner
    runner = BackendRunner()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, runner.signal_handler)
    signal.signal(signal.SIGTERM, runner.signal_handler)
    
    try:
        # Start services
        await runner.start_services()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        await runner.stop_services()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
