import asyncio
import json
import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
import websockets
from websockets.server import WebSocketServerProtocol
from backend.models.schemas import MCPMessage, MessageRole
from backend.services.groq_client import GroqClient

logger = logging.getLogger(__name__)

class MCPServer:
    """Model Context Protocol Server implementation"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8001):
        self.host = host
        self.port = port
        self.clients: Dict[str, WebSocketServerProtocol] = {}
        self.groq_client = GroqClient()
        self.server = None
        self.running = False
    
    async def start(self):
        """Start the MCP server"""
        try:
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port
            )
            self.running = True
            logger.info(f"MCP Server started on {self.host}:{self.port}")
            
            # Keep server running
            await self.server.wait_closed()
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise
    
    async def stop(self):
        """Stop the MCP server"""
        if self.server:
            self.running = False
            self.server.close()
            await self.server.wait_closed()
            await self.groq_client.close()
            logger.info("MCP Server stopped")
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle incoming client connections"""
        client_id = str(uuid.uuid4())
        self.clients[client_id] = websocket
        
        logger.info(f"Client {client_id} connected")
        
        try:
            # Send welcome message
            welcome_msg = {
                "id": str(uuid.uuid4()),
                "type": "welcome",
                "client_id": client_id,
                "server_info": {
                    "name": "AI Agent MCP Server",
                    "version": "1.0.0",
                    "capabilities": [
                        "text_generation",
                        "sentiment_analysis",
                        "summarization",
                        "embedding_generation"
                    ]
                }
            }
            await websocket.send(json.dumps(welcome_msg))
            
            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.process_message(client_id, data)
                except json.JSONDecodeError:
                    await self.send_error(client_id, "Invalid JSON format")
                except Exception as e:
                    logger.error(f"Error processing message from {client_id}: {e}")
                    await self.send_error(client_id, f"Processing error: {str(e)}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            if client_id in self.clients:
                del self.clients[client_id]
    
    async def process_message(self, client_id: str, data: Dict[str, Any]):
        """Process incoming MCP message"""
        try:
            message = MCPMessage(**data)
            
            # Route based on method
            if message.method == "generate_text":
                await self.handle_generate_text(client_id, message)
            elif message.method == "analyze_sentiment":
                await self.handle_analyze_sentiment(client_id, message)
            elif message.method == "summarize_text":
                await self.handle_summarize_text(client_id, message)
            elif message.method == "generate_embedding":
                await self.handle_generate_embedding(client_id, message)
            elif message.method == "chat_completion":
                await self.handle_chat_completion(client_id, message)
            else:
                await self.send_error(client_id, f"Unknown method: {message.method}")
        
        except Exception as e:
            logger.error(f"Error processing MCP message: {e}")
            await self.send_error(client_id, f"Message processing error: {str(e)}")
    
    async def handle_generate_text(self, client_id: str, message: MCPMessage):
        """Handle text generation request"""
        try:
            prompt = message.params.get("prompt", "")
            model = message.params.get("model", "mixtral-8x7b-32768")
            temperature = message.params.get("temperature", 0.7)
            max_tokens = message.params.get("max_tokens", 1024)
            
            if not prompt:
                await self.send_error(client_id, "Prompt is required")
                return
            
            # Generate text using Groq
            messages = [{"role": "user", "content": prompt}]
            result = await self.groq_client.generate_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Send response
            response = {
                "id": str(uuid.uuid4()),
                "type": "response",
                "request_id": message.id,
                "method": "generate_text",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            await self.send_message(client_id, response)
        
        except Exception as e:
            await self.send_error(client_id, f"Text generation error: {str(e)}")
    
    async def handle_analyze_sentiment(self, client_id: str, message: MCPMessage):
        """Handle sentiment analysis request"""
        try:
            text = message.params.get("text", "")
            
            if not text:
                await self.send_error(client_id, "Text is required")
                return
            
            # Analyze sentiment using Groq
            result = await self.groq_client.analyze_sentiment(text)
            
            # Send response
            response = {
                "id": str(uuid.uuid4()),
                "type": "response",
                "request_id": message.id,
                "method": "analyze_sentiment",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            await self.send_message(client_id, response)
        
        except Exception as e:
            await self.send_error(client_id, f"Sentiment analysis error: {str(e)}")
    
    async def handle_summarize_text(self, client_id: str, message: MCPMessage):
        """Handle text summarization request"""
        try:
            text = message.params.get("text", "")
            max_length = message.params.get("max_length", 200)
            
            if not text:
                await self.send_error(client_id, "Text is required")
                return
            
            # Summarize text using Groq
            result = await self.groq_client.summarize_text(text, max_length)
            
            # Send response
            response = {
                "id": str(uuid.uuid4()),
                "type": "response",
                "request_id": message.id,
                "method": "summarize_text",
                "result": {"summary": result},
                "timestamp": datetime.now().isoformat()
            }
            await self.send_message(client_id, response)
        
        except Exception as e:
            await self.send_error(client_id, f"Summarization error: {str(e)}")
    
    async def handle_generate_embedding(self, client_id: str, message: MCPMessage):
        """Handle embedding generation request"""
        try:
            text = message.params.get("text", "")
            
            if not text:
                await self.send_error(client_id, "Text is required")
                return
            
            # Generate embedding using Groq
            embedding = await self.groq_client.generate_embedding(text)
            
            # Send response
            response = {
                "id": str(uuid.uuid4()),
                "type": "response",
                "request_id": message.id,
                "method": "generate_embedding",
                "result": {"embedding": embedding, "dimension": len(embedding)},
                "timestamp": datetime.now().isoformat()
            }
            await self.send_message(client_id, response)
        
        except Exception as e:
            await self.send_error(client_id, f"Embedding generation error: {str(e)}")
    
    async def handle_chat_completion(self, client_id: str, message: MCPMessage):
        """Handle chat completion request"""
        try:
            messages = message.params.get("messages", [])
            model = message.params.get("model", "mixtral-8x7b-32768")
            temperature = message.params.get("temperature", 0.7)
            max_tokens = message.params.get("max_tokens", 1024)
            
            if not messages:
                await self.send_error(client_id, "Messages are required")
                return
            
            # Generate completion using Groq
            result = await self.groq_client.generate_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Send response
            response = {
                "id": str(uuid.uuid4()),
                "type": "response",
                "request_id": message.id,
                "method": "chat_completion",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            await self.send_message(client_id, response)
        
        except Exception as e:
            await self.send_error(client_id, f"Chat completion error: {str(e)}")
    
    async def send_message(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific client"""
        if client_id in self.clients:
            try:
                await self.clients[client_id].send(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
    
    async def send_error(self, client_id: str, error_message: str):
        """Send error message to client"""
        error_response = {
            "id": str(uuid.uuid4()),
            "type": "error",
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }
        await self.send_message(client_id, error_response)
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if self.clients:
            await asyncio.gather(
                *[self.send_message(client_id, message) for client_id in self.clients.keys()],
                return_exceptions=True
            )
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get server status information"""
        return {
            "running": self.running,
            "host": self.host,
            "port": self.port,
            "connected_clients": len(self.clients),
            "client_ids": list(self.clients.keys())
        }
