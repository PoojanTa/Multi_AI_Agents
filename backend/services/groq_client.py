import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
import aiohttp
from config import Config

logger = logging.getLogger(__name__)

class GroqClient:
    def __init__(self):
        self.api_key = Config.GROQ_API_KEY
        self.base_url = "https://api.groq.com/openai/v1"
        self.session = None
        
    async def _ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        if self.session:
            await self.session.close()
    
    async def generate_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate completion using Groq API"""
        await self._ensure_session()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "content": result["choices"][0]["message"]["content"],
                        "model": model,
                        "usage": result.get("usage", {}),
                        "response_time": result.get("response_time", 0)
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"Groq API error: {response.status} - {error_text}")
                    return {
                        "success": False,
                        "error": f"API Error: {response.status}",
                        "details": error_text
                    }
        except Exception as e:
            logger.error(f"Error calling Groq API: {str(e)}")
            return {
                "success": False,
                "error": "Connection error",
                "details": str(e)
            }
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using Groq API (placeholder - Groq doesn't have embeddings yet)"""
        # For now, we'll use a simple hash-based approach
        # In production, you'd use a proper embedding model
        import hashlib
        import struct
        
        # Create a simple embedding from text hash
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to float vector
        embedding = []
        for i in range(0, len(hash_bytes), 4):
            chunk = hash_bytes[i:i+4]
            if len(chunk) == 4:
                value = struct.unpack('>I', chunk)[0]
                embedding.append(float(value) / (2**32))
        
        # Normalize to 384 dimensions (standard for sentence transformers)
        while len(embedding) < 384:
            embedding.extend(embedding[:384-len(embedding)])
        
        return embedding[:384]
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using Groq API"""
        messages = [
            {
                "role": "system",
                "content": "You are a sentiment analysis expert. Analyze the sentiment of the given text and return a JSON response with 'sentiment' (positive/negative/neutral), 'confidence' (0-1), and 'reasoning'."
            },
            {
                "role": "user",
                "content": f"Analyze the sentiment of this text: {text}"
            }
        ]
        
        result = await self.generate_completion(
            model="mixtral-8x7b-32768",
            messages=messages,
            temperature=0.3
        )
        
        if result["success"]:
            try:
                sentiment_data = json.loads(result["content"])
                return sentiment_data
            except json.JSONDecodeError:
                return {
                    "sentiment": "neutral",
                    "confidence": 0.5,
                    "reasoning": "Unable to parse sentiment analysis"
                }
        else:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "reasoning": f"Error: {result.get('error', 'Unknown error')}"
            }
    
    async def summarize_text(self, text: str, max_length: int = 200) -> str:
        """Summarize text using Groq API"""
        messages = [
            {
                "role": "system",
                "content": f"You are a text summarization expert. Summarize the given text in no more than {max_length} words while preserving the key information."
            },
            {
                "role": "user",
                "content": f"Summarize this text: {text}"
            }
        ]
        
        result = await self.generate_completion(
            model="mixtral-8x7b-32768",
            messages=messages,
            temperature=0.3,
            max_tokens=max_length * 2
        )
        
        if result["success"]:
            return result["content"]
        else:
            return f"Error summarizing text: {result.get('error', 'Unknown error')}"
