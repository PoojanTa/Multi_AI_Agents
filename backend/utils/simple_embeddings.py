import hashlib
import numpy as np
from typing import List, Dict, Any

class SimpleEmbeddings:
    """Simple embeddings service using basic text features"""
    
    def __init__(self):
        self.vocab_size = 1000
        self.embedding_dim = 384
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate simple embedding from text"""
        # Simple hash-based embedding
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to numbers
        embedding = []
        for i in range(0, len(text_hash), 2):
            hex_val = text_hash[i:i+2]
            embedding.append(int(hex_val, 16) / 255.0)
        
        # Pad or truncate to desired size
        while len(embedding) < self.embedding_dim:
            embedding.extend(embedding[:self.embedding_dim - len(embedding)])
        
        return embedding[:self.embedding_dim]
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between embeddings"""
        if len(embedding1) != len(embedding2):
            return 0.0
        
        # Compute dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        
        # Compute magnitudes
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)