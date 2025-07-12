import logging
import json
import os
from typing import List, Dict, Any, Optional
from backend.utils.simple_embeddings import SimpleEmbeddings
from backend.utils.document_processor import DocumentProcessor
from backend.models.schemas import RAGQuery, RAGResult, DocumentChunk
from config import Config

logger = logging.getLogger(__name__)

class SimpleRAGService:
    """Simple RAG service using file-based storage"""
    
    def __init__(self):
        self.embedding_service = SimpleEmbeddings()
        self.document_processor = DocumentProcessor()
        self.storage_path = "rag_storage"
        self.documents = {}
        self.chunks = []
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize file-based storage"""
        try:
            os.makedirs(self.storage_path, exist_ok=True)
            
            # Load existing data
            chunks_file = os.path.join(self.storage_path, "chunks.json")
            if os.path.exists(chunks_file):
                with open(chunks_file, 'r') as f:
                    self.chunks = json.load(f)
            
            docs_file = os.path.join(self.storage_path, "documents.json")
            if os.path.exists(docs_file):
                with open(docs_file, 'r') as f:
                    self.documents = json.load(f)
                    
            logger.info("Simple RAG storage initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize storage: {e}")
    
    def _save_data(self):
        """Save data to files"""
        try:
            chunks_file = os.path.join(self.storage_path, "chunks.json")
            with open(chunks_file, 'w') as f:
                json.dump(self.chunks, f, indent=2)
            
            docs_file = os.path.join(self.storage_path, "documents.json")
            with open(docs_file, 'w') as f:
                json.dump(self.documents, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
    
    async def add_document(self, file_path: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """Add a document to the RAG system"""
        try:
            if document_id is None:
                document_id = f"doc_{len(self.documents) + 1}"
            
            # Process document
            processed_doc = self.document_processor.process_document(
                file_path, 
                chunk_size=Config.CHUNK_SIZE,
                overlap=Config.CHUNK_OVERLAP
            )
            
            if "error" in processed_doc:
                return {"success": False, "error": processed_doc["error"]}
            
            # Generate embeddings for chunks
            chunk_texts = [chunk["text"] for chunk in processed_doc["chunks"]]
            
            # Store chunks with embeddings
            chunk_count = 0
            for i, chunk in enumerate(processed_doc["chunks"]):
                chunk_id = f"{document_id}_{i}"
                embedding = self.embedding_service.generate_embedding(chunk["text"])
                
                chunk_data = {
                    "id": chunk_id,
                    "document_id": document_id,
                    "content": chunk["text"],
                    "embedding": embedding,
                    "metadata": {
                        "chunk_index": i,
                        "chunk_length": chunk["length"],
                        "start_index": chunk["start_index"],
                        "file_name": processed_doc["metadata"]["file_name"],
                        "file_path": processed_doc["metadata"]["file_path"]
                    }
                }
                
                self.chunks.append(chunk_data)
                chunk_count += 1
            
            # Store document info
            self.documents[document_id] = {
                "id": document_id,
                "file_name": processed_doc["metadata"]["file_name"],
                "file_path": processed_doc["metadata"]["file_path"],
                "chunk_count": chunk_count,
                "word_count": processed_doc["word_count"],
                "keywords": processed_doc["keywords"]
            }
            
            # Save data
            self._save_data()
            
            return {
                "success": True,
                "document_id": document_id,
                "chunk_count": chunk_count,
                "total_words": processed_doc["word_count"],
                "keywords": processed_doc["keywords"]
            }
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_documents(self, query: RAGQuery) -> RAGResult:
        """Search documents using simple similarity"""
        try:
            query_embedding = self.embedding_service.generate_embedding(query.query)
            
            # Calculate similarities
            similarities = []
            for chunk in self.chunks:
                similarity = self.embedding_service.compute_similarity(
                    query_embedding, 
                    chunk["embedding"]
                )
                similarities.append((chunk, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top results
            top_results = similarities[:query.top_k]
            
            # Format results
            results = []
            confidence_scores = []
            
            for chunk, score in top_results:
                doc_chunk = DocumentChunk(
                    id=chunk["id"],
                    content=chunk["content"],
                    metadata=chunk["metadata"],
                    embedding=chunk["embedding"]
                )
                results.append(doc_chunk)
                confidence_scores.append(score)
            
            # Create context
            context = "\n\n".join([chunk["content"] for chunk, _ in top_results])
            
            return RAGResult(
                query=query.query,
                results=results,
                context=context,
                confidence_scores=confidence_scores
            )
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return RAGResult(
                query=query.query,
                results=[],
                context="",
                confidence_scores=[]
            )
    
    async def get_document_list(self) -> List[Dict[str, Any]]:
        """Get list of all documents"""
        return list(self.documents.values())
    
    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document"""
        try:
            if document_id not in self.documents:
                return {"success": False, "error": "Document not found"}
            
            # Remove chunks
            self.chunks = [chunk for chunk in self.chunks if chunk["document_id"] != document_id]
            
            # Remove document
            del self.documents[document_id]
            
            # Save data
            self._save_data()
            
            return {"success": True, "message": "Document deleted successfully"}
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get RAG statistics"""
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "average_chunks_per_document": len(self.chunks) / max(len(self.documents), 1),
            "collection_name": "simple_rag"
        }