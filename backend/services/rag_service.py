import logging
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from backend.utils.embeddings import EmbeddingService
from backend.utils.document_processor import DocumentProcessor
from backend.models.schemas import DocumentChunk, RAGQuery, RAGResult
from config import Config

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.document_processor = DocumentProcessor()
        self.chroma_client = None
        self.collection = None
        self._initialize_vector_db()
    
    def _initialize_vector_db(self):
        """Initialize ChromaDB vector database"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=Config.CHROMA_DB_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    async def add_document(self, file_path: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """Add a document to the RAG system"""
        try:
            if document_id is None:
                document_id = str(uuid.uuid4())
            
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
            embeddings = self.embedding_service.encode_batch(chunk_texts)
            
            # Prepare data for ChromaDB
            chunk_ids = []
            chunk_metadatas = []
            
            for i, chunk in enumerate(processed_doc["chunks"]):
                chunk_id = f"{document_id}_{i}"
                chunk_ids.append(chunk_id)
                
                metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_length": chunk["length"],
                    "start_index": chunk["start_index"],
                    "file_name": processed_doc["metadata"]["file_name"],
                    "file_path": processed_doc["metadata"]["file_path"],
                    "keywords": ",".join(processed_doc["keywords"][:5])  # Top 5 keywords
                }
                chunk_metadatas.append(metadata)
            
            # Add to ChromaDB
            self.collection.add(
                documents=chunk_texts,
                embeddings=embeddings,
                metadatas=chunk_metadatas,
                ids=chunk_ids
            )
            
            return {
                "success": True,
                "document_id": document_id,
                "chunk_count": len(processed_doc["chunks"]),
                "total_words": processed_doc["word_count"],
                "keywords": processed_doc["keywords"]
            }
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_documents(self, query: RAGQuery) -> RAGResult:
        """Search documents using RAG"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.encode_text(query.query)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=query.top_k,
                where=query.filters if query.filters else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            document_chunks = []
            confidence_scores = []
            
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    # Convert distance to similarity score
                    similarity = 1 - distance
                    confidence_scores.append(similarity)
                    
                    chunk = DocumentChunk(
                        id=f"search_{i}",
                        content=doc,
                        metadata=metadata
                    )
                    document_chunks.append(chunk)
            
            # Create context from top results
            context = "\n\n".join([chunk.content for chunk in document_chunks])
            
            return RAGResult(
                query=query.query,
                results=document_chunks,
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
    
    async def get_document_info(self, document_id: str) -> Dict[str, Any]:
        """Get information about a specific document"""
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if not results["metadatas"]:
                return {"error": "Document not found"}
            
            # Aggregate information
            metadata = results["metadatas"][0]
            chunk_count = len(results["metadatas"])
            
            return {
                "document_id": document_id,
                "file_name": metadata["file_name"],
                "file_path": metadata["file_path"],
                "chunk_count": chunk_count,
                "keywords": metadata.get("keywords", "").split(",")
            }
            
        except Exception as e:
            logger.error(f"Error getting document info: {e}")
            return {"error": str(e)}
    
    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document from the RAG system"""
        try:
            # Get all chunks for this document
            results = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if not results["metadatas"]:
                return {"success": False, "error": "Document not found"}
            
            # Delete chunks
            chunk_ids = [f"{document_id}_{i}" for i in range(len(results["metadatas"]))]
            self.collection.delete(ids=chunk_ids)
            
            return {
                "success": True,
                "deleted_chunks": len(chunk_ids)
            }
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return {"success": False, "error": str(e)}
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the RAG system"""
        try:
            # Get all documents
            results = self.collection.get(include=["metadatas"])
            
            if not results["metadatas"]:
                return []
            
            # Group by document_id
            documents = {}
            for metadata in results["metadatas"]:
                doc_id = metadata["document_id"]
                if doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "file_name": metadata["file_name"],
                        "file_path": metadata["file_path"],
                        "chunk_count": 0,
                        "keywords": set()
                    }
                
                documents[doc_id]["chunk_count"] += 1
                if metadata.get("keywords"):
                    documents[doc_id]["keywords"].update(metadata["keywords"].split(","))
            
            # Convert to list and process keywords
            document_list = []
            for doc_info in documents.values():
                doc_info["keywords"] = list(doc_info["keywords"])
                document_list.append(doc_info)
            
            return document_list
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection"""
        try:
            collection_count = self.collection.count()
            documents = await self.list_documents()
            
            return {
                "total_chunks": collection_count,
                "total_documents": len(documents),
                "average_chunks_per_document": collection_count / len(documents) if documents else 0,
                "collection_name": self.collection.name
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
