import streamlit as st
import requests
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import tempfile
import os

class RAGInterface:
    """RAG (Retrieval-Augmented Generation) interface for document management"""
    
    def __init__(self, backend_url: str):
        self.backend_url = backend_url
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state for RAG interface"""
        if 'uploaded_documents' not in st.session_state:
            st.session_state.uploaded_documents = []
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
        if 'rag_context' not in st.session_state:
            st.session_state.rag_context = {}
    
    def upload_document(self, file) -> Dict[str, Any]:
        """Upload document to RAG system"""
        try:
            files = {"file": (file.name, file, file.type)}
            response = requests.post(
                f"{self.backend_url}/rag/upload",
                files=files,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"Upload failed with status {response.status_code}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_documents(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search documents in RAG system"""
        try:
            response = requests.post(
                f"{self.backend_url}/rag/search",
                json={
                    "query": query,
                    "top_k": top_k,
                    "filters": filters or {}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "results": [],
                    "error": f"Search failed with status {response.status_code}"
                }
        except Exception as e:
            return {
                "results": [],
                "error": str(e)
            }
    
    def get_document_list(self) -> List[Dict[str, Any]]:
        """Get list of uploaded documents"""
        try:
            response = requests.get(f"{self.backend_url}/rag/documents", timeout=10)
            if response.status_code == 200:
                return response.json()
            return []
        except:
            return []
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete document from RAG system"""
        try:
            response = requests.delete(
                f"{self.backend_url}/rag/documents/{document_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"Delete failed with status {response.status_code}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_rag_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        try:
            response = requests.get(f"{self.backend_url}/rag/stats", timeout=10)
            if response.status_code == 200:
                return response.json()
            return {}
        except:
            return {}
    
    def render_document_upload(self):
        """Render document upload section"""
        st.subheader("📁 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['txt', 'pdf', 'docx'],
            accept_multiple_files=True,
            help="Supported formats: TXT, PDF, DOCX"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s)")
            
            # Display file info
            for file in uploaded_files:
                with st.expander(f"📄 {file.name}"):
                    st.write(f"**Size:** {file.size:,} bytes")
                    st.write(f"**Type:** {file.type}")
            
            # Upload button
            if st.button("🚀 Upload Documents", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                successful_uploads = 0
                total_files = len(uploaded_files)
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Uploading {file.name}...")
                    
                    # Upload file
                    result = self.upload_document(file)
                    
                    if result.get("success"):
                        successful_uploads += 1
                        st.success(f"✅ {file.name} uploaded successfully")
                        
                        # Add to session state
                        st.session_state.uploaded_documents.append({
                            "name": file.name,
                            "document_id": result.get("document_id"),
                            "chunk_count": result.get("chunk_count", 0),
                            "upload_time": datetime.now()
                        })
                    else:
                        st.error(f"❌ Failed to upload {file.name}: {result.get('error', 'Unknown error')}")
                    
                    # Update progress
                    progress_bar.progress((i + 1) / total_files)
                
                status_text.text(f"Upload complete! {successful_uploads}/{total_files} files uploaded successfully.")
                
                if successful_uploads > 0:
                    st.balloons()
    
    def render_document_search(self):
        """Render document search section"""
        st.subheader("🔍 Document Search")
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "Search Query",
                placeholder="Enter your search query...",
                help="Search across all uploaded documents"
            )
        
        with col2:
            top_k = st.number_input(
                "Results Count",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of results to return"
            )
        
        # Advanced search options
        with st.expander("🔧 Advanced Search Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                file_type_filter = st.selectbox(
                    "File Type",
                    ["All", "PDF", "DOCX", "TXT"]
                )
            
            with col2:
                search_mode = st.selectbox(
                    "Search Mode",
                    ["Semantic", "Keyword", "Hybrid"]
                )
            
            # Date range filter
            st.write("**Upload Date Range:**")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("From", value=None)
            with col2:
                end_date = st.date_input("To", value=None)
        
        # Search button
        if st.button("🔍 Search Documents") and search_query:
            with st.spinner("Searching..."):
                # Prepare filters
                filters = {}
                if file_type_filter != "All":
                    filters["file_extension"] = f".{file_type_filter.lower()}"
                
                # Perform search
                results = self.search_documents(search_query, top_k, filters)
                
                # Store results
                st.session_state.search_results = results
                
                # Display results
                self.display_search_results(results, search_query)
    
    def display_search_results(self, results: Dict[str, Any], query: str):
        """Display search results"""
        if "error" in results:
            st.error(f"Search failed: {results['error']}")
            return
        
        search_results = results.get("results", [])
        
        if not search_results:
            st.info("No results found for your query.")
            return
        
        st.success(f"Found {len(search_results)} relevant results")
        
        # Display results
        for i, result in enumerate(search_results):
            with st.expander(f"📄 Result {i+1} - {result.get('metadata', {}).get('file_name', 'Unknown File')}"):
                # Content preview
                st.write("**Content:**")
                content = result.get("content", "")
                if len(content) > 500:
                    st.write(content[:500] + "...")
                else:
                    st.write(content)
                
                # Metadata
                metadata = result.get("metadata", {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**File Information:**")
                    st.write(f"File: {metadata.get('file_name', 'Unknown')}")
                    st.write(f"Chunk: {metadata.get('chunk_index', 'Unknown')}")
                    st.write(f"Length: {metadata.get('chunk_length', 'Unknown')} chars")
                
                with col2:
                    st.write("**Relevance:**")
                    confidence_scores = results.get("confidence_scores", [])
                    if i < len(confidence_scores):
                        st.progress(confidence_scores[i])
                        st.write(f"Score: {confidence_scores[i]:.2%}")
                    
                    # Keywords
                    keywords = metadata.get("keywords", "").split(",")
                    if keywords and keywords[0]:
                        st.write("**Keywords:** " + ", ".join(keywords[:3]))
                
                # Actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"📋 Copy Content", key=f"copy_{i}"):
                        st.code(content)
                
                with col2:
                    if st.button(f"🤖 Ask Agent", key=f"ask_{i}"):
                        # Pre-fill chat with context
                        context_prompt = f"Based on this document excerpt:\n\n{content}\n\nAnswer: {query}"
                        st.session_state.rag_context['selected_content'] = content
                        st.session_state.rag_context['query'] = query
                        st.info("Context saved! Go to Chat Interface to continue.")
    
    def render_document_management(self):
        """Render document management section"""
        st.subheader("📚 Document Management")
        
        # Get document list
        documents = self.get_document_list()
        
        if not documents:
            st.info("No documents uploaded yet.")
            return
        
        # Display documents in a table
        df = pd.DataFrame(documents)
        
        # Add selection column
        if len(df) > 0:
            # Display document table
            st.write(f"**Total Documents:** {len(documents)}")
            
            for i, doc in enumerate(documents):
                with st.expander(f"📄 {doc.get('file_name', 'Unknown')}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Document Info:**")
                        st.write(f"ID: {doc.get('document_id', 'Unknown')[:8]}...")
                        st.write(f"Chunks: {doc.get('chunk_count', 0)}")
                        st.write(f"Keywords: {len(doc.get('keywords', []))}")
                    
                    with col2:
                        st.write("**Statistics:**")
                        # Add more stats when available
                        st.write("Status: ✅ Indexed")
                        st.write("Type: Document")
                    
                    with col3:
                        st.write("**Actions:**")
                        
                        # Delete button
                        if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                            result = self.delete_document(doc.get('document_id'))
                            if result.get("success"):
                                st.success("Document deleted successfully!")
                                st.rerun()
                            else:
                                st.error(f"Delete failed: {result.get('error', 'Unknown error')}")
                        
                        # Download button (if available)
                        if st.button(f"📥 Download", key=f"download_{i}"):
                            st.info("Download functionality coming soon!")
    
    def render_rag_analytics(self):
        """Render RAG analytics section"""
        st.subheader("📊 RAG Analytics")
        
        # Get statistics
        stats = self.get_rag_statistics()
        
        if not stats:
            st.info("No analytics data available.")
            return
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Documents", stats.get("total_documents", 0))
        
        with col2:
            st.metric("Total Chunks", stats.get("total_chunks", 0))
        
        with col3:
            st.metric("Avg Chunks/Doc", f"{stats.get('average_chunks_per_document', 0):.1f}")
        
        with col4:
            st.metric("Collection", stats.get("collection_name", "Unknown"))
        
        # Usage statistics
        st.write("**Usage Statistics:**")
        
        # Mock usage data for demonstration
        usage_data = {
            "Date": pd.date_range(start="2024-01-01", periods=30, freq="D"),
            "Searches": [max(0, int(10 + 5 * (i % 7))) for i in range(30)],
            "Documents Added": [max(0, int(2 + (i % 3))) for i in range(30)],
            "Chunks Retrieved": [max(0, int(50 + 20 * (i % 5))) for i in range(30)]
        }
        
        df = pd.DataFrame(usage_data)
        
        # Display charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.line_chart(df.set_index("Date")["Searches"])
            st.caption("Daily Search Queries")
        
        with col2:
            st.line_chart(df.set_index("Date")["Documents Added"])
            st.caption("Documents Added Over Time")
        
        # Search performance
        st.write("**Search Performance:**")
        
        # Mock performance data
        performance_data = {
            "Query Type": ["Semantic", "Keyword", "Hybrid"],
            "Avg Response Time (ms)": [250, 150, 300],
            "Success Rate (%)": [95, 88, 97]
        }
        
        perf_df = pd.DataFrame(performance_data)
        st.dataframe(perf_df)
    
    def render_rag_settings(self):
        """Render RAG settings section"""
        st.subheader("⚙️ RAG Settings")
        
        # Chunking settings
        st.write("**Document Chunking:**")
        col1, col2 = st.columns(2)
        
        with col1:
            chunk_size = st.slider(
                "Chunk Size",
                min_value=200,
                max_value=2000,
                value=1000,
                help="Size of text chunks in characters"
            )
        
        with col2:
            chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=200,
                help="Overlap between chunks in characters"
            )
        
        # Search settings
        st.write("**Search Configuration:**")
        col1, col2 = st.columns(2)
        
        with col1:
            default_top_k = st.slider(
                "Default Results Count",
                min_value=1,
                max_value=20,
                value=5,
                help="Default number of search results"
            )
        
        with col2:
            similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                help="Minimum similarity score for results"
            )
        
        # Embedding settings
        st.write("**Embedding Model:**")
        embedding_model = st.selectbox(
            "Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L12-v2"],
            help="Sentence transformer model for embeddings"
        )
        
        # Save settings
        if st.button("💾 Save Settings"):
            settings = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "default_top_k": default_top_k,
                "similarity_threshold": similarity_threshold,
                "embedding_model": embedding_model
            }
            
            # Here you would normally save to backend
            st.success("Settings saved successfully!")
            st.json(settings)
    
    def render(self):
        """Render the complete RAG interface"""
        st.title("📚 RAG System")
        
        # Navigation tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📁 Upload", 
            "🔍 Search", 
            "📚 Management", 
            "📊 Analytics", 
            "⚙️ Settings"
        ])
        
        with tab1:
            self.render_document_upload()
        
        with tab2:
            self.render_document_search()
        
        with tab3:
            self.render_document_management()
        
        with tab4:
            self.render_rag_analytics()
        
        with tab5:
            self.render_rag_settings()
        
        # Quick actions
        st.sidebar.markdown("---")
        st.sidebar.subheader("🚀 Quick Actions")
        
        if st.sidebar.button("🔄 Refresh Documents"):
            st.rerun()
        
        if st.sidebar.button("🧹 Clear Search"):
            st.session_state.search_results = []
            st.rerun()
        
        if st.sidebar.button("📊 View Stats"):
            stats = self.get_rag_statistics()
            st.sidebar.json(stats)
