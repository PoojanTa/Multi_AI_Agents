import streamlit as st
import asyncio
import requests
import json
from typing import Dict, Any, List
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

# Import frontend components
from frontend.components.chat_interface import ChatInterface
from frontend.components.rag_interface import RAGInterface
from frontend.components.agent_monitor import AgentMonitor
from frontend.components.workflow_builder import WorkflowBuilder

# Configure Streamlit page
st.set_page_config(
    page_title="AI Agent Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    .agent-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online { background-color: #28a745; }
    .status-offline { background-color: #dc3545; }
    .status-busy { background-color: #ffc107; }
</style>
""", unsafe_allow_html=True)

class AIAgentPlatform:
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.mcp_url = "ws://localhost:8001"
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Dashboard"
        if 'agent_history' not in st.session_state:
            st.session_state.agent_history = []
        if 'active_workflows' not in st.session_state:
            st.session_state.active_workflows = []
        if 'system_metrics' not in st.session_state:
            st.session_state.system_metrics = {
                'active_agents': 0,
                'completed_tasks': 0,
                'failed_tasks': 0,
                'avg_response_time': 0.0
            }
    
    def check_backend_status(self) -> bool:
        """Check if backend is running"""
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics from backend"""
        try:
            response = requests.get(f"{self.backend_url}/metrics", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return st.session_state.system_metrics
        except:
            return st.session_state.system_metrics
    
    def get_agent_status(self) -> List[Dict[str, Any]]:
        """Get status of all agents"""
        try:
            response = requests.get(f"{self.backend_url}/agents/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get('agents', []) if isinstance(data, dict) else data
            else:
                return []
        except:
            return []
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 AI Agent Platform</h1>
            <p style="color: white; text-align: center; margin: 0;">
                Multi-Agent Orchestration • RAG Integration • MCP Server Connectivity
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar navigation"""
        st.sidebar.title("Navigation")
        
        # Backend status indicator
        backend_status = self.check_backend_status()
        status_color = "🟢" if backend_status else "🔴"
        st.sidebar.markdown(f"{status_color} Backend: {'Online' if backend_status else 'Offline'}")
        
        # Navigation menu
        pages = {
            "Dashboard": "📊",
            "Chat Interface": "💬",
            "RAG System": "📚",
            "Agent Monitor": "🔍",
            "Workflow Builder": "⚙️",
            "System Settings": "🛠️"
        }
        
        for page_name, icon in pages.items():
            if st.sidebar.button(f"{icon} {page_name}", key=f"nav_{page_name}"):
                st.session_state.current_page = page_name
                st.rerun()
        
        # Quick actions
        st.sidebar.markdown("---")
        st.sidebar.subheader("Quick Actions")
        
        if st.sidebar.button("🔄 Refresh Status"):
            st.rerun()
        
        if st.sidebar.button("🧹 Clear History"):
            st.session_state.agent_history = []
            st.success("History cleared!")
        
        # System info
        st.sidebar.markdown("---")
        st.sidebar.subheader("System Info")
        metrics = self.get_system_metrics()
        st.sidebar.metric("Active Agents", metrics.get('active_agents', 0))
        st.sidebar.metric("Completed Tasks", metrics.get('completed_tasks', 0))
        st.sidebar.metric("Avg Response Time", f"{metrics.get('avg_response_time', 0):.2f}s")
    
    def render_dashboard(self):
        """Render the main dashboard"""
        st.title("📊 System Dashboard")
        
        # System metrics
        metrics = self.get_system_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Active Agents",
                metrics.get('active_agents', 0),
                delta=None
            )
        
        with col2:
            st.metric(
                "Completed Tasks",
                metrics.get('completed_tasks', 0),
                delta=None
            )
        
        with col3:
            st.metric(
                "Failed Tasks",
                metrics.get('failed_tasks', 0),
                delta=None
            )
        
        with col4:
            st.metric(
                "Avg Response Time",
                f"{metrics.get('avg_response_time', 0):.2f}s",
                delta=None
            )
        
        st.markdown("---")
        
        # Agent status overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Agent Status Overview")
            agent_statuses = self.get_agent_status()
            
            if agent_statuses:
                for agent in agent_statuses:
                    status_class = "status-online" if not agent.get('is_busy', False) else "status-busy"
                    st.markdown(f"""
                    <div class="agent-card">
                        <span class="status-indicator {status_class}"></span>
                        <strong>{agent.get('name', 'Unknown Agent')}</strong>
                        <br>
                        <small>{agent.get('description', 'No description')}</small>
                        <br>
                        <small>Tasks: {agent.get('task_count', 0)} | Success Rate: {agent.get('success_rate', 0):.1%}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No agents available or backend is offline")
        
        with col2:
            st.subheader("Recent Activity")
            
            # Performance chart
            if len(st.session_state.agent_history) > 0:
                df = pd.DataFrame(st.session_state.agent_history)
                fig = px.line(
                    df, 
                    x='timestamp', 
                    y='response_time', 
                    title='Response Time Trends',
                    labels={'response_time': 'Response Time (s)', 'timestamp': 'Time'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No activity data available")
    
    def render_system_settings(self):
        """Render system settings page"""
        st.title("🛠️ System Settings")
        
        # API Configuration
        st.subheader("API Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input(
                "Groq API Key",
                type="password",
                placeholder="Enter your Groq API key",
                help="Required for LLM inference"
            )
            
            st.text_input(
                "HuggingFace API Key",
                type="password",
                placeholder="Enter your HuggingFace API key",
                help="Optional for additional models"
            )
        
        with col2:
            st.selectbox(
                "Default LLM Model",
                options=["mixtral-8x7b-32768", "llama2-70b-4096", "gemma-7b-it"],
                index=0
            )
            
            st.slider(
                "Default Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1
            )
        
        # Agent Configuration
        st.subheader("Agent Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input(
                "Max Concurrent Agents",
                min_value=1,
                max_value=50,
                value=10
            )
            
            st.number_input(
                "Agent Timeout (seconds)",
                min_value=30,
                max_value=600,
                value=300
            )
        
        with col2:
            st.number_input(
                "Max Task History",
                min_value=10,
                max_value=1000,
                value=100
            )
            
            st.selectbox(
                "Log Level",
                options=["DEBUG", "INFO", "WARNING", "ERROR"],
                index=1
            )
        
        # RAG Configuration
        st.subheader("RAG Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input(
                "Chunk Size",
                min_value=200,
                max_value=2000,
                value=1000
            )
            
            st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=200
            )
        
        with col2:
            st.number_input(
                "Top K Retrieval",
                min_value=1,
                max_value=20,
                value=5
            )
            
            st.text_input(
                "Vector DB Path",
                value="./chroma_db"
            )
        
        # System Controls
        st.subheader("System Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Restart Backend"):
                st.info("Backend restart initiated...")
        
        with col2:
            if st.button("🧹 Clear Vector DB"):
                st.warning("This will delete all stored documents!")
        
        with col3:
            if st.button("📊 Export Metrics"):
                st.info("Metrics export initiated...")
        
        # Save settings
        if st.button("💾 Save Settings", type="primary"):
            st.success("Settings saved successfully!")
    
    def run(self):
        """Main application runner"""
        self.render_header()
        self.render_sidebar()
        
        # Route to appropriate page
        if st.session_state.current_page == "Dashboard":
            self.render_dashboard()
        elif st.session_state.current_page == "Chat Interface":
            chat_interface = ChatInterface(self.backend_url)
            chat_interface.render()
        elif st.session_state.current_page == "RAG System":
            rag_interface = RAGInterface(self.backend_url)
            rag_interface.render()
        elif st.session_state.current_page == "Agent Monitor":
            agent_monitor = AgentMonitor(self.backend_url)
            agent_monitor.render()
        elif st.session_state.current_page == "Workflow Builder":
            workflow_builder = WorkflowBuilder(self.backend_url)
            workflow_builder.render()
        elif st.session_state.current_page == "System Settings":
            self.render_system_settings()

# Main application entry point
if __name__ == "__main__":
    app = AIAgentPlatform()
    app.run()
