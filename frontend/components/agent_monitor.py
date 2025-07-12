import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any

class AgentMonitor:
    """Agent monitoring and performance dashboard"""
    
    def __init__(self, backend_url: str):
        self.backend_url = backend_url
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state for monitoring"""
        if 'agent_metrics_history' not in st.session_state:
            st.session_state.agent_metrics_history = []
        if 'selected_agent' not in st.session_state:
            st.session_state.selected_agent = None
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
    
    def get_agents_status(self) -> List[Dict[str, Any]]:
        """Get current status of all agents"""
        try:
            response = requests.get(f"{self.backend_url}/agents/status", timeout=5)
            if response.status_code == 200:
                return response.json()
            return []
        except:
            return []
    
    def get_agent_details(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific agent"""
        try:
            response = requests.get(f"{self.backend_url}/agents/{agent_id}/info", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {}
        except:
            return {}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics"""
        try:
            response = requests.get(f"{self.backend_url}/metrics", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {}
        except:
            return {}
    
    def render_system_overview(self):
        """Render system overview section"""
        st.subheader("🔍 System Overview")
        
        # Get system metrics
        metrics = self.get_system_metrics()
        
        if not metrics:
            st.error("Unable to fetch system metrics. Backend may be offline.")
            return
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Agents", metrics.get('agent_count', 0))
        
        with col2:
            st.metric("Active Tasks", metrics.get('active_agents', 0))
        
        with col3:
            st.metric("Completed Tasks", metrics.get('completed_tasks', 0))
        
        with col4:
            st.metric("Success Rate", f"{metrics.get('success_rate', 0):.1%}")
        
        # System health indicators
        st.markdown("### System Health")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Response time chart
            if st.session_state.agent_metrics_history:
                df = pd.DataFrame(st.session_state.agent_metrics_history)
                fig = px.line(
                    df, 
                    x='timestamp', 
                    y='avg_response_time',
                    title='Average Response Time',
                    labels={'avg_response_time': 'Response Time (s)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No historical data available")
        
        with col2:
            # Task completion rate
            completed = metrics.get('completed_tasks', 0)
            failed = metrics.get('failed_tasks', 0)
            total = completed + failed
            
            if total > 0:
                fig = go.Figure(data=[
                    go.Pie(
                        labels=['Completed', 'Failed'],
                        values=[completed, failed],
                        hole=0.3
                    )
                ])
                fig.update_layout(title='Task Completion Rate')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No task data available")
    
    def render_agent_list(self):
        """Render list of agents with their status"""
        st.subheader("🤖 Agent Status")
        
        # Get agents data
        agents = self.get_agents_status()
        
        if not agents:
            st.warning("No agents found or backend is offline")
            return
        
        # Auto-refresh toggle
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔄 Refresh Status"):
                st.rerun()
        
        with col2:
            auto_refresh = st.toggle("Auto Refresh", value=st.session_state.auto_refresh)
            st.session_state.auto_refresh = auto_refresh
        
        # Agent cards
        for agent in agents:
            with st.expander(f"📊 {agent.get('name', 'Unknown Agent')}", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Agent Info**")
                    st.write(f"Type: {agent.get('agent_type', 'Unknown')}")
                    st.write(f"ID: {agent.get('agent_id', 'Unknown')[:8]}...")
                    
                    # Status indicator
                    if agent.get('is_busy', False):
                        st.markdown("🟡 **Status: Busy**")
                    else:
                        st.markdown("🟢 **Status: Available**")
                
                with col2:
                    st.write("**Performance**")
                    st.metric("Total Tasks", agent.get('task_count', 0))
                    st.metric("Success Rate", f"{agent.get('success_rate', 0):.1%}")
                    st.metric("Avg Response Time", f"{agent.get('average_execution_time', 0):.2f}s")
                
                with col3:
                    st.write("**Capabilities**")
                    capabilities = agent.get('capabilities', [])
                    if capabilities:
                        for cap in capabilities[:3]:  # Show first 3 capabilities
                            st.write(f"• {cap}")
                        if len(capabilities) > 3:
                            st.write(f"• +{len(capabilities) - 3} more...")
                    else:
                        st.write("No capabilities listed")
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"View Details", key=f"details_{agent.get('agent_id')}"):
                        st.session_state.selected_agent = agent.get('agent_id')
                        st.rerun()
                
                with col2:
                    if st.button(f"Test Agent", key=f"test_{agent.get('agent_id')}"):
                        self.test_agent(agent.get('agent_id'))
    
    def render_agent_details(self):
        """Render detailed view of selected agent"""
        if not st.session_state.selected_agent:
            st.info("Select an agent to view detailed information")
            return
        
        agent_details = self.get_agent_details(st.session_state.selected_agent)
        
        if not agent_details:
            st.error("Unable to fetch agent details")
            return
        
        agent_info = agent_details.get('info', {})
        agent_metrics = agent_details.get('metrics', {})
        
        st.subheader(f"📈 Agent Details: {agent_info.get('name', 'Unknown')}")
        
        # Basic information
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Information**")
            st.write(f"Agent ID: {agent_info.get('agent_id', 'Unknown')}")
            st.write(f"Type: {agent_info.get('agent_type', 'Unknown')}")
            st.write(f"Created: {agent_info.get('created_at', 'Unknown')}")
            st.write(f"Last Active: {agent_info.get('last_active', 'Unknown')}")
        
        with col2:
            st.write("**Performance Metrics**")
            st.metric("Total Tasks", agent_metrics.get('total_tasks', 0))
            st.metric("Successful Tasks", agent_metrics.get('completed_tasks', 0))
            st.metric("Failed Tasks", agent_metrics.get('failed_tasks', 0))
            st.metric("Success Rate", f"{agent_metrics.get('success_rate', 0):.1%}")
        
        # Capabilities
        st.write("**Capabilities**")
        capabilities = agent_details.get('capabilities', [])
        if capabilities:
            for i, cap in enumerate(capabilities, 1):
                st.write(f"{i}. {cap}")
        else:
            st.write("No capabilities listed")
        
        # Performance trends
        st.write("**Performance Trends**")
        
        # Mock performance data for demonstration
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        performance_data = {
            'date': dates,
            'tasks_completed': [max(0, int(10 + 5 * (i % 7) + (i % 3))) for i in range(30)],
            'response_time': [max(0.1, 2.0 + 0.5 * (i % 5) + 0.2 * (i % 3)) for i in range(30)]
        }
        
        df = pd.DataFrame(performance_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(df, x='date', y='tasks_completed', title='Daily Task Completion')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(df, x='date', y='response_time', title='Response Time Trend')
            st.plotly_chart(fig, use_container_width=True)
        
        # Back button
        if st.button("← Back to Agent List"):
            st.session_state.selected_agent = None
            st.rerun()
    
    def test_agent(self, agent_id: str):
        """Test agent functionality"""
        st.subheader(f"🧪 Testing Agent: {agent_id[:8]}...")
        
        # Test prompt
        test_prompt = st.text_area(
            "Test Prompt",
            placeholder="Enter a test prompt for the agent...",
            height=100
        )
        
        if st.button("Run Test"):
            if test_prompt:
                with st.spinner("Testing agent..."):
                    try:
                        # Get agent details to determine type
                        agent_details = self.get_agent_details(agent_id)
                        agent_type = agent_details.get('info', {}).get('agent_type', 'research')
                        
                        # Send test request
                        response = requests.post(
                            f"{self.backend_url}/agents/{agent_type}",
                            json={
                                "prompt": test_prompt,
                                "context": {"test_mode": True}
                            },
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success("Test completed successfully!")
                            
                            # Display results
                            st.write("**Response:**")
                            st.write(result.get('response', 'No response'))
                            
                            st.write("**Confidence:**")
                            st.progress(result.get('confidence', 0))
                            
                            st.write("**Reasoning:**")
                            st.write(result.get('reasoning', 'No reasoning provided'))
                            
                        else:
                            st.error(f"Test failed with status code: {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"Test failed: {str(e)}")
            else:
                st.warning("Please enter a test prompt")
    
    def render_performance_analytics(self):
        """Render performance analytics section"""
        st.subheader("📊 Performance Analytics")
        
        # Time range selector
        time_range = st.selectbox(
            "Select Time Range",
            ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days"]
        )
        
        # Agent type filter
        agent_types = ["All", "Research", "Analyst", "Coding", "Document"]
        selected_type = st.selectbox("Filter by Agent Type", agent_types)
        
        # Performance metrics
        agents = self.get_agents_status()
        
        if agents:
            # Create performance summary
            df = pd.DataFrame(agents)
            
            # Average performance by type
            if 'agent_type' in df.columns:
                type_performance = df.groupby('agent_type').agg({
                    'task_count': 'mean',
                    'success_rate': 'mean',
                    'average_execution_time': 'mean'
                }).round(2)
                
                st.write("**Average Performance by Agent Type**")
                st.dataframe(type_performance)
            
            # Top performers
            if len(df) > 0:
                st.write("**Top Performing Agents**")
                top_agents = df.nlargest(5, 'success_rate')[['name', 'agent_type', 'success_rate', 'task_count']]
                st.dataframe(top_agents)
        
        else:
            st.info("No performance data available")
    
    def render(self):
        """Render the complete agent monitor interface"""
        st.title("🔍 Agent Monitor")
        
        # Navigation tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "System Overview", 
            "Agent Status", 
            "Agent Details", 
            "Performance Analytics"
        ])
        
        with tab1:
            self.render_system_overview()
        
        with tab2:
            self.render_agent_list()
        
        with tab3:
            self.render_agent_details()
        
        with tab4:
            self.render_performance_analytics()
        
        # Auto-refresh functionality
        if st.session_state.auto_refresh:
            time.sleep(5)
            st.rerun()
        
        # Store current metrics for history
        current_metrics = self.get_system_metrics()
        if current_metrics:
            current_metrics['timestamp'] = datetime.now()
            st.session_state.agent_metrics_history.append(current_metrics)
            
            # Keep only last 100 entries
            if len(st.session_state.agent_metrics_history) > 100:
                st.session_state.agent_metrics_history = st.session_state.agent_metrics_history[-100:]
