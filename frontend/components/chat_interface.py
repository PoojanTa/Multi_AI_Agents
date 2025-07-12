import streamlit as st
import requests
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

class ChatInterface:
    """Interactive chat interface for AI agents"""
    
    def __init__(self, backend_url: str):
        self.backend_url = backend_url
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state for chat interface"""
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        if 'selected_agent_type' not in st.session_state:
            st.session_state.selected_agent_type = "research"
        if 'chat_context' not in st.session_state:
            st.session_state.chat_context = {}
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
    
    def get_available_models(self) -> Dict[str, str]:
        """Get available Groq models"""
        return {
            "mixtral-8x7b-32768": "Mixtral 8x7B (Best for reasoning)",
            "llama2-70b-4096": "Llama 2 70B (Complex analysis)",
            "gemma-7b-it": "Gemma 7B (Fast responses)"
        }
    
    def get_agent_types(self) -> Dict[str, str]:
        """Get available agent types"""
        return {
            "research": "🔍 Research Agent - Information gathering and analysis",
            "analyst": "📊 Analyst Agent - Data analysis and insights",
            "coding": "💻 Coding Agent - Code generation and review",
            "document": "📄 Document Agent - Document processing and generation"
        }
    
    def send_message_to_agent(self, message: str, agent_type: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send message to specific agent"""
        try:
            response = requests.post(
                f"{self.backend_url}/agents/{agent_type}",
                json={
                    "prompt": message,
                    "context": context or {}
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"Request failed with status {response.status_code}",
                    "response": "Sorry, I encountered an error while processing your request."
                }
        except Exception as e:
            return {
                "error": str(e),
                "response": "I'm having trouble connecting to the backend. Please try again."
            }
    
    def render_chat_settings(self):
        """Render chat settings sidebar"""
        st.sidebar.subheader("🎛️ Chat Settings")
        
        # Agent type selection
        agent_types = self.get_agent_types()
        selected_type = st.sidebar.selectbox(
            "Select Agent Type",
            options=list(agent_types.keys()),
            format_func=lambda x: agent_types[x],
            index=list(agent_types.keys()).index(st.session_state.selected_agent_type)
        )
        st.session_state.selected_agent_type = selected_type
        
        # Model selection
        models = self.get_available_models()
        selected_model = st.sidebar.selectbox(
            "Select Model",
            options=list(models.keys()),
            format_func=lambda x: models[x],
            index=0
        )
        
        # Parameters
        temperature = st.sidebar.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make output more random"
        )
        
        max_tokens = st.sidebar.slider(
            "Max Tokens",
            min_value=100,
            max_value=2048,
            value=1024,
            step=100,
            help="Maximum length of response"
        )
        
        return {
            "agent_type": selected_type,
            "model": selected_model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    
    def render_conversation_history(self):
        """Render conversation history"""
        st.subheader("💬 Conversation")
        
        # Display messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                else:
                    # Agent response
                    st.write(message["content"])
                    
                    # Show metadata if available
                    if "metadata" in message:
                        with st.expander("Response Details"):
                            metadata = message["metadata"]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Confidence", f"{metadata.get('confidence', 0):.1%}")
                                st.write(f"**Agent:** {metadata.get('agent_type', 'Unknown')}")
                            
                            with col2:
                                st.metric("Response Time", f"{metadata.get('execution_time', 0):.2f}s")
                                st.write(f"**Tools Used:** {', '.join(metadata.get('tools_used', []))}")
                            
                            if metadata.get('reasoning'):
                                st.write("**Reasoning:**")
                                st.write(metadata['reasoning'])
    
    def render_quick_actions(self):
        """Render quick action buttons"""
        st.subheader("⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📚 Summarize Document"):
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": "Please help me summarize a document.",
                    "timestamp": datetime.now()
                })
                st.rerun()
        
        with col2:
            if st.button("🔍 Research Topic"):
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": "I need help researching a topic.",
                    "timestamp": datetime.now()
                })
                st.rerun()
        
        with col3:
            if st.button("💻 Generate Code"):
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": "I need help generating code.",
                    "timestamp": datetime.now()
                })
                st.rerun()
        
        # Additional quick actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Analyze Data"):
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": "I need help analyzing data.",
                    "timestamp": datetime.now()
                })
                st.rerun()
        
        with col2:
            if st.button("🧠 Brainstorm Ideas"):
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": "Help me brainstorm ideas for my project.",
                    "timestamp": datetime.now()
                })
                st.rerun()
        
        with col3:
            if st.button("🔧 Debug Code"):
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": "I need help debugging code.",
                    "timestamp": datetime.now()
                })
                st.rerun()
    
    def render_advanced_features(self):
        """Render advanced chat features"""
        st.subheader("🚀 Advanced Features")
        
        # Multi-agent conversation
        with st.expander("Multi-Agent Conversation"):
            st.write("Enable multiple agents to collaborate on your request")
            
            agent_types = self.get_agent_types()
            selected_agents = st.multiselect(
                "Select Agents for Collaboration",
                options=list(agent_types.keys()),
                format_func=lambda x: agent_types[x],
                default=[st.session_state.selected_agent_type]
            )
            
            if st.button("Start Multi-Agent Session"):
                if len(selected_agents) > 1:
                    st.session_state.chat_context['multi_agent_mode'] = True
                    st.session_state.chat_context['selected_agents'] = selected_agents
                    st.success(f"Multi-agent mode enabled with {len(selected_agents)} agents")
                else:
                    st.warning("Please select at least 2 agents for collaboration")
        
        # Context management
        with st.expander("Context Management"):
            st.write("Manage conversation context and memory")
            
            # Show current context
            if st.session_state.chat_context:
                st.write("**Current Context:**")
                st.json(st.session_state.chat_context)
            
            # Clear context
            if st.button("Clear Context"):
                st.session_state.chat_context = {}
                st.success("Context cleared")
        
        # Export conversation
        with st.expander("Export Conversation"):
            st.write("Export your conversation history")
            
            export_format = st.selectbox(
                "Export Format",
                ["JSON", "Markdown", "Plain Text"]
            )
            
            if st.button("Export"):
                if st.session_state.chat_messages:
                    exported_data = self.export_conversation(export_format)
                    st.download_button(
                        label="Download Conversation",
                        data=exported_data,
                        file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}",
                        mime="application/octet-stream"
                    )
                else:
                    st.warning("No conversation to export")
    
    def export_conversation(self, format_type: str) -> str:
        """Export conversation in specified format"""
        if format_type == "JSON":
            return json.dumps(st.session_state.chat_messages, indent=2, default=str)
        
        elif format_type == "Markdown":
            markdown = "# Chat Conversation\n\n"
            for message in st.session_state.chat_messages:
                role = "**User**" if message["role"] == "user" else "**Assistant**"
                markdown += f"{role}: {message['content']}\n\n"
            return markdown
        
        elif format_type == "Plain Text":
            text = "Chat Conversation\n" + "="*50 + "\n\n"
            for message in st.session_state.chat_messages:
                role = "User" if message["role"] == "user" else "Assistant"
                text += f"{role}: {message['content']}\n\n"
            return text
        
        return ""
    
    def handle_multi_agent_request(self, user_input: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Handle multi-agent collaborative request"""
        if not st.session_state.chat_context.get('multi_agent_mode'):
            return self.send_message_to_agent(user_input, settings['agent_type'])
        
        selected_agents = st.session_state.chat_context.get('selected_agents', [])
        responses = {}
        
        # Send request to each selected agent
        for agent_type in selected_agents:
            response = self.send_message_to_agent(user_input, agent_type, st.session_state.chat_context)
            responses[agent_type] = response
        
        # Combine responses
        combined_response = "**Multi-Agent Response:**\n\n"
        for agent_type, response in responses.items():
            agent_name = self.get_agent_types()[agent_type].split(' - ')[0]
            combined_response += f"**{agent_name}:**\n{response.get('response', 'No response')}\n\n"
        
        return {
            "response": combined_response,
            "multi_agent_responses": responses,
            "confidence": sum(r.get('confidence', 0) for r in responses.values()) / len(responses)
        }
    
    def render_chat_input(self, settings: Dict[str, Any]):
        """Render chat input area"""
        # Chat input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message
            st.session_state.chat_messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now()
            })
            
            # Show processing indicator
            with st.spinner("Thinking..."):
                # Handle request (multi-agent or single agent)
                response = self.handle_multi_agent_request(user_input, settings)
                
                # Add assistant response
                assistant_message = {
                    "role": "assistant",
                    "content": response.get('response', 'No response received'),
                    "timestamp": datetime.now(),
                    "metadata": {
                        "agent_type": settings['agent_type'],
                        "confidence": response.get('confidence', 0),
                        "execution_time": response.get('execution_time', 0),
                        "tools_used": response.get('tools_used', []),
                        "reasoning": response.get('reasoning', '')
                    }
                }
                
                st.session_state.chat_messages.append(assistant_message)
                
                # Update context
                st.session_state.chat_context['last_response'] = response
                st.session_state.chat_context['last_agent'] = settings['agent_type']
            
            st.rerun()
    
    def render(self):
        """Render the complete chat interface"""
        st.title("💬 Chat Interface")
        
        # Get settings from sidebar
        settings = self.render_chat_settings()
        
        # Main chat area
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Conversation history
            self.render_conversation_history()
            
            # Chat input
            self.render_chat_input(settings)
        
        with col2:
            # Quick actions
            self.render_quick_actions()
            
            # Advanced features
            self.render_advanced_features()
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_messages = []
                st.session_state.chat_context = {}
                st.rerun()
        
        with col2:
            if st.button("💾 Save Session"):
                # Save current session
                session_data = {
                    "messages": st.session_state.chat_messages,
                    "context": st.session_state.chat_context,
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.conversation_history.append(session_data)
                st.success("Session saved!")
        
        with col3:
            if st.button("📋 Load Session"):
                if st.session_state.conversation_history:
                    # Load most recent session
                    latest_session = st.session_state.conversation_history[-1]
                    st.session_state.chat_messages = latest_session["messages"]
                    st.session_state.chat_context = latest_session["context"]
                    st.success("Session loaded!")
                    st.rerun()
                else:
                    st.warning("No saved sessions found")
