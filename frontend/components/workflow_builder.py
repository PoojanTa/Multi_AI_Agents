import streamlit as st
import requests
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import graphviz

class WorkflowBuilder:
    """Workflow builder for creating multi-agent workflows"""
    
    def __init__(self, backend_url: str):
        self.backend_url = backend_url
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state for workflow builder"""
        if 'workflow_steps' not in st.session_state:
            st.session_state.workflow_steps = []
        if 'current_workflow' not in st.session_state:
            st.session_state.current_workflow = None
        if 'workflow_history' not in st.session_state:
            st.session_state.workflow_history = []
        if 'workflow_templates' not in st.session_state:
            st.session_state.workflow_templates = self.get_default_templates()
    
    def get_default_templates(self) -> List[Dict[str, Any]]:
        """Get default workflow templates"""
        return [
            {
                "name": "Research & Analysis",
                "description": "Comprehensive research followed by detailed analysis",
                "steps": [
                    {
                        "id": "research",
                        "agent_type": "research",
                        "prompt": "Conduct comprehensive research on the given topic",
                        "dependencies": [],
                        "context_keys": []
                    },
                    {
                        "id": "analysis",
                        "agent_type": "analyst",
                        "prompt": "Analyze the research findings and provide insights",
                        "dependencies": ["research"],
                        "context_keys": ["research"]
                    }
                ]
            },
            {
                "name": "Code Development",
                "description": "Research requirements, develop code, and create documentation",
                "steps": [
                    {
                        "id": "requirements",
                        "agent_type": "research",
                        "prompt": "Research technical requirements and best practices",
                        "dependencies": [],
                        "context_keys": []
                    },
                    {
                        "id": "coding",
                        "agent_type": "coding",
                        "prompt": "Develop code based on requirements",
                        "dependencies": ["requirements"],
                        "context_keys": ["requirements"]
                    },
                    {
                        "id": "documentation",
                        "agent_type": "document",
                        "prompt": "Create comprehensive documentation",
                        "dependencies": ["coding"],
                        "context_keys": ["requirements", "coding"]
                    }
                ]
            },
            {
                "name": "Document Analysis",
                "description": "Process document, extract insights, and generate summary",
                "steps": [
                    {
                        "id": "processing",
                        "agent_type": "document",
                        "prompt": "Process and analyze the document",
                        "dependencies": [],
                        "context_keys": []
                    },
                    {
                        "id": "insights",
                        "agent_type": "analyst",
                        "prompt": "Extract key insights and patterns",
                        "dependencies": ["processing"],
                        "context_keys": ["processing"]
                    },
                    {
                        "id": "summary",
                        "agent_type": "document",
                        "prompt": "Create executive summary",
                        "dependencies": ["insights"],
                        "context_keys": ["processing", "insights"]
                    }
                ]
            }
        ]
    
    def get_agent_types(self) -> Dict[str, str]:
        """Get available agent types"""
        return {
            "research": "🔍 Research Agent",
            "analyst": "📊 Analyst Agent",
            "coding": "💻 Coding Agent",
            "document": "📄 Document Agent"
        }
    
    def execute_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow"""
        try:
            response = requests.post(
                f"{self.backend_url}/agents/workflow",
                json=workflow,
                timeout=300  # 5 minutes timeout for workflows
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"Request failed with status {response.status_code}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_collaborative_workflow(self, objective: str, agent_types: List[str]) -> Dict[str, Any]:
        """Create a collaborative workflow"""
        try:
            response = requests.post(
                f"{self.backend_url}/agents/collaborative_workflow",
                json={
                    "objective": objective,
                    "agent_types": agent_types
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"Request failed with status {response.status_code}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def render_workflow_builder(self):
        """Render workflow builder interface"""
        st.subheader("🔧 Workflow Builder")
        
        # Workflow basic info
        col1, col2 = st.columns(2)
        
        with col1:
            workflow_name = st.text_input(
                "Workflow Name",
                placeholder="Enter workflow name...",
                value="New Workflow"
            )
        
        with col2:
            workflow_description = st.text_area(
                "Description",
                placeholder="Describe what this workflow does...",
                height=100
            )
        
        # Step management
        st.write("**Workflow Steps:**")
        
        # Add step button
        if st.button("➕ Add Step"):
            new_step = {
                "id": f"step_{len(st.session_state.workflow_steps) + 1}",
                "agent_type": "research",
                "prompt": "",
                "dependencies": [],
                "context_keys": []
            }
            st.session_state.workflow_steps.append(new_step)
            st.rerun()
        
        # Display and edit steps
        agent_types = self.get_agent_types()
        
        for i, step in enumerate(st.session_state.workflow_steps):
            with st.expander(f"Step {i+1}: {step['id']}", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Step ID
                    step_id = st.text_input(
                        "Step ID",
                        value=step["id"],
                        key=f"step_id_{i}"
                    )
                    step["id"] = step_id
                    
                    # Agent type
                    agent_type = st.selectbox(
                        "Agent Type",
                        options=list(agent_types.keys()),
                        format_func=lambda x: agent_types[x],
                        index=list(agent_types.keys()).index(step["agent_type"]),
                        key=f"agent_type_{i}"
                    )
                    step["agent_type"] = agent_type
                
                with col2:
                    # Dependencies
                    available_steps = [s["id"] for j, s in enumerate(st.session_state.workflow_steps) if j < i]
                    dependencies = st.multiselect(
                        "Dependencies",
                        options=available_steps,
                        default=[dep for dep in step["dependencies"] if dep in available_steps],
                        key=f"dependencies_{i}"
                    )
                    step["dependencies"] = dependencies
                    
                    # Context keys
                    context_keys = st.multiselect(
                        "Context Keys",
                        options=available_steps,
                        default=[key for key in step["context_keys"] if key in available_steps],
                        key=f"context_keys_{i}"
                    )
                    step["context_keys"] = context_keys
                
                # Prompt
                prompt = st.text_area(
                    "Prompt",
                    value=step["prompt"],
                    placeholder="Enter the prompt for this step...",
                    height=100,
                    key=f"prompt_{i}"
                )
                step["prompt"] = prompt
                
                # Actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🔼 Move Up", key=f"up_{i}", disabled=i == 0):
                        st.session_state.workflow_steps[i], st.session_state.workflow_steps[i-1] = \
                            st.session_state.workflow_steps[i-1], st.session_state.workflow_steps[i]
                        st.rerun()
                
                with col2:
                    if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                        st.session_state.workflow_steps.pop(i)
                        st.rerun()
        
        # Workflow actions
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Workflow"):
                if workflow_name and st.session_state.workflow_steps:
                    workflow = {
                        "id": str(uuid.uuid4()),
                        "name": workflow_name,
                        "description": workflow_description,
                        "steps": st.session_state.workflow_steps,
                        "created_at": datetime.now().isoformat()
                    }
                    
                    # Save to history
                    st.session_state.workflow_history.append(workflow)
                    st.success("Workflow saved successfully!")
                else:
                    st.warning("Please provide a name and at least one step")
        
        with col2:
            if st.button("🚀 Execute Workflow"):
                if st.session_state.workflow_steps:
                    workflow = {
                        "id": str(uuid.uuid4()),
                        "name": workflow_name,
                        "description": workflow_description,
                        "steps": st.session_state.workflow_steps
                    }
                    
                    with st.spinner("Executing workflow..."):
                        result = self.execute_workflow(workflow)
                    
                    if result.get("success", True):
                        st.success("Workflow executed successfully!")
                        st.json(result)
                    else:
                        st.error(f"Workflow execution failed: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("Please add at least one step")
        
        with col3:
            if st.button("🧹 Clear Workflow"):
                st.session_state.workflow_steps = []
                st.rerun()
    
    def render_workflow_templates(self):
        """Render workflow templates section"""
        st.subheader("📋 Workflow Templates")
        
        # Template selection
        template_names = [template["name"] for template in st.session_state.workflow_templates]
        
        if template_names:
            selected_template = st.selectbox(
                "Select Template",
                options=template_names,
                index=0
            )
            
            # Show template details
            template = next(t for t in st.session_state.workflow_templates if t["name"] == selected_template)
            
            with st.expander("Template Details", expanded=True):
                st.write(f"**Description:** {template['description']}")
                st.write(f"**Steps:** {len(template['steps'])}")
                
                # Show steps
                for i, step in enumerate(template['steps']):
                    st.write(f"**Step {i+1}:** {step['agent_type'].title()} - {step['id']}")
                    if step['dependencies']:
                        st.write(f"  Dependencies: {', '.join(step['dependencies'])}")
            
            # Actions
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Load Template"):
                    st.session_state.workflow_steps = template['steps'].copy()
                    st.success(f"Template '{selected_template}' loaded!")
                    st.rerun()
            
            with col2:
                if st.button("🚀 Execute Template"):
                    workflow = {
                        "id": str(uuid.uuid4()),
                        "name": template["name"],
                        "description": template["description"],
                        "steps": template["steps"]
                    }
                    
                    with st.spinner("Executing template..."):
                        result = self.execute_workflow(workflow)
                    
                    if result.get("success", True):
                        st.success("Template executed successfully!")
                        st.json(result)
                    else:
                        st.error(f"Template execution failed: {result.get('error', 'Unknown error')}")
        
        # Create custom template
        st.markdown("---")
        st.write("**Create Custom Template:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            template_name = st.text_input("Template Name", placeholder="Enter template name...")
        
        with col2:
            template_desc = st.text_input("Template Description", placeholder="Enter description...")
        
        if st.button("💾 Save as Template"):
            if template_name and st.session_state.workflow_steps:
                new_template = {
                    "name": template_name,
                    "description": template_desc,
                    "steps": st.session_state.workflow_steps.copy()
                }
                
                st.session_state.workflow_templates.append(new_template)
                st.success(f"Template '{template_name}' saved!")
            else:
                st.warning("Please provide a name and create some workflow steps")
    
    def render_workflow_history(self):
        """Render workflow execution history"""
        st.subheader("📈 Workflow History")
        
        if not st.session_state.workflow_history:
            st.info("No workflow history available")
            return
        
        # Display history
        for i, workflow in enumerate(reversed(st.session_state.workflow_history)):
            with st.expander(f"🔄 {workflow['name']} - {workflow.get('created_at', 'Unknown time')}"):
                st.write(f"**Description:** {workflow.get('description', 'No description')}")
                st.write(f"**Steps:** {len(workflow.get('steps', []))}")
                st.write(f"**Status:** {workflow.get('status', 'Saved')}")
                
                # Show steps
                for j, step in enumerate(workflow.get('steps', [])):
                    st.write(f"  {j+1}. {step['agent_type'].title()} - {step['id']}")
                
                # Actions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button(f"📥 Load", key=f"load_{i}"):
                        st.session_state.workflow_steps = workflow['steps'].copy()
                        st.success("Workflow loaded!")
                        st.rerun()
                
                with col2:
                    if st.button(f"🚀 Execute", key=f"execute_{i}"):
                        with st.spinner("Executing workflow..."):
                            result = self.execute_workflow(workflow)
                        
                        if result.get("success", True):
                            st.success("Workflow executed successfully!")
                        else:
                            st.error(f"Execution failed: {result.get('error', 'Unknown error')}")
                
                with col3:
                    if st.button(f"🗑️ Delete", key=f"delete_history_{i}"):
                        st.session_state.workflow_history.pop(-(i+1))
                        st.rerun()
    
    def render_workflow_visualization(self):
        """Render workflow visualization"""
        st.subheader("🎨 Workflow Visualization")
        
        if not st.session_state.workflow_steps:
            st.info("No workflow steps to visualize")
            return
        
        # Create workflow graph
        try:
            dot = graphviz.Digraph(comment='Workflow')
            dot.attr(rankdir='TB')
            
            # Add nodes
            for step in st.session_state.workflow_steps:
                agent_type = step["agent_type"]
                color = {
                    "research": "lightblue",
                    "analyst": "lightgreen",
                    "coding": "lightcoral",
                    "document": "lightyellow"
                }.get(agent_type, "lightgray")
                
                dot.node(step["id"], f"{step['id']}\n({agent_type})", 
                        style='filled', fillcolor=color)
            
            # Add edges for dependencies
            for step in st.session_state.workflow_steps:
                for dep in step["dependencies"]:
                    dot.edge(dep, step["id"])
            
            # Display graph
            st.graphviz_chart(dot.source)
            
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            st.info("Workflow visualization requires graphviz to be installed")
        
        # Alternative text-based visualization
        st.write("**Workflow Structure:**")
        for i, step in enumerate(st.session_state.workflow_steps):
            indent = "  " * len(step["dependencies"])
            st.write(f"{indent}{i+1}. {step['id']} ({step['agent_type']})")
            if step["dependencies"]:
                st.write(f"{indent}   Dependencies: {', '.join(step['dependencies'])}")
    
    def render_collaborative_workflow(self):
        """Render collaborative workflow creation"""
        st.subheader("🤝 Collaborative Workflow")
        
        st.write("Create workflows that automatically coordinate multiple agents based on your objective.")
        
        # Objective input
        objective = st.text_area(
            "Objective",
            placeholder="Describe what you want to accomplish...",
            height=100,
            help="The system will automatically create a workflow with the appropriate agents"
        )
        
        # Agent selection
        agent_types = self.get_agent_types()
        selected_agents = st.multiselect(
            "Select Agents",
            options=list(agent_types.keys()),
            format_func=lambda x: agent_types[x],
            default=list(agent_types.keys()),
            help="Choose which agents should collaborate"
        )
        
        # Create and execute
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔧 Create Workflow"):
                if objective and selected_agents:
                    with st.spinner("Creating collaborative workflow..."):
                        result = self.create_collaborative_workflow(objective, selected_agents)
                    
                    if result.get("success", True):
                        st.success("Collaborative workflow created!")
                        # Load the created workflow
                        if "workflow" in result:
                            st.session_state.workflow_steps = result["workflow"]["steps"]
                            st.rerun()
                    else:
                        st.error(f"Failed to create workflow: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("Please provide an objective and select at least one agent")
        
        with col2:
            if st.button("🚀 Create & Execute"):
                if objective and selected_agents:
                    with st.spinner("Creating and executing workflow..."):
                        # Create workflow
                        workflow = {
                            "id": str(uuid.uuid4()),
                            "name": f"Collaborative: {objective[:50]}...",
                            "description": f"Auto-generated collaborative workflow for: {objective}",
                            "steps": []  # Will be generated by backend
                        }
                        
                        # This would normally call a collaborative endpoint
                        result = self.execute_workflow(workflow)
                    
                    if result.get("success", True):
                        st.success("Collaborative workflow executed successfully!")
                        st.json(result)
                    else:
                        st.error(f"Execution failed: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("Please provide an objective and select at least one agent")
    
    def render(self):
        """Render the complete workflow builder interface"""
        st.title("⚙️ Workflow Builder")
        
        # Navigation tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔧 Builder",
            "📋 Templates", 
            "📈 History",
            "🎨 Visualization",
            "🤝 Collaborative"
        ])
        
        with tab1:
            self.render_workflow_builder()
        
        with tab2:
            self.render_workflow_templates()
        
        with tab3:
            self.render_workflow_history()
        
        with tab4:
            self.render_workflow_visualization()
        
        with tab5:
            self.render_collaborative_workflow()
