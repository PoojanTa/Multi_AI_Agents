import asyncio
import json
import logging
from typing import Dict, Any, List
from backend.agents.base_agent import BaseAgent
from backend.models.schemas import AgentTask, AgentResponse, AgentType

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    """Agent specialized in research and information gathering"""
    
    def __init__(self):
        super().__init__(
            agent_type=AgentType.RESEARCH,
            name="Research Agent",
            description="Specialized in conducting research, fact-checking, and gathering information from various sources"
        )
    
    def get_system_prompt(self) -> str:
        return """You are a professional research agent with expertise in:
        - Conducting comprehensive research on any topic
        - Fact-checking and verifying information
        - Synthesizing information from multiple sources
        - Providing well-structured, accurate, and detailed research reports
        - Identifying reliable sources and credible information
        - Analyzing trends and patterns in data
        
        Your responses should be:
        - Factual and well-researched
        - Structured with clear headings and bullet points
        - Include relevant context and background information
        - Cite sources when applicable
        - Objective and unbiased
        
        When conducting research, always provide:
        1. Executive summary
        2. Key findings
        3. Detailed analysis
        4. Recommendations (if applicable)
        5. Sources and references
        """
    
    def get_capabilities(self) -> List[str]:
        return [
            "Information research and gathering",
            "Fact-checking and verification",
            "Trend analysis",
            "Market research",
            "Academic research",
            "Competitive analysis",
            "Data synthesis",
            "Report generation",
            "Source verification",
            "Literature review"
        ]
    
    async def execute_task(self, task: AgentTask) -> AgentResponse:
        """Execute research task"""
        try:
            # Analyze the research request
            research_type = await self._identify_research_type(task.prompt)
            
            # Generate research plan
            research_plan = await self._create_research_plan(task.prompt, research_type)
            
            # Conduct research
            research_results = await self._conduct_research(task.prompt, research_plan)
            
            # Synthesize findings
            final_report = await self._synthesize_findings(research_results)
            
            # Calculate confidence based on research quality
            confidence = self._calculate_confidence(research_results)
            
            return AgentResponse(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                response=final_report,
                confidence=confidence,
                reasoning=f"Research conducted using {research_type} methodology",
                tools_used=["research_planner", "information_gatherer", "fact_checker", "synthesizer"],
                metadata={
                    "research_type": research_type,
                    "research_plan": research_plan,
                    "sources_analyzed": len(research_results.get("sources", [])),
                    "key_findings_count": len(research_results.get("key_findings", []))
                }
            )
            
        except Exception as e:
            logger.error(f"Research agent error: {e}")
            return AgentResponse(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                response=f"Research task failed: {str(e)}",
                confidence=0.0,
                reasoning=f"Error in research process: {str(e)}"
            )
    
    async def _identify_research_type(self, prompt: str) -> str:
        """Identify the type of research needed"""
        analysis_prompt = f"""
        Analyze this research request and identify the type of research needed:
        
        Request: {prompt}
        
        Research types:
        - Market Research: Business, market analysis, competitor analysis
        - Academic Research: Scientific, scholarly, theoretical topics
        - Factual Research: Fact-checking, verification, current events
        - Trend Analysis: Patterns, forecasting, data analysis
        - Technical Research: Technology, engineering, software topics
        - General Research: Broad topics, general information gathering
        
        Respond with just the research type and a brief explanation.
        """
        
        result = await self.generate_response(analysis_prompt, temperature=0.3)
        
        if result["success"]:
            return result["content"]
        else:
            return "General Research"
    
    async def _create_research_plan(self, prompt: str, research_type: str) -> Dict[str, Any]:
        """Create a structured research plan"""
        plan_prompt = f"""
        Create a detailed research plan for this request:
        
        Topic: {prompt}
        Research Type: {research_type}
        
        Provide a structured plan with:
        1. Research objectives
        2. Key questions to answer
        3. Information sources to explore
        4. Research methodology
        5. Expected deliverables
        
        Format as JSON with these fields: objectives, key_questions, sources, methodology, deliverables
        """
        
        result = await self.generate_response(plan_prompt, temperature=0.3)
        
        if result["success"]:
            try:
                return json.loads(result["content"])
            except json.JSONDecodeError:
                return {
                    "objectives": ["Gather comprehensive information"],
                    "key_questions": ["What are the main aspects to explore?"],
                    "sources": ["Academic papers", "Industry reports", "News articles"],
                    "methodology": "Systematic information gathering and analysis",
                    "deliverables": "Detailed research report"
                }
        else:
            return {}
    
    async def _conduct_research(self, prompt: str, research_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct the actual research"""
        research_prompt = f"""
        Conduct comprehensive research on: {prompt}
        
        Research Plan: {json.dumps(research_plan, indent=2)}
        
        Provide detailed research findings including:
        1. Executive Summary
        2. Key Findings (at least 5)
        3. Detailed Analysis
        4. Current Trends and Patterns
        5. Implications and Insights
        6. Potential Sources and References
        7. Limitations and Considerations
        
        Make this a comprehensive research report with specific details and insights.
        """
        
        result = await self.generate_response(research_prompt, temperature=0.4, max_tokens=2048)
        
        if result["success"]:
            return {
                "content": result["content"],
                "sources": research_plan.get("sources", []),
                "key_findings": self._extract_key_findings(result["content"]),
                "methodology": research_plan.get("methodology", ""),
                "quality_score": self._assess_research_quality(result["content"])
            }
        else:
            return {"content": "Research could not be completed", "sources": [], "key_findings": []}
    
    async def _synthesize_findings(self, research_results: Dict[str, Any]) -> str:
        """Synthesize research findings into a final report"""
        synthesis_prompt = f"""
        Synthesize these research findings into a comprehensive, well-structured report:
        
        Research Content: {research_results.get('content', '')}
        Key Findings: {research_results.get('key_findings', [])}
        
        Create a final report with:
        1. Executive Summary
        2. Methodology
        3. Key Findings and Insights
        4. Detailed Analysis
        5. Trends and Patterns
        6. Recommendations
        7. Limitations
        8. Conclusion
        
        Make it professional, comprehensive, and actionable.
        """
        
        result = await self.generate_response(synthesis_prompt, temperature=0.3, max_tokens=2048)
        
        if result["success"]:
            return result["content"]
        else:
            return research_results.get("content", "Research synthesis failed")
    
    def _extract_key_findings(self, content: str) -> List[str]:
        """Extract key findings from research content"""
        try:
            # Simple extraction based on common patterns
            lines = content.split('\n')
            key_findings = []
            
            for line in lines:
                if any(keyword in line.lower() for keyword in ['key finding', 'finding:', 'conclusion:', 'insight:']):
                    key_findings.append(line.strip())
                elif line.strip().startswith(('•', '-', '*')) and len(line.strip()) > 20:
                    key_findings.append(line.strip())
            
            return key_findings[:10]  # Limit to top 10 findings
        except Exception:
            return []
    
    def _assess_research_quality(self, content: str) -> float:
        """Assess the quality of research content"""
        try:
            # Simple quality assessment based on content characteristics
            word_count = len(content.split())
            
            quality_indicators = {
                'comprehensive': 0.2 if word_count > 500 else 0.0,
                'structured': 0.2 if any(header in content.lower() for header in ['summary', 'analysis', 'findings', 'conclusion']) else 0.0,
                'detailed': 0.2 if word_count > 1000 else 0.1,
                'specific': 0.2 if content.count('%') > 0 or content.count('$') > 0 or any(num in content for num in ['2023', '2024', '2025']) else 0.0,
                'objective': 0.2 if not any(subjective in content.lower() for subjective in ['i think', 'i believe', 'personally']) else 0.0
            }
            
            return sum(quality_indicators.values())
        except Exception:
            return 0.5
    
    def _calculate_confidence(self, research_results: Dict[str, Any]) -> float:
        """Calculate confidence score based on research quality"""
        try:
            quality_score = research_results.get("quality_score", 0.5)
            sources_count = len(research_results.get("sources", []))
            findings_count = len(research_results.get("key_findings", []))
            
            # Weighted confidence calculation
            confidence = (
                quality_score * 0.4 +
                min(sources_count / 5, 1.0) * 0.3 +
                min(findings_count / 5, 1.0) * 0.3
            )
            
            return min(confidence, 1.0)
        except Exception:
            return 0.5
    
    async def conduct_fact_check(self, statement: str) -> Dict[str, Any]:
        """Conduct fact-checking on a specific statement"""
        fact_check_prompt = f"""
        Fact-check this statement: "{statement}"
        
        Provide:
        1. Verification status (True/False/Partially True/Unverifiable)
        2. Detailed explanation
        3. Supporting evidence
        4. Contradicting evidence (if any)
        5. Context and nuance
        6. Confidence level (0-1)
        
        Be thorough and objective in your analysis.
        """
        
        result = await self.generate_response(fact_check_prompt, temperature=0.2)
        
        if result["success"]:
            return {
                "statement": statement,
                "verification": result["content"],
                "timestamp": self.last_active.isoformat()
            }
        else:
            return {
                "statement": statement,
                "verification": "Fact-check could not be completed",
                "error": result.get("error", "Unknown error")
            }
