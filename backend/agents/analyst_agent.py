import asyncio
import json
import logging
from typing import Dict, Any, List
from backend.agents.base_agent import BaseAgent
from backend.models.schemas import AgentTask, AgentResponse, AgentType

logger = logging.getLogger(__name__)

class AnalystAgent(BaseAgent):
    """Agent specialized in data analysis, insights generation, and pattern recognition"""
    
    def __init__(self):
        super().__init__(
            agent_type=AgentType.ANALYST,
            name="Analyst Agent",
            description="Specialized in data analysis, statistical insights, pattern recognition, and predictive modeling"
        )
    
    def get_system_prompt(self) -> str:
        return """You are a professional data analyst with expertise in:
        - Statistical analysis and interpretation
        - Pattern recognition and trend analysis
        - Data visualization recommendations
        - Predictive modeling and forecasting
        - Business intelligence and insights
        - Risk analysis and assessment
        - Performance metrics and KPIs
        - Market analysis and competitive intelligence
        
        Your responses should be:
        - Data-driven and analytical
        - Include specific metrics and numbers when possible
        - Provide actionable insights and recommendations
        - Explain statistical concepts clearly
        - Identify trends, patterns, and anomalies
        - Consider multiple perspectives and scenarios
        
        When analyzing data, always provide:
        1. Executive summary of key insights
        2. Detailed analysis with supporting data
        3. Trend identification and patterns
        4. Risk factors and considerations
        5. Recommendations and next steps
        6. Confidence intervals and limitations
        """
    
    def get_capabilities(self) -> List[str]:
        return [
            "Statistical analysis",
            "Trend analysis",
            "Pattern recognition",
            "Predictive modeling",
            "Business intelligence",
            "Risk assessment",
            "Performance analysis",
            "Market analysis",
            "Data visualization guidance",
            "Anomaly detection",
            "Correlation analysis",
            "Forecasting"
        ]
    
    async def execute_task(self, task: AgentTask) -> AgentResponse:
        """Execute analysis task"""
        try:
            # Determine analysis type
            analysis_type = await self._identify_analysis_type(task.prompt)
            
            # Extract and structure data context
            data_context = await self._extract_data_context(task.prompt)
            
            # Perform analysis
            analysis_results = await self._perform_analysis(task.prompt, analysis_type, data_context)
            
            # Generate insights and recommendations
            insights = await self._generate_insights(analysis_results, analysis_type)
            
            # Calculate confidence based on data quality and analysis depth
            confidence = self._calculate_analysis_confidence(analysis_results, data_context)
            
            return AgentResponse(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                response=insights,
                confidence=confidence,
                reasoning=f"Analysis conducted using {analysis_type} methodology",
                tools_used=["statistical_analyzer", "pattern_detector", "trend_analyzer", "insight_generator"],
                metadata={
                    "analysis_type": analysis_type,
                    "data_points_analyzed": data_context.get("data_points", 0),
                    "patterns_identified": len(analysis_results.get("patterns", [])),
                    "recommendations_count": len(analysis_results.get("recommendations", []))
                }
            )
            
        except Exception as e:
            logger.error(f"Analyst agent error: {e}")
            return AgentResponse(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                response=f"Analysis task failed: {str(e)}",
                confidence=0.0,
                reasoning=f"Error in analysis process: {str(e)}"
            )
    
    async def _identify_analysis_type(self, prompt: str) -> str:
        """Identify the type of analysis needed"""
        analysis_prompt = f"""
        Analyze this request and identify the type of analysis needed:
        
        Request: {prompt}
        
        Analysis types:
        - Descriptive Analysis: What happened? Summary statistics, trends
        - Diagnostic Analysis: Why did it happen? Root cause analysis
        - Predictive Analysis: What will happen? Forecasting, modeling
        - Prescriptive Analysis: What should we do? Recommendations, optimization
        - Exploratory Analysis: What patterns exist? Data exploration, discovery
        - Comparative Analysis: How do things compare? Benchmarking, A/B testing
        
        Respond with just the analysis type and key focus areas.
        """
        
        result = await self.generate_response(analysis_prompt, temperature=0.3)
        
        if result["success"]:
            return result["content"]
        else:
            return "Descriptive Analysis"
    
    async def _extract_data_context(self, prompt: str) -> Dict[str, Any]:
        """Extract data context from the prompt"""
        context_prompt = f"""
        Extract data context from this analysis request:
        
        Request: {prompt}
        
        Identify:
        1. Data sources mentioned
        2. Time periods or ranges
        3. Key metrics or variables
        4. Target audience or stakeholders
        5. Business context or domain
        6. Constraints or limitations
        
        Format as JSON with fields: sources, time_period, metrics, audience, domain, constraints
        """
        
        result = await self.generate_response(context_prompt, temperature=0.2)
        
        if result["success"]:
            try:
                return json.loads(result["content"])
            except json.JSONDecodeError:
                return {
                    "sources": ["Not specified"],
                    "time_period": "Not specified",
                    "metrics": ["General metrics"],
                    "audience": "General audience",
                    "domain": "General business",
                    "constraints": ["No specific constraints mentioned"]
                }
        else:
            return {}
    
    async def _perform_analysis(self, prompt: str, analysis_type: str, data_context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the actual analysis"""
        analysis_prompt = f"""
        Perform comprehensive {analysis_type} on: {prompt}
        
        Data Context: {json.dumps(data_context, indent=2)}
        
        Provide detailed analysis including:
        1. Data Summary and Overview
        2. Key Metrics and Statistics
        3. Trend Analysis and Patterns
        4. Anomalies and Outliers
        5. Correlations and Relationships
        6. Segments and Classifications
        7. Risk Factors and Considerations
        8. Predictive Insights (if applicable)
        9. Comparative Analysis (if applicable)
        10. Actionable Recommendations
        
        Include specific numbers, percentages, and quantitative insights where possible.
        """
        
        result = await self.generate_response(analysis_prompt, temperature=0.4, max_tokens=2048)
        
        if result["success"]:
            return {
                "content": result["content"],
                "analysis_type": analysis_type,
                "patterns": self._extract_patterns(result["content"]),
                "recommendations": self._extract_recommendations(result["content"]),
                "metrics": self._extract_metrics(result["content"]),
                "quality_score": self._assess_analysis_quality(result["content"])
            }
        else:
            return {"content": "Analysis could not be completed", "patterns": [], "recommendations": []}
    
    async def _generate_insights(self, analysis_results: Dict[str, Any], analysis_type: str) -> str:
        """Generate insights and recommendations from analysis"""
        insights_prompt = f"""
        Generate actionable insights and recommendations from this analysis:
        
        Analysis Results: {analysis_results.get('content', '')}
        Analysis Type: {analysis_type}
        Key Patterns: {analysis_results.get('patterns', [])}
        
        Create a comprehensive insights report with:
        1. Executive Summary
        2. Key Findings and Insights
        3. Trend Analysis
        4. Risk Assessment
        5. Opportunities Identified
        6. Strategic Recommendations
        7. Implementation Priorities
        8. Success Metrics
        9. Monitoring and Review Plan
        10. Conclusion
        
        Make insights actionable and business-focused.
        """
        
        result = await self.generate_response(insights_prompt, temperature=0.3, max_tokens=2048)
        
        if result["success"]:
            return result["content"]
        else:
            return analysis_results.get("content", "Insights generation failed")
    
    def _extract_patterns(self, content: str) -> List[str]:
        """Extract patterns from analysis content"""
        try:
            lines = content.split('\n')
            patterns = []
            
            for line in lines:
                if any(keyword in line.lower() for keyword in ['pattern', 'trend', 'correlation', 'relationship']):
                    patterns.append(line.strip())
                elif line.strip().startswith(('•', '-', '*')) and any(word in line.lower() for word in ['increase', 'decrease', 'correlation', 'trend']):
                    patterns.append(line.strip())
            
            return patterns[:8]  # Limit to top 8 patterns
        except Exception:
            return []
    
    def _extract_recommendations(self, content: str) -> List[str]:
        """Extract recommendations from analysis content"""
        try:
            lines = content.split('\n')
            recommendations = []
            
            for line in lines:
                if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should', 'action']):
                    recommendations.append(line.strip())
                elif line.strip().startswith(('•', '-', '*')) and any(word in line.lower() for word in ['implement', 'consider', 'improve', 'optimize']):
                    recommendations.append(line.strip())
            
            return recommendations[:6]  # Limit to top 6 recommendations
        except Exception:
            return []
    
    def _extract_metrics(self, content: str) -> List[str]:
        """Extract metrics from analysis content"""
        try:
            import re
            
            # Find percentages, numbers, and metrics
            percentage_pattern = r'\b\d+\.?\d*%\b'
            number_pattern = r'\b\d+\.?\d*[KMB]?\b'
            
            percentages = re.findall(percentage_pattern, content)
            numbers = re.findall(number_pattern, content)
            
            metrics = []
            lines = content.split('\n')
            
            for line in lines:
                if any(metric in line.lower() for metric in ['kpi', 'metric', 'rate', 'ratio', 'score']):
                    metrics.append(line.strip())
            
            return list(set(metrics + percentages[:5] + numbers[:5]))[:10]
        except Exception:
            return []
    
    def _assess_analysis_quality(self, content: str) -> float:
        """Assess the quality of analysis content"""
        try:
            word_count = len(content.split())
            
            quality_indicators = {
                'comprehensive': 0.2 if word_count > 800 else 0.1,
                'quantitative': 0.2 if any(indicator in content for indicator in ['%', '$', '±', 'correlation']) else 0.0,
                'structured': 0.2 if sum(1 for header in ['summary', 'analysis', 'findings', 'recommendations'] if header in content.lower()) >= 2 else 0.0,
                'actionable': 0.2 if any(action in content.lower() for action in ['recommend', 'should', 'implement', 'action']) else 0.0,
                'insightful': 0.2 if any(insight in content.lower() for insight in ['pattern', 'trend', 'correlation', 'insight']) else 0.0
            }
            
            return sum(quality_indicators.values())
        except Exception:
            return 0.5
    
    def _calculate_analysis_confidence(self, analysis_results: Dict[str, Any], data_context: Dict[str, Any]) -> float:
        """Calculate confidence score based on analysis quality"""
        try:
            quality_score = analysis_results.get("quality_score", 0.5)
            patterns_count = len(analysis_results.get("patterns", []))
            recommendations_count = len(analysis_results.get("recommendations", []))
            
            # Context quality assessment
            context_quality = 0.0
            if data_context.get("sources") and data_context["sources"] != ["Not specified"]:
                context_quality += 0.2
            if data_context.get("time_period") and data_context["time_period"] != "Not specified":
                context_quality += 0.2
            if data_context.get("metrics") and data_context["metrics"] != ["General metrics"]:
                context_quality += 0.1
            
            # Weighted confidence calculation
            confidence = (
                quality_score * 0.4 +
                min(patterns_count / 5, 1.0) * 0.2 +
                min(recommendations_count / 4, 1.0) * 0.2 +
                context_quality * 0.2
            )
            
            return min(confidence, 1.0)
        except Exception:
            return 0.5
    
    async def perform_risk_analysis(self, scenario: str) -> Dict[str, Any]:
        """Perform risk analysis on a specific scenario"""
        risk_prompt = f"""
        Perform comprehensive risk analysis for: {scenario}
        
        Analyze:
        1. Risk Identification and Categories
        2. Probability Assessment (High/Medium/Low)
        3. Impact Assessment (High/Medium/Low)
        4. Risk Matrix and Priority Ranking
        5. Mitigation Strategies
        6. Contingency Planning
        7. Monitoring and Review Requirements
        8. Overall Risk Score
        
        Provide quantitative assessments where possible.
        """
        
        result = await self.generate_response(risk_prompt, temperature=0.3)
        
        if result["success"]:
            return {
                "scenario": scenario,
                "risk_analysis": result["content"],
                "timestamp": self.last_active.isoformat()
            }
        else:
            return {
                "scenario": scenario,
                "risk_analysis": "Risk analysis could not be completed",
                "error": result.get("error", "Unknown error")
            }
