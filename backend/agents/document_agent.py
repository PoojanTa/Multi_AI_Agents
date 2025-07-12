import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from backend.agents.base_agent import BaseAgent
from backend.models.schemas import AgentTask, AgentResponse, AgentType
from backend.utils.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class DocumentAgent(BaseAgent):
    """Agent specialized in document processing, analysis, and content generation"""
    
    def __init__(self):
        super().__init__(
            agent_type=AgentType.DOCUMENT,
            name="Document Agent",
            description="Specialized in document processing, content analysis, summarization, and document generation"
        )
        self.document_processor = DocumentProcessor()
    
    def get_system_prompt(self) -> str:
        return """You are a professional document analyst with expertise in:
        - Document processing and analysis
        - Content summarization and extraction
        - Document structure and formatting
        - Information extraction and organization
        - Content quality assessment
        - Document classification and categorization
        - Technical writing and documentation
        - Report generation and formatting
        
        Your responses should be:
        - Well-structured and organized
        - Clear and concise
        - Properly formatted with headings and sections
        - Include relevant details and context
        - Maintain professional tone and style
        - Consider document purpose and audience
        
        When processing documents, always provide:
        1. Document summary and key points
        2. Structural analysis and organization
        3. Content quality assessment
        4. Key insights and findings
        5. Recommendations for improvement
        6. Actionable next steps
        """
    
    def get_capabilities(self) -> List[str]:
        return [
            "Document processing and parsing",
            "Content summarization",
            "Information extraction",
            "Document analysis",
            "Content generation",
            "Document formatting",
            "Report writing",
            "Content optimization",
            "Document classification",
            "Quality assessment",
            "Keyword extraction",
            "Document comparison"
        ]
    
    async def execute_task(self, task: AgentTask) -> AgentResponse:
        """Execute document processing task"""
        try:
            # Determine document task type
            task_type = await self._identify_document_task(task.prompt)
            
            # Extract document context and requirements
            doc_context = await self._extract_document_context(task.prompt)
            
            # Process document based on task type
            if "file_path" in task.context:
                # Process existing document
                doc_results = await self._process_existing_document(
                    task.context["file_path"], task.prompt, task_type
                )
            else:
                # Generate new document content
                doc_results = await self._generate_document_content(
                    task.prompt, task_type, doc_context
                )
            
            # Enhance and format results
            formatted_results = await self._format_document_results(
                doc_results, task_type, doc_context
            )
            
            # Calculate confidence based on document quality
            confidence = self._calculate_document_confidence(doc_results, doc_context)
            
            return AgentResponse(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                response=formatted_results,
                confidence=confidence,
                reasoning=f"Document processing completed using {task_type} approach",
                tools_used=["document_processor", "content_analyzer", "formatter", "optimizer"],
                metadata={
                    "task_type": task_type,
                    "document_type": doc_context.get("document_type", "Unknown"),
                    "word_count": self._count_words(formatted_results),
                    "sections_count": self._count_sections(formatted_results),
                    "processing_method": "existing_document" if "file_path" in task.context else "content_generation"
                }
            )
            
        except Exception as e:
            logger.error(f"Document agent error: {e}")
            return AgentResponse(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                response=f"Document processing failed: {str(e)}",
                confidence=0.0,
                reasoning=f"Error in document processing: {str(e)}"
            )
    
    async def _identify_document_task(self, prompt: str) -> str:
        """Identify the type of document task"""
        analysis_prompt = f"""
        Analyze this document request and identify the task type:
        
        Request: {prompt}
        
        Task types:
        - Document Analysis: Analyze and summarize existing document
        - Content Generation: Create new document content
        - Document Summarization: Extract key points and create summary
        - Information Extraction: Extract specific information or data
        - Document Comparison: Compare multiple documents
        - Content Optimization: Improve existing content
        - Report Generation: Create structured reports
        - Document Classification: Categorize and organize documents
        
        Respond with just the task type and key objectives.
        """
        
        result = await self.generate_response(analysis_prompt, temperature=0.3)
        
        if result["success"]:
            return result["content"]
        else:
            return "Document Analysis"
    
    async def _extract_document_context(self, prompt: str) -> Dict[str, Any]:
        """Extract document context and requirements"""
        context_prompt = f"""
        Extract document context from this request:
        
        Request: {prompt}
        
        Identify:
        1. Document type (report, article, manual, etc.)
        2. Target audience
        3. Purpose and objectives
        4. Tone and style requirements
        5. Length or scope requirements
        6. Specific sections or structure needed
        7. Key topics or themes
        8. Output format preferences
        
        Format as JSON with fields: document_type, audience, purpose, tone, length, structure, topics, format
        """
        
        result = await self.generate_response(context_prompt, temperature=0.2)
        
        if result["success"]:
            try:
                return json.loads(result["content"])
            except json.JSONDecodeError:
                return {
                    "document_type": "General document",
                    "audience": "General audience",
                    "purpose": "Information sharing",
                    "tone": "Professional",
                    "length": "Medium",
                    "structure": "Standard structure",
                    "topics": ["General topics"],
                    "format": "Text format"
                }
        else:
            return {}
    
    async def _process_existing_document(self, file_path: str, prompt: str, task_type: str) -> Dict[str, Any]:
        """Process an existing document"""
        try:
            # Process document using document processor
            processed_doc = self.document_processor.process_document(file_path)
            
            if "error" in processed_doc:
                return {"error": processed_doc["error"]}
            
            # Analyze document content based on task type
            analysis_prompt = f"""
            Perform {task_type} on this document:
            
            Document Content: {processed_doc["text"][:4000]}...
            Document Metadata: {json.dumps(processed_doc["metadata"], indent=2)}
            Keywords: {processed_doc["keywords"]}
            
            Task: {prompt}
            
            Provide comprehensive analysis including:
            1. Document Overview and Summary
            2. Key Findings and Insights
            3. Content Structure Analysis
            4. Important Sections and Topics
            5. Quality Assessment
            6. Recommendations and Action Items
            7. Relevant Quotes or Examples
            8. Conclusion and Next Steps
            """
            
            result = await self.generate_response(analysis_prompt, temperature=0.4, max_tokens=2048)
            
            if result["success"]:
                return {
                    "content": result["content"],
                    "original_doc": processed_doc,
                    "analysis_type": task_type,
                    "quality_score": self._assess_document_quality(result["content"])
                }
            else:
                return {"error": "Document analysis failed"}
                
        except Exception as e:
            logger.error(f"Error processing existing document: {e}")
            return {"error": str(e)}
    
    async def _generate_document_content(self, prompt: str, task_type: str, doc_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate new document content"""
        content_prompt = f"""
        Generate document content for: {prompt}
        
        Task Type: {task_type}
        Document Context: {json.dumps(doc_context, indent=2)}
        
        Create a comprehensive document with:
        1. Executive Summary or Introduction
        2. Main Content Sections (organized logically)
        3. Key Points and Supporting Details
        4. Data, Examples, or Case Studies (if applicable)
        5. Analysis and Insights
        6. Recommendations or Action Items
        7. Conclusion or Summary
        8. Next Steps or Follow-up Items
        
        Ensure the content is:
        - Well-structured and organized
        - Appropriate for the target audience
        - Professional and engaging
        - Comprehensive and detailed
        - Actionable and practical
        """
        
        result = await self.generate_response(content_prompt, temperature=0.5, max_tokens=2048)
        
        if result["success"]:
            return {
                "content": result["content"],
                "generation_type": task_type,
                "context": doc_context,
                "quality_score": self._assess_document_quality(result["content"])
            }
        else:
            return {"error": "Content generation failed"}
    
    async def _format_document_results(self, doc_results: Dict[str, Any], task_type: str, doc_context: Dict[str, Any]) -> str:
        """Format document results for presentation"""
        if "error" in doc_results:
            return f"Document processing error: {doc_results['error']}"
        
        formatting_prompt = f"""
        Format this document content for professional presentation:
        
        Content: {doc_results.get('content', '')}
        Task Type: {task_type}
        Document Context: {json.dumps(doc_context, indent=2)}
        
        Format with:
        1. Professional document structure
        2. Clear headings and sections
        3. Proper paragraph breaks
        4. Bullet points where appropriate
        5. Emphasis on key points
        6. Consistent formatting style
        7. Professional tone and language
        8. Logical flow and organization
        
        Make it ready for professional use.
        """
        
        result = await self.generate_response(formatting_prompt, temperature=0.3, max_tokens=2048)
        
        if result["success"]:
            return result["content"]
        else:
            return doc_results.get("content", "Document formatting failed")
    
    def _count_words(self, text: str) -> int:
        """Count words in text"""
        try:
            return len(text.split())
        except Exception:
            return 0
    
    def _count_sections(self, text: str) -> int:
        """Count sections in formatted text"""
        try:
            # Count headings (lines starting with #, ##, etc.)
            lines = text.split('\n')
            sections = 0
            
            for line in lines:
                line = line.strip()
                if line.startswith('#') or line.endswith(':') or line.isupper():
                    sections += 1
            
            return sections
        except Exception:
            return 0
    
    def _assess_document_quality(self, content: str) -> float:
        """Assess document quality"""
        try:
            word_count = len(content.split())
            
            quality_indicators = {
                'comprehensive': 0.2 if word_count > 500 else 0.1,
                'structured': 0.2 if any(header in content.lower() for header in ['summary', 'introduction', 'conclusion', 'recommendations']) else 0.0,
                'detailed': 0.2 if word_count > 1000 else 0.1,
                'professional': 0.2 if not any(informal in content.lower() for informal in ['gonna', 'wanna', 'kinda']) else 0.0,
                'actionable': 0.2 if any(action in content.lower() for action in ['recommend', 'should', 'action', 'next steps']) else 0.0
            }
            
            return sum(quality_indicators.values())
        except Exception:
            return 0.5
    
    def _calculate_document_confidence(self, doc_results: Dict[str, Any], doc_context: Dict[str, Any]) -> float:
        """Calculate confidence score for document processing"""
        try:
            if "error" in doc_results:
                return 0.0
            
            quality_score = doc_results.get("quality_score", 0.5)
            
            # Context completeness
            context_completeness = 0.0
            required_fields = ["document_type", "audience", "purpose", "tone"]
            for field in required_fields:
                if doc_context.get(field) and doc_context[field] != f"General {field}":
                    context_completeness += 0.1
            
            # Content quality
            content_quality = 0.0
            if doc_results.get("content"):
                content_length = len(doc_results["content"])
                if content_length > 500:
                    content_quality = 0.3
                elif content_length > 200:
                    content_quality = 0.2
                else:
                    content_quality = 0.1
            
            # Weighted confidence calculation
            confidence = (
                quality_score * 0.4 +
                context_completeness * 0.3 +
                content_quality * 0.3
            )
            
            return min(confidence, 1.0)
        except Exception:
            return 0.5
    
    async def summarize_document(self, file_path: str, summary_type: str = "executive") -> Dict[str, Any]:
        """Summarize a document"""
        try:
            processed_doc = self.document_processor.process_document(file_path)
            
            if "error" in processed_doc:
                return {"error": processed_doc["error"]}
            
            summary_prompt = f"""
            Create a {summary_type} summary of this document:
            
            Document: {processed_doc["text"][:3000]}...
            Keywords: {processed_doc["keywords"]}
            
            Provide:
            1. Executive Summary (2-3 paragraphs)
            2. Key Points (5-7 bullet points)
            3. Main Themes and Topics
            4. Important Statistics or Data
            5. Conclusions and Recommendations
            6. Action Items (if applicable)
            
            Keep it concise but comprehensive.
            """
            
            result = await self.generate_response(summary_prompt, temperature=0.3)
            
            if result["success"]:
                return {
                    "file_path": file_path,
                    "summary_type": summary_type,
                    "summary": result["content"],
                    "original_length": processed_doc["word_count"],
                    "keywords": processed_doc["keywords"],
                    "timestamp": self.last_active.isoformat()
                }
            else:
                return {"error": "Summarization failed"}
                
        except Exception as e:
            return {"error": str(e)}
    
    async def extract_information(self, file_path: str, extraction_criteria: str) -> Dict[str, Any]:
        """Extract specific information from a document"""
        try:
            processed_doc = self.document_processor.process_document(file_path)
            
            if "error" in processed_doc:
                return {"error": processed_doc["error"]}
            
            extraction_prompt = f"""
            Extract specific information from this document:
            
            Document: {processed_doc["text"][:3000]}...
            
            Extraction Criteria: {extraction_criteria}
            
            Provide:
            1. Requested Information (organized by criteria)
            2. Context and Supporting Details
            3. Location in Document (if applicable)
            4. Related Information or Cross-references
            5. Confidence Level for Each Extract
            6. Any Limitations or Gaps
            
            Be precise and accurate in extraction.
            """
            
            result = await self.generate_response(extraction_prompt, temperature=0.2)
            
            if result["success"]:
                return {
                    "file_path": file_path,
                    "extraction_criteria": extraction_criteria,
                    "extracted_information": result["content"],
                    "timestamp": self.last_active.isoformat()
                }
            else:
                return {"error": "Information extraction failed"}
                
        except Exception as e:
            return {"error": str(e)}
