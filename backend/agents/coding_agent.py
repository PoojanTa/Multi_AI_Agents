import asyncio
import json
import logging
import re
from typing import Dict, Any, List
from backend.agents.base_agent import BaseAgent
from backend.models.schemas import AgentTask, AgentResponse, AgentType

logger = logging.getLogger(__name__)

class CodingAgent(BaseAgent):
    """Agent specialized in code generation, review, debugging, and software development"""
    
    def __init__(self):
        super().__init__(
            agent_type=AgentType.CODING,
            name="Coding Agent",
            description="Specialized in code generation, review, debugging, testing, and software architecture"
        )
    
    def get_system_prompt(self) -> str:
        return """You are a senior software engineer with expertise in:
        - Code generation and implementation
        - Code review and optimization
        - Debugging and troubleshooting
        - Software architecture and design patterns
        - Testing and quality assurance
        - Performance optimization
        - Security best practices
        - Documentation and commenting
        
        Your responses should be:
        - Technically accurate and well-structured
        - Include proper error handling
        - Follow coding best practices and conventions
        - Provide clear explanations and comments
        - Consider security and performance implications
        - Include testing considerations
        
        When working with code, always provide:
        1. Clean, readable, and maintainable code
        2. Proper error handling and validation
        3. Comprehensive comments and documentation
        4. Testing recommendations
        5. Performance and security considerations
        6. Alternative approaches when applicable
        """
    
    def get_capabilities(self) -> List[str]:
        return [
            "Code generation",
            "Code review and optimization",
            "Debugging and troubleshooting",
            "Architecture design",
            "Testing and QA",
            "Performance optimization",
            "Security analysis",
            "Documentation generation",
            "Refactoring",
            "API design",
            "Database design",
            "Algorithm implementation"
        ]
    
    async def execute_task(self, task: AgentTask) -> AgentResponse:
        """Execute coding task"""
        try:
            # Determine coding task type
            task_type = await self._identify_coding_task(task.prompt)
            
            # Extract technical requirements
            tech_requirements = await self._extract_tech_requirements(task.prompt)
            
            # Generate code solution
            code_solution = await self._generate_code_solution(task.prompt, task_type, tech_requirements)
            
            # Perform code review and optimization
            reviewed_code = await self._review_and_optimize(code_solution, tech_requirements)
            
            # Calculate confidence based on code quality
            confidence = self._calculate_code_confidence(reviewed_code, tech_requirements)
            
            return AgentResponse(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                response=reviewed_code,
                confidence=confidence,
                reasoning=f"Code solution generated using {task_type} approach",
                tools_used=["code_generator", "code_reviewer", "optimizer", "validator"],
                metadata={
                    "task_type": task_type,
                    "language": tech_requirements.get("language", "Unknown"),
                    "complexity": tech_requirements.get("complexity", "Medium"),
                    "lines_of_code": self._count_lines_of_code(reviewed_code),
                    "functions_count": self._count_functions(reviewed_code)
                }
            )
            
        except Exception as e:
            logger.error(f"Coding agent error: {e}")
            return AgentResponse(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                response=f"Coding task failed: {str(e)}",
                confidence=0.0,
                reasoning=f"Error in coding process: {str(e)}"
            )
    
    async def _identify_coding_task(self, prompt: str) -> str:
        """Identify the type of coding task"""
        analysis_prompt = f"""
        Analyze this coding request and identify the task type:
        
        Request: {prompt}
        
        Task types:
        - Code Generation: Create new code from scratch
        - Code Review: Review and improve existing code
        - Debugging: Fix bugs and errors
        - Refactoring: Improve code structure and maintainability
        - Testing: Create tests or test strategies
        - Architecture: Design system architecture
        - Documentation: Generate code documentation
        - Optimization: Improve performance or efficiency
        
        Respond with just the task type and key focus areas.
        """
        
        result = await self.generate_response(analysis_prompt, temperature=0.3)
        
        if result["success"]:
            return result["content"]
        else:
            return "Code Generation"
    
    async def _extract_tech_requirements(self, prompt: str) -> Dict[str, Any]:
        """Extract technical requirements from the prompt"""
        requirements_prompt = f"""
        Extract technical requirements from this coding request:
        
        Request: {prompt}
        
        Identify:
        1. Programming language (Python, JavaScript, Java, etc.)
        2. Frameworks or libraries needed
        3. Complexity level (Simple, Medium, Complex)
        4. Performance requirements
        5. Security considerations
        6. Platform or environment
        7. Input/output specifications
        8. Error handling requirements
        
        Format as JSON with fields: language, frameworks, complexity, performance, security, platform, input_output, error_handling
        """
        
        result = await self.generate_response(requirements_prompt, temperature=0.2)
        
        if result["success"]:
            try:
                return json.loads(result["content"])
            except json.JSONDecodeError:
                return {
                    "language": "Python",
                    "frameworks": ["Standard library"],
                    "complexity": "Medium",
                    "performance": "Standard",
                    "security": "Basic",
                    "platform": "Cross-platform",
                    "input_output": "Standard I/O",
                    "error_handling": "Basic exception handling"
                }
        else:
            return {}
    
    async def _generate_code_solution(self, prompt: str, task_type: str, tech_requirements: Dict[str, Any]) -> str:
        """Generate code solution"""
        code_prompt = f"""
        Generate a complete code solution for: {prompt}
        
        Task Type: {task_type}
        Technical Requirements: {json.dumps(tech_requirements, indent=2)}
        
        Provide:
        1. Complete, working code with proper structure
        2. Clear comments explaining functionality
        3. Error handling and validation
        4. Input/output handling
        5. Function/method documentation
        6. Usage examples
        7. Security considerations
        8. Performance optimization where applicable
        
        Make the code production-ready and maintainable.
        """
        
        result = await self.generate_response(code_prompt, temperature=0.3, max_tokens=2048)
        
        if result["success"]:
            return result["content"]
        else:
            return "# Code generation failed\n# Error: " + result.get("error", "Unknown error")
    
    async def _review_and_optimize(self, code_solution: str, tech_requirements: Dict[str, Any]) -> str:
        """Review and optimize the generated code"""
        review_prompt = f"""
        Review and optimize this code solution:
        
        Code:
        {code_solution}
        
        Technical Requirements: {json.dumps(tech_requirements, indent=2)}
        
        Provide improved version with:
        1. Code quality improvements
        2. Performance optimizations
        3. Security enhancements
        4. Better error handling
        5. Improved documentation
        6. Code structure optimization
        7. Best practices implementation
        8. Testing recommendations
        
        Return the complete optimized code with explanations of changes made.
        """
        
        result = await self.generate_response(review_prompt, temperature=0.2, max_tokens=2048)
        
        if result["success"]:
            return result["content"]
        else:
            return code_solution  # Return original if review fails
    
    def _count_lines_of_code(self, code: str) -> int:
        """Count lines of code (excluding comments and empty lines)"""
        try:
            lines = code.split('\n')
            code_lines = 0
            
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith('#') and not stripped.startswith('//'):
                    code_lines += 1
            
            return code_lines
        except Exception:
            return 0
    
    def _count_functions(self, code: str) -> int:
        """Count number of functions/methods in the code"""
        try:
            # Count Python functions
            python_functions = len(re.findall(r'def\s+\w+\s*\(', code))
            
            # Count JavaScript functions
            js_functions = len(re.findall(r'function\s+\w+\s*\(', code))
            
            # Count class methods
            methods = len(re.findall(r'^\s*def\s+\w+\s*\(', code, re.MULTILINE))
            
            return max(python_functions, js_functions, methods)
        except Exception:
            return 0
    
    def _calculate_code_confidence(self, code: str, tech_requirements: Dict[str, Any]) -> float:
        """Calculate confidence score based on code quality"""
        try:
            # Code quality indicators
            quality_indicators = {
                'has_comments': 0.2 if '#' in code or '//' in code or '/*' in code else 0.0,
                'has_error_handling': 0.2 if any(keyword in code.lower() for keyword in ['try', 'except', 'catch', 'throw']) else 0.0,
                'has_functions': 0.2 if any(keyword in code for keyword in ['def ', 'function ', 'class ']) else 0.0,
                'has_documentation': 0.2 if any(keyword in code for keyword in ['"""', "'''", '/**']) else 0.0,
                'proper_structure': 0.2 if len(code.split('\n')) > 5 else 0.1
            }
            
            # Technical requirements match
            lang = tech_requirements.get("language", "").lower()
            if lang in code.lower():
                quality_indicators['language_match'] = 0.1
            
            return sum(quality_indicators.values())
        except Exception:
            return 0.5
    
    async def debug_code(self, code: str, error_description: str) -> Dict[str, Any]:
        """Debug code and provide solutions"""
        debug_prompt = f"""
        Debug this code and fix the issues:
        
        Code:
        {code}
        
        Error Description: {error_description}
        
        Provide:
        1. Error analysis and root cause
        2. Fixed code with corrections
        3. Explanation of changes made
        4. Prevention strategies
        5. Testing recommendations
        
        Make sure the fixed code is robust and handles edge cases.
        """
        
        result = await self.generate_response(debug_prompt, temperature=0.2)
        
        if result["success"]:
            return {
                "original_code": code,
                "error_description": error_description,
                "debug_solution": result["content"],
                "timestamp": self.last_active.isoformat()
            }
        else:
            return {
                "original_code": code,
                "error_description": error_description,
                "debug_solution": "Debug analysis could not be completed",
                "error": result.get("error", "Unknown error")
            }
    
    async def generate_tests(self, code: str, test_type: str = "unit") -> Dict[str, Any]:
        """Generate tests for the given code"""
        test_prompt = f"""
        Generate comprehensive {test_type} tests for this code:
        
        Code:
        {code}
        
        Provide:
        1. Test cases covering normal scenarios
        2. Edge cases and boundary conditions
        3. Error handling tests
        4. Performance tests (if applicable)
        5. Test setup and teardown
        6. Assertions and expected results
        7. Test documentation
        
        Use appropriate testing framework and best practices.
        """
        
        result = await self.generate_response(test_prompt, temperature=0.3)
        
        if result["success"]:
            return {
                "original_code": code,
                "test_type": test_type,
                "test_code": result["content"],
                "timestamp": self.last_active.isoformat()
            }
        else:
            return {
                "original_code": code,
                "test_type": test_type,
                "test_code": "Test generation could not be completed",
                "error": result.get("error", "Unknown error")
            }
