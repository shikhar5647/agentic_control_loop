from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from typing import Dict, Any, List
import logging
import os

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for all agents in the control loop prediction system"""
    
    def __init__(self, agent_name: str, temperature: float = 0.3):
        """
        Initialize base agent
        
        Args:
            agent_name: Name of the agent
            temperature: Temperature for LLM generation
        """
        self.agent_name = agent_name
        self.temperature = temperature
        
        # Initialize Gemini model
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=temperature,
            google_api_key=api_key
        )
        
        logger.info(f"Initialized {agent_name} with temperature {temperature}")
    
    def create_system_prompt(self) -> str:
        """Create system prompt for the agent - to be overridden by subclasses"""
        return f"You are {self.agent_name}, an expert in chemical process control."
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for the agent
        
        Args:
            state: Current state from LangGraph
            
        Returns:
            Updated state
        """
        raise NotImplementedError("Subclasses must implement invoke method")
    
    def call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """
        Call LLM with given prompt
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            
        Returns:
            LLM response
        """
        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error calling LLM in {self.agent_name}: {e}")
            return f"Error: {str(e)}"
    
    def format_matrix(self, matrix, var_names_row=None, var_names_col=None) -> str:
        """
        Format matrix for LLM consumption
        
        Args:
            matrix: Numpy array
            var_names_row: Row variable names
            var_names_col: Column variable names
            
        Returns:
            Formatted string
        """
        import numpy as np
        
        lines = []
        if var_names_col:
            header = "     " + "  ".join(f"{name:>8}" for name in var_names_col)
            lines.append(header)
            lines.append("-" * len(header))
        
        for i, row in enumerate(matrix):
            row_name = var_names_row[i] if var_names_row else f"Row {i}"
            row_str = f"{row_name:>4} " + "  ".join(f"{val:>8.4f}" for val in row)
            lines.append(row_str)
        
        return "\n".join(lines)