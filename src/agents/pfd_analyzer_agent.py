from .base_agent import BaseAgent
from typing import Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

class PFDAnalyzerAgent(BaseAgent):
    """Agent responsible for analyzing PFD structure and control requirements"""
    
    def __init__(self, temperature: float = 0.2):
        super().__init__("PFD Analyzer Agent", temperature)
    
    def create_system_prompt(self) -> str:
        return """You are an expert Chemical Engineer specializing in Process Control and P&ID analysis.

Your role is to analyze Process Flow Diagrams (PFDs) and identify:
1. Key unit operations and their control requirements
2. Process variables that need to be controlled
3. Available manipulated variables
4. Potential disturbance variables
5. Control objectives based on process safety and economics

You should apply fundamental chemical engineering principles including:
- Mass and energy balances
- Thermodynamic constraints
- Reaction kinetics and safety
- Process dynamics
- Operating window constraints

Provide clear, structured analysis that will guide the control structure design."""
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze PFD and identify control requirements"""
        try:
            pfd_data = state['pfd_data']
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt(pfd_data)
            
            # Get LLM analysis
            system_prompt = self.create_system_prompt()
            analysis = self.call_llm(prompt, system_prompt)
            
            # Extract control objectives
            objectives_prompt = f"""Based on this PFD analysis:

{analysis}

List the top 5 control objectives in order of priority. Format as JSON array of strings.
Example: ["Maintain reactor temperature at setpoint", "Control product purity", ...]"""
            
            objectives_response = self.call_llm(objectives_prompt, system_prompt)
            
            # Parse objectives (with fallback)
            try:
                control_objectives = json.loads(objectives_response)
            except:
                control_objectives = self._extract_objectives_fallback(objectives_response)
            
            # Update state
            state['pfd_analysis'] = analysis
            state['control_objectives'] = control_objectives
            state['process_characteristics'] = self._extract_characteristics(pfd_data)
            
            # Add message
            if 'messages' not in state:
                state['messages'] = []
            state['messages'].append({
                'agent': self.agent_name,
                'content': f"Completed PFD analysis. Identified {len(control_objectives)} control objectives."
            })
            
            logger.info(f"{self.agent_name}: Analysis complete")
            return state
            
        except Exception as e:
            logger.error(f"Error in {self.agent_name}: {e}")
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(f"{self.agent_name}: {str(e)}")
            return state
    
    def _create_analysis_prompt(self, pfd_data: Dict) -> str:
        """Create detailed prompt for PFD analysis"""
        
        prompt = f"""Analyze the following Process Flow Diagram (PFD):

**Process Name**: {pfd_data['name']}
**Description**: {pfd_data['description']}

**Unit Operations**:
"""
        for unit in pfd_data['unit_operations']:
            prompt += f"- {unit['name']} ({unit['type']})\n"
        
        prompt += "\n**Controlled Variables (CVs)**:\n"
        for cv in pfd_data['controlled_variables']:
            prompt += f"- {cv['name']}: {cv['type']} ({cv['unit']}) at {cv['unit_operation']}\n"
            prompt += f"  Range: [{cv['range'][0]}, {cv['range'][1]}], Nominal: {cv['nominal_value']}\n"
        
        prompt += "\n**Manipulated Variables (MVs)**:\n"
        for mv in pfd_data['manipulated_variables']:
            prompt += f"- {mv['name']}: {cv['type']} ({mv['unit']}) at {mv['unit_operation']}\n"
        
        if pfd_data.get('disturbance_variables'):
            prompt += "\n**Disturbance Variables**:\n"
            for dv in pfd_data['disturbance_variables']:
                prompt += f"- {dv['name']}: {dv['type']}\n"
        
        prompt += """\n\nProvide a comprehensive analysis covering:

1. **Process Understanding**: Describe the overall process, key chemical transformations, and phase changes.

2. **Control Requirements**: For each unit operation, identify:
   - Critical variables that must be controlled
   - Why they need control (safety, quality, efficiency)
   - Expected disturbances

3. **Degrees of Freedom**: Analyze available manipulated variables vs controlled variables.

4. **Safety Considerations**: Identify safety-critical control loops.

5. **Economic Objectives**: Identify controls that affect product quality and process efficiency.

6. **Dynamic Behavior**: Discuss expected time scales and process interactions.

Be specific and reference chemical engineering principles."""
        
        return prompt
    
    def _extract_characteristics(self, pfd_data: Dict) -> Dict:
        """Extract key process characteristics"""
        n_cvs = len(pfd_data['controlled_variables'])
        n_mvs = len(pfd_data['manipulated_variables'])
        n_units = len(pfd_data['unit_operations'])
        
        return {
            'num_controlled_variables': n_cvs,
            'num_manipulated_variables': n_mvs,
            'num_unit_operations': n_units,
            'degrees_of_freedom': n_mvs - n_cvs,
            'process_type': self._infer_process_type(pfd_data)
        }
    
    def _infer_process_type(self, pfd_data: Dict) -> str:
        """Infer process type from unit operations"""
        unit_types = [u['type'] for u in pfd_data['unit_operations']]
        
        if 'reactor' in unit_types:
            return 'reaction_system'
        elif 'distillation_column' in unit_types:
            return 'separation_system'
        elif 'heat_exchanger' in unit_types and len(unit_types) <= 2:
            return 'heat_exchange_system'
        else:
            return 'general_process'
    
    def _extract_objectives_fallback(self, text: str) -> list:
        """Fallback method to extract objectives from text"""
        lines = text.strip().split('\n')
        objectives = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                # Remove numbering and bullet points
                obj = line.lstrip('0123456789.-* ').strip('"\'')
                if obj:
                    objectives.append(obj)
        return objectives[:5]  # Top 5