# src/agents/pfd_analyzer_agent.py - COMPLETE VERSION
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
- Unit operation specific control strategies

IMPORTANT GUIDELINES:
- Focus on practical, implementable control strategies
- Prioritize safety-critical controls first
- Consider economic optimization objectives
- Account for process interactions
- Think about startup, shutdown, and abnormal operations

Provide clear, structured analysis that will guide the control structure design."""
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze PFD and identify control requirements"""
        try:
            pfd_data = state['pfd_data']
            
            logger.info(f"{self.agent_name}: Starting PFD analysis for {pfd_data['name']}")
            
            # Create comprehensive analysis prompt
            prompt = self._create_analysis_prompt(pfd_data)
            
            # Get LLM analysis
            system_prompt = self.create_system_prompt()
            analysis = self.call_llm(prompt, system_prompt)
            
            logger.info(f"{self.agent_name}: Analysis complete, extracting control objectives")
            
            # Extract control objectives
            objectives_prompt = f"""Based on this PFD analysis:

{analysis}

List the top 5-7 control objectives in order of priority (most critical first). 
Each objective should be specific and actionable.

Format as a simple JSON array of strings:
["Objective 1", "Objective 2", ...]

Only return the JSON array, nothing else."""
            
            objectives_response = self.call_llm(objectives_prompt, system_prompt)
            
            # Parse objectives (with fallback)
            try:
                # Try to extract JSON from response
                objectives_response = objectives_response.strip()
                if '```json' in objectives_response:
                    objectives_response = objectives_response.split('```json')[1].split('```')[0].strip()
                elif '```' in objectives_response:
                    objectives_response = objectives_response.split('```')[1].split('```')[0].strip()
                
                control_objectives = json.loads(objectives_response)
            except:
                logger.warning(f"{self.agent_name}: Could not parse objectives JSON, using fallback")
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
            
            logger.info(f"{self.agent_name}: Analysis complete with {len(control_objectives)} objectives")
            return state
            
        except Exception as e:
            logger.error(f"Error in {self.agent_name}: {e}", exc_info=True)
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
        for i, unit in enumerate(pfd_data['unit_operations'], 1):
            prompt += f"{i}. {unit['name']} - {unit['type']}"
            if 'description' in unit:
                prompt += f" ({unit['description']})"
            prompt += "\n"
        
        prompt += "\n**Controlled Variables (CVs)**:\n"
        for i, cv in enumerate(pfd_data['controlled_variables'], 1):
            prompt += f"{i}. {cv['name']}: {cv['type']} ({cv['unit']})\n"
            prompt += f"   - Range: [{cv['range'][0]}, {cv['range'][1]}]\n"
            prompt += f"   - Nominal: {cv['nominal_value']}\n"
            prompt += f"   - Location: {cv['unit_operation']}\n"
            if 'description' in cv:
                prompt += f"   - Description: {cv['description']}\n"
        
        prompt += "\n**Manipulated Variables (MVs)**:\n"
        for i, mv in enumerate(pfd_data['manipulated_variables'], 1):
            prompt += f"{i}. {mv['name']}: {mv['type']} ({mv['unit']})\n"
            prompt += f"   - Range: [{mv['range'][0]}, {mv['range'][1]}]\n"
            prompt += f"   - Nominal: {mv['nominal_value']}\n"
            prompt += f"   - Location: {mv['unit_operation']}\n"
            if 'description' in mv:
                prompt += f"   - Description: {mv['description']}\n"
        
        if pfd_data.get('disturbance_variables'):
            prompt += "\n**Disturbance Variables**:\n"
            for i, dv in enumerate(pfd_data['disturbance_variables'], 1):
                prompt += f"{i}. {dv['name']}: {dv['type']}"
                if 'description' in dv:
                    prompt += f" - {dv['description']}"
                prompt += "\n"
        
        prompt += """\n\nProvide a comprehensive analysis covering:

1. **Process Understanding**:
   - Describe the overall process and its purpose
   - Identify key chemical/physical transformations
   - Explain the role of each unit operation
   - Discuss material and energy flows

2. **Control Requirements Analysis**:
   For each unit operation, identify:
   - Which variables MUST be controlled (safety, product quality)
   - WHY each variable needs control (specific reasons)
   - Expected disturbances and their impact
   - Dynamic behavior (fast/slow responses)

3. **Degrees of Freedom Analysis**:
   - Count available manipulated variables vs controlled variables
   - Identify if system is square, over-determined, or under-determined
   - Discuss implications for control structure

4. **Safety Considerations**:
   - Identify safety-critical variables (temperature limits, pressure limits, etc.)
   - Specify fail-safe requirements
   - Highlight potential hazards (runaway reactions, overpressure, etc.)

5. **Economic Objectives**:
   - Identify controls affecting product quality/purity
   - Discuss energy efficiency considerations
   - Identify throughput-limiting factors

6. **Process Interactions**:
   - Identify expected interactions between control loops
   - Discuss potential control conflicts
   - Suggest cascade or advanced control opportunities

7. **Dynamic Behavior**:
   - Characterize time scales (fast: seconds, moderate: minutes, slow: hours)
   - Identify integrating processes (level, inventory)
   - Discuss measurement and actuation delays

Be specific, technical, and reference chemical engineering principles. 
Your analysis will guide the selection of control pairings."""
        
        return prompt
    
    def _extract_characteristics(self, pfd_data: Dict) -> Dict:
        """Extract key process characteristics"""
        n_cvs = len(pfd_data['controlled_variables'])
        n_mvs = len(pfd_data['manipulated_variables'])
        n_units = len(pfd_data['unit_operations'])
        
        # Analyze variable types
        cv_types = {}
        for cv in pfd_data['controlled_variables']:
            cv_type = cv['type']
            cv_types[cv_type] = cv_types.get(cv_type, 0) + 1
        
        mv_types = {}
        for mv in pfd_data['manipulated_variables']:
            mv_type = mv['type']
            mv_types[mv_type] = mv_types.get(mv_type, 0) + 1
        
        return {
            'num_controlled_variables': n_cvs,
            'num_manipulated_variables': n_mvs,
            'num_unit_operations': n_units,
            'degrees_of_freedom': n_mvs - n_cvs,
            'process_type': self._infer_process_type(pfd_data),
            'cv_type_distribution': cv_types,
            'mv_type_distribution': mv_types,
            'has_disturbances': len(pfd_data.get('disturbance_variables', [])) > 0,
            'has_time_constants': 'time_constants' in pfd_data
        }
    
    def _infer_process_type(self, pfd_data: Dict) -> str:
        """Infer process type from unit operations"""
        unit_types = [u['type'] for u in pfd_data['unit_operations']]
        
        if 'reactor' in unit_types:
            return 'reaction_system'
        elif 'distillation_column' in unit_types or 'distillation' in unit_types:
            return 'separation_system'
        elif 'heat_exchanger' in unit_types and len(unit_types) <= 2:
            return 'heat_exchange_system'
        elif 'separator' in unit_types:
            return 'separation_system'
        elif 'mixer' in unit_types:
            return 'mixing_system'
        else:
            return 'general_process'
    
    def _extract_objectives_fallback(self, text: str) -> list:
        """Fallback method to extract objectives from text"""
        lines = text.strip().split('\n')
        objectives = []
        
        for line in lines:
            line = line.strip()
            # Look for numbered or bulleted items
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*') or line.startswith('•')):
                # Remove numbering and bullet points
                obj = line.lstrip('0123456789.-*• ').strip('"\'[]')
                if obj and len(obj) > 10:  # Ensure it's substantial
                    objectives.append(obj)
        
        # If no structured objectives found, create defaults
        if not objectives:
            objectives = [
                "Maintain product quality within specifications",
                "Ensure safe operation within operating limits",
                "Maximize process efficiency and throughput",
                "Minimize energy consumption",
                "Reject disturbances effectively"
            ]
        
        return objectives[:7]  # Top 7 max