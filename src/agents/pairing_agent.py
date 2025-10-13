from .base_agent import BaseAgent
from src.utils.chemical_engineering import ChemicalEngineeringUtils
from typing import Dict, Any, List
import numpy as np
import logging
import json

logger = logging.getLogger(__name__)

class PairingAgent(BaseAgent):
    """Agent responsible for optimizing control pairings using multiple heuristics"""
    
    def __init__(self, temperature: float = 0.3):
        super().__init__("Pairing Optimizer Agent", temperature)
        self.chem_utils = ChemicalEngineeringUtils()
    
    def create_system_prompt(self) -> str:
        return """You are an expert in control structure synthesis and optimization.

Your role is to determine the optimal control loop pairings by integrating multiple analysis methods:

1. **RGA Analysis**: Variable pairing based on interaction measures
2. **Controllability (SVD)**: Ensuring pairings use strong control directions
3. **Interaction Minimization**: Reducing loop coupling
4. **Chemical Engineering Principles**: Process-specific control strategies

Pairing Optimization Criteria:
- Maximize RGA diagonal elements (minimize interactions)
- Align pairings with dominant singular value directions
- Minimize interaction index
- Follow unit operation-specific control strategies
- Consider practical implementation and maintenance

You must provide:
1. Final optimized CV-MV pairings with justification
2. Controller type recommendations (PI, PID, Cascade, etc.)
3. Chemical engineering rationale for each pairing
4. Interaction warnings and mitigation strategies
5. Tuning guidance based on process dynamics

Balance theoretical optimality with practical engineering judgment."""
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize control pairings"""
        try:
            # Get all previous analyses
            gain_matrix = state['gain_matrix']
            rga_matrix = state.get('rga_matrix')
            rga_pairings = state.get('rga_pairings', [])
            svd_metrics = state.get('controllability_metrics', {})
            pfd_data = state['pfd_data']
            pfd_analysis = state.get('pfd_analysis', '')
            control_objectives = state.get('control_objectives', [])
            
            # Calculate interaction index
            interaction_index = self.chem_utils.calculate_interaction_index(gain_matrix)
            
            # Get maximum weight matching
            mw_pairings = self.chem_utils.maximum_weight_matching(gain_matrix)
            
            # Create optimization prompt
            prompt = self._create_optimization_prompt(
                gain_matrix, rga_matrix, rga_pairings, svd_metrics,
                interaction_index, mw_pairings, pfd_data,
                pfd_analysis, control_objectives
            )
            
            # Get LLM optimization
            system_prompt = self.create_system_prompt()
            optimization_analysis = self.call_llm(prompt, system_prompt)
            
            # Extract final pairings
            pairings_prompt = f"""Based on the comprehensive analysis:

{optimization_analysis}

Provide the FINAL OPTIMAL control loop pairings in JSON format:
[
    {{
        "controlled_variable": "CV_name",
        "manipulated_variable": "MV_name",
        "controller_type": "PID/PI/CASCADE/etc",
        "rga_value": 0.xx,
        "controllability_score": 0.xx,
        "interaction_score": 0.xx,
        "overall_confidence": 0.xx,
        "reasoning": "brief justification",
        "chemical_eng_rationale": "process-specific justification",
        "tuning_guidance": "controller tuning recommendations"
    }},
    ...
]

Include ALL control loops (one per controlled variable)."""
            
            pairings_response = self.call_llm(pairings_prompt, system_prompt)
            
            # Parse pairings
            try:
                optimal_pairings = json.loads(pairings_response)
            except:
                optimal_pairings = self._create_fallback_pairings(
                    rga_pairings, svd_metrics, pfd_data
                )
            
            # Validate pairings
            optimal_pairings = self._validate_and_enhance_pairings(
                optimal_pairings, gain_matrix, rga_matrix, pfd_data
            )
            
            # Update state
            state['interaction_index'] = interaction_index
            state['optimal_pairings'] = optimal_pairings
            state['pairing_reasoning'] = optimization_analysis
            
            # Add message
            if 'messages' not in state:
                state['messages'] = []
            state['messages'].append({
                'agent': self.agent_name,
                'content': f"Optimized {len(optimal_pairings)} control loop pairings. Interaction index: {interaction_index:.3f}"
            })
            
            logger.info(f"{self.agent_name}: Pairing optimization complete")
            return state
            
        except Exception as e:
            logger.error(f"Error in {self.agent_name}: {e}")
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(f"{self.agent_name}: {str(e)}")
            return state
    
    def _create_optimization_prompt(
        self, gain_matrix, rga_matrix, rga_pairings, svd_metrics,
        interaction_index, mw_pairings, pfd_data, pfd_analysis, control_objectives
    ) -> str:
        """Create comprehensive optimization prompt"""
        
        cv_names = [cv['name'] for cv in pfd_data['controlled_variables']]
        mv_names = [mv['name'] for mv in pfd_data['manipulated_variables']]
        
        prompt = f"""Optimize control loop pairings by integrating multiple analysis methods:

**Process Overview:**
{pfd_data['description']}

**Control Objectives:**
"""
        for i, obj in enumerate(control_objectives, 1):
            prompt += f"{i}. {obj}\n"
        
        prompt += f"""\n**Key Process Analysis:**
{pfd_analysis[:500]}...

**Gain Matrix:**
{self.format_matrix(gain_matrix, cv_names, mv_names)}

**RGA-Based Pairings:**
"""
        for p in rga_pairings:
            prompt += f"- {p['cv']} ← {p['mv']} (λ = {p['rga_value']:.3f})\n"
        
        prompt += f"""\n**Controllability Metrics (SVD):**
- Condition Number: {svd_metrics.get('condition_number', 'N/A')}
- Singular Values: {[f"{s:.3f}" for s in svd_metrics.get('singular_values', [])]}
- Controllability Score: {svd_metrics.get('controllability_score', 'N/A')}

**Interaction Index:** {interaction_index:.4f}
(0 = no interaction, 1 = full interaction)

**Maximum Weight Matching Suggestions:**
"""
        for cv_idx, mv_idx in mw_pairings:
            if cv_idx < len(cv_names) and mv_idx < len(mv_names):
                prompt += f"- {cv_names[cv_idx]} ← {mv_names[mv_idx]} (gain: {abs(gain_matrix[cv_idx, mv_idx]):.3f})\n"
        
        prompt += """\n**Unit Operation Control Strategies:**
"""
        for unit in pfd_data['unit_operations']:
            strategies = self.chem_utils.get_unit_operation_control_strategy(unit['type'])
            prompt += f"\n{unit['name']} ({unit['type']}):\n"
            for strategy in strategies[:3]:  # Top 3
                prompt += f"  - {strategy}\n"
        
        prompt += """\n\n**Optimization Task:**

Synthesize an optimal control structure by:

1. **Evaluating Trade-offs**:
   - RGA optimality vs controllability vs interaction minimization
   - Weight the criteria appropriately for this process

2. **Pairing Selection**:
   - Select ONE manipulated variable for each controlled variable
   - Justify each pairing using multiple criteria
   - Ensure pairings are physically realizable and maintainable

3. **Controller Type Selection**:
   - Recommend controller type (PI, PID, Cascade, Ratio, Feedforward)
   - Base on process dynamics and control objectives
   - Consider measurement availability and quality

4. **Chemical Engineering Validation**:
   - Verify pairings follow process physics and thermodynamics
   - Check against industry best practices for similar units
   - Ensure safety-critical variables are properly controlled

5. **Interaction Management**:
   - If interaction index > 0.3, recommend mitigation strategies
   - Consider cascade or decoupling if needed
   - Identify loops that may require detuning

6. **Practical Considerations**:
   - Operator familiarity and ease of operation
   - Maintenance and instrumentation requirements
   - Startup and shutdown considerations

Provide a comprehensive optimization that balances all factors."""
        
        return prompt
    
    def _create_fallback_pairings(self, rga_pairings, svd_metrics, pfd_data) -> List[Dict]:
        """Create fallback pairings if LLM parsing fails"""
        pairings = []
        for p in rga_pairings:
            pairings.append({
                'controlled_variable': p['cv'],
                'manipulated_variable': p['mv'],
                'controller_type': 'PID',
                'rga_value': p['rga_value'],
                'controllability_score': svd_metrics.get('controllability_score', 0.5),
                'interaction_score': 0.5,
                'overall_confidence': 0.7,
                'reasoning': 'Based on RGA analysis',
                'chemical_eng_rationale': 'Standard pairing',
                'tuning_guidance': 'Start with conservative tuning'
            })
        return pairings
    
    def _validate_and_enhance_pairings(
        self, pairings: List[Dict], gain_matrix, rga_matrix, pfd_data
    ) -> List[Dict]:
        """Validate and enhance pairings with computed metrics"""
        cv_names = [cv['name'] for cv in pfd_data['controlled_variables']]
        mv_names = [mv['name'] for mv in pfd_data['manipulated_variables']]
        
        for pairing in pairings:
            try:
                cv_idx = cv_names.index(pairing['controlled_variable'])
                mv_idx = mv_names.index(pairing['manipulated_variable'])
                
                # Add/verify numerical metrics
                if rga_matrix is not None:
                    pairing['rga_value'] = float(rga_matrix[cv_idx, mv_idx])
                
                pairing['steady_state_gain'] = float(gain_matrix[cv_idx, mv_idx])
                
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not validate pairing: {e}")
        
        return pairings