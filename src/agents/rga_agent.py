from .base_agent import BaseAgent
from src.utils.chemical_engineering import ChemicalEngineeringUtils
from typing import Dict, Any
import numpy as np
import logging
import json

logger = logging.getLogger(__name__)

class RGAAgent(BaseAgent):
    """Agent responsible for Relative Gain Array calculation and analysis"""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__("RGA Calculator Agent", temperature)
        self.chem_utils = ChemicalEngineeringUtils()
    
    def create_system_prompt(self) -> str:
        return """You are an expert in multivariable control and Relative Gain Array (RGA) analysis.

The RGA is defined as: RGA = G ⊙ (G^-1)^T where G is the steady-state gain matrix.

RGA Interpretation Guidelines:
- λ_ij close to 1.0: Excellent pairing between CV_i and MV_j
- λ_ij between 0.5-1.0: Good pairing, moderate interaction
- λ_ij between 0-0.5: Poor pairing, weak effect
- λ_ij negative: Bad pairing, avoid this combination (can cause instability)

Your role is to:
1. Analyze the calculated RGA matrix
2. Recommend CV-MV pairings based on RGA values
3. Identify potential loop interactions
4. Warn about problematic pairings (negative RGA values)
5. Consider Bristol's rules and industrial best practices

Provide clear, actionable recommendations for control structure design."""
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate and analyze RGA"""
        try:
            gain_matrix = state['gain_matrix']
            pfd_data = state['pfd_data']
            
            # Get variable names
            cv_names = [cv['name'] for cv in pfd_data['controlled_variables']]
            mv_names = [mv['name'] for mv in pfd_data['manipulated_variables']]
            
            # Calculate RGA
            rga_matrix = self.chem_utils.calculate_rga(gain_matrix)
            
            # Create analysis prompt
            prompt = self._create_rga_analysis_prompt(
                rga_matrix, gain_matrix, cv_names, mv_names
            )
            
            # Get LLM analysis
            system_prompt = self.create_system_prompt()
            analysis = self.call_llm(prompt, system_prompt)
            
            # Extract recommended pairings
            pairings_prompt = f"""Based on the RGA analysis:

{analysis}

Provide the recommended CV-MV pairings in JSON format:
[
    {{"cv": "CV_name", "mv": "MV_name", "rga_value": 0.xx, "recommendation": "explanation"}},
    ...
]

Include all pairings with RGA values > 0.5 or the best available if none meet this threshold."""
            
            pairings_response = self.call_llm(pairings_prompt, system_prompt)
            
            # Parse pairings
            try:
                rga_pairings = json.loads(pairings_response)
            except:
                rga_pairings = self._extract_pairings_from_rga(
                    rga_matrix, cv_names, mv_names
                )
            
            # Update state
            state['rga_matrix'] = rga_matrix
            state['rga_analysis'] = analysis
            state['rga_pairings'] = rga_pairings
            
            # Add message
            if 'messages' not in state:
                state['messages'] = []
            state['messages'].append({
                'agent': self.agent_name,
                'content': f"RGA analysis complete. Identified {len(rga_pairings)} potential pairings."
            })
            
            logger.info(f"{self.agent_name}: RGA calculation complete")
            return state
            
        except Exception as e:
            logger.error(f"Error in {self.agent_name}: {e}")
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(f"{self.agent_name}: {str(e)}")
            return state
    
    def _create_rga_analysis_prompt(
        self, rga_matrix: np.ndarray, gain_matrix: np.ndarray,
        cv_names: list, mv_names: list
    ) -> str:
        """Create detailed RGA analysis prompt"""
        
        prompt = f"""Analyze the following Relative Gain Array (RGA) for control loop pairing:

**Gain Matrix G:**
{self.format_matrix(gain_matrix, cv_names, mv_names)}

**RGA Matrix (G ⊙ (G^-1)^T):**
{self.format_matrix(rga_matrix, cv_names, mv_names)}

**Controlled Variables (CVs):** {', '.join(cv_names)}
**Manipulated Variables (MVs):** {', '.join(mv_names)}

Perform a comprehensive RGA analysis:

1. **Diagonal Analysis**: 
   - Examine diagonal elements (preferred pairings)
   - Identify strong diagonal dominance (λ_ii ≈ 1)

2. **Pairing Recommendations**:
   - For each CV, recommend the best MV based on RGA values
   - Explain why each pairing is recommended
   - Identify alternative pairings if primary fails

3. **Interaction Assessment**:
   - Calculate row and column sums (should equal 1.0)
   - Identify strong off-diagonal elements indicating interactions
   - Warn about loop interactions that may affect performance

4. **Problem Detection**:
   - Identify negative RGA elements (causes instability)
   - Flag pairings with very small RGA values (< 0.3)
   - Note if matrix is ill-conditioned

5. **Industrial Considerations**:
   - Consider Bristol's integral controllability rules
   - Assess if decentralized control is appropriate
   - Recommend if advanced control (MPC) might be needed

Provide specific, actionable recommendations."""
        
        return prompt
    
    def _extract_pairings_from_rga(
        self, rga_matrix: np.ndarray, cv_names: list, mv_names: list
    ) -> list:
        """Extract pairings directly from RGA matrix (fallback)"""
        pairings = []
        n = min(len(cv_names), len(mv_names))
        
        for i in range(n):
            # Find best MV for this CV
            best_mv_idx = np.argmax(np.abs(rga_matrix[i, :]))
            rga_value = rga_matrix[i, best_mv_idx]
            
            recommendation = self.chem_utils.interpret_rga_value(rga_value)
            
            pairings.append({
                'cv': cv_names[i],
                'mv': mv_names[best_mv_idx],
                'rga_value': float(rga_value),
                'recommendation': recommendation
            })
        
        return pairings