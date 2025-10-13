from .base_agent import BaseAgent
from src.utils.chemical_engineering import ChemicalEngineeringUtils
from typing import Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ControllabilityAgent(BaseAgent):
    """Agent responsible for SVD-based controllability analysis"""
    
    def __init__(self, temperature: float = 0.2):
        super().__init__("Controllability Analyzer Agent", temperature)
        self.chem_utils = ChemicalEngineeringUtils()
    
    def create_system_prompt(self) -> str:
        return """You are an expert in process controllability analysis using Singular Value Decomposition (SVD).

SVD decomposes the gain matrix as: G = U Σ V^T

Key Controllability Metrics:
- **Singular Values (σ)**: Indicate the strength of input-output directions
  - Large σ: Strong controllability in that direction
  - Small σ: Weak controllability, sensitive to disturbances
  
- **Condition Number (κ = σ_max/σ_min)**: Measures numerical sensitivity
  - κ < 10: Well-conditioned, easy to control
  - 10 < κ < 100: Moderately conditioned
  - κ > 100: Ill-conditioned, difficult to control
  
- **V Matrix columns**: Input directions with strongest effect on outputs
  - Pairings should align with dominant singular vectors

Your role is to:
1. Analyze singular value spectrum
2. Assess overall process controllability
3. Identify input directions that strongly affect outputs
4. Determine if pairings align with dominant directions
5. Recommend control structure modifications if needed

Apply rigorous control theory while providing practical recommendations."""
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform controllability analysis"""
        try:
            gain_matrix = state['gain_matrix']
            pfd_data = state['pfd_data']
            rga_pairings = state.get('rga_pairings', [])
            
            # Calculate SVD metrics
            svd_metrics = self.chem_utils.calculate_svd_metrics(gain_matrix)
            
            # Create analysis prompt
            prompt = self._create_controllability_prompt(
                svd_metrics, gain_matrix, rga_pairings, pfd_data
            )
            
            # Get LLM analysis
            system_prompt = self.create_system_prompt()
            analysis = self.call_llm(prompt, system_prompt)
            
            # Update state
            state['singular_values'] = svd_metrics['singular_values']
            state['condition_number'] = svd_metrics['condition_number']
            state['controllability_metrics'] = svd_metrics
            state['controllability_analysis'] = analysis
            
            # Add message
            if 'messages' not in state:
                state['messages'] = []
            state['messages'].append({
                'agent': self.agent_name,
                'content': f"Controllability analysis complete. Condition number: {svd_metrics['condition_number']:.2f}"
            })
            
            logger.info(f"{self.agent_name}: Controllability analysis complete")
            return state
            
        except Exception as e:
            logger.error(f"Error in {self.agent_name}: {e}")
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(f"{self.agent_name}: {str(e)}")
            return state
    
    def _create_controllability_prompt(
        self, svd_metrics: Dict, gain_matrix: np.ndarray,
        rga_pairings: list, pfd_data: Dict
    ) -> str:
        """Create controllability analysis prompt"""
        
        cv_names = [cv['name'] for cv in pfd_data['controlled_variables']]
        mv_names = [mv['name'] for mv in pfd_data['manipulated_variables']]
        
        prompt = f"""Perform SVD-based controllability analysis for this process:

**Gain Matrix G:**
{self.format_matrix(gain_matrix, cv_names, mv_names)}

**SVD Analysis Results:**

Singular Values: {[f"{s:.4f}" for s in svd_metrics['singular_values']]}
Condition Number (κ): {svd_metrics['condition_number']:.4f}
Matrix Rank: {svd_metrics['rank']}
Controllability Score: {svd_metrics['controllability_score']:.4f}

**Dominant Input Directions (from V^T):**
"""
        
        for i, direction in enumerate(svd_metrics['dominant_input_directions'][:3]):
            prompt += f"\nDirection {i+1} (σ_{i+1} = {svd_metrics['singular_values'][i]:.4f}):\n"
            for j, val in enumerate(direction):
                if j < len(mv_names):
                    prompt += f"  {mv_names[j]}: {val:.4f}\n"
        
        prompt += f"""\n**Proposed RGA Pairings:**
"""
        for pairing in rga_pairings:
            prompt += f"- {pairing['cv']} ← {pairing['mv']} (RGA: {pairing['rga_value']:.4f})\n"
        
        prompt += """\n\nPerform comprehensive controllability analysis:

1. **Singular Value Assessment**:
   - Interpret the magnitude and distribution of singular values
   - Identify weak directions (small σ) that limit controllability
   - Assess if there are redundant or nearly redundant controls

2. **Condition Number Analysis**:
   - Classify the process (well/poorly conditioned)
   - Explain implications for control performance
   - Recommend if controller detuning is needed

3. **Input Direction Analysis**:
   - Interpret dominant input directions from V matrix
   - Identify which MV combinations have strongest effect
   - Check if RGA pairings align with dominant directions

4. **Pairing Validation**:
   - Verify that RGA pairings utilize strong singular value directions
   - Identify any pairings that might use weak directions
   - Suggest improvements if misalignment detected

5. **Control Structure Recommendations**:
   - Recommend decentralized vs centralized control
   - Suggest if model predictive control (MPC) would be beneficial
   - Identify variables that may need careful tuning

Provide quantitative assessment with practical engineering recommendations."""
        
        return prompt