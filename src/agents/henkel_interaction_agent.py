from .base_agent import BaseAgent
from typing import Dict, Any, List
import numpy as np
import logging
import json

logger = logging.getLogger(__name__)

class HankelInteractionAgent(BaseAgent):
    """Agent responsible for Hankel Interaction Index analysis for dynamic loop interactions"""
    
    def __init__(self, temperature: float = 0.2):
        super().__init__("Hankel Interaction Agent", temperature)
    
    def create_system_prompt(self) -> str:
        return """You are an expert in dynamic process control and interaction analysis using Hankel matrices.

The Hankel Interaction Index (HII) measures dynamic interactions between control loops, accounting for both 
steady-state gains and process dynamics (time constants). Unlike RGA which only considers steady-state behavior,
HII provides insight into transient interactions.

Key Concepts:
- **Hankel Matrix**: Captures the impulse response of the system over time
- **Hankel Singular Values**: Indicate the strength of dynamic interactions
- **HII Elements**: Similar interpretation to RGA but for dynamic behavior
  - HII close to 1.0: Minimal dynamic interaction, good pairing
  - HII < 0.5 or > 2.0: Significant dynamic interaction, careful tuning needed
  - Negative HII: Problematic dynamic coupling

Your role is to:
1. Analyze Hankel singular values and their distribution
2. Calculate Hankel Interaction Index for each potential pairing
3. Compare HII with RGA to identify dynamic vs steady-state discrepancies
4. Identify pairings with strong transient interactions
5. Recommend controller tuning considerations based on dynamic coupling
6. Assess if cascade control or dynamic decoupling is needed

Provide quantitative analysis with practical recommendations for controller design."""
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Hankel interaction analysis"""
        try:
            gain_matrix = state['gain_matrix']
            pfd_data = state['pfd_data']
            time_constants = pfd_data.get('time_constants')
            rga_matrix = state.get('rga_matrix')
            rga_pairings = state.get('rga_pairings', [])
            
            # Validate time constants availability
            if time_constants is None:
                logger.warning(f"{self.agent_name}: No time constants provided, skipping Hankel analysis")
                state['hankel_analysis'] = "Hankel analysis skipped: time constants not available"
                state['hii_matrix'] = None
                state['hankel_pairings'] = []
                return state
            
            time_constants = np.array(time_constants)
            
            # Get variable names
            cv_names = [cv['name'] for cv in pfd_data['controlled_variables']]
            mv_names = [mv['name'] for mv in pfd_data['manipulated_variables']]
            
            # Calculate Hankel Interaction Index
            hii_matrix, hankel_singular_values = self._calculate_hii(
                gain_matrix, time_constants
            )
            
            # Create analysis prompt
            prompt = self._create_hankel_analysis_prompt(
                hii_matrix, hankel_singular_values, gain_matrix, 
                time_constants, rga_matrix, cv_names, mv_names, rga_pairings
            )
            
            # Get LLM analysis
            system_prompt = self.create_system_prompt()
            analysis = self.call_llm(prompt, system_prompt)
            
            # Extract recommended pairings considering dynamics
            pairings_prompt = f"""Based on the Hankel interaction analysis:

{analysis}

Provide the recommended CV-MV pairings considering dynamic interactions in JSON format:
[
    {{
        "cv": "CV_name", 
        "mv": "MV_name", 
        "hii_value": 0.xx, 
        "dynamic_coupling": "weak/moderate/strong",
        "recommendation": "explanation",
        "tuning_consideration": "specific guidance for this pairing"
    }},
    ...
]

Include all pairings, prioritizing those with HII values between 0.7-1.3."""
            
            pairings_response = self.call_llm(pairings_prompt, system_prompt)
            
            # Parse pairings
            try:
                hankel_pairings = json.loads(pairings_response)
            except:
                hankel_pairings = self._extract_pairings_from_hii(
                    hii_matrix, cv_names, mv_names
                )
            
            # Update state
            state['hii_matrix'] = hii_matrix
            state['hankel_singular_values'] = hankel_singular_values.tolist()
            state['hankel_analysis'] = analysis
            state['hankel_pairings'] = hankel_pairings
            
            # Add message
            if 'messages' not in state:
                state['messages'] = []
            state['messages'].append({
                'agent': self.agent_name,
                'content': f"Hankel analysis complete. Identified {len(hankel_pairings)} pairings with dynamic interaction assessment."
            })
            
            logger.info(f"{self.agent_name}: Hankel Interaction Index calculation complete")
            return state
            
        except Exception as e:
            logger.error(f"Error in {self.agent_name}: {e}", exc_info=True)
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(f"{self.agent_name}: {str(e)}")
            return state
    
    def _calculate_hii(self, gain_matrix: np.ndarray, 
                       time_constants: np.ndarray) -> tuple:
        """
        Calculate Hankel Interaction Index using first-order approximation
        
        For first-order systems: G(s) = K / (τs + 1)
        Impulse response: g(t) = (K/τ) * exp(-t/τ)
        
        Args:
            gain_matrix: Steady-state gain matrix K
            time_constants: Time constant matrix τ
            
        Returns:
            hii_matrix: Hankel Interaction Index matrix
            hankel_sv: Hankel singular values
        """
        n_outputs, n_inputs = gain_matrix.shape
        
        # Time horizon for Hankel matrix (use maximum time constant as reference)
        tau_max = np.max(time_constants)
        t_horizon = 5 * tau_max  # Capture ~99% of response
        
        # Number of time samples
        n_samples = 100
        dt = t_horizon / n_samples
        t = np.linspace(0, t_horizon, n_samples)
        
        # Build Hankel matrix for the system
        # H[i,j] represents the impulse response from input j to output i
        hankel_matrices = []
        
        for i in range(n_outputs):
            for j in range(n_inputs):
                K = gain_matrix[i, j]
                tau = time_constants[i, j]
                
                # First-order impulse response
                if tau > 0:
                    impulse_response = (K / tau) * np.exp(-t / tau)
                else:
                    impulse_response = np.zeros_like(t)
                
                # Build Hankel matrix from impulse response
                hankel_size = n_samples // 2
                H = np.zeros((hankel_size, hankel_size))
                
                for row in range(hankel_size):
                    for col in range(hankel_size):
                        idx = row + col
                        if idx < n_samples:
                            H[row, col] = impulse_response[idx]
                
                hankel_matrices.append(H)
        
        # Compute Hankel singular values (average across all elements)
        all_singular_values = []
        for H in hankel_matrices:
            try:
                _, sv, _ = np.linalg.svd(H)
                all_singular_values.append(sv)
            except:
                pass
        
        if all_singular_values:
            # Take mean singular values as representative
            hankel_sv = np.mean(all_singular_values, axis=0)
        else:
            hankel_sv = np.array([1.0])
        
        # Calculate Hankel Interaction Index
        # HII approximation using dynamic gain matrix
        # For first-order systems at characteristic frequency ω = 1/τ_avg
        tau_avg = np.mean(time_constants)
        omega = 1.0 / tau_avg if tau_avg > 0 else 1.0
        
        # Dynamic gain matrix at frequency ω
        G_dynamic = np.zeros_like(gain_matrix, dtype=complex)
        for i in range(n_outputs):
            for j in range(n_inputs):
                K = gain_matrix[i, j]
                tau = time_constants[i, j]
                # G(jω) = K / (jωτ + 1)
                G_dynamic[i, j] = K / (1j * omega * tau + 1)
        
        # HII using dynamic gains (similar to RGA but with dynamic transfer functions)
        try:
            G_dynamic_inv = np.linalg.pinv(G_dynamic)
            hii_complex = G_dynamic * G_dynamic_inv.T
            hii_matrix = np.abs(hii_complex)  # Take magnitude
        except:
            logger.warning("HII calculation failed, falling back to RGA-like approximation")
            # Fallback: use steady-state RGA
            try:
                G_inv = np.linalg.pinv(gain_matrix)
                hii_matrix = np.abs(gain_matrix * G_inv.T)
            except:
                hii_matrix = np.eye(min(n_outputs, n_inputs))
        
        return hii_matrix, hankel_sv
    
    def _create_hankel_analysis_prompt(
        self, hii_matrix: np.ndarray, hankel_sv: np.ndarray,
        gain_matrix: np.ndarray, time_constants: np.ndarray,
        rga_matrix: np.ndarray, cv_names: list, mv_names: list,
        rga_pairings: list
    ) -> str:
        """Create comprehensive Hankel analysis prompt"""
        
        prompt = f"""Analyze the dynamic interactions using Hankel Interaction Index:

**System Dynamics Overview:**
- Number of CVs: {len(cv_names)}
- Number of MVs: {len(mv_names)}
- Time Constant Range: [{np.min(time_constants):.2f}, {np.max(time_constants):.2f}] time units
- Average Time Constant: {np.mean(time_constants):.2f} time units

**Steady-State Gain Matrix K:**
{self.format_matrix(gain_matrix, cv_names, mv_names)}

**Time Constant Matrix τ:**
{self.format_matrix(time_constants, cv_names, mv_names)}

**Hankel Interaction Index (HII) Matrix:**
{self.format_matrix(hii_matrix, cv_names, mv_names)}
"""
        
        if rga_matrix is not None:
            prompt += f"""
**Comparison with RGA (Steady-State):**
{self.format_matrix(rga_matrix, cv_names, mv_names)}
"""
        
        prompt += f"""
**Hankel Singular Values:**
{[f"{sv:.4f}" for sv in hankel_sv[:10]]}  # First 10 values

**RGA-Based Pairings (for comparison):**
"""
        for p in rga_pairings:
            prompt += f"- {p['cv']} ← {p['mv']} (RGA: {p['rga_value']:.3f})\n"
        
        prompt += """

**Analysis Tasks:**

1. **Hankel Singular Value Assessment**:
   - Interpret the magnitude and decay of Hankel singular values
   - Identify dominant modes in the dynamic response
   - Assess model reduction potential
   - Determine if system has well-separated time scales

2. **Dynamic Interaction Analysis (HII)**:
   - Compare HII with RGA element by element
   - Identify pairings where dynamic behavior differs from steady-state
   - Flag cases where HII ≠ RGA (indicates time-scale separation issues)
   - Assess overall dynamic coupling strength

3. **Pairing Recommendations**:
   - For each CV, recommend MV based on HII values
   - Prioritize HII values close to 1.0 (0.7 - 1.3 range)
   - Warn about pairings with HII < 0.5 or > 2.0
   - Identify pairings with negative or near-zero HII

4. **RGA vs HII Comparison**:
   - Identify pairings where RGA and HII disagree significantly
   - Explain implications of such disagreements
   - Determine if steady-state analysis is sufficient or if dynamics matter
   - Example: If RGA says pair CV1-MV1 but HII suggests CV1-MV2, why?

5. **Time-Scale Analysis**:
   - Identify fast vs slow loops based on time constants
   - Recommend if cascade control structures are beneficial
   - Assess if dynamic decoupling is needed
   - Suggest loop timing/tuning priorities

6. **Controller Design Implications**:
   - For strong dynamic coupling (HII > 1.5), recommend:
     * Detuned controllers
     * Sequential loop closing
     * Dynamic decoupling
   - For weak coupling (HII near 1.0), confirm decentralized control is OK
   - Suggest specific tuning constraints based on time constants

7. **Practical Recommendations**:
   - Recommend sampling rates based on fastest time constants
   - Suggest controller types (PI vs PID) based on dynamics
   - Identify loops requiring feedforward or cascade
   - Warn about potential windup or oscillation issues

Provide detailed quantitative analysis with specific engineering recommendations.
Compare your findings with the RGA-based analysis and explain any significant differences."""
        
        return prompt
    
    def _extract_pairings_from_hii(
        self, hii_matrix: np.ndarray, cv_names: list, mv_names: list
    ) -> list:
        """Extract pairings directly from HII matrix (fallback)"""
        pairings = []
        n = min(len(cv_names), len(mv_names))
        
        for i in range(n):
            # Find best MV for this CV based on HII
            hii_row = hii_matrix[i, :]
            best_mv_idx = np.argmax(hii_row)  # Want HII close to 1.0
            hii_value = hii_matrix[i, best_mv_idx]
            
            # Assess dynamic coupling strength
            if 0.7 <= hii_value <= 1.3:
                coupling = "weak"
                recommendation = "Good dynamic pairing with minimal interaction"
            elif 0.5 <= hii_value < 0.7 or 1.3 < hii_value <= 2.0:
                coupling = "moderate"
                recommendation = "Moderate dynamic coupling, careful tuning required"
            else:
                coupling = "strong"
                recommendation = "Strong dynamic interaction, consider advanced control"
            
            pairings.append({
                'cv': cv_names[i],
                'mv': mv_names[best_mv_idx],
                'hii_value': float(hii_value),
                'dynamic_coupling': coupling,
                'recommendation': recommendation,
                'tuning_consideration': f"Account for dynamic interaction (HII={hii_value:.3f})"
            })
        
        return pairings
    
    def _interpret_hii_value(self, hii_value: float) -> str:
        """Interpret HII value for pairing recommendation"""
        if 0.8 <= hii_value <= 1.2:
            return "Excellent dynamic pairing - minimal transient interaction"
        elif 0.6 <= hii_value < 0.8 or 1.2 < hii_value <= 1.5:
            return "Good dynamic pairing - moderate transient interaction"
        elif 0.3 <= hii_value < 0.6 or 1.5 < hii_value <= 2.0:
            return "Fair dynamic pairing - significant transient coupling"
        elif hii_value < 0.3:
            return "Poor dynamic pairing - weak dynamic response"
        else:
            return "Problematic pairing - strong dynamic interaction, avoid or use advanced control"