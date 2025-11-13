import numpy as np
from scipy import linalg
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class ChemicalEngineeringUtils:
    """Utilities for chemical engineering calculations"""
    
    @staticmethod
    def calculate_rga(gain_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate Relative Gain Array (RGA)
        RGA = G ⊙ (G^-1)^T where ⊙ is element-wise multiplication
        
        Args:
            gain_matrix: Steady-state gain matrix G (n×m)
            
        Returns:
            RGA matrix (n×n)
        """
        try:
            if gain_matrix.shape[0] != gain_matrix.shape[1]:
                logger.warning("Non-square gain matrix. RGA may not be meaningful.")
            
            G_inv = np.linalg.pinv(gain_matrix)
            rga = gain_matrix * G_inv.T
            
            return rga
        except Exception as e:
            logger.error(f"Error calculating RGA: {e}")
            return np.eye(gain_matrix.shape[0])
    
    @staticmethod
    def interpret_rga_value(lambda_ij: float) -> str:
        """
        Interpret RGA value for pairing recommendation
        
        Args:
            lambda_ij: RGA element value
            
        Returns:
            Interpretation string
        """
        if lambda_ij > 0.7:
            return "Excellent pairing - strong positive interaction"
        elif 0.3 < lambda_ij <= 0.7:
            return "Good pairing - moderate positive interaction"
        elif 0 <= lambda_ij <= 0.3:
            return "Poor pairing - weak interaction"
        elif -0.3 <= lambda_ij < 0:
            return "Poor pairing - weak negative interaction"
        else:
            return "Bad pairing - strong negative interaction, avoid"
    
    @staticmethod
    def calculate_svd_metrics(gain_matrix: np.ndarray) -> Dict:
        """
        Perform SVD analysis for controllability
        G = U Σ V^T
        
        Args:
            gain_matrix: Gain matrix
            
        Returns:
            Dictionary with SVD metrics
        """
        try:
            U, sigma, Vt = np.linalg.svd(gain_matrix)
            
            condition_number = sigma[0] / sigma[-1] if sigma[-1] != 0 else np.inf
            
            # Dominant directions in input space
            dominant_directions = Vt[:3] if len(Vt) >= 3 else Vt
            
            return {
                'singular_values': sigma.tolist(),
                'condition_number': condition_number,
                'dominant_input_directions': dominant_directions.tolist(),
                'rank': np.sum(sigma > 1e-10),
                'controllability_score': 1.0 / condition_number if condition_number < np.inf else 0.0
            }
        except Exception as e:
            logger.error(f"Error in SVD calculation: {e}")
            return {
                'singular_values': [],
                'condition_number': np.inf,
                'dominant_input_directions': [],
                'rank': 0,
                'controllability_score': 0.0
            }
    
    @staticmethod
    def calculate_interaction_index(gain_matrix: np.ndarray) -> float:
        """
        Calculate interaction index: I = ||G - diag(G)|| / ||G||
        
        Args:
            gain_matrix: Gain matrix
            
        Returns:
            Interaction index (0 = no interaction, 1 = full interaction)
        """
        try:
            # Handle non-square matrices by building a diagonal matrix of the
            # same shape as gain_matrix. For non-square G (n_rows x n_cols),
            # only the first min(n_rows, n_cols) diagonal elements are defined.
            n_rows, n_cols = gain_matrix.shape
            min_dim = min(n_rows, n_cols)

            G_diag = np.zeros_like(gain_matrix, dtype=float)
            diag_elems = np.diag(gain_matrix)
            for i in range(min_dim):
                G_diag[i, i] = diag_elems[i]

            G_off_diag = gain_matrix - G_diag

            norm_off = np.linalg.norm(G_off_diag, 'fro')
            norm_total = np.linalg.norm(gain_matrix, 'fro')

            interaction_index = norm_off / norm_total if norm_total != 0 else 0.0

            return float(interaction_index)
        except Exception as e:
            logger.error(f"Error calculating interaction index: {e}")
            return 0.0
    
    @staticmethod
    def maximum_weight_matching(gain_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """
        Maximum weight matching for variable pairing
        Uses greedy approach for simplicity
        
        Args:
            gain_matrix: Gain matrix (absolute values used as weights)
            
        Returns:
            List of (CV_index, MV_index) pairings
        """
        weights = np.abs(gain_matrix)
        n_rows, n_cols = weights.shape
        
        pairings = []
        used_rows = set()
        used_cols = set()
        
        # Greedy matching: iteratively pick highest weight
        for _ in range(min(n_rows, n_cols)):
            # Mask already used rows/cols
            masked_weights = weights.copy()
            for r in used_rows:
                masked_weights[r, :] = -np.inf
            for c in used_cols:
                masked_weights[:, c] = -np.inf
            
            if np.all(masked_weights == -np.inf):
                break
            
            # Find maximum weight
            max_idx = np.unravel_index(np.argmax(masked_weights), masked_weights.shape)
            pairings.append(max_idx)
            used_rows.add(max_idx[0])
            used_cols.add(max_idx[1])
        
        return pairings
    
    @staticmethod
    def analyze_process_dynamics(time_constants: np.ndarray) -> Dict:
        """
        Analyze process dynamics based on time constants
        
        Args:
            time_constants: Matrix of time constants
            
        Returns:
            Dictionary with dynamics analysis
        """
        try:
            tau_avg = np.mean(time_constants)
            tau_range = (np.min(time_constants), np.max(time_constants))
            
            # Classification based on time constants
            if tau_avg < 1.0:
                speed = "fast"
            elif tau_avg < 10.0:
                speed = "moderate"
            else:
                speed = "slow"
            
            return {
                'average_time_constant': tau_avg,
                'time_constant_range': tau_range,
                'process_speed': speed,
                'recommendation': f"Process is {speed}. Appropriate controller tuning needed."
            }
        except Exception as e:
            logger.error(f"Error analyzing dynamics: {e}")
            return {
                'average_time_constant': 0,
                'time_constant_range': (0, 0),
                'process_speed': 'unknown',
                'recommendation': 'Unable to analyze dynamics'
            }
    
    @staticmethod
    def get_unit_operation_control_strategy(unit_type: str) -> List[str]:
        """
        Get typical control strategies for unit operations
        
        Args:
            unit_type: Type of unit operation
            
        Returns:
            List of control strategy recommendations
        """
        strategies = {
            'distillation_column': [
                "Control top composition via reflux ratio",
                "Control bottom composition via reboiler duty",
                "Maintain column pressure for stable operation",
                "Level control on reflux drum and column sump",
                "Consider dual composition control for high-purity separations"
            ],
            'reactor': [
                "Temperature control critical for reaction rate and selectivity",
                "Exothermic reactions require tight temperature control",
                "Flow ratio control for maintaining stoichiometry",
                "Pressure control for gas-phase reactions",
                "Level control for liquid-phase batch/semi-batch",
                "Consider cascade control for better disturbance rejection"
            ],
            'heat_exchanger': [
                "Control outlet temperature via hot/cold fluid flow",
                "Monitor pressure drop across exchanger",
                "Consider bypass control for precise temperature control",
                "Fouling monitoring through heat transfer coefficient"
            ],
            'separator': [
                "Level control for each phase",
                "Pressure control affects vapor-liquid equilibrium",
                "Interface level control for liquid-liquid separation"
            ],
            'mixer': [
                "Flow ratio control for blending",
                "Total flow control at mixer outlet",
                "Composition control if inline analyzer available"
            ]
        }
        
        return strategies.get(unit_type, ["Standard regulatory control recommended"])