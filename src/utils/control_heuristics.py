import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class ControlHeuristics:
    """Chemical engineering control heuristics and best practices"""
    
    # Skogestad's self-optimizing control hierarchy
    CONTROL_HIERARCHY = {
        1: "Regulatory control (stabilization)",
        2: "Constraint control (maximize throughput)",
        3: "Optimization control (minimize cost)",
        4: "Economic optimization"
    }
    
    # Unit operation specific control strategies
    UNIT_CONTROL_STRATEGIES = {
        'distillation_column': {
            'primary_objectives': [
                'Product purity control',
                'Product recovery maximization',
                'Energy efficiency'
            ],
            'typical_pairings': [
                ('Top composition', 'Reflux ratio or Reflux flow'),
                ('Bottom composition', 'Reboiler duty or Boilup rate'),
                ('Column pressure', 'Condenser duty or Vent valve'),
                ('Reflux drum level', 'Distillate flow'),
                ('Sump level', 'Bottoms flow')
            ],
            'control_structure_options': [
                'LV configuration (Reflux, Boilup)',
                'DV configuration (Distillate, Boilup)',
                'Ratio control schemes',
                'Dual composition control'
            ],
            'common_issues': [
                'Dual composition control often infeasible (RGA issues)',
                'Temperature control as proxy for composition',
                'Pressure affects relative volatility'
            ]
        },
        'reactor': {
            'primary_objectives': [
                'Temperature control (reaction rate, selectivity, safety)',
                'Pressure control (gas-phase reactions)',
                'Residence time control',
                'Stoichiometry maintenance'
            ],
            'typical_pairings': [
                ('Temperature', 'Cooling/heating duty'),
                ('Level', 'Product flow'),
                ('Pressure', 'Vent or feed rate'),
                ('Feed ratio', 'Individual feed flows')
            ],
            'control_structure_options': [
                'Cascade temperature control',
                'Ratio control for feeds',
                'Feedforward from feed composition',
                'Override controls for safety'
            ],
            'common_issues': [
                'Exothermic reactions need tight temperature control',
                'Runaway risk if temperature control fails',
                'Selectivity sensitive to temperature'
            ]
        },
        'heat_exchanger': {
            'primary_objectives': [
                'Outlet temperature control',
                'Fouling monitoring',
                'Energy efficiency'
            ],
            'typical_pairings': [
                ('Outlet temperature', 'Utility flow'),
                ('Bypass control', 'Temperature control')
            ],
            'control_structure_options': [
                'Split-range control',
                'Bypass control',
                'Cascade control with intermediate temperature'
            ],
            'common_issues': [
                'Fouling degrades performance over time',
                'Nonlinear dynamics at low flow rates'
            ]
        },
        'separator': {
            'primary_objectives': [
                'Phase separation efficiency',
                'Interface level control',
                'Pressure control'
            ],
            'typical_pairings': [
                ('Liquid level', 'Liquid outlet flow'),
                ('Pressure', 'Vapor outlet or vent'),
                ('Interface level', 'Draw-off rate')
            ],
            'control_structure_options': [
                'Averaging level control',
                'Tight level control for small vessels'
            ],
            'common_issues': [
                'Interface detection can be difficult',
                'Emulsion formation'
            ]
        }
    }
    
    @staticmethod
    def get_luyben_plantwide_rules() -> List[str]:
        """
        Luyben's plantwide control design procedure rules
        
        Returns:
            List of plantwide control rules
        """
        return [
            "1. Establish control objectives (product quality, safety, environmental)",
            "2. Determine what variables need to be controlled",
            "3. Establish energy management system",
            "4. Set production rate (fix throughput manipulator)",
            "5. Control product quality and handle safety/environmental constraints",
            "6. Fix flow in recycle streams",
            "7. Check component balances",
            "8. Control individual unit operations",
            "9. Optimize economic performance"
        ]
    
    @staticmethod
    def apply_mpc_criteria(gain_matrix: np.ndarray, 
                          interaction_index: float,
                          condition_number: float) -> Dict:
        """
        Determine if MPC is beneficial based on system characteristics
        
        Args:
            gain_matrix: Process gain matrix
            interaction_index: Loop interaction measure
            condition_number: Matrix condition number
            
        Returns:
            Dictionary with MPC recommendation
        """
        mpc_score = 0
        reasons = []
        
        # High interaction suggests MPC
        if interaction_index > 0.4:
            mpc_score += 3
            reasons.append(f"High interaction (I={interaction_index:.3f}) suggests MPC")
        
        # Ill-conditioned system
        if condition_number > 100:
            mpc_score += 2
            reasons.append(f"Ill-conditioned system (κ={condition_number:.1f})")
        
        # Many variables
        n_vars = gain_matrix.shape[0]
        if n_vars >= 5:
            mpc_score += 2
            reasons.append(f"Large number of variables ({n_vars})")
        
        # Recommendation
        if mpc_score >= 5:
            recommendation = "STRONGLY RECOMMENDED"
        elif mpc_score >= 3:
            recommendation = "RECOMMENDED"
        elif mpc_score >= 1:
            recommendation = "CONSIDER"
        else:
            recommendation = "NOT NECESSARY"
        
        return {
            'recommendation': recommendation,
            'score': mpc_score,
            'reasons': reasons,
            'alternative': 'Decentralized PID control may be sufficient' if mpc_score < 3 else 'Consider cascade or decoupling'
        }
    
    @staticmethod
    def pairing_priority_rules(cv_type: str, process_type: str) -> List[str]:
        """
        Get pairing priority rules based on variable and process type
        
        Args:
            cv_type: Controlled variable type
            process_type: Process/unit type
            
        Returns:
            List of priority rules
        """
        rules = {
            'temperature': [
                "Prefer direct heat duty manipulation for tight control",
                "Consider cascade with flow control for better disturbance rejection",
                "For exothermic reactors, use cooling as primary manipulator"
            ],
            'pressure': [
                "Use vent/inlet for gas pressure control",
                "For distillation, condenser duty or vent valve",
                "Maintain above bubble point, below dew point"
            ],
            'level': [
                "Use averaging control unless inventory critical",
                "Pair with outlet flow for liquid vessels",
                "Interface level control critical for separators"
            ],
            'composition': [
                "Direct composition control often slow",
                "Consider temperature as proxy measurement",
                "Inferential control using multiple temperatures"
            ],
            'flow': [
                "Flow control typically fast and straightforward",
                "Use for throughput manipulation",
                "Ratio control for stoichiometry"
            ]
        }
        
        return rules.get(cv_type, ["Apply standard control principles"])
    
    @staticmethod
    def calculate_pairing_score(rga_value: float, 
                               controllability: float,
                               interaction: float,
                               steady_state_gain: float) -> float:
        """
        Calculate overall pairing score using weighted criteria
        
        Args:
            rga_value: RGA element value
            controllability: Controllability metric
            interaction: Interaction measure
            steady_state_gain: Steady-state gain magnitude
            
        Returns:
            Overall score (0-1, higher is better)
        """
        # RGA score (peak at 1.0, penalize negative)
        if rga_value < 0:
            rga_score = 0.0
        elif 0.8 <= rga_value <= 1.2:
            rga_score = 1.0
        elif 0.5 <= rga_value < 0.8:
            rga_score = 0.7
        elif rga_value < 0.5:
            rga_score = 0.3
        else:  # rga_value > 1.2
            rga_score = max(0, 1.0 - 0.3 * (rga_value - 1.2))
        
        # Controllability score (higher is better)
        controllability_score = min(1.0, controllability)
        
        # Interaction score (lower is better)
        interaction_score = max(0, 1.0 - interaction)
        
        # Gain magnitude score (prefer significant gains)
        gain_score = min(1.0, abs(steady_state_gain) / 1.0)
        
        # Weighted combination
        weights = {
            'rga': 0.4,
            'controllability': 0.3,
            'interaction': 0.2,
            'gain': 0.1
        }
        
        total_score = (
            weights['rga'] * rga_score +
            weights['controllability'] * controllability_score +
            weights['interaction'] * interaction_score +
            weights['gain'] * gain_score
        )
        
        return total_score
    
    @staticmethod
    def controller_type_recommendation(process_dynamics: Dict) -> str:
        """
        Recommend controller type based on process dynamics
        
        Args:
            process_dynamics: Dictionary with time constants, dead times, etc.
            
        Returns:
            Recommended controller type
        """
        tau = process_dynamics.get('time_constant', 10.0)
        theta = process_dynamics.get('dead_time', 0.0)
        
        # Calculate dead-time to time-constant ratio
        if tau > 0:
            ratio = theta / tau
        else:
            ratio = 0
        
        if ratio < 0.1:
            return "PID (fast, minimal dead time)"
        elif ratio < 0.3:
            return "PID with derivative action"
        elif ratio < 0.5:
            return "PI controller (moderate dead time)"
        else:
            return "Consider cascade or Smith predictor (large dead time)"
    
    @staticmethod
    def tuning_recommendations(time_constant: float, 
                              dead_time: float,
                              gain: float) -> Dict:
        """
        Provide controller tuning recommendations
        
        Args:
            time_constant: Process time constant
            dead_time: Process dead time
            gain: Process gain
            
        Returns:
            Tuning parameter recommendations
        """
        # Ziegler-Nichols tuning for FOPTD model
        if dead_time > 0 and time_constant > 0:
            # PI controller
            Kc_pi = 0.9 * time_constant / (gain * dead_time)
            Ti_pi = 3.33 * dead_time
            
            # PID controller
            Kc_pid = 1.2 * time_constant / (gain * dead_time)
            Ti_pid = 2.0 * dead_time
            Td_pid = 0.5 * dead_time
            
            return {
                'PI': {
                    'Kc': Kc_pi,
                    'Ti': Ti_pi,
                    'method': 'Ziegler-Nichols FOPTD'
                },
                'PID': {
                    'Kc': Kc_pid,
                    'Ti': Ti_pid,
                    'Td': Td_pid,
                    'method': 'Ziegler-Nichols FOPTD'
                },
                'note': 'Start with 50% of calculated Kc for safety'
            }
        else:
            return {
                'note': 'Insufficient dynamic information for tuning',
                'recommendation': 'Start with conservative tuning and iterate'
            }
    
    @staticmethod
    def inventory_control_philosophy(vessel_size: str, 
                                     process_criticality: str) -> str:
        """
        Recommend inventory (level) control philosophy
        
        Args:
            vessel_size: 'small', 'medium', 'large'
            process_criticality: 'high', 'medium', 'low'
            
        Returns:
            Control philosophy recommendation
        """
        if vessel_size == 'small' or process_criticality == 'high':
            return "TIGHT level control - maintain level within narrow band"
        elif vessel_size == 'large' and process_criticality == 'low':
            return "AVERAGING level control - allow level to float, smooth flow disturbances"
        else:
            return "MODERATE level control - balance between tight and averaging"