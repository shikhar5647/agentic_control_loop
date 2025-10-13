from .base_agent import BaseAgent
from typing import Dict, Any, List
import numpy as np
import logging
import json

logger = logging.getLogger(__name__)

class ValidationAgent(BaseAgent):
    """Agent responsible for validating final control structure"""
    
    def __init__(self, temperature: float = 0.2):
        super().__init__("Validation Agent", temperature)
    
    def create_system_prompt(self) -> str:
        return """You are an expert in process control validation and safety review.

Your role is to perform final validation of the proposed control structure:

**Safety Checks:**
- All safety-critical variables have reliable control
- Fail-safe actions are defined
- Emergency shutdown sequences are clear
- Interlocks and alarms are considered

**Engineering Checks:**
- Control structure is physically realizable
- Instrumentation is available/practical
- Pairings respect process constraints
- No conflicting control actions

**Performance Checks:**
- Expected closed-loop performance is adequate
- Interaction effects are manageable
- Disturbance rejection is sufficient
- Robustness to model uncertainty

**Operational Checks:**
- Operators can understand and operate the system
- Startup and shutdown procedures are clear
- Maintenance requirements are reasonable
- Tuning and commissioning are feasible

You must provide:
1. Validation results (PASS/FAIL for each check)
2. Critical warnings that must be addressed
3. Recommendations for improvement
4. Final confidence score for the control structure

Be thorough and conservative - safety is paramount."""
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate final control structure"""
        try:
            optimal_pairings = state['optimal_pairings']
            pfd_data = state['pfd_data']
            gain_matrix = state['gain_matrix']
            rga_matrix = state.get('rga_matrix')
            interaction_index = state.get('interaction_index', 0.0)
            condition_number = state.get('condition_number', 1.0)
            pfd_analysis = state.get('pfd_analysis', '')
            
            # Create validation prompt
            prompt = self._create_validation_prompt(
                optimal_pairings, pfd_data, gain_matrix, rga_matrix,
                interaction_index, condition_number, pfd_analysis
            )
            
            # Get LLM validation
            system_prompt = self.create_system_prompt()
            validation_analysis = self.call_llm(prompt, system_prompt)
            
            # Extract structured validation results
            results_prompt = f"""Based on the validation analysis:

{validation_analysis}

Provide structured validation results in JSON format:
{{
    "safety_check": "PASS/FAIL",
    "engineering_check": "PASS/FAIL",
    "performance_check": "PASS/FAIL",
    "operational_check": "PASS/FAIL",
    "overall_status": "APPROVED/CONDITIONAL/REJECTED",
    "confidence_score": 0.xx,
    "critical_warnings": ["warning1", "warning2", ...],
    "recommendations": ["rec1", "rec2", ...],
    "summary": "brief overall assessment"
}}"""
            
            results_response = self.call_llm(results_prompt, system_prompt)
            
            # Parse results
            try:
                validation_results = json.loads(results_response)
            except:
                validation_results = self._create_fallback_validation(optimal_pairings)
            
            # Perform automated checks
            automated_checks = self._perform_automated_checks(
                optimal_pairings, gain_matrix, rga_matrix,
                interaction_index, condition_number
            )
            
            # Merge results
            validation_results['automated_checks'] = automated_checks
            
            # Generate final recommendations
            final_recommendations = self._generate_recommendations(
                validation_results, automated_checks, optimal_pairings
            )
            
            # Update state
            state['validation_results'] = validation_results
            state['final_recommendations'] = final_recommendations
            state['warnings'] = validation_results.get('critical_warnings', [])
            
            # Add message
            if 'messages' not in state:
                state['messages'] = []
            state['messages'].append({
                'agent': self.agent_name,
                'content': f"Validation complete: {validation_results.get('overall_status', 'UNKNOWN')} "
                          f"(Confidence: {validation_results.get('confidence_score', 0.0):.2f})"
            })
            
            logger.info(f"{self.agent_name}: Validation complete")
            return state
            
        except Exception as e:
            logger.error(f"Error in {self.agent_name}: {e}")
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(f"{self.agent_name}: {str(e)}")
            return state
    
    def _create_validation_prompt(
        self, optimal_pairings, pfd_data, gain_matrix, rga_matrix,
        interaction_index, condition_number, pfd_analysis
    ) -> str:
        """Create comprehensive validation prompt"""
        
        prompt = f"""Validate the following control structure design:

**Process:** {pfd_data['name']}
{pfd_data['description']}

**Proposed Control Structure:**
"""
        for i, pairing in enumerate(optimal_pairings, 1):
            prompt += f"\n{i}. Control Loop:\n"
            prompt += f"   CV: {pairing['controlled_variable']}\n"
            prompt += f"   MV: {pairing['manipulated_variable']}\n"
            prompt += f"   Controller: {pairing.get('controller_type', 'PID')}\n"
            prompt += f"   RGA Value: {pairing.get('rga_value', 'N/A')}\n"
            prompt += f"   Confidence: {pairing.get('overall_confidence', 'N/A')}\n"
            prompt += f"   Rationale: {pairing.get('reasoning', 'N/A')}\n"
        
        prompt += f"""\n**System Metrics:**
- Interaction Index: {interaction_index:.4f}
- Condition Number: {condition_number:.4f}
- Number of Control Loops: {len(optimal_pairings)}

**Process Analysis Summary:**
{pfd_analysis[:600]}...

**Validation Tasks:**

1. **Safety Validation**:
   - Are all safety-critical variables controlled?
   - Are there adequate fail-safe mechanisms?
   - Can the system handle emergency situations?
   - Are there any single points of failure?

2. **Engineering Validation**:
   - Are all pairings physically realizable?
   - Is required instrumentation available?
   - Do pairings respect thermodynamic constraints?
   - Are there conflicting control objectives?
   - Is the control structure complete?

3. **Performance Validation**:
   - Will the system meet control objectives?
   - Can disturbances be adequately rejected?
   - Is interaction index acceptable (< 0.4 preferred)?
   - Is condition number acceptable (< 100 preferred)?
   - Will loop interactions cause instability?

4. **Operational Validation**:
   - Can operators understand this structure?
   - Is startup/shutdown procedure clear?
   - Are maintenance requirements reasonable?
   - Can controllers be tuned practically?

5. **Completeness Check**:
   - Is each controlled variable assigned exactly one manipulated variable?
   - Are all critical process variables addressed?
   - Are there unused manipulated variables?

For each validation area, provide:
- PASS/FAIL status
- Specific issues identified
- Severity (CRITICAL/HIGH/MEDIUM/LOW)
- Recommendations for resolution

Be thorough and identify any potential problems."""
        
        return prompt
    
    def _perform_automated_checks(
        self, pairings, gain_matrix, rga_matrix,
        interaction_index, condition_number
    ) -> Dict:
        """Perform automated numerical checks"""
        checks = {
            'completeness': True,
            'rga_negative_values': False,
            'high_interaction': False,
            'poor_conditioning': False,
            'weak_pairings': [],
            'issues': []
        }
        
        # Check for negative RGA values
        if rga_matrix is not None:
            for pairing in pairings:
                rga_val = pairing.get('rga_value', 1.0)
                if rga_val < 0:
                    checks['rga_negative_values'] = True
                    checks['issues'].append(
                        f"CRITICAL: Negative RGA value ({rga_val:.3f}) for "
                        f"{pairing['controlled_variable']} ← {pairing['manipulated_variable']}"
                    )
        
        # Check interaction index
        if interaction_index > 0.4:
            checks['high_interaction'] = True
            checks['issues'].append(
                f"HIGH: Interaction index ({interaction_index:.3f}) exceeds recommended threshold (0.4)"
            )
        
        # Check condition number
        if condition_number > 100:
            checks['poor_conditioning'] = True
            checks['issues'].append(
                f"HIGH: Condition number ({condition_number:.2f}) indicates ill-conditioned system"
            )
        elif condition_number > 50:
            checks['issues'].append(
                f"MEDIUM: Condition number ({condition_number:.2f}) suggests moderate conditioning issues"
            )
        
        # Check for weak pairings
        for pairing in pairings:
            rga_val = abs(pairing.get('rga_value', 1.0))
            if rga_val < 0.3:
                checks['weak_pairings'].append(pairing['controlled_variable'])
                checks['issues'].append(
                    f"MEDIUM: Weak pairing (RGA={rga_val:.3f}) for {pairing['controlled_variable']}"
                )
        
        return checks
    
    def _generate_recommendations(
        self, validation_results, automated_checks, pairings
    ) -> List[str]:
        """Generate final recommendations"""
        recommendations = []
        
        # From validation results
        if validation_results.get('recommendations'):
            recommendations.extend(validation_results['recommendations'])
        
        # From automated checks
        if automated_checks.get('rga_negative_values'):
            recommendations.append(
                "CRITICAL: Reconsider pairings with negative RGA values - these may cause instability"
            )
        
        if automated_checks.get('high_interaction'):
            recommendations.append(
                "Consider advanced control strategies (cascade, decoupling, or MPC) to handle loop interactions"
            )
        
        if automated_checks.get('poor_conditioning'):
            recommendations.append(
                "System is ill-conditioned. Consider: (1) Adding measurement filters, "
                "(2) Conservative controller tuning, (3) Sensor selection review"
            )
        
        if automated_checks.get('weak_pairings'):
            recommendations.append(
                f"Review weak pairings for: {', '.join(automated_checks['weak_pairings'])}. "
                "Consider alternative manipulated variables."
            )
        
        # General recommendations
        recommendations.append(
            "Perform dynamic simulation to validate controller tuning before implementation"
        )
        recommendations.append(
            "Develop comprehensive operating procedures including startup and shutdown sequences"
        )
        
        return recommendations
    
    def _create_fallback_validation(self, pairings) -> Dict:
        """Create fallback validation if LLM parsing fails"""
        return {
            'safety_check': 'CONDITIONAL',
            'engineering_check': 'CONDITIONAL',
            'performance_check': 'CONDITIONAL',
            'operational_check': 'CONDITIONAL',
            'overall_status': 'CONDITIONAL',
            'confidence_score': 0.7,
            'critical_warnings': ['Manual review required'],
            'recommendations': ['Verify all pairings', 'Perform dynamic simulation'],
            'summary': 'Validation completed with standard checks'
        }