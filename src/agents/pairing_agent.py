from src.agents.base_agent import BaseAgent
from src.utils.chemical_engineering import ChemicalEngineeringUtils
from typing import Dict, Any, List, Optional
import numpy as np
import logging
import json

logger = logging.getLogger(__name__)


class PairingAgent(BaseAgent):
    """Agent responsible for optimizing control pairings using multiple
    heuristics including Hankel — now with critic-feedback awareness."""

    def __init__(self, temperature: float = 0.3):
        super().__init__("Pairing Optimizer Agent", temperature)
        self.chem_utils = ChemicalEngineeringUtils()

    def create_system_prompt(self) -> str:
        return """You are an expert in control structure synthesis and optimization.

Your role is to determine the optimal control loop pairings by integrating multiple analysis methods:

1. **RGA Analysis**: Variable pairing based on steady-state interaction measures
2. **Hankel Interaction Index**: Dynamic interaction analysis accounting for process time constants
3. **Controllability (SVD)**: Ensuring pairings use strong control directions
4. **Interaction Minimization**: Reducing loop coupling
5. **Chemical Engineering Principles**: Process-specific control strategies

Pairing Optimization Criteria:
- Maximize RGA diagonal elements (minimize steady-state interactions)
- Minimize Hankel dynamic coupling (good HII values: 0.7-1.3)
- Align pairings with dominant singular value directions
- Minimize overall interaction index
- Follow unit operation-specific control strategies
- Consider practical implementation and maintenance
- Account for time-scale separation

Key Integration:
- When RGA and HII disagree, prioritize based on system dynamics
- For fast processes, HII matters more (transient behavior critical)
- For slow processes, RGA may be sufficient
- Strong HII coupling (> 1.5) requires special tuning considerations

IMPORTANT — REVISION ROUNDS:
If you receive CRITIC FEEDBACK in the prompt it means a previous version
of your pairings was reviewed and found wanting.  You MUST address every
flagged issue.  Changed pairings should be justified with numerical
evidence (RGA / HII values for old vs new pairing).

You must provide:
1. Final optimized CV-MV pairings with justification
2. Controller type recommendations (PI, PID, Cascade, etc.)
3. Chemical engineering rationale for each pairing
4. Dynamic interaction warnings and mitigation strategies
5. Tuning guidance based on both steady-state and dynamic analysis

Balance theoretical optimality with practical engineering judgment."""

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize control pairings"""
        try:
            # Get all previous analyses
            gain_matrix = state['gain_matrix']
            rga_matrix = state.get('rga_matrix')
            rga_pairings = state.get('rga_pairings', [])
            hii_matrix = state.get('hii_matrix')
            hankel_pairings = state.get('hankel_pairings', [])
            hankel_analysis = state.get('hankel_analysis', '')
            svd_metrics = state.get('controllability_metrics', {})
            pfd_data = state['pfd_data']
            pfd_analysis = state.get('pfd_analysis', '')
            control_objectives = state.get('control_objectives', [])

            # --- NEW: Critic feedback for revision rounds ---
            critic_feedback = state.get('critic_feedback')

            # Calculate interaction index
            interaction_index = self.chem_utils.calculate_interaction_index(gain_matrix)

            # Get maximum weight matching
            mw_pairings = self.chem_utils.maximum_weight_matching(gain_matrix)

            # Create optimization prompt (now with optional critic feedback)
            prompt = self._create_optimization_prompt(
                gain_matrix, rga_matrix, rga_pairings,
                hii_matrix, hankel_pairings, hankel_analysis,
                svd_metrics, interaction_index, mw_pairings,
                pfd_data, pfd_analysis, control_objectives,
                critic_feedback  # <-- NEW
            )

            # Get LLM optimization
            system_prompt = self.create_system_prompt()
            optimization_analysis = self.call_llm(prompt, system_prompt)

            # Extract final pairings
            pairings_prompt = f"""Based on the comprehensive analysis including Hankel dynamics:

{optimization_analysis}

Provide the FINAL OPTIMAL control loop pairings in JSON format:
[
    {{
        "controlled_variable": "CV_name",
        "manipulated_variable": "MV_name",
        "controller_type": "PID/PI/CASCADE/etc",
        "rga_value": 0.xx,
        "hii_value": 0.xx,
        "controllability_score": 0.xx,
        "interaction_score": 0.xx,
        "overall_confidence": 0.xx,
        "reasoning": "brief justification integrating RGA, HII, and controllability",
        "chemical_eng_rationale": "process-specific justification",
        "dynamic_consideration": "specific guidance based on Hankel analysis",
        "tuning_guidance": "controller tuning recommendations considering dynamics"
    }},
    ...
]

Include ALL control loops (one per controlled variable). Ensure pairings balance steady-state (RGA) and dynamic (HII) considerations."""

            pairings_response = self.call_llm(pairings_prompt, system_prompt)

            # Parse pairings
            try:
                optimal_pairings = json.loads(pairings_response)
            except Exception:
                optimal_pairings = self._create_fallback_pairings(
                    rga_pairings, hankel_pairings, svd_metrics, pfd_data
                )

            # Validate pairings
            optimal_pairings = self._validate_and_enhance_pairings(
                optimal_pairings, gain_matrix, rga_matrix, hii_matrix, pfd_data
            )

            # Update state
            state['interaction_index'] = interaction_index
            state['optimal_pairings'] = optimal_pairings
            state['pairing_reasoning'] = optimization_analysis

            # Add message
            if 'messages' not in state:
                state['messages'] = []

            revision_note = ""
            if critic_feedback:
                revision_note = " (revised after critic feedback)"

            state['messages'].append({
                'agent': self.agent_name,
                'content': (
                    f"Optimized {len(optimal_pairings)} control loop pairings "
                    f"considering RGA, Hankel dynamics, and controllability. "
                    f"Interaction index: {interaction_index:.3f}{revision_note}"
                )
            })

            logger.info(f"{self.agent_name}: Pairing optimization complete{revision_note}")
            return state

        except Exception as e:
            logger.error(f"Error in {self.agent_name}: {e}")
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(f"{self.agent_name}: {str(e)}")
            return state

    def _create_optimization_prompt(
        self, gain_matrix, rga_matrix, rga_pairings,
        hii_matrix, hankel_pairings, hankel_analysis,
        svd_metrics, interaction_index, mw_pairings,
        pfd_data, pfd_analysis, control_objectives,
        critic_feedback: Optional[str] = None  # <-- NEW
    ) -> str:
        """Create comprehensive optimization prompt"""

        cv_names = [cv['name'] for cv in pfd_data['controlled_variables']]
        mv_names = [mv['name'] for mv in pfd_data['manipulated_variables']]

        prompt = f"""Optimize control loop pairings by integrating RGA, Hankel dynamics, and controllability analysis:

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

**RGA-Based Pairings (Steady-State Analysis):**
"""
        for p in rga_pairings:
            prompt += f"- {p['cv']} ← {p['mv']} (RGA λ = {p['rga_value']:.3f}) - {p.get('recommendation', 'N/A')}\n"

        # Add Hankel analysis if available
        if hii_matrix is not None and len(hankel_pairings) > 0:
            prompt += f"""\n**Hankel Interaction Index (Dynamic Analysis):**
"""
            if hii_matrix is not None:
                prompt += self.format_matrix(hii_matrix, cv_names, mv_names) + "\n"

            prompt += "\n**Hankel-Based Pairings (Dynamic Coupling):**\n"
            for p in hankel_pairings:
                prompt += f"- {p['cv']} ← {p['mv']} (HII = {p['hii_value']:.3f}, Coupling: {p.get('dynamic_coupling', 'N/A')}) - {p.get('recommendation', 'N/A')}\n"

            prompt += f"""\n**Key Insights from Hankel Analysis:**
{hankel_analysis[:800]}...

**RGA vs HII Comparison:**
Critical Question: Where do RGA and HII disagree, and why?
"""
            disagreements = []
            for rga_p in rga_pairings:
                rga_cv, rga_mv = rga_p['cv'], rga_p['mv']
                hii_p = next((h for h in hankel_pairings if h['cv'] == rga_cv), None)
                if hii_p and hii_p['mv'] != rga_mv:
                    disagreements.append(f"  - {rga_cv}: RGA suggests {rga_mv}, but HII suggests {hii_p['mv']}")

            if disagreements:
                prompt += "\n" + "\n".join(disagreements) + "\n"
            else:
                prompt += "  - RGA and HII pairings are in agreement\n"
        else:
            prompt += "\n**Note:** Hankel analysis not available (time constants missing). Relying on RGA and controllability.\n"

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
                prompt += f"- {cv_names[cv_idx]} ← {mv_names[mv_idx]} (|gain|: {abs(gain_matrix[cv_idx, mv_idx]):.3f})\n"

        prompt += """\n**Unit Operation Control Strategies:**
"""
        for unit in pfd_data['unit_operations']:
            strategies = self.chem_utils.get_unit_operation_control_strategy(unit['type'])
            prompt += f"\n{unit['name']} ({unit['type']}):\n"
            for strategy in strategies[:3]:
                prompt += f"  - {strategy}\n"

        prompt += """\n\n**Optimization Task:**

Synthesize an optimal control structure by:

1. **Multi-Criteria Integration**:
   - RGA optimality (40% weight)
   - HII dynamic coupling (25% weight)
   - Controllability (25% weight)
   - Interaction minimization (10% weight)

2. **Resolving RGA vs HII Conflicts**:
   - If RGA and HII disagree: check process time scales
   - Fast processes: favor HII
   - Slow processes: favor RGA
   - Moderate: compromise or suggest cascade

3. **Pairing Selection**:
   - ONE manipulated variable per controlled variable
   - Justify each using ALL analyses (RGA, HII, SVD)
   - Ensure physically realizable and maintainable

4. **Chemical Engineering Validation**:
   - Verify pairings follow process physics
   - Safety-critical variables must be adequately controlled
   - Inventory loops use direct outflows

Provide a comprehensive optimization that balances steady-state and dynamic factors."""

        # ----- NEW: Append critic feedback if in revision round -----
        if critic_feedback:
            prompt += f"""

============================================================
⚠️  CRITIC REVISION FEEDBACK — YOU MUST ADDRESS THESE ISSUES
============================================================

The Critic Agent has reviewed your previous pairing proposal and
identified issues that MUST be addressed in this revision.

{critic_feedback}

IMPORTANT:
- You MUST change pairings that were flagged as CRITICAL or HIGH severity.
- For MEDIUM issues, provide explicit justification if you keep the same pairing.
- Explain how your revised pairings address each flagged issue.
- If a suggested alternative MV is available, evaluate it numerically
  (RGA, HII) and adopt it if it is superior.
- Cite specific RGA and HII values for old vs new pairings to justify changes.
============================================================
"""

        return prompt

    def _create_fallback_pairings(self, rga_pairings, hankel_pairings,
                                   svd_metrics, pfd_data) -> List[Dict]:
        """Create fallback pairings if LLM parsing fails"""
        pairings = []
        primary_pairings = hankel_pairings if hankel_pairings else rga_pairings

        for p in primary_pairings:
            pairing_dict = {
                'controlled_variable': p['cv'],
                'manipulated_variable': p['mv'],
                'controller_type': 'PID',
                'rga_value': p.get('rga_value', 0.0),
                'hii_value': p.get('hii_value', 0.0),
                'controllability_score': svd_metrics.get('controllability_score', 0.5),
                'interaction_score': 0.5,
                'overall_confidence': 0.7,
                'reasoning': 'Based on RGA and Hankel analysis',
                'chemical_eng_rationale': 'Standard pairing',
                'dynamic_consideration': p.get('tuning_consideration', 'Standard tuning'),
                'tuning_guidance': 'Start with conservative tuning'
            }
            pairings.append(pairing_dict)

        return pairings

    def _validate_and_enhance_pairings(
        self, pairings: List[Dict], gain_matrix, rga_matrix,
        hii_matrix, pfd_data
    ) -> List[Dict]:
        """Validate and enhance pairings with computed metrics"""
        cv_names = [cv['name'] for cv in pfd_data['controlled_variables']]
        mv_names = [mv['name'] for mv in pfd_data['manipulated_variables']]

        for pairing in pairings:
            try:
                cv_idx = cv_names.index(pairing['controlled_variable'])
                mv_idx = mv_names.index(pairing['manipulated_variable'])

                if rga_matrix is not None:
                    pairing['rga_value'] = float(rga_matrix[cv_idx, mv_idx])

                if hii_matrix is not None:
                    pairing['hii_value'] = float(hii_matrix[cv_idx, mv_idx])
                else:
                    pairing['hii_value'] = None

                pairing['steady_state_gain'] = float(gain_matrix[cv_idx, mv_idx])

            except (ValueError, IndexError) as e:
                logger.warning(f"Could not validate pairing: {e}")

        return pairings