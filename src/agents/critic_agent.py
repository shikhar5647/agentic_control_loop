from .base_agent import BaseAgent
from typing import Dict, Any, List
import numpy as np
import logging
import json

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """
    Critic Agent that performs reflection-based evaluation of proposed control
    pairings.  It acts as an adversarial reviewer: instead of contributing to
    the solution it *challenges* it, probing for weaknesses in disturbance
    rejection, interaction under transient conditions, and heuristic
    consistency.

    The agent produces a structured critique with per-pairing assessments and
    an overall accept / revise verdict.  When issues are found it returns
    actionable feedback that the Pairing Optimizer can use in a subsequent
    refinement pass.
    """

    MAX_REVISION_ROUNDS = 2  # safety cap so we don't loop forever

    def __init__(self, temperature: float = 0.2):
        super().__init__("Critic Agent", temperature)

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------
    def create_system_prompt(self) -> str:
        return """You are a senior process-control reviewer whose SOLE purpose is to
CHALLENGE a proposed control structure — not to build one.

You must evaluate the proposed pairings against five criteria:

1. **Disturbance Rejection**
   - For each CV, identify the most likely disturbances.
   - Compare the disturbance-to-CV gain/dynamics against the MV-to-CV
     control pathway (gain magnitude, HII, time constant).
   - Flag any pairing where the disturbance pathway is stronger or
     faster than the control pathway.

2. **Interaction Under Transient Conditions**
   - Using off-diagonal RGA and HII elements, assess whether a
     disturbance entering one loop will propagate excessively to
     other loops through the proposed pairing topology.
   - Flag pairings where off-diagonal HII > 1.5 for the paired
     configuration.

3. **Heuristic Consistency**
   - Verify that safety-critical variables (temperature in exothermic
     reactors, pressure in vessels) are paired with the MV that has the
     largest physical capacity to handle them.
   - Verify inventory (level) loops use direct outflows.
   - Verify the plantwide control hierarchy (safety → inventory →
     quality → economics) is respected.

4. **Worst-Case Vulnerability**
   - Identify the single weakest pairing in the configuration — the
     one most likely to fail under realistic operating disturbances.
   - Explain *why* it is the weakest and what could go wrong.

5. **Overall Verdict**
   - ACCEPT: all pairings pass with at most minor observations.
   - REVISE: one or more pairings have significant issues; provide
     specific, actionable suggestions for the Pairing Optimizer.

Be rigorous, quantitative (cite RGA / HII values), and concise.
Do NOT propose a full alternative structure — only flag problems and
suggest targeted fixes."""

    # ------------------------------------------------------------------
    # Main invoke
    # ------------------------------------------------------------------
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the critique pipeline."""
        try:
            optimal_pairings = state.get("optimal_pairings", [])
            if not optimal_pairings:
                logger.warning(f"{self.agent_name}: No pairings to critique.")
                state["critique_result"] = {
                    "verdict": "SKIP",
                    "reason": "No pairings provided to critique.",
                }
                return state

            # ---- deterministic checks first ----
            deterministic = self._deterministic_checks(state)

            # ---- LLM-based critique ----
            prompt = self._build_critique_prompt(state, deterministic)
            system_prompt = self.create_system_prompt()
            critique_text = self.call_llm(prompt, system_prompt)

            # ---- extract structured verdict ----
            verdict = self._extract_verdict(critique_text, system_prompt)

            # ---- assemble result ----
            critique_result = {
                "critique_text": critique_text,
                "deterministic_checks": deterministic,
                "verdict": verdict.get("verdict", "ACCEPT"),
                "per_pairing_issues": verdict.get("per_pairing_issues", []),
                "worst_case_pairing": verdict.get("worst_case_pairing", None),
                "revision_suggestions": verdict.get("revision_suggestions", []),
                "confidence_adjustment": verdict.get("confidence_adjustment", 0.0),
            }

            state["critique_result"] = critique_result

            # Track how many revision rounds have occurred
            revision_count = state.get("revision_count", 0)
            state["revision_count"] = revision_count

            # ---- messages ----
            if "messages" not in state:
                state["messages"] = []
            state["messages"].append(
                {
                    "agent": self.agent_name,
                    "content": (
                        f"Critique complete — verdict: {critique_result['verdict']}. "
                        f"Issues found: {len(critique_result['per_pairing_issues'])}. "
                        f"Revision round: {revision_count}."
                    ),
                }
            )

            logger.info(
                f"{self.agent_name}: verdict={critique_result['verdict']}, "
                f"issues={len(critique_result['per_pairing_issues'])}"
            )
            return state

        except Exception as e:
            logger.error(f"Error in {self.agent_name}: {e}", exc_info=True)
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(f"{self.agent_name}: {str(e)}")
            # On error, default to ACCEPT so the pipeline doesn't stall
            state["critique_result"] = {"verdict": "ACCEPT", "error": str(e)}
            return state

    # ------------------------------------------------------------------
    # Deterministic (numerical) checks
    # ------------------------------------------------------------------
    def _deterministic_checks(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fast, code-based checks that don't need an LLM."""
        checks: Dict[str, Any] = {
            "negative_rga_pairings": [],
            "weak_hii_pairings": [],
            "strong_offdiag_interaction": [],
            "high_condition_number": False,
            "issues_count": 0,
        }

        pairings = state.get("optimal_pairings", [])
        rga_matrix = state.get("rga_matrix")
        hii_matrix = state.get("hii_matrix")
        condition_number = state.get("condition_number", 1.0)
        pfd_data = state.get("pfd_data", {})

        cv_names = [cv["name"] for cv in pfd_data.get("controlled_variables", [])]
        mv_names = [mv["name"] for mv in pfd_data.get("manipulated_variables", [])]

        # 1. Negative RGA on paired elements
        for p in pairings:
            rga_val = p.get("rga_value", 1.0)
            if rga_val is not None and rga_val < 0:
                checks["negative_rga_pairings"].append(
                    {
                        "cv": p["controlled_variable"],
                        "mv": p["manipulated_variable"],
                        "rga_value": rga_val,
                    }
                )
                checks["issues_count"] += 1

        # 2. Weak HII on paired elements (< 0.3)
        for p in pairings:
            hii_val = p.get("hii_value")
            if hii_val is not None and hii_val < 0.3:
                checks["weak_hii_pairings"].append(
                    {
                        "cv": p["controlled_variable"],
                        "mv": p["manipulated_variable"],
                        "hii_value": hii_val,
                    }
                )
                checks["issues_count"] += 1

        # 3. Strong off-diagonal interaction in the *paired* sub-matrix
        if hii_matrix is not None and len(pairings) > 1:
            paired_cv_indices = []
            paired_mv_indices = []
            for p in pairings:
                try:
                    ci = cv_names.index(p["controlled_variable"])
                    mi = mv_names.index(p["manipulated_variable"])
                    paired_cv_indices.append(ci)
                    paired_mv_indices.append(mi)
                except ValueError:
                    continue

            for i, ci in enumerate(paired_cv_indices):
                for j, mj in enumerate(paired_mv_indices):
                    if i != j:
                        hii_offdiag = float(hii_matrix[ci, mj])
                        if hii_offdiag > 1.5:
                            checks["strong_offdiag_interaction"].append(
                                {
                                    "cv": cv_names[ci],
                                    "interfering_mv": mv_names[mj],
                                    "hii_offdiag": hii_offdiag,
                                }
                            )
                            checks["issues_count"] += 1

        # 4. High condition number
        if condition_number is not None and condition_number > 100:
            checks["high_condition_number"] = True
            checks["issues_count"] += 1

        return checks

    # ------------------------------------------------------------------
    # LLM prompt construction
    # ------------------------------------------------------------------
    def _build_critique_prompt(
        self, state: Dict[str, Any], deterministic: Dict[str, Any]
    ) -> str:
        """Build the full critique prompt combining all available info."""
        pfd_data = state.get("pfd_data", {})
        pairings = state.get("optimal_pairings", [])
        rga_matrix = state.get("rga_matrix")
        hii_matrix = state.get("hii_matrix")
        gain_matrix = state.get("gain_matrix")
        condition_number = state.get("condition_number", 0)
        interaction_index = state.get("interaction_index", 0)
        pairing_reasoning = state.get("pairing_reasoning", "")

        cv_names = [cv["name"] for cv in pfd_data.get("controlled_variables", [])]
        mv_names = [mv["name"] for mv in pfd_data.get("manipulated_variables", [])]

        prompt = f"""## CRITIQUE TASK

You are reviewing the following proposed control structure for:
**{pfd_data.get('name', 'Unknown Process')}**
{pfd_data.get('description', '')}

---

### Proposed Pairings
"""
        for i, p in enumerate(pairings, 1):
            prompt += (
                f"\n{i}. **{p.get('controlled_variable')}** ← "
                f"**{p.get('manipulated_variable')}**\n"
                f"   RGA = {p.get('rga_value', 'N/A')}, "
                f"HII = {p.get('hii_value', 'N/A')}, "
                f"Controller = {p.get('controller_type', 'PID')}\n"
                f"   Reasoning: {p.get('reasoning', 'N/A')}\n"
            )

        # Gain matrix
        if gain_matrix is not None:
            prompt += "\n### Gain Matrix\n"
            prompt += self.format_matrix(gain_matrix, cv_names, mv_names) + "\n"

        # RGA matrix
        if rga_matrix is not None:
            prompt += "\n### RGA Matrix\n"
            prompt += self.format_matrix(rga_matrix, cv_names, mv_names) + "\n"

        # HII matrix
        if hii_matrix is not None:
            prompt += "\n### Hankel Interaction Index (HII) Matrix\n"
            prompt += self.format_matrix(hii_matrix, cv_names, mv_names) + "\n"

        # System metrics
        prompt += f"""
### System Metrics
- Condition Number (κ): {condition_number:.4f}
- Interaction Index: {interaction_index:.4f}
"""

        # Deterministic flags
        if deterministic["issues_count"] > 0:
            prompt += "\n### Automated Flags (deterministic checks)\n"
            if deterministic["negative_rga_pairings"]:
                prompt += "**Negative RGA on paired elements:**\n"
                for item in deterministic["negative_rga_pairings"]:
                    prompt += f"  - {item['cv']} ← {item['mv']}: RGA = {item['rga_value']:.3f}\n"
            if deterministic["weak_hii_pairings"]:
                prompt += "**Weak HII on paired elements (< 0.3):**\n"
                for item in deterministic["weak_hii_pairings"]:
                    prompt += f"  - {item['cv']} ← {item['mv']}: HII = {item['hii_value']:.3f}\n"
            if deterministic["strong_offdiag_interaction"]:
                prompt += "**Strong off-diagonal HII (> 1.5) in paired topology:**\n"
                for item in deterministic["strong_offdiag_interaction"]:
                    prompt += (
                        f"  - {item['cv']} affected by {item['interfering_mv']}: "
                        f"HII = {item['hii_offdiag']:.3f}\n"
                    )
            if deterministic["high_condition_number"]:
                prompt += f"**System is ill-conditioned** (κ = {condition_number:.2f})\n"

        # Previous pairing reasoning (truncated)
        if pairing_reasoning:
            prompt += f"\n### Pairing Optimizer Reasoning (excerpt)\n{pairing_reasoning[:1000]}...\n"

        # Disturbance info
        disturbances = pfd_data.get("disturbance_variables", [])
        if disturbances:
            prompt += "\n### Known Disturbances\n"
            for dv in disturbances:
                prompt += f"- {dv['name']} ({dv.get('type', '')}): {dv.get('description', '')}\n"

        # Time constants
        time_constants = pfd_data.get("time_constants")
        if time_constants is not None:
            tc = np.array(time_constants)
            prompt += f"\n### Time Constant Statistics\n"
            prompt += f"- Range: [{np.min(tc):.2f}, {np.max(tc):.2f}]\n"
            prompt += f"- Mean: {np.mean(tc):.2f}\n"

        prompt += """
---

### Your Critique Must Cover

1. **Disturbance Rejection Assessment** — for each pairing, is the
   control pathway strong and fast enough relative to the expected
   disturbance pathways?

2. **Transient Interaction Risk** — will disturbances propagate
   excessively across loops given the chosen pairing topology?

3. **Heuristic Consistency** — do the pairings respect safety-first
   hierarchy, inventory control best practices, and unit-operation
   specific guidelines?

4. **Worst-Case Identification** — which single pairing is most
   vulnerable and why?

5. **Verdict** — ACCEPT (minor issues only) or REVISE (significant
   issues with specific actionable suggestions).

Be quantitative. Cite RGA and HII values. Be concise but thorough.
"""
        return prompt

    # ------------------------------------------------------------------
    # Extract structured verdict from LLM critique
    # ------------------------------------------------------------------
    def _extract_verdict(self, critique_text: str, system_prompt: str) -> Dict:
        """Ask the LLM to distill the free-text critique into structured JSON."""
        extraction_prompt = f"""Based on the following critique:

{critique_text}

Extract the results into this exact JSON structure (return ONLY valid JSON):
{{
    "verdict": "ACCEPT or REVISE",
    "per_pairing_issues": [
        {{
            "cv": "variable name",
            "mv": "variable name",
            "severity": "CRITICAL / HIGH / MEDIUM / LOW",
            "issue": "brief description",
            "suggestion": "what to change"
        }}
    ],
    "worst_case_pairing": {{
        "cv": "variable name",
        "mv": "variable name",
        "reason": "why this is the weakest link"
    }},
    "revision_suggestions": [
        "actionable suggestion 1",
        "actionable suggestion 2"
    ],
    "confidence_adjustment": -0.05
}}

Rules:
- confidence_adjustment is a number between -0.20 and 0.0
  (0.0 = no change, -0.20 = serious problems reduce confidence by 20%)
- If verdict is ACCEPT, per_pairing_issues should only contain LOW/MEDIUM items
- If verdict is REVISE, at least one CRITICAL or HIGH item must be present
"""
        response = self.call_llm(extraction_prompt, system_prompt)

        try:
            # Clean markdown fences if present
            cleaned = response.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            return json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            logger.warning(
                f"{self.agent_name}: Could not parse verdict JSON, "
                "defaulting to ACCEPT."
            )
            return {
                "verdict": "ACCEPT",
                "per_pairing_issues": [],
                "worst_case_pairing": None,
                "revision_suggestions": [],
                "confidence_adjustment": 0.0,
            }