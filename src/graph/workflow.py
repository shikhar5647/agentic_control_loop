from langgraph.graph import StateGraph, END
from src.graph.state import AgentState
from src.agents.pfd_analyzer_agent import PFDAnalyzerAgent
from src.agents.rga_agent import RGAAgent
from src.agents.hankel_interaction_agent import HankelInteractionAgent
from src.agents.controllability_agent import ControllabilityAgent
from src.agents.pairing_agent import PairingAgent
from src.agents.critic_agent import CriticAgent
from src.agents.validation_agent import ValidationAgent
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ControlLoopWorkflow:
    """
    LangGraph workflow for control loop prediction with Hankel interaction
    analysis and a Critic Agent reflection loop.

    Pipeline:
        PFD → RGA → Hankel → Controllability → Pairing → Critic
            ↓ (ACCEPT)                                  ↑ (REVISE, up to MAX rounds)
        Validation → Finalize → END

    The Critic Agent evaluates the proposed pairings and either ACCEPTs
    them (proceeding to Validation) or issues a REVISE verdict, sending
    structured feedback back to the Pairing Optimizer for refinement.
    A maximum of MAX_REVISION_ROUNDS iterations is enforced to guarantee
    termination.
    """

    MAX_REVISION_ROUNDS = 2  # max critic→optimizer loops

    def __init__(self, config: dict = None):
        self.config = config or {}
        agent_config = self.config.get("agents", {})

        # ---- initialise agents ----
        self.pfd_analyzer = PFDAnalyzerAgent(
            temperature=agent_config.get("pfd_analyzer", {}).get("temperature", 0.2)
        )
        self.rga_agent = RGAAgent(
            temperature=agent_config.get("rga_calculator", {}).get("temperature", 0.1)
        )
        self.hankel_agent = HankelInteractionAgent(
            temperature=agent_config.get("hankel_analyzer", {}).get("temperature", 0.2)
        )
        self.controllability_agent = ControllabilityAgent(
            temperature=agent_config.get("controllability_analyzer", {}).get(
                "temperature", 0.2
            )
        )
        self.pairing_agent = PairingAgent(
            temperature=agent_config.get("pairing_optimizer", {}).get(
                "temperature", 0.3
            )
        )
        self.critic_agent = CriticAgent(
            temperature=agent_config.get("critic_agent", {}).get("temperature", 0.2)
        )
        self.validation_agent = ValidationAgent(
            temperature=agent_config.get("validation_agent", {}).get(
                "temperature", 0.2
            )
        )

        # Build the graph
        self.graph = self._build_graph()
        logger.info(
            "Control Loop Workflow initialised with 7 agents "
            "(including Critic with reflection loop)"
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("pfd_analysis", self._pfd_analysis_node)
        workflow.add_node("rga_calculation", self._rga_calculation_node)
        workflow.add_node("hankel_interaction", self._hankel_interaction_node)
        workflow.add_node("controllability_analysis", self._controllability_analysis_node)
        workflow.add_node("pairing_optimization", self._pairing_optimization_node)
        workflow.add_node("critique", self._critique_node)
        workflow.add_node("validation", self._validation_node)
        workflow.add_node("finalize", self._finalize_node)

        # Entry
        workflow.set_entry_point("pfd_analysis")

        # Linear edges up to pairing
        workflow.add_edge("pfd_analysis", "rga_calculation")
        workflow.add_edge("rga_calculation", "hankel_interaction")
        workflow.add_edge("hankel_interaction", "controllability_analysis")
        workflow.add_edge("controllability_analysis", "pairing_optimization")
        workflow.add_edge("pairing_optimization", "critique")

        # Conditional edge after critique: REVISE → pairing, ACCEPT → validation
        workflow.add_conditional_edges(
            "critique",
            self._should_revise,
            {
                "revise": "pairing_optimization",
                "accept": "validation",
            },
        )

        workflow.add_edge("validation", "finalize")
        workflow.add_edge("finalize", END)

        logger.info("LangGraph workflow built with critic reflection loop")
        return workflow.compile()

    # ------------------------------------------------------------------
    # Routing decision
    # ------------------------------------------------------------------
    def _should_revise(self, state: AgentState) -> str:
        """Decide whether to loop back to the Pairing Optimizer."""
        critique = state.get("critique_result", {})
        verdict = critique.get("verdict", "ACCEPT").upper()
        revision_count = state.get("revision_count", 0)

        if verdict == "REVISE" and revision_count < self.MAX_REVISION_ROUNDS:
            logger.info(
                f"Critic verdict: REVISE (round {revision_count + 1}/"
                f"{self.MAX_REVISION_ROUNDS})"
            )
            return "revise"

        if verdict == "REVISE":
            logger.info(
                f"Critic verdict: REVISE but max rounds reached "
                f"({self.MAX_REVISION_ROUNDS}). Proceeding to validation."
            )
        else:
            logger.info("Critic verdict: ACCEPT — proceeding to validation.")

        return "accept"

    # ------------------------------------------------------------------
    # Node wrappers
    # ------------------------------------------------------------------
    def _pfd_analysis_node(self, state: AgentState) -> AgentState:
        logger.info("=" * 60)
        logger.info("EXECUTING: PFD Analysis Node")
        logger.info("=" * 60)
        return self.pfd_analyzer.invoke(state)

    def _rga_calculation_node(self, state: AgentState) -> AgentState:
        logger.info("=" * 60)
        logger.info("EXECUTING: RGA Calculation Node")
        logger.info("=" * 60)
        return self.rga_agent.invoke(state)

    def _hankel_interaction_node(self, state: AgentState) -> AgentState:
        logger.info("=" * 60)
        logger.info("EXECUTING: Hankel Interaction Node")
        logger.info("=" * 60)
        return self.hankel_agent.invoke(state)

    def _controllability_analysis_node(self, state: AgentState) -> AgentState:
        logger.info("=" * 60)
        logger.info("EXECUTING: Controllability Analysis Node")
        logger.info("=" * 60)
        return self.controllability_agent.invoke(state)

    def _pairing_optimization_node(self, state: AgentState) -> AgentState:
        """
        Pairing Optimizer node — enhanced to incorporate critic feedback
        when running in a revision round.
        """
        logger.info("=" * 60)
        revision_count = state.get("revision_count", 0)
        critique_result = state.get("critique_result")

        if critique_result and critique_result.get("verdict") == "REVISE":
            # Increment revision counter
            state["revision_count"] = revision_count + 1
            logger.info(
                f"EXECUTING: Pairing Optimization Node "
                f"(REVISION round {state['revision_count']})"
            )

            # Inject critic feedback into state so the pairing agent can
            # read it.  We append the revision suggestions to the
            # control_objectives (or a dedicated field) so that the
            # pairing agent's prompt picks them up naturally.
            self._inject_critic_feedback(state)
        else:
            logger.info("EXECUTING: Pairing Optimization Node (initial)")

        logger.info("=" * 60)
        return self.pairing_agent.invoke(state)

    def _critique_node(self, state: AgentState) -> AgentState:
        logger.info("=" * 60)
        logger.info("EXECUTING: Critic Node")
        logger.info("=" * 60)
        return self.critic_agent.invoke(state)

    def _validation_node(self, state: AgentState) -> AgentState:
        """Validation node — adjusts confidence based on critic feedback."""
        logger.info("=" * 60)
        logger.info("EXECUTING: Validation Node")
        logger.info("=" * 60)
        state = self.validation_agent.invoke(state)

        # Apply confidence adjustment from critic
        critique = state.get("critique_result", {})
        adjustment = critique.get("confidence_adjustment", 0.0)
        if adjustment and state.get("validation_results"):
            original = state["validation_results"].get("confidence_score", 0.8)
            adjusted = max(0.0, min(1.0, original + adjustment))
            state["validation_results"]["confidence_score"] = adjusted
            logger.info(
                f"Confidence adjusted by critic: "
                f"{original:.2f} → {adjusted:.2f} ({adjustment:+.2f})"
            )

        return state

    def _finalize_node(self, state: AgentState) -> AgentState:
        """Finalize and package results including critique data."""
        logger.info("=" * 60)
        logger.info("FINALIZING RESULTS")
        logger.info("=" * 60)

        try:
            critique = state.get("critique_result", {})

            control_structure = {
                "pairings": state.get("optimal_pairings", []),
                "rga_matrix": self._convert_to_list(state.get("rga_matrix")),
                "hii_matrix": self._convert_to_list(state.get("hii_matrix")),
                "hankel_singular_values": state.get("hankel_singular_values", []),
                "singular_values": state.get("singular_values", []),
                "condition_number": float(state.get("condition_number", 0.0)),
                "interaction_index": float(state.get("interaction_index", 0.0)),
                "validation_results": state.get("validation_results", {}),
                "recommendations": state.get("final_recommendations", []),
                "warnings": state.get("warnings", []),
                "confidence_score": state.get("validation_results", {}).get(
                    "confidence_score", 0.0
                ),
                # Detailed analyses
                "pfd_analysis": state.get("pfd_analysis", ""),
                "rga_analysis": state.get("rga_analysis", ""),
                "hankel_analysis": state.get("hankel_analysis", ""),
                "controllability_analysis": state.get("controllability_analysis", ""),
                "pairing_reasoning": state.get("pairing_reasoning", ""),
                # Critic results
                "critique_result": {
                    "verdict": critique.get("verdict", "N/A"),
                    "per_pairing_issues": critique.get("per_pairing_issues", []),
                    "worst_case_pairing": critique.get("worst_case_pairing"),
                    "revision_suggestions": critique.get("revision_suggestions", []),
                    "revision_rounds_used": state.get("revision_count", 0),
                    "critique_text": critique.get("critique_text", ""),
                },
                # Pairing sources
                "rga_pairings": state.get("rga_pairings", []),
                "hankel_pairings": state.get("hankel_pairings", []),
                # Meta
                "messages": state.get("messages", []),
                "errors": state.get("errors", []),
            }

            state["control_structure"] = control_structure

            if "messages" not in state:
                state["messages"] = []
            state["messages"].append(
                {
                    "agent": "Workflow",
                    "content": (
                        f"Control structure prediction complete! "
                        f"{len(control_structure['pairings'])} pairings. "
                        f"Critic verdict: {critique.get('verdict', 'N/A')}. "
                        f"Revision rounds: {state.get('revision_count', 0)}."
                    ),
                }
            )

            logger.info(
                f"Workflow completed: {len(control_structure['pairings'])} pairings, "
                f"critic={critique.get('verdict', 'N/A')}, "
                f"revisions={state.get('revision_count', 0)}"
            )

        except Exception as e:
            logger.error(f"Error in finalize node: {e}", exc_info=True)
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(f"Finalize: {str(e)}")

        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _inject_critic_feedback(self, state: Dict) -> None:
        """
        Inject the critic's revision suggestions into the state so that
        the Pairing Optimizer sees them on the next pass.

        We store them in a dedicated 'critic_feedback' field.  The
        PairingAgent's prompt builder should check for this field and
        incorporate the feedback.
        """
        critique = state.get("critique_result", {})
        suggestions = critique.get("revision_suggestions", [])
        issues = critique.get("per_pairing_issues", [])

        feedback_text = "## CRITIC FEEDBACK (address these issues):\n\n"

        if issues:
            feedback_text += "### Specific Pairing Issues:\n"
            for issue in issues:
                feedback_text += (
                    f"- **{issue.get('cv', '?')} ← {issue.get('mv', '?')}** "
                    f"[{issue.get('severity', '?')}]: {issue.get('issue', '')}\n"
                    f"  Suggestion: {issue.get('suggestion', 'N/A')}\n"
                )

        if suggestions:
            feedback_text += "\n### General Revision Suggestions:\n"
            for s in suggestions:
                feedback_text += f"- {s}\n"

        state["critic_feedback"] = feedback_text
        logger.info(
            f"Injected critic feedback: {len(issues)} issues, "
            f"{len(suggestions)} suggestions"
        )

    def _convert_to_list(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, pfd_data: dict, gain_matrix: np.ndarray) -> dict:
        try:
            initial_state = {
                "pfd_data": pfd_data,
                "gain_matrix": gain_matrix,
                "messages": [],
                "errors": [],
                "revision_count": 0,
                "critic_feedback": None,
                "critique_result": None,
                # All other fields initialised to None
                "pfd_analysis": None,
                "process_characteristics": None,
                "control_objectives": None,
                "rga_matrix": None,
                "rga_analysis": None,
                "rga_pairings": None,
                "hii_matrix": None,
                "hankel_singular_values": None,
                "hankel_analysis": None,
                "hankel_pairings": None,
                "singular_values": None,
                "condition_number": None,
                "controllability_metrics": None,
                "controllability_analysis": None,
                "interaction_index": None,
                "interaction_matrix": None,
                "interaction_analysis": None,
                "optimal_pairings": None,
                "pairing_reasoning": None,
                "chemical_eng_validation": None,
                "validation_results": None,
                "final_recommendations": None,
                "warnings": None,
                "control_structure": None,
            }

            logger.info("=" * 80)
            logger.info(
                f"STARTING CONTROL LOOP PREDICTION: {pfd_data['name']}"
            )
            logger.info(
                "Pipeline: PFD → RGA → Hankel → Controllability → "
                "Pairing ⇄ Critic → Validation"
            )
            logger.info("=" * 80)

            final_state = self.graph.invoke(initial_state)

            logger.info("=" * 80)
            logger.info("WORKFLOW EXECUTION COMPLETE")
            logger.info("=" * 80)

            control_structure = final_state.get("control_structure", {})

            if control_structure:
                cr = control_structure.get("critique_result", {})
                logger.info("Results Summary:")
                logger.info(
                    f"  - Pairings: "
                    f"{len(control_structure.get('pairings', []))}"
                )
                logger.info(
                    f"  - Confidence: "
                    f"{control_structure.get('confidence_score', 0):.1%}"
                )
                logger.info(
                    f"  - Critic Verdict: {cr.get('verdict', 'N/A')}"
                )
                logger.info(
                    f"  - Revision Rounds: "
                    f"{cr.get('revision_rounds_used', 0)}"
                )
                logger.info(
                    f"  - Warnings: "
                    f"{len(control_structure.get('warnings', []))}"
                )

            return control_structure

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "pairings": [],
                "warnings": [f"Workflow failed: {str(e)}"],
                "messages": [],
                "errors": [str(e)],
            }

    async def run_async(self, pfd_data: dict, gain_matrix: np.ndarray) -> dict:
        try:
            initial_state = {
                "pfd_data": pfd_data,
                "gain_matrix": gain_matrix,
                "messages": [],
                "errors": [],
                "revision_count": 0,
                "critic_feedback": None,
                "critique_result": None,
                "pfd_analysis": None,
                "process_characteristics": None,
                "control_objectives": None,
                "rga_matrix": None,
                "rga_analysis": None,
                "rga_pairings": None,
                "hii_matrix": None,
                "hankel_singular_values": None,
                "hankel_analysis": None,
                "hankel_pairings": None,
                "singular_values": None,
                "condition_number": None,
                "controllability_metrics": None,
                "controllability_analysis": None,
                "interaction_index": None,
                "interaction_matrix": None,
                "interaction_analysis": None,
                "optimal_pairings": None,
                "pairing_reasoning": None,
                "chemical_eng_validation": None,
                "validation_results": None,
                "final_recommendations": None,
                "warnings": None,
                "control_structure": None,
            }

            logger.info(f"Starting async workflow for {pfd_data['name']}")
            final_state = await self.graph.ainvoke(initial_state)
            return final_state.get("control_structure", {})

        except Exception as e:
            logger.error(f"Async workflow failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "pairings": [],
                "warnings": [f"Workflow failed: {str(e)}"],
                "messages": [],
                "errors": [str(e)],
            }