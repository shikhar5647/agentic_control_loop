from typing import TypedDict, List, Dict, Optional, Annotated
from typing_extensions import TypedDict
import numpy as np


class AgentState(TypedDict):
    """State shared across all agents in the LangGraph workflow"""

    # Input data
    pfd_data: Dict
    gain_matrix: np.ndarray

    # PFD Analysis results
    pfd_analysis: Optional[str]
    process_characteristics: Optional[Dict]
    control_objectives: Optional[List[str]]

    # RGA Analysis results
    rga_matrix: Optional[np.ndarray]
    rga_analysis: Optional[str]
    rga_pairings: Optional[List[Dict]]

    # Controllability Analysis results
    singular_values: Optional[List[float]]
    condition_number: Optional[float]
    controllability_metrics: Optional[Dict]
    controllability_analysis: Optional[str]

    # Hankel Interaction Analysis results
    hii_matrix: Optional[np.ndarray]
    hankel_singular_values: Optional[List[float]]
    hankel_analysis: Optional[str]
    hankel_pairings: Optional[List[Dict]]

    # Interaction Analysis results
    interaction_index: Optional[float]
    interaction_matrix: Optional[np.ndarray]
    interaction_analysis: Optional[str]

    # Pairing Optimization results
    optimal_pairings: Optional[List[Dict]]
    pairing_reasoning: Optional[str]
    chemical_eng_validation: Optional[str]

    # ---- Critic / Reflection fields (NEW) ----
    critique_result: Optional[Dict]       # structured output from CriticAgent
    critic_feedback: Optional[str]        # text feedback injected for revision
    revision_count: int                   # number of critic→optimizer loops done

    # Validation results
    validation_results: Optional[Dict]
    final_recommendations: Optional[List[str]]
    warnings: Optional[List[str]]

    # Final output
    control_structure: Optional[Dict]

    # Agent messages and conversation history
    messages: Annotated[List, "messages"]

    # Error tracking
    errors: Optional[List[str]]