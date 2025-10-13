from langgraph.graph import StateGraph, END
from src.graph.state import AgentState
from src.agents.pfd_analyzer_agent import PFDAnalyzerAgent
from src.agents.rga_agent import RGAAgent
from src.agents.controllability_agent import ControllabilityAgent
from src.agents.pairing_agent import PairingAgent
from src.agents.validation_agent import ValidationAgent
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ControlLoopWorkflow:
    """LangGraph workflow for control loop prediction"""
    
    def __init__(self, config: dict = None):
        """
        Initialize workflow
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize agents with configuration
        agent_config = self.config.get('agents', {})
        
        self.pfd_analyzer = PFDAnalyzerAgent(
            temperature=agent_config.get('pfd_analyzer', {}).get('temperature', 0.2)
        )
        self.rga_agent = RGAAgent(
            temperature=agent_config.get('rga_calculator', {}).get('temperature', 0.1)
        )
        self.controllability_agent = ControllabilityAgent(
            temperature=agent_config.get('controllability_analyzer', {}).get('temperature', 0.2)
        )
        self.pairing_agent = PairingAgent(
            temperature=agent_config.get('pairing_optimizer', {}).get('temperature', 0.3)
        )
        self.validation_agent = ValidationAgent(
            temperature=agent_config.get('validation_agent', {}).get('temperature', 0.2)
        )
        
        # Build graph
        self.graph = self._build_graph()
        
        logger.info("Control Loop Workflow initialized with 5 agents")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create graph with AgentState
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("pfd_analysis", self._pfd_analysis_node)
        workflow.add_node("rga_calculation", self._rga_calculation_node)
        workflow.add_node("controllability_analysis", self._controllability_analysis_node)
        workflow.add_node("pairing_optimization", self._pairing_optimization_node)
        workflow.add_node("validation", self._validation_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Set entry point
        workflow.set_entry_point("pfd_analysis")
        
        # Add edges (sequential flow)
        workflow.add_edge("pfd_analysis", "rga_calculation")
        workflow.add_edge("rga_calculation", "controllability_analysis")
        workflow.add_edge("controllability_analysis", "pairing_optimization")
        workflow.add_edge("pairing_optimization", "validation")
        workflow.add_edge("validation", "finalize")
        workflow.add_edge("finalize", END)
        
        logger.info("LangGraph workflow built successfully")
        return workflow.compile()
    
    def _pfd_analysis_node(self, state: AgentState) -> AgentState:
        """PFD Analysis Node"""
        logger.info("=" * 60)
        logger.info("EXECUTING: PFD Analysis Node")
        logger.info("=" * 60)
        result = self.pfd_analyzer.invoke(state)
        logger.info("PFD Analysis Node completed")
        return result
    
    def _rga_calculation_node(self, state: AgentState) -> AgentState:
        """RGA Calculation Node"""
        logger.info("=" * 60)
        logger.info("EXECUTING: RGA Calculation Node")
        logger.info("=" * 60)
        result = self.rga_agent.invoke(state)
        logger.info("RGA Calculation Node completed")
        return result
    
    def _controllability_analysis_node(self, state: AgentState) -> AgentState:
        """Controllability Analysis Node"""
        logger.info("=" * 60)
        logger.info("EXECUTING: Controllability Analysis Node")
        logger.info("=" * 60)
        result = self.controllability_agent.invoke(state)
        logger.info("Controllability Analysis Node completed")
        return result
    
    def _pairing_optimization_node(self, state: AgentState) -> AgentState:
        """Pairing Optimization Node"""
        logger.info("=" * 60)
        logger.info("EXECUTING: Pairing Optimization Node")
        logger.info("=" * 60)
        result = self.pairing_agent.invoke(state)
        logger.info("Pairing Optimization Node completed")
        return result
    
    def _validation_node(self, state: AgentState) -> AgentState:
        """Validation Node"""
        logger.info("=" * 60)
        logger.info("EXECUTING: Validation Node")
        logger.info("=" * 60)
        result = self.validation_agent.invoke(state)
        logger.info("Validation Node completed")
        return result
    
    def _finalize_node(self, state: AgentState) -> AgentState:
        """Finalize and package results"""
        logger.info("=" * 60)
        logger.info("FINALIZING RESULTS")
        logger.info("=" * 60)
        
        try:
            # Package final control structure
            control_structure = {
                'pairings': state.get('optimal_pairings', []),
                'rga_matrix': self._convert_to_list(state.get('rga_matrix')),
                'singular_values': state.get('singular_values', []),
                'condition_number': float(state.get('condition_number', 0.0)),
                'interaction_index': float(state.get('interaction_index', 0.0)),
                'validation_results': state.get('validation_results', {}),
                'recommendations': state.get('final_recommendations', []),
                'warnings': state.get('warnings', []),
                'confidence_score': state.get('validation_results', {}).get('confidence_score', 0.0),
                
                # Include detailed analyses
                'pfd_analysis': state.get('pfd_analysis', ''),
                'rga_analysis': state.get('rga_analysis', ''),
                'controllability_analysis': state.get('controllability_analysis', ''),
                'pairing_reasoning': state.get('pairing_reasoning', ''),
                
                # Include messages
                'messages': state.get('messages', []),
                'errors': state.get('errors', [])
            }
            
            state['control_structure'] = control_structure
            
            # Add final message
            if 'messages' not in state:
                state['messages'] = []
            state['messages'].append({
                'agent': 'Workflow',
                'content': f'Control structure prediction complete! Generated {len(control_structure["pairings"])} control loop pairings.'
            })
            
            logger.info(f"Workflow completed successfully with {len(control_structure['pairings'])} pairings")
            
        except Exception as e:
            logger.error(f"Error in finalize node: {e}", exc_info=True)
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(f"Finalize: {str(e)}")
        
        return state
    
    def _convert_to_list(self, obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def run(self, pfd_data: dict, gain_matrix: np.ndarray) -> dict:
        """
        Run the complete workflow
        
        Args:
            pfd_data: PFD data dictionary
            gain_matrix: Gain matrix (numpy array)
            
        Returns:
            Final control structure
        """
        try:
            # Initialize state
            initial_state = {
                'pfd_data': pfd_data,
                'gain_matrix': gain_matrix,
                'messages': [],
                'errors': [],
                
                # Initialize all other fields to None
                'pfd_analysis': None,
                'process_characteristics': None,
                'control_objectives': None,
                'rga_matrix': None,
                'rga_analysis': None,
                'rga_pairings': None,
                'singular_values': None,
                'condition_number': None,
                'controllability_metrics': None,
                'controllability_analysis': None,
                'interaction_index': None,
                'interaction_matrix': None,
                'interaction_analysis': None,
                'optimal_pairings': None,
                'pairing_reasoning': None,
                'chemical_eng_validation': None,
                'validation_results': None,
                'final_recommendations': None,
                'warnings': None,
                'control_structure': None
            }
            
            # Run workflow
            logger.info("=" * 80)
            logger.info(f"STARTING CONTROL LOOP PREDICTION WORKFLOW: {pfd_data['name']}")
            logger.info("=" * 80)
            
            final_state = self.graph.invoke(initial_state)
            
            logger.info("=" * 80)
            logger.info("WORKFLOW EXECUTION COMPLETE")
            logger.info("=" * 80)
            
            # Return control structure
            control_structure = final_state.get('control_structure', {})
            
            # Log summary
            if control_structure:
                logger.info(f"Results Summary:")
                logger.info(f"  - Pairings: {len(control_structure.get('pairings', []))}")
                logger.info(f"  - Confidence: {control_structure.get('confidence_score', 0):.1%}")
                logger.info(f"  - Warnings: {len(control_structure.get('warnings', []))}")
                logger.info(f"  - Errors: {len(control_structure.get('errors', []))}")
            
            return control_structure
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            return {
                'error': str(e),
                'pairings': [],
                'warnings': [f"Workflow failed: {str(e)}"],
                'messages': [],
                'errors': [str(e)]
            }
    
    async def run_async(self, pfd_data: dict, gain_matrix: np.ndarray) -> dict:
        """
        Run workflow asynchronously
        
        Args:
            pfd_data: PFD data dictionary
            gain_matrix: Gain matrix
            
        Returns:
            Final control structure
        """
        try:
            initial_state = {
                'pfd_data': pfd_data,
                'gain_matrix': gain_matrix,
                'messages': [],
                'errors': [],
                'pfd_analysis': None,
                'process_characteristics': None,
                'control_objectives': None,
                'rga_matrix': None,
                'rga_analysis': None,
                'rga_pairings': None,
                'singular_values': None,
                'condition_number': None,
                'controllability_metrics': None,
                'controllability_analysis': None,
                'interaction_index': None,
                'interaction_matrix': None,
                'interaction_analysis': None,
                'optimal_pairings': None,
                'pairing_reasoning': None,
                'chemical_eng_validation': None,
                'validation_results': None,
                'final_recommendations': None,
                'warnings': None,
                'control_structure': None
            }
            
            logger.info(f"Starting async workflow for {pfd_data['name']}")
            final_state = await self.graph.ainvoke(initial_state)
            
            return final_state.get('control_structure', {})
            
        except Exception as e:
            logger.error(f"Async workflow execution failed: {e}", exc_info=True)
            return {
                'error': str(e),
                'pairings': [],
                'warnings': [f"Workflow failed: {str(e)}"],
                'messages': [],
                'errors': [str(e)]
            }