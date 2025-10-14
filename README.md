# PFD Control Loop Prediction System - Technical Documentation

## 📋 Project Overview

This system uses a **multi-agent AI architecture** built with **LangGraph** and **Google Gemini** to automatically predict optimal control loop pairings for Process Flow Diagrams (PFDs). It combines classical control theory (RGA, SVD) with chemical engineering heuristics and AI reasoning to recommend control structures for industrial chemical processes.

### Key Technologies
- **LangGraph**: Multi-agent workflow orchestration
- **Google Gemini**: Large Language Model for AI reasoning
- **Streamlit**: Web-based user interface
- **NumPy/SciPy**: Numerical computations
- **Pydantic**: Data validation

---

## 🎯 System Architecture

### Workflow Overview

```
Input PFD Data → PFD Analyzer → RGA Calculator → Controllability Analyzer 
→ Pairing Optimizer → Validation → Control Structure Output
```

### Agent Pipeline

1. **PFD Analyzer Agent**: Understands process structure
2. **RGA Agent**: Calculates Relative Gain Array
3. **Controllability Agent**: Performs SVD analysis
4. **Pairing Optimizer Agent**: Synthesizes optimal pairings
5. **Validation Agent**: Validates final control structure

---

## 📁 Detailed File Documentation

### 1. Core Application

#### `app.py` - Streamlit Web Application

**Purpose**: Main user interface for the control loop prediction system.

**Key Features**:
- Interactive web interface with 5 tabs (Input, Analysis, Results, Agent Activity, Help)
- File upload and sample data loading
- Real-time progress tracking during analysis
- Visualization of results (heatmaps, charts, metrics)
- Export functionality (JSON, CSV, Markdown)

**Inputs**:
- PFD data (JSON file or sample selection)
- User configuration (model selection, temperature settings)

**Outputs**:
- Interactive dashboards with control loop recommendations
- Downloadable reports and data files

**Key Components**:
```python
# Tab 1: Input Data - Upload/load PFD data
# Tab 2: Run Analysis - Execute workflow
# Tab 3: Results - View control structure
# Tab 4: Agent Activity - View agent logs
# Tab 5: Help - Documentation
```

**Usage**:
```bash
streamlit run app.py
```

---

### 2. Agent System (`src/agents/`)

#### `base_agent.py` - Base Agent Class

**Purpose**: Abstract base class providing common functionality for all agents.

**Key Features**:
- Gemini LLM initialization and configuration
- Common prompt handling methods
- Matrix formatting utilities
- Error handling framework

**Inputs**:
- `agent_name`: Name identifier for the agent
- `temperature`: LLM temperature parameter (0.0-1.0)

**Key Methods**:
```python
def call_llm(prompt: str, system_prompt: str) -> str:
    """Call Gemini LLM with prompts"""
    
def format_matrix(matrix, var_names_row, var_names_col) -> str:
    """Format matrices for LLM consumption"""
```

---

#### `pfd_analyzer_agent.py` - PFD Analyzer Agent

**Purpose**: Analyzes PFD structure and identifies control requirements using chemical engineering principles.

**Inputs**:
- `pfd_data`: Dictionary containing process information
  - Unit operations (reactors, distillation columns, etc.)
  - Controlled variables (temperature, pressure, flow, level, composition)
  - Manipulated variables (valve positions, flow rates, duties)
  - Gain matrix (steady-state process gains)

**Processing**:
1. Analyzes each unit operation and its control requirements
2. Identifies safety-critical variables
3. Determines control objectives (safety, quality, efficiency)
4. Assesses degrees of freedom
5. Evaluates process interactions and dynamics

**Outputs**:
- `pfd_analysis`: Detailed text analysis of process structure
- `control_objectives`: List of prioritized control objectives
- `process_characteristics`: Dictionary with:
  ```python
  {
      'num_controlled_variables': int,
      'num_manipulated_variables': int,
      'degrees_of_freedom': int,
      'process_type': str,  # 'reaction_system', 'separation_system', etc.
      'cv_type_distribution': dict,
      'mv_type_distribution': dict
  }
  ```

**Chemical Engineering Principles Applied**:
- Mass and energy balances
- Thermodynamic constraints
- Reaction kinetics and safety
- Unit operation specific strategies

**Example Output**:
```
Control Objectives:
1. Maintain reactor temperature at 95°C for reaction rate control
2. Control product composition at 90 mol% for quality
3. Maintain reactor level for inventory control
4. Control pressure for safety
```

---

#### `rga_agent.py` - RGA Calculator Agent

**Purpose**: Computes Relative Gain Array (RGA) and recommends CV-MV pairings based on interaction analysis.

**Inputs**:
- `gain_matrix`: Steady-state gain matrix G (n×m numpy array)
- `pfd_data`: Process variable names and information

**Processing**:
1. Calculates RGA matrix: `RGA = G ⊙ (G⁻¹)ᵀ`
2. Analyzes RGA elements for pairing recommendations
3. Applies Bristol's rules:
   - λᵢⱼ ≈ 1.0: Excellent pairing
   - λᵢⱼ < 0: Avoid (causes instability)
   - λᵢⱼ ≈ 0: Poor pairing (weak interaction)
4. Identifies loop interactions
5. Validates row/column sums (should equal 1.0)

**Outputs**:
- `rga_matrix`: RGA matrix (n×n numpy array)
- `rga_analysis`: Detailed text analysis
- `rga_pairings`: List of recommended pairings
  ```python
  [
      {
          'cv': 'T_reactor',
          'mv': 'F_coolant',
          'rga_value': 0.92,
          'recommendation': 'Excellent pairing - strong positive interaction'
      },
      ...
  ]
  ```

**Mathematical Background**:
```
RGA(i,j) = (∂yᵢ/∂uⱼ)ₐₗₗ ₒₜₕₑᵣ ᵤ ₑₓ𝒸ₑₚₜ ᵤⱼ / (∂yᵢ/∂uⱼ)ₐₗₗ ₒₜₕₑᵣ ᵧ ₑₓ𝒸ₑₚₜ ᵧᵢ

Where:
- yᵢ = controlled variable i
- uⱼ = manipulated variable j
```

**Key Features**:
- Handles non-square matrices using pseudo-inverse
- Detects problematic pairings (negative RGA)
- Considers industrial best practices

---

#### `controllability_agent.py` - Controllability Analyzer Agent

**Purpose**: Performs Singular Value Decomposition (SVD) to assess system controllability and validate RGA pairings.

**Inputs**:
- `gain_matrix`: Process gain matrix G
- `rga_pairings`: Pairings from RGA analysis

**Processing**:
1. Performs SVD: `G = U Σ Vᵀ`
2. Calculates condition number: `κ = σₘₐₓ / σₘᵢₙ`
3. Analyzes singular value spectrum
4. Identifies dominant input directions (from V matrix)
5. Validates RGA pairings align with dominant directions
6. Assesses overall controllability

**Outputs**:
- `singular_values`: Array of singular values [σ₁, σ₂, ..., σₙ]
- `condition_number`: System condition number κ
- `controllability_metrics`: Dictionary with:
  ```python
  {
      'singular_values': list,
      'condition_number': float,
      'dominant_input_directions': list,
      'rank': int,
      'controllability_score': float  # 1/κ
  }
  ```
- `controllability_analysis`: Detailed text analysis

**Interpretation**:
- **Condition Number < 10**: Well-conditioned (easy to control)
- **10 < κ < 100**: Moderately conditioned
- **κ > 100**: Ill-conditioned (difficult to control)

**Key Insights**:
- Large singular values indicate strong controllability
- Small singular values indicate sensitivity to disturbances
- V matrix columns show which MV combinations affect outputs most

---

#### `pairing_agent.py` - Pairing Optimizer Agent

**Purpose**: Synthesizes optimal control loop pairings by integrating RGA, SVD, interaction analysis, and chemical engineering heuristics.

**Inputs**:
- `gain_matrix`: Process gain matrix
- `rga_matrix`: RGA matrix
- `rga_pairings`: Initial RGA recommendations
- `controllability_metrics`: SVD results
- `pfd_analysis`: Process understanding
- `control_objectives`: Control goals

**Processing**:
1. Calculates interaction index: `I = ||G - diag(G)|| / ||G||`
2. Performs maximum weight matching on gain matrix
3. Integrates multiple criteria:
   - RGA optimality (weight: 40%)
   - Controllability (weight: 30%)
   - Interaction minimization (weight: 20%)
   - Gain magnitude (weight: 10%)
4. Applies chemical engineering heuristics:
   - Unit operation specific strategies
   - Luyben's plantwide control rules
   - Skogestad's self-optimizing control
5. Recommends controller types (PI, PID, Cascade, Ratio)
6. Provides tuning guidance

**Outputs**:
- `optimal_pairings`: List of final control loop pairings
  ```python
  [
      {
          'controlled_variable': 'T_reactor',
          'manipulated_variable': 'F_coolant',
          'controller_type': 'PID',
          'rga_value': 0.92,
          'controllability_score': 0.85,
          'interaction_score': 0.78,
          'overall_confidence': 0.88,
          'reasoning': 'Strong RGA value and good controllability...',
          'chemical_eng_rationale': 'Temperature control via coolant is standard...',
          'tuning_guidance': 'Start with Kc=2.5, Ti=5 min, Td=1 min'
      },
      ...
  ]
  ```
- `interaction_index`: Loop coupling measure
- `pairing_reasoning`: Detailed optimization rationale

**Optimization Criteria**:
```python
score = 0.4 * rga_score + 
        0.3 * controllability_score + 
        0.2 * (1 - interaction_score) + 
        0.1 * gain_score
```

**Chemical Engineering Heuristics Applied**:
- Distillation: LV, DV, or LV/DV configurations
- Reactors: Temperature cascade, feed ratio control
- Heat Exchangers: Bypass or split-range control
- Separators: Level averaging control

---

#### `validation_agent.py` - Validation Agent

**Purpose**: Validates the final control structure against safety, engineering, performance, and operational criteria.

**Inputs**:
- `optimal_pairings`: Proposed control structure
- `pfd_data`: Process information
- `gain_matrix`, `rga_matrix`: Analysis matrices
- `interaction_index`, `condition_number`: System metrics

**Processing**:
1. **Safety Validation**:
   - All safety-critical variables controlled
   - Fail-safe mechanisms defined
   - Emergency scenarios addressed
   
2. **Engineering Validation**:
   - Pairings are physically realizable
   - Instrumentation is available
   - No conflicting control actions
   - Thermodynamic constraints respected
   
3. **Performance Validation**:
   - Expected closed-loop performance adequate
   - Disturbance rejection sufficient
   - Interaction index acceptable
   - Condition number reasonable
   
4. **Operational Validation**:
   - Operators can understand system
   - Startup/shutdown procedures clear
   - Maintenance requirements reasonable
   - Tuning is practical

5. **Automated Checks**:
   - Negative RGA values detection
   - High interaction warning
   - Poor conditioning alert
   - Weak pairing identification

**Outputs**:
- `validation_results`: Dictionary with:
  ```python
  {
      'safety_check': 'PASS' | 'FAIL',
      'engineering_check': 'PASS' | 'FAIL',
      'performance_check': 'PASS' | 'FAIL',
      'operational_check': 'PASS' | 'FAIL',
      'overall_status': 'APPROVED' | 'CONDITIONAL' | 'REJECTED',
      'confidence_score': 0.88,
      'critical_warnings': [...],
      'recommendations': [...],
      'summary': 'Overall assessment text'
  }
  ```
- `final_recommendations`: List of actionable recommendations
- `warnings`: List of critical warnings

**Validation Criteria**:
- Interaction Index < 0.4 (preferred)
- Condition Number < 100 (acceptable)
- RGA values > 0.5 (good pairings)
- All CVs have exactly one MV

---

### 3. Workflow Management (`src/graph/`)

#### `state.py` - LangGraph State Definition

**Purpose**: Defines the shared state structure passed between agents in the LangGraph workflow.

**State Schema**:
```python
class AgentState(TypedDict):
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
    
    # Controllability results
    singular_values: Optional[List[float]]
    condition_number: Optional[float]
    controllability_metrics: Optional[Dict]
    controllability_analysis: Optional[str]
    
    # Interaction results
    interaction_index: Optional[float]
    interaction_matrix: Optional[np.ndarray]
    interaction_analysis: Optional[str]
    
    # Pairing results
    optimal_pairings: Optional[List[Dict]]
    pairing_reasoning: Optional[str]
    
    # Validation results
    validation_results: Optional[Dict]
    final_recommendations: Optional[List[str]]
    warnings: Optional[List[str]]
    
    # Final output
    control_structure: Optional[Dict]
    
    # Metadata
    messages: List[Dict]
    errors: Optional[List[str]]
```

**Key Features**:
- Type-safe state definition
- Tracks all intermediate results
- Maintains message history
- Error tracking

---

#### `workflow.py` - LangGraph Workflow

**Purpose**: Orchestrates the multi-agent workflow using LangGraph's StateGraph.

**Workflow Structure**:
```python
PFD Analysis → RGA Calculation → Controllability Analysis 
→ Pairing Optimization → Validation → Finalize → END
```

**Key Components**:

1. **Workflow Initialization**:
```python
workflow = ControlLoopWorkflow(config)
```

2. **Node Definitions**:
- Each agent is wrapped in a node function
- Nodes receive and return `AgentState`
- Sequential execution with state passing

3. **Execution**:
```python
result = workflow.run(pfd_data, gain_matrix)
```

**Inputs**:
- `pfd_data`: Dictionary with process information
- `gain_matrix`: Numpy array of steady-state gains
- `config`: Optional configuration dictionary

**Outputs**:
- `control_structure`: Complete analysis results
  ```python
  {
      'pairings': [...],  # Control loop recommendations
      'rga_matrix': [...],
      'singular_values': [...],
      'condition_number': float,
      'interaction_index': float,
      'confidence_score': float,
      'recommendations': [...],
      'warnings': [...],
      'messages': [...],  # Agent activity log
      'errors': [...]
  }
  ```

**Features**:
- Automatic state propagation
- Error handling at each node
- Logging and progress tracking
- Synchronous and asynchronous execution
- Result finalization and packaging

---

### 4. Data Models (`src/models/`)

#### `pfd_models.py` - Pydantic Data Models

**Purpose**: Define validated data structures for type safety and data integrity.

**Key Models**:

1. **VariableType (Enum)**:
```python
TEMPERATURE, PRESSURE, FLOW, LEVEL, COMPOSITION, pH, DENSITY
```

2. **UnitOperation (Enum)**:
```python
REACTOR, DISTILLATION, HEAT_EXCHANGER, SEPARATOR, 
PUMP, COMPRESSOR, VALVE, MIXER, SPLITTER
```

3. **ControllerType (Enum)**:
```python
PID, PI, PD, CASCADE, RATIO, FEEDFORWARD
```

4. **ProcessVariable**:
```python
{
    'id': str,
    'name': str,
    'type': VariableType,
    'unit': str,
    'range': Tuple[float, float],
    'nominal_value': float,
    'is_controlled': bool,
    'is_manipulated': bool,
    'is_disturbance': bool,
    'unit_operation': str,
    'steady_state_gain': Optional[float]
}
```

5. **ControlPairing**:
```python
{
    'controlled_variable': str,
    'manipulated_variable': str,
    'controller_type': ControllerType,
    'rga_value': float,
    'interaction_index': float,
    'controllability_score': float,
    'confidence': float,
    'reasoning': str,
    'chemical_eng_justification': str
}
```

6. **PFDData**:
```python
{
    'name': str,
    'description': str,
    'unit_operations': List[Dict],
    'controlled_variables': List[ProcessVariable],
    'manipulated_variables': List[ProcessVariable],
    'disturbance_variables': List[ProcessVariable],
    'gain_matrix': List[List[float]],
    'time_constants': Optional[List[List[float]]]
}
```

7. **ControlStructure**:
```python
{
    'pairings': List[ControlPairing],
    'rga_matrix': List[List[float]],
    'singular_values': List[float],
    'condition_number': float,
    'interaction_index': float,
    'overall_confidence': float,
    'recommendations': List[str],
    'warnings': List[str]
}
```

**Features**:
- Automatic validation
- Type checking
- Default values
- Serialization support

---

### 5. Utilities (`src/utils/`)

#### `chemical_engineering.py` - Chemical Engineering Utilities

**Purpose**: Core chemical engineering calculations and analysis functions.

**Key Functions**:

1. **RGA Calculation**:
```python
def calculate_rga(gain_matrix: np.ndarray) -> np.ndarray:
    """Calculate RGA = G ⊙ (G⁻¹)ᵀ"""
```

2. **SVD Metrics**:
```python
def calculate_svd_metrics(gain_matrix: np.ndarray) -> Dict:
    """
    Returns:
    - singular_values
    - condition_number
    - dominant_directions
    - rank
    - controllability_score
    """
```

3. **Interaction Index**:
```python
def calculate_interaction_index(gain_matrix: np.ndarray) -> float:
    """I = ||G - diag(G)|| / ||G||"""
```

4. **Maximum Weight Matching**:
```python
def maximum_weight_matching(gain_matrix: np.ndarray) -> List[Tuple]:
    """Greedy bipartite matching for pairing"""
```

5. **Unit Operation Strategies**:
```python
def get_unit_operation_control_strategy(unit_type: str) -> List[str]:
    """
    Returns control strategies for:
    - Distillation columns
    - Reactors
    - Heat exchangers
    - Separators
    - Mixers
    """
```

**Chemical Engineering Principles**:
- Mass balance considerations
- Energy balance requirements
- Thermodynamic constraints
- Reaction kinetics
- Phase equilibrium

---

#### `control_heuristics.py` - Control Design Heuristics

**Purpose**: Implements classical and modern control design heuristics.

**Key Features**:

1. **Luyben's Plantwide Control Rules**:
```python
def get_luyben_plantwide_rules() -> List[str]:
    """9-step procedure for plantwide control"""
```

2. **Skogestad's Control Hierarchy**:
- Regulatory control (stabilization)
- Constraint control (maximize throughput)
- Optimization control (minimize cost)
- Economic optimization

3. **MPC Applicability**:
```python
def apply_mpc_criteria(gain_matrix, interaction_index, 
                       condition_number) -> Dict:
    """
    Recommends MPC if:
    - High interaction (I > 0.4)
    - Ill-conditioned (κ > 100)
    - Many variables (n >= 5)
    """
```

4. **Pairing Priority Rules**:
```python
def pairing_priority_rules(cv_type: str, 
                          process_type: str) -> List[str]:
    """Variable-specific pairing recommendations"""
```

5. **Controller Tuning**:
```python
def tuning_recommendations(time_constant, dead_time, gain) -> Dict:
    """
    Returns Ziegler-Nichols tuning parameters:
    - Kc (proportional gain)
    - Ti (integral time)
    - Td (derivative time)
    """
```

6. **Inventory Control Philosophy**:
```python
def inventory_control_philosophy(vessel_size, criticality) -> str:
    """
    Recommends:
    - TIGHT control (small vessels, critical)
    - AVERAGING control (large vessels, non-critical)
    - MODERATE control (in between)
    """
```

**Heuristics Sources**:
- Luyben et al. (1997) - Plantwide Control
- Skogestad (2004) - Control Structure Design
- Bristol (1966) - RGA Rules
- Ziegler-Nichols (1942) - Tuning

---

#### `matrix_operations.py` - Advanced Matrix Operations

**Purpose**: Specialized matrix calculations for control analysis.

**Key Functions**:

1. **RGA Calculation** (alternative implementation)
2. **SVD Decomposition**
3. **Condition Number**
4. **Frobenius Norm**
5. **Niederlinski Index**:
```python
def niederlinski_index(gain_matrix, rga_matrix) -> float:
    """
    NI = det(G) / prod(diag(G))
    For stability: NI > 0
    """
```

6. **Gramian-Based Controllability**:
```python
def gramian_based_controllability(A, B, time_horizon) -> np.ndarray:
    """Controllability Gramian for state-space models"""
```

7. **Bristol's RGA Rules**:
```python
def bristol_rga_rules(rga_matrix) -> List[str]:
    """Apply Bristol's rules for pairing selection"""
```

8. **Participation Matrix**:
```python
def participation_matrix(A) -> Tuple[np.ndarray, np.ndarray]:
    """Shows which states participate in which modes"""
```

9. **Skogestad's Half Rule**:
```python
def skogestad_half_rule(gain_matrix, time_constants) -> np.ndarray:
    """Approximates time delays: θ ≈ 0.5τ"""
```

10. **Morari's Resilience Index**:
```python
def morari_resilience_index(gain_matrix) -> float:
    """RI = σₘᵢₙ / σₘₐₓ"""
```

11. **Relative Disturbance Gain**:
```python
def relative_disturbance_gain(G, Gd) -> np.ndarray:
    """RDG = Gd ⊙ (G⁻¹)ᵀ"""
```

---

#### `data_loader.py` - Data Loading and Validation

**Purpose**: Handle PFD data loading, validation, and serialization.

**Key Functions**:

1. **Load Data**:
```python
def load_json(file_path: str) -> Dict:
    """Load JSON file"""

def load_yaml(file_path: str) -> Dict:
    """Load YAML file"""
```

2. **Validate PFD Data**:
```python
def validate_pfd_data(data: Dict) -> bool:
    """
    Validates:
    - Required fields present
    - Gain matrix dimensions match
    - Variable structures correct
    - Ranges valid
    - Nominal values within ranges
    """
```

3. **Convert to Pydantic**:
```python
def convert_to_pydantic(data: Dict) -> PFDData:
    """Convert dict to validated PFDData model"""
```

4. **Save Results**:
```python
def save_results(results: Dict, output_path: str) -> None:
    """Save with numpy array serialization"""
```

5. **Load Samples**:
```python
def load_sample_data(sample_name: str) -> Dict:
    """Load predefined samples: distillation, cstr, heat_exchanger"""
```

**Validation Checks**:
- Field presence
- Type correctness
- Dimensional consistency
- Value range validation
- Nominal value bounds

---

### 6. Configuration

#### `config/config.yaml` - System Configuration

**Purpose**: Central configuration for models, agents, and control parameters.

**Structure**:
```yaml
model:
  name: "gemini-1.5-pro"
  temperature: 0.3
  max_tokens: 8000

agents:
  pfd_analyzer:
    temperature: 0.2
    description: "Analyzes PFD structure"
  
  rga_calculator:
    temperature: 0.1
    description: "Calculates RGA"
  
  controllability_analyzer:
    temperature: 0.2
    description: "SVD analysis"
  
  pairing_optimizer:
    temperature: 0.3
    description: "Optimizes pairings"
  
  validation_agent:
    temperature: 0.2
    description: "Validates structure"

control_parameters:
  rga_threshold_good: 0.7
  rga_threshold_poor: 0.3
  interaction_index_threshold: 0.3
  min_singular_value_ratio: 0.01
  
  # Pairing weights
  weight_rga: 0.4
  weight_controllability: 0.3
  weight_interaction: 0.3

chemical_engineering_principles:
  distillation:
    - "Control reflux ratio for product composition"
    - "Level control on condenser and reboiler"
    - "Pressure control for stable operation"
  
  reactor:
    - "Temperature control critical for rate and safety"
    - "Pressure control for gas-phase reactions"
    - "Flow ratio control for stoichiometry"
```

---

### 7. Sample Data (`data/`)

#### Input Data Format

**JSON Structure**:
```json
{
  "name": "Process Name",
  "description": "Detailed process description",
  "unit_operations": [
    {
      "name": "R-101",
      "type": "reactor",
      "description": "Main CSTR reactor"
    }
  ],
  "controlled_variables": [
    {
      "name": "T_reactor",
      "type": "temperature",
      "unit": "°C",
      "range": [50.0, 150.0],
      "nominal_value": 100.0,
      "unit_operation": "R-101",
      "description": "Reactor temperature"
    }
  ],
  "manipulated_variables": [
    {
      "name": "F_coolant",
      "type": "flow",
      "unit": "kg/h",
      "range": [0.0, 5000.0],
      "nominal_value": 2500.0,
      "unit_operation": "R-101",
      "description": "Coolant flow rate"
    }
  ],
  "disturbance_variables": [
    {
      "name": "T_feed",
      "type": "temperature",
      "unit": "°C",
      "description": "Feed temperature variation"
    }
  ],
  "gain_matrix": [
    [0.92, -0.08],
    [0.15, 0.95]
  ],
  "time_constants": [
    [5.0, 8.0],
    [3.0, 2.0]
  ]
}
```

#### Sample Datasets

1. **sample_distillation.json**:
   - Binary distillation column (Benzene-Toluene)
   - 5 CVs × 5 MVs
   - Tests dual composition control
   - Interaction analysis showcase

2. **sample_cstr.json**:
   - Exothermic CSTR with cooling
   - 4 CVs × 4 MVs
   - Tests temperature control
   - Ratio control example

3. **sample_heat_exchanger.json**:
   - Heat exchanger network
   - 3 CVs × 3 MVs
   - Tests bypass control
   - Simple SISO loops

---
