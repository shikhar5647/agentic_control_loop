from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np

class VariableType(str, Enum):
    """Types of process variables"""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW = "flow"
    LEVEL = "level"
    COMPOSITION = "composition"
    pH = "pH"
    DENSITY = "density"
    
class UnitOperation(str, Enum):
    """Types of unit operations"""
    REACTOR = "reactor"
    DISTILLATION = "distillation_column"
    HEAT_EXCHANGER = "heat_exchanger"
    SEPARATOR = "separator"
    PUMP = "pump"
    COMPRESSOR = "compressor"
    VALVE = "valve"
    MIXER = "mixer"
    SPLITTER = "splitter"

class ControllerType(str, Enum):
    """Types of controllers"""
    PID = "PID"
    PI = "PI"
    PD = "PD"
    CASCADE = "cascade"
    RATIO = "ratio"
    FEEDFORWARD = "feedforward"

class ProcessVariable(BaseModel):
    """Represents a process variable"""
    id: str
    name: str
    type: VariableType
    unit: str
    range: Tuple[float, float]
    nominal_value: float
    is_controlled: bool = False
    is_manipulated: bool = False
    is_disturbance: bool = False
    unit_operation: str
    steady_state_gain: Optional[float] = None

class ControlPairing(BaseModel):
    """Represents a control loop pairing"""
    controlled_variable: str
    manipulated_variable: str
    controller_type: ControllerType
    rga_value: float
    interaction_index: float
    controllability_score: float
    confidence: float
    reasoning: str
    chemical_eng_justification: str

class PFDData(BaseModel):
    """Complete PFD data structure"""
    name: str
    description: str
    unit_operations: List[Dict[str, str]]
    controlled_variables: List[ProcessVariable]
    manipulated_variables: List[ProcessVariable]
    disturbance_variables: Optional[List[ProcessVariable]] = []
    gain_matrix: List[List[float]]  # G matrix
    time_constants: Optional[List[List[float]]] = None
    
    class Config:
        arbitrary_types_allowed = True

class ControlStructure(BaseModel):
    """Final control structure output"""
    pairings: List[ControlPairing]
    rga_matrix: List[List[float]]
    singular_values: List[float]
    condition_number: float
    interaction_index: float
    overall_confidence: float
    recommendations: List[str]
    warnings: List[str]