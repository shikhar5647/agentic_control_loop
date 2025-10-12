from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
from enum import Enum

class VariableType(str, Enum):
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW = "flow"
    LEVEL = "level"
    COMPOSITION = "composition"
    pH = "pH"
    
class UnitOperation(str, Enum):
    REACTOR = "reactor"
    DISTILLATION = "distillation_column"
    HEAT_EXCHANGER = "heat_exchanger"
    SEPARATOR = "separator"
    PUMP = "pump"
    COMPRESSOR = "compressor"
    VALVE = "valve"
    MIXER = "mixer"

class ProcessVariable(BaseModel):
    id: str
    name: str
    type: VariableType
    unit: str
    range: Tuple[float, float]
    is_controlled: bool = False
    is_manipulated: bool = False
    unit_operation: str

class ControlPairing(BaseModel):
    controlled_variable: str
    manipulated_variable: str
    rga_value: float
    interaction_index: float
    confidence: float
    reasoning: str

class PFDData(BaseModel):
    name: str
    description: str
    unit_operations: List[str]
    controlled_variables: List[ProcessVariable]
    manipulated_variables: List[ProcessVariable]
    disturbance_variables: Optional[List[ProcessVariable]] = []
    gain_matrix: Optional[List[List[float]]] = None