import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging
from src.models.pfd_models import PFDData, ProcessVariable

logger = logging.getLogger(__name__)

class DataLoader:
    """Utilities for loading and validating PFD data"""
    
    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """
        Load JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON data
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded JSON file: {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            raise
    
    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """
        Load YAML file
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Parsed YAML data
        """
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            logger.info(f"Loaded YAML file: {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading YAML file {file_path}: {e}")
            raise
    
    @staticmethod
    def validate_pfd_data(data: Dict[str, Any]) -> bool:
        """
        Validate PFD data structure
        
        Args:
            data: PFD data dictionary
            
        Returns:
            True if valid, raises exception otherwise
        """
        required_fields = [
            'name',
            'description',
            'unit_operations',
            'controlled_variables',
            'manipulated_variables',
            'gain_matrix'
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate gain matrix dimensions
        n_cvs = len(data['controlled_variables'])
        n_mvs = len(data['manipulated_variables'])
        gain_matrix = np.array(data['gain_matrix'])
        
        if gain_matrix.shape != (n_cvs, n_mvs):
            raise ValueError(
                f"Gain matrix shape {gain_matrix.shape} doesn't match "
                f"CVs ({n_cvs}) × MVs ({n_mvs})"
            )
        
        # Validate variable structures
        for cv in data['controlled_variables']:
            DataLoader._validate_variable(cv, 'controlled')
        
        for mv in data['manipulated_variables']:
            DataLoader._validate_variable(mv, 'manipulated')
        
        logger.info("PFD data validation successful")
        return True
    
    @staticmethod
    def _validate_variable(var: Dict, var_type: str) -> bool:
        """Validate individual variable structure"""
        required = ['name', 'type', 'unit', 'range', 'nominal_value', 'unit_operation']
        
        for field in required:
            if field not in var:
                raise ValueError(f"Missing field '{field}' in {var_type} variable")
        
        # Validate range
        if len(var['range']) != 2:
            raise ValueError(f"Range must have 2 elements for {var['name']}")
        
        if var['range'][0] >= var['range'][1]:
            raise ValueError(f"Invalid range for {var['name']}: min >= max")
        
        # Validate nominal value is within range
        if not (var['range'][0] <= var['nominal_value'] <= var['range'][1]):
            logger.warning(
                f"Nominal value {var['nominal_value']} outside range "
                f"{var['range']} for {var['name']}"
            )
        
        return True
    
    @staticmethod
    def convert_to_pydantic(data: Dict[str, Any]) -> PFDData:
        """
        Convert dictionary to Pydantic model
        
        Args:
            data: PFD data dictionary
            
        Returns:
            PFDData pydantic model
        """
        try:
            # Convert controlled variables
            cvs = [
                ProcessVariable(
                    id=cv.get('id', f"cv_{i}"),
                    name=cv['name'],
                    type=cv['type'],
                    unit=cv['unit'],
                    range=tuple(cv['range']),
                    nominal_value=cv['nominal_value'],
                    is_controlled=True,
                    unit_operation=cv['unit_operation']
                )
                for i, cv in enumerate(data['controlled_variables'])
            ]
            
            # Convert manipulated variables
            mvs = [
                ProcessVariable(
                    id=mv.get('id', f"mv_{i}"),
                    name=mv['name'],
                    type=mv['type'],
                    unit=mv['unit'],
                    range=tuple(mv['range']),
                    nominal_value=mv['nominal_value'],
                    is_manipulated=True,
                    unit_operation=mv['unit_operation']
                )
                for i, mv in enumerate(data['manipulated_variables'])
            ]
            
            # Convert disturbance variables if present
            dvs = []
            if 'disturbance_variables' in data:
                dvs = [
                    ProcessVariable(
                        id=dv.get('id', f"dv_{i}"),
                        name=dv['name'],
                        type=dv['type'],
                        unit=dv['unit'],
                        range=dv.get('range', (0.0, 100.0)),
                        nominal_value=dv.get('nominal_value', 50.0),
                        is_disturbance=True,
                        unit_operation=dv.get('unit_operation', 'external')
                    )
                    for i, dv in enumerate(data['disturbance_variables'])
                ]
            
            # Create PFDData model
            pfd_data = PFDData(
                name=data['name'],
                description=data['description'],
                unit_operations=data['unit_operations'],
                controlled_variables=cvs,
                manipulated_variables=mvs,
                disturbance_variables=dvs,
                gain_matrix=data['gain_matrix'],
                time_constants=data.get('time_constants')
            )
            
            return pfd_data
            
        except Exception as e:
            logger.error(f"Error converting to Pydantic model: {e}")
            raise
    
    @staticmethod
    def save_results(results: Dict[str, Any], output_path: str) -> None:
        """
        Save analysis results to file
        
        Args:
            results: Results dictionary
            output_path: Output file path
        """
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = DataLoader._make_serializable(results)
            
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: DataLoader._make_serializable(value) 
                   for key, value in obj.items()}
        elif isinstance(obj, list):
            return [DataLoader._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    @staticmethod
    def load_sample_data(sample_name: str) -> Dict[str, Any]:
        """
        Load predefined sample data
        
        Args:
            sample_name: Name of sample ('distillation', 'cstr', 'heat_exchanger')
            
        Returns:
            Sample PFD data
        """
        samples_dir = Path(__file__).parent.parent.parent / 'data'
        sample_files = {
            'distillation': 'sample_distillation.json',
            'cstr': 'sample_cstr.json',
            'heat_exchanger': 'sample_heat_exchanger.json'
        }
        
        if sample_name not in sample_files:
            raise ValueError(f"Unknown sample: {sample_name}")
        
        file_path = samples_dir / sample_files[sample_name]
        return DataLoader.load_json(str(file_path))