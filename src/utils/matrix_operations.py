import numpy as np
from scipy import linalg
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class MatrixOperations:
    """Advanced matrix operations for control analysis"""
    
    @staticmethod
    def calculate_rga(gain_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate Relative Gain Array
        RGA = G ⊙ (G^-1)^T
        
        Args:
            gain_matrix: Steady-state gain matrix
            
        Returns:
            RGA matrix
        """
        try:
            # For square matrices
            if gain_matrix.shape[0] == gain_matrix.shape[1]:
                G_inv = np.linalg.inv(gain_matrix)
                rga = gain_matrix * G_inv.T
            else:
                # For non-square, use pseudo-inverse
                G_inv = np.linalg.pinv(gain_matrix)
                rga = gain_matrix @ G_inv.T
            
            return rga
        except np.linalg.LinAlgError as e:
            logger.error(f"Matrix inversion failed: {e}")
            # Return identity matrix as fallback
            n = min(gain_matrix.shape)
            return np.eye(n)
    
    @staticmethod
    def svd_decomposition(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Singular Value Decomposition
        A = U Σ V^T
        
        Args:
            matrix: Input matrix
            
        Returns:
            U, Sigma, V^T matrices
        """
        try:
            U, sigma, Vt = np.linalg.svd(matrix, full_matrices=False)
            return U, sigma, Vt
        except Exception as e:
            logger.error(f"SVD failed: {e}")
            raise
    
    @staticmethod
    def condition_number(matrix: np.ndarray) -> float:
        """
        Calculate condition number (ratio of max to min singular value)
        
        Args:
            matrix: Input matrix
            
        Returns:
            Condition number
        """
        try:
            _, sigma, _ = MatrixOperations.svd_decomposition(matrix)
            if sigma[-1] > 1e-10:
                return sigma[0] / sigma[-1]
            else:
                return np.inf
        except Exception as e:
            logger.error(f"Condition number calculation failed: {e}")
            return np.inf
    
    @staticmethod
    def frobenius_norm(matrix: np.ndarray) -> float:
        """
        Calculate Frobenius norm
        
        Args:
            matrix: Input matrix
            
        Returns:
            Frobenius norm
        """
        return np.linalg.norm(matrix, 'fro')
    
    @staticmethod
    def interaction_measure(gain_matrix: np.ndarray) -> float:
        """
        Calculate interaction measure
        I = ||G - diag(G)|| / ||G||
        
        Args:
            gain_matrix: Gain matrix
            
        Returns:
            Interaction index
        """
        G_diag = np.diag(np.diag(gain_matrix))
        G_off = gain_matrix - G_diag
        
        norm_off = MatrixOperations.frobenius_norm(G_off)
        norm_total = MatrixOperations.frobenius_norm(gain_matrix)
        
        if norm_total > 1e-10:
            return norm_off / norm_total
        return 0.0
    
    @staticmethod
    def niederlinski_index(gain_matrix: np.ndarray, rga_matrix: np.ndarray) -> float:
        """
        Calculate Niederlinski index for stability check
        NI = det(G) / (product of diagonal elements of G)
        
        For decentralized control to be stable: NI > 0
        
        Args:
            gain_matrix: Gain matrix
            rga_matrix: RGA matrix
            
        Returns:
            Niederlinski index
        """
        try:
            det_G = np.linalg.det(gain_matrix)
            prod_diag = np.prod(np.diag(gain_matrix))
            
            if abs(prod_diag) > 1e-10:
                ni = det_G / prod_diag
            else:
                ni = 0.0
            
            return ni
        except Exception as e:
            logger.error(f"Niederlinski index calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def gramian_based_controllability(A: np.ndarray, B: np.ndarray, 
                                     time_horizon: float = 10.0) -> np.ndarray:
        """
        Calculate controllability Gramian
        Wc = integral(exp(At) * B * B^T * exp(A^T*t) dt) from 0 to T
        
        Args:
            A: State matrix
            B: Input matrix
            time_horizon: Integration time horizon
            
        Returns:
            Controllability Gramian
        """
        try:
            # Solve Lyapunov equation: A*Wc + Wc*A^T + B*B^T = 0
            Q = -B @ B.T
            Wc = linalg.solve_continuous_lyapunov(A, Q)
            return Wc
        except Exception as e:
            logger.error(f"Gramian calculation failed: {e}")
            return np.eye(A.shape[0])
    
    @staticmethod
    def bristol_rga_rules(rga_matrix: np.ndarray) -> List[str]:
        """
        Apply Bristol's RGA rules for pairing
        
        Args:
            rga_matrix: RGA matrix
            
        Returns:
            List of recommendations
        """
        recommendations = []
        n = rga_matrix.shape[0]
        
        # Rule 1: Pair on positive elements close to 1
        for i in range(n):
            for j in range(n):
                if 0.7 < rga_matrix[i, j] <= 1.5:
                    recommendations.append(
                        f"Good pairing: CV{i+1} with MV{j+1} (λ = {rga_matrix[i,j]:.3f})"
                    )
        
        # Rule 2: Avoid negative pairings
        for i in range(n):
            for j in range(n):
                if rga_matrix[i, j] < 0:
                    recommendations.append(
                        f"WARNING: Avoid pairing CV{i+1} with MV{j+1} (negative RGA: {rga_matrix[i,j]:.3f})"
                    )
        
        # Rule 3: Check row and column sums (should be 1.0)
        row_sums = np.sum(rga_matrix, axis=1)
        col_sums = np.sum(rga_matrix, axis=0)
        
        for i, rs in enumerate(row_sums):
            if abs(rs - 1.0) > 0.1:
                recommendations.append(
                    f"WARNING: Row {i+1} sum is {rs:.3f}, should be 1.0"
                )
        
        return recommendations
    
    @staticmethod
    def participation_matrix(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate participation matrix from eigenvalue decomposition
        Shows which states participate in which modes
        
        Args:
            A: State matrix
            
        Returns:
            Eigenvalues and participation matrix
        """
        try:
            eigenvalues, eigenvectors = np.linalg.eig(A)
            
            # Participation matrix
            participation = np.abs(eigenvectors) ** 2
            
            # Normalize columns
            col_sums = np.sum(participation, axis=0)
            participation = participation / col_sums
            
            return eigenvalues, participation
        except Exception as e:
            logger.error(f"Participation matrix calculation failed: {e}")
            n = A.shape[0]
            return np.zeros(n), np.eye(n)
    
    @staticmethod
    def skogestad_half_rule(gain_matrix: np.ndarray, 
                           time_constants: np.ndarray) -> np.ndarray:
        """
        Apply Skogestad's half rule for approximating transfer functions
        
        Args:
            gain_matrix: Steady-state gains
            time_constants: Time constants matrix
            
        Returns:
            Effective time delays
        """
        try:
            # Half rule: delay ≈ 0.5 * time_constant
            delays = 0.5 * time_constants
            return delays
        except Exception as e:
            logger.error(f"Half rule calculation failed: {e}")
            return np.zeros_like(time_constants)
    
    @staticmethod
    def morari_resilience_index(gain_matrix: np.ndarray) -> float:
        """
        Calculate Morari's resilience index
        RI = min(singular values) / max(singular values)
        
        Higher values indicate better resilience to uncertainty
        
        Args:
            gain_matrix: Gain matrix
            
        Returns:
            Resilience index (0 to 1)
        """
        try:
            _, sigma, _ = MatrixOperations.svd_decomposition(gain_matrix)
            if sigma[0] > 1e-10:
                return sigma[-1] / sigma[0]
            return 0.0
        except Exception as e:
            logger.error(f"Resilience index calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def relative_disturbance_gain(G: np.ndarray, Gd: np.ndarray) -> np.ndarray:
        """
        Calculate Relative Disturbance Gain (RDG)
        Measures effect of disturbances relative to manipulated variables
        
        Args:
            G: Process gain matrix (outputs × inputs)
            Gd: Disturbance gain matrix (outputs × disturbances)
            
        Returns:
            RDG matrix
        """
        try:
            G_inv = np.linalg.pinv(G)
            RDG = Gd @ G_inv.T
            return np.abs(RDG)
        except Exception as e:
            logger.error(f"RDG calculation failed: {e}")
            return np.zeros_like(Gd)