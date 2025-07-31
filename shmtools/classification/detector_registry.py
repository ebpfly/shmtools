"""
Detector Registry System for Custom Outlier Detector Assembly

This module provides a registry of available detector learning/scoring function 
pairs organized by detector type (parametric, non-parametric, semi-parametric).
Used by the assemble_outlier_detector_shm function to create custom detectors.
"""

from typing import Dict, List, Tuple, Any, Optional
import inspect
from . import outlier_detection, nonparametric, semiparametric


class DetectorRegistry:
    """Registry of available detector learning/scoring function pairs."""
    
    def __init__(self):
        """Initialize the detector registry with available detectors."""
        self._parametric_detectors = {
            'pca': {
                'learn_function': 'learn_pca_shm',
                'score_function': 'score_pca_shm',
                'display_name': 'Principal Component Analysis',
                'description': 'PCA-based outlier detection using principal component scores'
            },
            'mahalanobis': {
                'learn_function': 'learn_mahalanobis_shm',
                'score_function': 'score_mahalanobis_shm', 
                'display_name': 'Mahalanobis Distance',
                'description': 'Mahalanobis distance-based outlier detection'
            },
            'svd': {
                'learn_function': 'learn_svd_shm',
                'score_function': 'score_svd_shm',
                'display_name': 'Singular Value Decomposition',
                'description': 'SVD-based outlier detection using reconstruction errors'
            },
            'factor_analysis': {
                'learn_function': 'learn_factor_analysis_shm',
                'score_function': 'score_factor_analysis_shm',
                'display_name': 'Factor Analysis',
                'description': 'Factor analysis-based outlier detection'
            }
        }
        
        self._nonparametric_detectors = {
            'kernel_density': {
                'learn_function': 'learn_kernel_density_shm',
                'score_function': 'score_kernel_density_shm',
                'display_name': 'Kernel Density Estimation',
                'description': 'Non-parametric kernel density estimation for outlier detection'
            }
        }
        
        self._semiparametric_detectors = {
            'gmm_semi': {
                'learn_function': 'learn_gmm_semiparametric_model_shm',
                'score_function': 'score_gmm_semiparametric_model_shm',
                'display_name': 'Gaussian Mixture Model (Semi-parametric)',
                'description': 'Semi-parametric GMM-based outlier detection with partitioning'
            }
        }
        
        # Available kernels for non-parametric detectors
        self._available_kernels = [
            'gaussian', 'epanechnikov', 'quartic', 'triangle', 
            'triweight', 'uniform', 'cosine'
        ]
        
        # Available partitioning algorithms for semi-parametric detectors  
        self._partitioning_algorithms = [
            'kmeans', 'kmedians', 'kdtree', 'pdtree', 'rptree'
        ]
    
    @property
    def parametric_detectors(self) -> Dict[str, Dict[str, str]]:
        """Get dictionary of available parametric detectors."""
        return self._parametric_detectors.copy()
    
    @property 
    def nonparametric_detectors(self) -> Dict[str, Dict[str, str]]:
        """Get dictionary of available non-parametric detectors."""
        return self._nonparametric_detectors.copy()
    
    @property
    def semiparametric_detectors(self) -> Dict[str, Dict[str, str]]:
        """Get dictionary of available semi-parametric detectors."""
        return self._semiparametric_detectors.copy()
    
    @property
    def available_kernels(self) -> List[str]:
        """Get list of available kernels for non-parametric detectors."""
        return self._available_kernels.copy()
    
    @property
    def partitioning_algorithms(self) -> List[str]:
        """Get list of available partitioning algorithms."""
        return self._partitioning_algorithms.copy()
    
    def get_detector_types(self) -> List[str]:
        """Get list of available detector types."""
        return ['parametric', 'nonparametric', 'semiparametric']
    
    def get_detectors_by_type(self, detector_type: str) -> Dict[str, Dict[str, str]]:
        """
        Get detectors by type.
        
        Parameters
        ----------
        detector_type : str
            Type of detector ('parametric', 'nonparametric', 'semiparametric')
            
        Returns
        -------
        detectors : dict
            Dictionary of available detectors for the specified type
        """
        detector_type = detector_type.lower()
        
        if detector_type == 'parametric':
            return self.parametric_detectors
        elif detector_type == 'nonparametric':
            return self.nonparametric_detectors
        elif detector_type == 'semiparametric':
            return self.semiparametric_detectors
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
    
    def get_detector_info(self, detector_type: str, detector_name: str) -> Dict[str, str]:
        """
        Get information about a specific detector.
        
        Parameters
        ----------
        detector_type : str
            Type of detector ('parametric', 'nonparametric', 'semiparametric')
        detector_name : str
            Name of the detector (e.g., 'pca', 'mahalanobis')
            
        Returns
        -------
        detector_info : dict
            Dictionary with detector information
        """
        detectors = self.get_detectors_by_type(detector_type)
        
        if detector_name not in detectors:
            available = list(detectors.keys())
            raise ValueError(f"Unknown {detector_type} detector: {detector_name}. "
                           f"Available: {available}")
        
        return detectors[detector_name].copy()
    
    def get_function_pair(self, detector_type: str, detector_name: str) -> Tuple[str, str]:
        """
        Get the learning and scoring function names for a detector.
        
        Parameters
        ----------
        detector_type : str
            Type of detector
        detector_name : str
            Name of the detector
            
        Returns
        -------
        learn_function : str
            Name of the learning function
        score_function : str
            Name of the scoring function
        """
        detector_info = self.get_detector_info(detector_type, detector_name)
        return detector_info['learn_function'], detector_info['score_function']
    
    def validate_detector_availability(self, detector_type: str, detector_name: str) -> bool:
        """
        Check if a detector is available and its functions can be imported.
        
        Parameters
        ----------
        detector_type : str
            Type of detector
        detector_name : str
            Name of the detector
            
        Returns
        -------
        available : bool
            True if detector is available and functions can be imported
        """
        try:
            learn_func, score_func = self.get_function_pair(detector_type, detector_name)
            
            # Try to import the functions to verify they exist
            if detector_type == 'parametric':
                module = outlier_detection
            elif detector_type == 'nonparametric':
                module = nonparametric
            elif detector_type == 'semiparametric':
                module = semiparametric
            else:
                return False
            
            # Check if functions exist in the module
            return (hasattr(module, learn_func) and hasattr(module, score_func))
            
        except Exception:
            return False
    
    def get_function_signature(self, detector_type: str, detector_name: str, 
                             function_type: str) -> inspect.Signature:
        """
        Get the function signature for a detector function.
        
        Parameters
        ----------
        detector_type : str
            Type of detector
        detector_name : str 
            Name of the detector
        function_type : str
            Type of function ('learn' or 'score')
            
        Returns
        -------
        signature : inspect.Signature
            Function signature
        """
        learn_func, score_func = self.get_function_pair(detector_type, detector_name)
        
        if function_type == 'learn':
            func_name = learn_func
        elif function_type == 'score':
            func_name = score_func
        else:
            raise ValueError("function_type must be 'learn' or 'score'")
        
        # Get the appropriate module
        if detector_type == 'parametric':
            module = outlier_detection
        elif detector_type == 'nonparametric':
            module = nonparametric
        elif detector_type == 'semiparametric':
            module = semiparametric
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
        
        # Get the function and its signature
        func = getattr(module, func_name)
        return inspect.signature(func)


# Global registry instance
detector_registry = DetectorRegistry()