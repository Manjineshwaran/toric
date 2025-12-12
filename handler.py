"""
Handler/Bridge between UI and Main Pipeline Code
==================================================

This module manages shared state between the PyQt5 UI and the main pipeline code.
It allows the UI to update configuration values that the main pipeline uses.

Author: Toric Lens Surgery Pipeline
"""

import threading
from typing import Optional
from pathlib import Path


class PipelineConfigHandler:
    """
    Thread-safe configuration handler that bridges UI and main pipeline.
    
    This class stores configuration values that can be updated from the UI
    and read by the main pipeline code.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Initialize with default values from main.py"""
        # Default model paths from main.py
        self._preop_yolo_model = "model\\intraop_latest.pt"
        self._intraop_yolo_model = "model\\intraop_latest.pt"
        
        # YOLO confidence score (0.0 - 1.0)
        self._yolo_confidence = 0.85
        
        # Matching confidence threshold (0.0 - 1.0)
        self._matching_confidence_threshold = 0.85
        
        # Freeze confidence threshold (0.0 - 1.0) - when match confidence exceeds this, skip analysis for 5000 frames
        self._freeze_confidence_threshold = 0.85
        
        # Application mode: 'normal' or 'demo'
        self._app_mode = 'normal'
        
        # Flag to indicate if configuration has been submitted
        self._config_submitted = False
        
        # Pre-op image path
        self._preop_image_path = None
        
        # Intra-op video path
        self._intraop_video_path = None
        
        # Preprocessing parameters (defaults from Config)
        self._inner_exclude_ratio = 0.85
        self._outer_include_ratio = 1.40
        self._crop_width_ratio = 4.0
        self._crop_height_ratio = 3.0
        self._eyelid_trim_upper_ratio = 0.85
        self._eyelid_trim_lower_ratio = 0.85
        
        # Preprocessed preop result (stored after preprocessing)
        self._preop_result = None
        
        # Manual axis offsets for preop limbus center adjustment (in pixels)
        self._manual_x_offset = 0
        self._manual_y_offset = 0
        
        # Reference, toric, and incision angles (in degrees, default 0)
        self._reference_angle = 0.0
        self._toric_angle = 0.0
        self._incision_angle = 0.0
        
        # Lock for thread-safe access
        self._config_lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance (thread-safe)"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    # ========== Pre-op Model Path ==========
    def get_preop_model_path(self) -> str:
        """Get pre-op YOLO model path"""
        with self._config_lock:
            return self._preop_yolo_model
    
    def set_preop_model_path(self, path: str):
        """Set pre-op YOLO model path"""
        with self._config_lock:
            if path and Path(path).exists():
                self._preop_yolo_model = path
            elif path:  # Allow setting even if file doesn't exist yet (for UI)
                self._preop_yolo_model = path
    
    # ========== Intra-op Model Path ==========
    def get_intraop_model_path(self) -> str:
        """Get intra-op YOLO model path"""
        with self._config_lock:
            return self._intraop_yolo_model
    
    def set_intraop_model_path(self, path: str):
        """Set intra-op YOLO model path"""
        with self._config_lock:
            if path and Path(path).exists():
                self._intraop_yolo_model = path
            elif path:  # Allow setting even if file doesn't exist yet (for UI)
                self._intraop_yolo_model = path
    
    # ========== YOLO Confidence Score ==========
    def get_yolo_confidence(self) -> float:
        """Get YOLO confidence score"""
        with self._config_lock:
            return self._yolo_confidence
    
    def set_yolo_confidence(self, confidence: float):
        """Set YOLO confidence score (0.0 - 1.0)"""
        with self._config_lock:
            self._yolo_confidence = max(0.0, min(1.0, float(confidence)))
    
    # ========== Matching Confidence Threshold ==========
    def get_matching_confidence_threshold(self) -> float:
        """Get matching confidence threshold"""
        with self._config_lock:
            return self._matching_confidence_threshold
    
    def set_matching_confidence_threshold(self, threshold: float):
        """Set matching confidence threshold (0.0 - 1.0)"""
        with self._config_lock:
            self._matching_confidence_threshold = max(0.0, min(1.0, float(threshold)))
    
    # ========== Freeze Confidence Threshold ==========
    def get_freeze_confidence_threshold(self) -> float:
        """Get freeze confidence threshold"""
        with self._config_lock:
            return self._freeze_confidence_threshold
    
    def set_freeze_confidence_threshold(self, threshold: float):
        """Set freeze confidence threshold (0.0 - 1.0)"""
        with self._config_lock:
            self._freeze_confidence_threshold = max(0.0, min(1.0, float(threshold)))
    
    # ========== Application Mode ==========
    def get_app_mode(self) -> str:
        """Get application mode: 'normal' or 'demo'"""
        with self._config_lock:
            return self._app_mode
    
    def set_app_mode(self, mode: str):
        """Set application mode: 'normal' or 'demo'"""
        with self._config_lock:
            if mode.lower() in ('normal', 'demo'):
                self._app_mode = mode.lower()
    
    # ========== Configuration Submitted Flag ==========
    def is_config_submitted(self) -> bool:
        """Check if configuration has been submitted"""
        with self._config_lock:
            return self._config_submitted
    
    def set_config_submitted(self, submitted: bool = True):
        """Mark configuration as submitted"""
        with self._config_lock:
            self._config_submitted = submitted
    
    # ========== Pre-op Image Path ==========
    def get_preop_image_path(self) -> Optional[str]:
        """Get pre-op image path"""
        with self._config_lock:
            return self._preop_image_path
    
    def set_preop_image_path(self, path: str):
        """Set pre-op image path"""
        with self._config_lock:
            self._preop_image_path = path
    
    # ========== Intra-op Video Path ==========
    def get_intraop_video_path(self) -> Optional[str]:
        """Get intra-op video path"""
        with self._config_lock:
            return self._intraop_video_path
    
    def set_intraop_video_path(self, path: str):
        """Set intra-op video path"""
        with self._config_lock:
            self._intraop_video_path = path
    
    # ========== Preprocessing Parameters ==========
    def get_inner_exclude_ratio(self) -> float:
        """Get inner exclude ratio"""
        with self._config_lock:
            return self._inner_exclude_ratio
    
    def set_inner_exclude_ratio(self, ratio: float):
        """Set inner exclude ratio"""
        with self._config_lock:
            self._inner_exclude_ratio = max(0.0, float(ratio))
    
    def get_outer_include_ratio(self) -> float:
        """Get outer include ratio"""
        with self._config_lock:
            return self._outer_include_ratio
    
    def set_outer_include_ratio(self, ratio: float):
        """Set outer include ratio"""
        with self._config_lock:
            self._outer_include_ratio = max(0.0, float(ratio))
    
    def get_crop_width_ratio(self) -> float:
        """Get crop width ratio"""
        with self._config_lock:
            return self._crop_width_ratio
    
    def set_crop_width_ratio(self, ratio: float):
        """Set crop width ratio"""
        with self._config_lock:
            self._crop_width_ratio = max(0.0, float(ratio))
    
    def get_crop_height_ratio(self) -> float:
        """Get crop height ratio"""
        with self._config_lock:
            return self._crop_height_ratio
    
    def set_crop_height_ratio(self, ratio: float):
        """Set crop height ratio"""
        with self._config_lock:
            self._crop_height_ratio = max(0.0, float(ratio))
    
    def get_eyelid_trim_upper_ratio(self) -> float:
        """Get eyelid trim upper ratio"""
        with self._config_lock:
            return self._eyelid_trim_upper_ratio
    
    def set_eyelid_trim_upper_ratio(self, ratio: float):
        """Set eyelid trim upper ratio"""
        with self._config_lock:
            self._eyelid_trim_upper_ratio = max(0.0, float(ratio))
    
    def get_eyelid_trim_lower_ratio(self) -> float:
        """Get eyelid trim lower ratio"""
        with self._config_lock:
            return self._eyelid_trim_lower_ratio
    
    def set_eyelid_trim_lower_ratio(self, ratio: float):
        """Set eyelid trim lower ratio"""
        with self._config_lock:
            self._eyelid_trim_lower_ratio = max(0.0, float(ratio))
    
    # ========== Pre-op Result ==========
    def get_preop_result(self):
        """Get preprocessed pre-op result"""
        with self._config_lock:
            return self._preop_result
    
    def set_preop_result(self, result):
        """Set preprocessed pre-op result"""
        with self._config_lock:
            self._preop_result = result
    
    # ========== Manual Axis Offsets (Pre-op only) ==========
    def get_manual_x_offset(self) -> int:
        """Get manual X-axis offset for pre-op limbus center (in pixels)"""
        with self._config_lock:
            return self._manual_x_offset
    
    def set_manual_x_offset(self, offset: int):
        """Set manual X-axis offset for pre-op limbus center (in pixels)"""
        with self._config_lock:
            self._manual_x_offset = int(offset)
    
    def get_manual_y_offset(self) -> int:
        """Get manual Y-axis offset for pre-op limbus center (in pixels)"""
        with self._config_lock:
            return self._manual_y_offset
    
    def set_manual_y_offset(self, offset: int):
        """Set manual Y-axis offset for pre-op limbus center (in pixels)"""
        with self._config_lock:
            self._manual_y_offset = int(offset)
    
    # ========== Reference, Toric, and Incision Angles ==========
    def get_reference_angle(self) -> float:
        """Get reference angle (in degrees, default 0)"""
        with self._config_lock:
            return self._reference_angle
    
    def set_reference_angle(self, angle: float):
        """Set reference angle (in degrees)"""
        with self._config_lock:
            self._reference_angle = float(angle)
    
    def get_toric_angle(self) -> float:
        """Get toric angle (in degrees, default 0)"""
        with self._config_lock:
            return self._toric_angle
    
    def set_toric_angle(self, angle: float):
        """Set toric angle (in degrees)"""
        with self._config_lock:
            self._toric_angle = float(angle)
    
    def get_incision_angle(self) -> float:
        """Get incision angle (in degrees, default 0)"""
        with self._config_lock:
            return self._incision_angle
    
    def set_incision_angle(self, angle: float):
        """Set incision angle (in degrees)"""
        with self._config_lock:
            self._incision_angle = float(angle)
    
    # ========== Reset Configuration ==========
    def reset_to_defaults(self):
        """Reset all configuration to default values"""
        with self._config_lock:
            self._preop_yolo_model = "model\\intraop_latest.pt"
            self._intraop_yolo_model = "model\\intraop_latest.pt"
            self._yolo_confidence = 0.85
            self._matching_confidence_threshold = 0.85
            self._freeze_confidence_threshold = 0.85
            self._app_mode = 'normal'
            self._config_submitted = False
            self._preop_image_path = None
            self._intraop_video_path = None
            self._inner_exclude_ratio = 0.85
            self._outer_include_ratio = 1.40
            self._crop_width_ratio = 4.0
            self._crop_height_ratio = 3.0
            self._eyelid_trim_upper_ratio = 0.85
            self._eyelid_trim_lower_ratio = 0.85
            self._preop_result = None
            self._manual_x_offset = 0
            self._manual_y_offset = 0
            self._reference_angle = 0.0
            self._toric_angle = 0.0
            self._incision_angle = 0.0
    
    # ========== Get All Configuration ==========
    def get_all_config(self) -> dict:
        """Get all configuration as dictionary"""
        with self._config_lock:
            return {
                'preop_model_path': self._preop_yolo_model,
                'intraop_model_path': self._intraop_yolo_model,
                'yolo_confidence': self._yolo_confidence,
                'matching_confidence_threshold': self._matching_confidence_threshold,
                'freeze_confidence_threshold': self._freeze_confidence_threshold,
                'app_mode': self._app_mode,
                'config_submitted': self._config_submitted,
                'preop_image_path': self._preop_image_path,
                'intraop_video_path': self._intraop_video_path,
                'inner_exclude_ratio': self._inner_exclude_ratio,
                'outer_include_ratio': self._outer_include_ratio,
                'crop_width_ratio': self._crop_width_ratio,
                'crop_height_ratio': self._crop_height_ratio,
                'eyelid_trim_upper_ratio': self._eyelid_trim_upper_ratio,
                'eyelid_trim_lower_ratio': self._eyelid_trim_lower_ratio,
                'manual_x_offset': self._manual_x_offset,
                'manual_y_offset': self._manual_y_offset,
                'reference_angle': self._reference_angle,
                'toric_angle': self._toric_angle,
                'incision_angle': self._incision_angle
            }
    
    def print_config(self):
        """Print current configuration (for debugging)"""
        config = self.get_all_config()
        print("\n" + "="*50)
        print("Current Pipeline Configuration:")
        print("="*50)
        for key, value in config.items():
            print(f"  {key}: {value}")
        print("="*50 + "\n")


# Convenience function to get the handler instance
def get_config_handler() -> PipelineConfigHandler:
    """Get the global configuration handler instance"""
    return PipelineConfigHandler.get_instance()

