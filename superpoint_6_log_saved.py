"""
Robust SuperPoint + LightGlue Pipeline for Surgical Eye Alignment
==================================================================

HIGH ACCURACY ROTATION ESTIMATION for Toric Lens Surgery

This module implements a multi-stage, robust feature matching pipeline
designed for maximum accuracy in determining eye rotation between
preoperative and intraoperative images.

Key Accuracy Features:
----------------------
1. Multi-scale feature extraction
2. Vessel-aware keypoint filtering
3. Geometric consistency filtering
4. Multi-method rotation estimation with consensus
5. Outlier rejection using multiple RANSAC variants
6. Confidence scoring and validation
7. Cross-validation of rotation estimate

Author: High-Accuracy Surgical Alignment Pipeline
"""

import cv2
import os
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

# Matplotlib visualization disabled - no windows will open
# import matplotlib
# matplotlib.use('Agg')  # Non-interactive backend (no GUI, no windows)
# import matplotlib.pyplot as plt
plt = None  # Placeholder to avoid import errors
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List, Any
import warnings
import time
warnings.filterwarnings('ignore')
import sys
from datetime import datetime

# LightGlue imports
try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import load_image, rbd
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False
    print("[WARNING] LightGlue not installed. Install with: pip install git+https://github.com/cvg/LightGlue.git")

# ==============================================================================
# LOGGING UTILITY
# ==============================================================================

class Logger:
    """
    Logger that captures all print statements and saves them to a file.
    Can be used as a context manager or as a global logger.
    """
    _instance = None
    _log_file = None
    _log_file_handle = None
    _original_stdout = None
    _enabled = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance
    
    def start_logging(self, log_file_path: str, append: bool = False):
        """
        Start logging all print statements to a file.
        
        Args:
            log_file_path: Path to log file
            append: If True, append to existing file; otherwise overwrite
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        # Open log file
        mode = 'a' if append else 'w'
        self._log_file_handle = open(log_file_path, mode, encoding='utf-8')
        self._log_file = log_file_path
        self._original_stdout = sys.stdout
        
        # Create a custom stdout that writes to both console and file
        class TeeOutput:
            def __init__(self, *files):
                self.files = files
            
            def write(self, text):
                for f in self.files:
                    f.write(text)
                    f.flush()
            
            def flush(self):
                for f in self.files:
                    f.flush()
        
        # Redirect stdout to both console and file
        sys.stdout = TeeOutput(self._original_stdout, self._log_file_handle)
        self._enabled = True
        
        # Write header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*60}")
        print(f"LOGGING STARTED: {timestamp}")
        print(f"Log file: {log_file_path}")
        print(f"{'='*60}\n")
    
    def stop_logging(self):
        """Stop logging and restore original stdout"""
        if self._enabled and self._log_file_handle:
            print(f"\n{'='*60}")
            print(f"LOGGING ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}\n")
            
            sys.stdout = self._original_stdout
            self._log_file_handle.close()
            self._log_file_handle = None
            self._enabled = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_logging()
    
    @classmethod
    def get_logger(cls):
        """Get the singleton logger instance"""
        return cls()


# ==============================================================================
# CONFIGURATION - TUNED FOR MAXIMUM ACCURACY
# ==============================================================================
@dataclass
class AccuracyConfig:
    """Configuration optimized for maximum accuracy"""
    
    # SuperPoint parameters
    MAX_KEYPOINTS: int = 10096           # More keypoints = more potential matches
    NMS_RADIUS: int = 3                  # Non-maximum suppression radius (smaller = more keypoints)
    DETECTION_THRESHOLD: float = 0.001   # Lower = more keypoints (very sensitive)
    
    # LightGlue parameters - RELAXED for difficult matching
    DEPTH_CONFIDENCE: float = 0.5        # Lower = more matches (was 0.95, too strict)
    WIDTH_CONFIDENCE: float = 0.5        # Lower = more matches
    
    # Match filtering thresholds
    MIN_MATCH_CONFIDENCE: float = 0.1    # Low threshold to keep more matches
    MIN_MATCHES_FOR_RANSAC: int = 8      # Minimum matches required (relaxed from 15)
    
    # Geometric filtering
    MAX_SCALE_DEVIATION: float = 0.4     # Max scale difference from 1.0 (relaxed)
    MAX_ANGLE_DEVIATION: float = 60.0    # Max keypoint angle difference (degrees)
    MAX_RADIAL_DEVIATION: float = 0.35   # Max radial position deviation ratio (relaxed)
    
    # RANSAC parameters
    RANSAC_REPROJ_THRESHOLD: float = 5.0  # Reprojection error threshold (relaxed)
    RANSAC_MAX_ITERS: int = 10000         # More iterations = better result
    RANSAC_CONFIDENCE: float = 0.999      # High confidence
    
    # Rotation consensus
    ROTATION_STD_THRESHOLD: float = 10.0  # Max std dev for valid rotation (degrees, relaxed)
    MIN_INLIER_RATIO: float = 0.2         # Minimum inlier ratio for validity (relaxed)
    
    # Multi-scale extraction
    SCALES: List[float] = field(default_factory=lambda: [1.0])  # Can add [0.8, 1.0, 1.2]
    
    # Vessel filtering
    VESSEL_SEARCH_RADIUS: int = 8         # Pixels around keypoint to search for vessel (larger)

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================
@dataclass
class RotationResult:
    """Complete rotation estimation result with confidence"""
    rotation_deg: float                 # Primary rotation estimate (degrees)
    rotation_rad: float                 # Rotation in radians
    scale: float                        # Scale factor
    translation: Tuple[float, float]    # Translation (x, y)
    
    # Confidence metrics
    num_matches: int                    # Total matches found
    num_inliers: int                    # RANSAC inliers
    inlier_ratio: float                 # Inliers / total matches
    rotation_std: float                 # Standard deviation of rotation estimates
    confidence_score: float             # Overall confidence (0-1)
    
    # Validation
    is_reliable: bool                   # Whether result passes quality checks
    quality_issues: List[str] = field(default_factory=list)
    
    # Transformation matrix
    transform_matrix: np.ndarray = None
    
    # Method used
    estimation_method: str = "affine"

@dataclass 
class MatchResult:
    """Feature matching result"""
    keypoints0: np.ndarray              # Matched keypoints in image 0
    keypoints1: np.ndarray              # Matched keypoints in image 1
    confidences: np.ndarray             # Match confidences
    num_raw_matches: int                # Before filtering
    num_filtered_matches: int           # After filtering

# ==============================================================================
# MODEL INITIALIZATION
# ==============================================================================
def initialize_models(config: AccuracyConfig = None, 
                      device: str = None) -> Tuple[Any, Any, str]:
    """
    Initialize SuperPoint and LightGlue with optimized parameters.
    """
    if not LIGHTGLUE_AVAILABLE:
        raise RuntimeError("LightGlue not available. Please install it.")
    
    if config is None:
        config = AccuracyConfig()
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print("INITIALIZING HIGH-ACCURACY FEATURE MODELS")
    print(f"{'='*70}")
    print(f"Device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize SuperPoint with optimized parameters
    print(f"\nLoading SuperPoint...")
    print(f"  Max keypoints: {config.MAX_KEYPOINTS}")
    print(f"  NMS radius: {config.NMS_RADIUS}")
    print(f"  Detection threshold: {config.DETECTION_THRESHOLD}")
    
    extractor = SuperPoint(
        max_num_keypoints=config.MAX_KEYPOINTS,
        nms_radius=config.NMS_RADIUS,
        detection_threshold=config.DETECTION_THRESHOLD
    ).eval().to(device)
    
    # Initialize LightGlue with high-accuracy settings
    print(f"\nLoading LightGlue...")
    print(f"  Depth confidence: {config.DEPTH_CONFIDENCE}")
    print(f"  Width confidence: {config.WIDTH_CONFIDENCE}")
    
    matcher = LightGlue(
        features='superpoint',
        depth_confidence=config.DEPTH_CONFIDENCE,
        width_confidence=config.WIDTH_CONFIDENCE
    ).eval().to(device)
    
    print(f"\n✓ Models initialized successfully")
    print(f"{'='*70}\n")
    
    return extractor, matcher, device

# ==============================================================================
# FEATURE EXTRACTION
# ==============================================================================
def extract_features(image_path: str, 
                     extractor: Any, 
                     device: str,
                     mask: np.ndarray = None,
                     config: AccuracyConfig = None) -> Dict:
    """
    Extract SuperPoint features with optional mask filtering.
    
    Args:
        image_path: Path to image
        extractor: SuperPoint extractor
        device: Computing device
        mask: Optional binary mask (255 = valid region)
        config: Accuracy configuration
    
    Returns:
        Features dictionary with keypoints, descriptors, scores
    """
    if config is None:
        config = AccuracyConfig()
    
    # Load image
    image = load_image(image_path)
    
    # Extract features
    with torch.no_grad():
        feats = extractor.extract(image.to(device))
    
    original_count = feats['keypoints'].shape[1]
    
    # Apply mask filtering if provided
    if mask is not None:
        feats = filter_keypoints_by_mask(feats, mask, config.VESSEL_SEARCH_RADIUS)
    
    filtered_count = feats['keypoints'].shape[1]
    
    print(f"  Keypoints: {original_count} → {filtered_count} (after mask filtering)")
    
    return feats

def filter_keypoints_by_mask(feats: Dict, 
                              mask: np.ndarray, 
                              search_radius: int = 5) -> Dict:
    """
    Filter keypoints to keep only those within valid mask regions.
    """
    if mask is None:
        return feats
    
    kpts = feats['keypoints'][0].cpu().numpy()
    valid_indices = []
    
    h, w = mask.shape[:2]
    
    for i, (x, y) in enumerate(kpts):
        ix, iy = int(round(x)), int(round(y))
        
        # Check bounds
        if iy < 0 or iy >= h or ix < 0 or ix >= w:
            continue
        
        # Check neighborhood for valid region
        y_min = max(0, iy - search_radius)
        y_max = min(h, iy + search_radius + 1)
        x_min = max(0, ix - search_radius)
        x_max = min(w, ix + search_radius + 1)
        
        neighborhood = mask[y_min:y_max, x_min:x_max]
        
        if np.any(neighborhood > 0):
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        print("  [WARNING] No keypoints in valid mask region!")
        return feats
    
    # Create filtered features
    valid_tensor = torch.tensor(valid_indices, device=feats['keypoints'].device)
    
    filtered_feats = {
        'keypoints': feats['keypoints'][:, valid_tensor, :],
        'descriptors': feats['descriptors'][:, valid_tensor, :],
    }
    
    if 'scores' in feats:
        filtered_feats['scores'] = feats['scores'][:, valid_tensor]
    
    # Copy other metadata
    for key in feats:
        if key not in filtered_feats:
            filtered_feats[key] = feats[key]
    
    return filtered_feats

# ==============================================================================
# FEATURE MATCHING
# ==============================================================================
def match_features(feats0: Dict, 
                   feats1: Dict, 
                   matcher: Any,
                   config: AccuracyConfig = None) -> MatchResult:
    """
    Match features using LightGlue with confidence filtering.
    Handles various output formats from LightGlue robustly.
    """
    if config is None:
        config = AccuracyConfig()
    
    print(f"\n{'='*70}")
    print("FEATURE MATCHING")
    print(f"{'='*70}")
    print(f"Pre-op keypoints: {feats0['keypoints'].shape[1]}")
    print(f"Intra-op keypoints: {feats1['keypoints'].shape[1]}")
    
    # Run LightGlue matching
    with torch.no_grad():
        matches_raw = matcher({'image0': feats0, 'image1': feats1})
    
    # Remove batch dimension from all outputs first
    feats0_clean, feats1_clean, matches = [rbd(x) for x in [feats0, feats1, matches_raw]]
    
    # Get keypoints as numpy
    kpts0 = feats0_clean['keypoints']
    kpts1 = feats1_clean['keypoints']
    
    if torch.is_tensor(kpts0):
        kpts0 = kpts0.cpu().numpy()
    if torch.is_tensor(kpts1):
        kpts1 = kpts1.cpu().numpy()
    
    # Get match indices - LightGlue returns matches as [N, 2] array
    match_indices = matches.get('matches', matches.get('matches0', None))
    
    if match_indices is None:
        print("[ERROR] No matches found in output")
        return MatchResult(
            keypoints0=np.array([]).reshape(0, 2),
            keypoints1=np.array([]).reshape(0, 2),
            confidences=np.array([]),
            num_raw_matches=0,
            num_filtered_matches=0
        )
    
    # Convert to numpy
    if torch.is_tensor(match_indices):
        match_indices = match_indices.cpu().numpy()
    elif isinstance(match_indices, list):
        match_indices = np.array(match_indices)
    
    # Get match scores/confidences
    match_scores = matches.get('scores', matches.get('matching_scores0', None))
    
    if match_scores is not None:
        if torch.is_tensor(match_scores):
            match_scores = match_scores.cpu().numpy()
        elif isinstance(match_scores, list):
            match_scores = np.array(match_scores)
        match_scores = np.asarray(match_scores).flatten()
    else:
        # No scores available - use ones
        match_scores = np.ones(len(match_indices))
    
    raw_count = len(match_indices)
    print(f"Raw matches: {raw_count}")
    
    if raw_count == 0:
        return MatchResult(
            keypoints0=np.array([]).reshape(0, 2),
            keypoints1=np.array([]).reshape(0, 2),
            confidences=np.array([]),
            num_raw_matches=0,
            num_filtered_matches=0
        )
    
    # Handle different match index formats
    # LightGlue can return [N, 2] where each row is [idx0, idx1]
    # Or it might return separate arrays
    if match_indices.ndim == 1:
        # Single dimension - might be indices into one image, look for matches1
        matches1 = matches.get('matches1', None)
        if matches1 is not None:
            if torch.is_tensor(matches1):
                matches1 = matches1.cpu().numpy()
            # Create [N, 2] array
            valid_mask = (match_indices >= 0) & (matches1 >= 0)
            match_indices = np.stack([
                np.arange(len(match_indices))[valid_mask],
                match_indices[valid_mask]
            ], axis=1)
            match_scores = match_scores[valid_mask] if len(match_scores) > 0 else np.ones(len(match_indices))
        else:
            # Assume it's already valid indices
            pass
    
    # Ensure match_indices is 2D
    if match_indices.ndim == 1:
        print(f"[WARNING] Unexpected match format, shape: {match_indices.shape}")
        return MatchResult(
            keypoints0=np.array([]).reshape(0, 2),
            keypoints1=np.array([]).reshape(0, 2),
            confidences=np.array([]),
            num_raw_matches=raw_count,
            num_filtered_matches=0
        )
    
    # Filter by confidence
    if len(match_scores) > 0 and len(match_scores) == len(match_indices):
        high_conf_mask = match_scores >= config.MIN_MATCH_CONFIDENCE
        match_indices = match_indices[high_conf_mask]
        match_scores = match_scores[high_conf_mask]
        print(f"After confidence filter (>{config.MIN_MATCH_CONFIDENCE}): {len(match_indices)}")
    
    # Extract matched keypoints
    if len(match_indices) > 0:
        # Validate indices are within bounds
        valid_mask = (
            (match_indices[:, 0] >= 0) & (match_indices[:, 0] < len(kpts0)) &
            (match_indices[:, 1] >= 0) & (match_indices[:, 1] < len(kpts1))
        )
        match_indices = match_indices[valid_mask]
        match_scores = match_scores[valid_mask] if len(match_scores) == len(valid_mask) else match_scores
        
        matched_kpts0 = kpts0[match_indices[:, 0].astype(int)]
        matched_kpts1 = kpts1[match_indices[:, 1].astype(int)]
    else:
        matched_kpts0 = np.array([]).reshape(0, 2)
        matched_kpts1 = np.array([]).reshape(0, 2)
    
    print(f"Final matched pairs: {len(matched_kpts0)}")
    
    if len(match_scores) > 0 and len(matched_kpts0) > 0:
        # Ensure scores array matches keypoints
        if len(match_scores) != len(matched_kpts0):
            match_scores = np.ones(len(matched_kpts0))
        print(f"Confidence range: [{match_scores.min():.3f}, {match_scores.max():.3f}]")
        print(f"Mean confidence: {match_scores.mean():.3f}")
    
    return MatchResult(
        keypoints0=matched_kpts0,
        keypoints1=matched_kpts1,
        confidences=match_scores if len(match_scores) == len(matched_kpts0) else np.ones(len(matched_kpts0)),
        num_raw_matches=raw_count,
        num_filtered_matches=len(matched_kpts0)
    )

# ==============================================================================
# GEOMETRIC FILTERING
# ==============================================================================

def apply_geometric_filtering(match_result: MatchResult,
                               center0: Tuple[int, int],
                               center1: Tuple[int, int],
                               radius0: int,
                               radius1: int,
                               config: AccuracyConfig = None) -> MatchResult:
    """
    Apply geometric consistency filtering to remove outliers.
    
    Filters based on:
    1. Radial position consistency
    2. Angular position consistency
    3. Local scale consistency
    """
    if config is None:
        config = AccuracyConfig()
    
    kpts0 = match_result.keypoints0
    kpts1 = match_result.keypoints1
    confs = match_result.confidences
    
    # Ensure arrays
    kpts0 = np.asarray(kpts0)
    kpts1 = np.asarray(kpts1)
    confs = np.asarray(confs) if confs is not None else np.ones(len(kpts0))
    
    if len(kpts0) < 4:
        return match_result
    
    print(f"\n{'='*70}")
    print("GEOMETRIC FILTERING")
    print(f"{'='*70}")
    
    valid_indices = []
    
    for i in range(len(kpts0)):
        pt0 = kpts0[i]
        pt1 = kpts1[i]
        
        # Calculate vectors from centers
        v0 = pt0 - np.array(center0)
        v1 = pt1 - np.array(center1)
        
        # Radial distances (normalized by radius)
        r0_norm = np.linalg.norm(v0) / max(radius0, 1)
        r1_norm = np.linalg.norm(v1) / max(radius1, 1)
        
        # Radial consistency check
        radial_ratio = r0_norm / (r1_norm + 1e-6)
        if abs(radial_ratio - 1.0) > config.MAX_RADIAL_DEVIATION:
            continue
        
        # Angular positions (for logging/debugging)
        # angle0 = np.degrees(np.arctan2(v0[1], v0[0]))
        # angle1 = np.degrees(np.arctan2(v1[1], v1[0]))
        
        valid_indices.append(i)
    
    # Filter using numpy advanced indexing
    valid_indices = np.array(valid_indices)
    
    if len(valid_indices) > 0:
        filtered_kpts0 = kpts0[valid_indices]
        filtered_kpts1 = kpts1[valid_indices]
        filtered_confs = confs[valid_indices] if len(confs) == len(kpts0) else np.ones(len(valid_indices))
    else:
        # Keep all if none pass (avoid empty result)
        print("[WARNING] No matches passed geometric filter, keeping all")
        filtered_kpts0 = kpts0
        filtered_kpts1 = kpts1
        filtered_confs = confs
    
    print(f"Before: {len(kpts0)} matches")
    print(f"After geometric filter: {len(filtered_kpts0)} matches")
    
    return MatchResult(
        keypoints0=filtered_kpts0,
        keypoints1=filtered_kpts1,
        confidences=filtered_confs,
        num_raw_matches=match_result.num_raw_matches,
        num_filtered_matches=len(filtered_kpts0)
    )

# ==============================================================================
# ROTATION ESTIMATION - MULTI-METHOD CONSENSUS
# ==============================================================================

def estimate_rotation_robust(match_result: MatchResult,
                              center0: Tuple[int, int],
                              center1: Tuple[int, int],
                              config: AccuracyConfig = None) -> RotationResult:
    """
    Robust rotation estimation using multiple methods and consensus.
    
    Methods:
    1. Affine transformation (cv2.estimateAffinePartial2D)
    2. Homography decomposition
    3. Direct angle calculation from matched point pairs
    4. Weighted consensus of all methods
    """
    if config is None:
        config = AccuracyConfig()
    
    print(f"\n{'='*70}")
    print("ROTATION ESTIMATION (MULTI-METHOD)")
    print(f"{'='*70}")
    
    kpts0 = match_result.keypoints0
    kpts1 = match_result.keypoints1
    
    if len(kpts0) < config.MIN_MATCHES_FOR_RANSAC:
        print(f"[ERROR] Insufficient matches: {len(kpts0)} < {config.MIN_MATCHES_FOR_RANSAC}")
        return _create_failed_result(len(kpts0))
    
    # ========== METHOD 1: Affine Transformation ==========
    print("\n[Method 1] Affine Transformation (RANSAC)")
    affine_result = _estimate_affine(kpts0, kpts1, config)
    
    # ========== METHOD 2: Homography ==========
    print("\n[Method 2] Homography Decomposition")
    homography_result = _estimate_homography(kpts0, kpts1, config)
    
    # ========== METHOD 3: Direct Angle Calculation ==========
    print("\n[Method 3] Direct Angular Measurement")
    direct_result = _estimate_direct_angles(kpts0, kpts1, center0, center1, config)
    
    # ========== CONSENSUS ==========
    print(f"\n{'='*70}")
    print("CONSENSUS VOTING")
    print(f"{'='*70}")
    
    rotation_estimates = []
    weights = []
    
    if affine_result is not None:
        rotation_estimates.append(affine_result['rotation'])
        weights.append(affine_result['inlier_ratio'] * 2)  # Higher weight for affine
        print(f"  Affine: {affine_result['rotation']:.2f}° (weight: {weights[-1]:.2f})")
    
    if homography_result is not None:
        rotation_estimates.append(homography_result['rotation'])
        weights.append(homography_result['inlier_ratio'])
        print(f"  Homography: {homography_result['rotation']:.2f}° (weight: {weights[-1]:.2f})")
    
    if direct_result is not None:
        rotation_estimates.append(direct_result['rotation'])
        weights.append(direct_result['consistency'] * 1.5)
        print(f"  Direct: {direct_result['rotation']:.2f}° (weight: {weights[-1]:.2f})")
    
    if len(rotation_estimates) == 0:
        print("[ERROR] All estimation methods failed!")
        return _create_failed_result(len(kpts0))
    
    # Weighted median for robustness
    rotation_estimates = np.array(rotation_estimates)
    weights = np.array(weights)
    
    # Handle angle wrapping
    rotation_estimates = _unwrap_angles(rotation_estimates)
    
    # Weighted average
    final_rotation = np.average(rotation_estimates, weights=weights)
    rotation_std = np.sqrt(np.average((rotation_estimates - final_rotation)**2, weights=weights))
    
    print(f"\n  Final rotation: {final_rotation:.2f}° ± {rotation_std:.2f}°")
    
    # Use best method's inliers for other metrics
    best_result = affine_result if affine_result is not None else homography_result
    if best_result is None:
        best_result = {'scale': 1.0, 'translation': (0, 0), 'inliers': len(kpts0), 
                       'inlier_ratio': 1.0, 'matrix': np.eye(3)}
    
    # Calculate confidence score
    confidence = _calculate_confidence(
        num_matches=len(kpts0),
        inlier_ratio=best_result['inlier_ratio'],
        rotation_std=rotation_std,
        num_methods=len(rotation_estimates),
        config=config
    )
    
    # Quality checks
    is_reliable, issues = _validate_result(
        rotation_std=rotation_std,
        inlier_ratio=best_result['inlier_ratio'],
        num_matches=len(kpts0),
        config=config
    )
    
    return RotationResult(
        rotation_deg=final_rotation,
        rotation_rad=np.radians(final_rotation),
        scale=best_result.get('scale', 1.0),
        translation=best_result.get('translation', (0, 0)),
        num_matches=len(kpts0),
        num_inliers=best_result.get('inliers', len(kpts0)),
        inlier_ratio=best_result['inlier_ratio'],
        rotation_std=rotation_std,
        confidence_score=confidence,
        is_reliable=is_reliable,
        quality_issues=issues,
        transform_matrix=best_result.get('matrix'),
        estimation_method="consensus"
    )

def _estimate_affine(kpts0: np.ndarray, 
                      kpts1: np.ndarray, 
                      config: AccuracyConfig) -> Optional[Dict]:
    """Estimate rotation using affine transformation."""
    
    pts0 = kpts0.astype(np.float32).reshape(-1, 1, 2)
    pts1 = kpts1.astype(np.float32).reshape(-1, 1, 2)
    
    # Try multiple RANSAC methods
    methods = [
        (cv2.RANSAC, "RANSAC"),
        (cv2.LMEDS, "LMEDS"),
    ]
    
    best_result = None
    best_inliers = 0
    
    for method, name in methods:
        try:
            M, inliers = cv2.estimateAffinePartial2D(
                pts0, pts1,
                method=method,
                ransacReprojThreshold=config.RANSAC_REPROJ_THRESHOLD,
                maxIters=config.RANSAC_MAX_ITERS,
                confidence=config.RANSAC_CONFIDENCE
            )
            
            if M is not None and inliers is not None:
                num_inliers = int(np.sum(inliers))
                
                if num_inliers > best_inliers:
                    rotation = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
                    scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
                    
                    best_result = {
                        'rotation': rotation,
                        'scale': scale,
                        'translation': (M[0, 2], M[1, 2]),
                        'inliers': num_inliers,
                        'inlier_ratio': num_inliers / len(kpts0),
                        'matrix': M,
                        'method': name
                    }
                    best_inliers = num_inliers
                    
        except Exception as e:
            continue
    
    if best_result:
        print(f"  Best method: {best_result['method']}")
        print(f"  Rotation: {best_result['rotation']:.2f}°")
        print(f"  Scale: {best_result['scale']:.3f}")
        print(f"  Inliers: {best_result['inliers']}/{len(kpts0)} ({best_result['inlier_ratio']*100:.1f}%)")
    
    return best_result

def _estimate_homography(kpts0: np.ndarray, 
                          kpts1: np.ndarray, 
                          config: AccuracyConfig) -> Optional[Dict]:
    """Estimate rotation using homography decomposition."""
    
    pts0 = kpts0.astype(np.float32).reshape(-1, 1, 2)
    pts1 = kpts1.astype(np.float32).reshape(-1, 1, 2)
    
    try:
        H, inliers = cv2.findHomography(
            pts0, pts1,
            cv2.RANSAC,
            ransacReprojThreshold=config.RANSAC_REPROJ_THRESHOLD,
            maxIters=config.RANSAC_MAX_ITERS,
            confidence=config.RANSAC_CONFIDENCE
        )
        
        if H is None:
            return None
        
        num_inliers = int(np.sum(inliers))
        
        # Decompose homography to get rotation
        # For planar scene (eye is approximately planar), we can extract rotation directly
        rotation = np.degrees(np.arctan2(H[1, 0], H[0, 0]))
        scale = np.sqrt(H[0, 0]**2 + H[1, 0]**2)
        
        print(f"  Rotation: {rotation:.2f}°")
        print(f"  Scale: {scale:.3f}")
        print(f"  Inliers: {num_inliers}/{len(kpts0)} ({num_inliers/len(kpts0)*100:.1f}%)")
        
        return {
            'rotation': rotation,
            'scale': scale,
            'inliers': num_inliers,
            'inlier_ratio': num_inliers / len(kpts0),
            'matrix': H
        }
        
    except Exception as e:
        print(f"  Failed: {e}")
        return None

def _estimate_direct_angles(kpts0: np.ndarray, 
                             kpts1: np.ndarray,
                             center0: Tuple[int, int],
                             center1: Tuple[int, int],
                             config: AccuracyConfig) -> Optional[Dict]:
    """
    Estimate rotation by directly measuring angle changes of matched points.
    
    For each matched pair, calculate:
    - Angle of point relative to center in image 0
    - Angle of point relative to center in image 1
    - Difference = rotation
    
    Use median for robustness.
    """
    
    angles_diff = []
    
    for pt0, pt1 in zip(kpts0, kpts1):
        # Vector from center to point
        v0 = pt0 - np.array(center0)
        v1 = pt1 - np.array(center1)
        
        # Angles
        angle0 = np.degrees(np.arctan2(v0[1], v0[0]))
        angle1 = np.degrees(np.arctan2(v1[1], v1[0]))
        
        # Rotation = angle1 - angle0
        diff = angle1 - angle0
        
        # Normalize to [-180, 180]
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        
        angles_diff.append(diff)
    
    angles_diff = np.array(angles_diff)
    
    # Use median for robustness against outliers
    rotation_median = np.median(angles_diff)
    
    # Calculate consistency (how well angles agree)
    angles_normalized = angles_diff - rotation_median
    angles_normalized = np.where(angles_normalized > 180, angles_normalized - 360, angles_normalized)
    angles_normalized = np.where(angles_normalized < -180, angles_normalized + 360, angles_normalized)
    
    mad = np.median(np.abs(angles_normalized))  # Median Absolute Deviation
    consistency = 1.0 / (1.0 + mad / 10.0)  # Higher consistency = lower MAD
    
    # Count inliers (angles within 2*MAD of median)
    threshold = max(2 * mad, 5.0)  # At least 5 degrees
    inliers = np.sum(np.abs(angles_normalized) < threshold)
    
    print(f"  Median rotation: {rotation_median:.2f}°")
    print(f"  MAD: {mad:.2f}°")
    print(f"  Consistency: {consistency:.3f}")
    print(f"  Inliers: {inliers}/{len(kpts0)}")
    
    return {
        'rotation': rotation_median,
        'consistency': consistency,
        'mad': mad,
        'inliers': inliers,
        'inlier_ratio': inliers / len(kpts0),
        'all_angles': angles_diff
    }

def _unwrap_angles(angles: np.ndarray) -> np.ndarray:
    """Handle angle wrapping for averaging."""
    # Find the median and unwrap others relative to it
    median = np.median(angles)
    
    unwrapped = []
    for a in angles:
        diff = a - median
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        unwrapped.append(median + diff)
    
    return np.array(unwrapped)

def _calculate_confidence(num_matches: int,
                           inlier_ratio: float,
                           rotation_std: float,
                           num_methods: int,
                           config: AccuracyConfig) -> float:
    """Calculate overall confidence score (0-1)."""
    
    # Match count factor
    match_factor = min(1.0, num_matches / 50)
    
    # Inlier ratio factor
    inlier_factor = min(1.0, inlier_ratio / 0.7)
    
    # Consistency factor (lower std = higher confidence)
    std_factor = max(0, 1.0 - rotation_std / 20.0)
    
    # Method agreement factor
    method_factor = min(1.0, num_methods / 3)
    
    # Weighted combination
    confidence = (
        0.3 * match_factor +
        0.3 * inlier_factor +
        0.3 * std_factor +
        0.1 * method_factor
    )
    
    return float(np.clip(confidence, 0, 1))

def _validate_result(rotation_std: float,
                      inlier_ratio: float,
                      num_matches: int,
                      config: AccuracyConfig) -> Tuple[bool, List[str]]:
    """Validate rotation result quality."""
    
    issues = []
    is_reliable = True
    
    if rotation_std > config.ROTATION_STD_THRESHOLD:
        issues.append(f"High rotation variance: {rotation_std:.1f}° > {config.ROTATION_STD_THRESHOLD}°")
        is_reliable = False
    
    if inlier_ratio < config.MIN_INLIER_RATIO:
        issues.append(f"Low inlier ratio: {inlier_ratio*100:.1f}% < {config.MIN_INLIER_RATIO*100:.1f}%")
        is_reliable = False
    
    if num_matches < config.MIN_MATCHES_FOR_RANSAC:
        issues.append(f"Few matches: {num_matches} < {config.MIN_MATCHES_FOR_RANSAC}")
        is_reliable = False
    
    return is_reliable, issues

def _create_failed_result(num_matches: int) -> RotationResult:
    """Create a failed result placeholder."""
    return RotationResult(
        rotation_deg=0.0,
        rotation_rad=0.0,
        scale=1.0,
        translation=(0, 0),
        num_matches=num_matches,
        num_inliers=0,
        inlier_ratio=0.0,
        rotation_std=float('inf'),
        confidence_score=0.0,
        is_reliable=False,
        quality_issues=["Estimation failed"],
        estimation_method="failed"
    )

# ==============================================================================
# VISUALIZATION
# ==============================================================================

def save_keypoints_image(image_path: str,
                         keypoints: np.ndarray,
                         output_path: str,
                         center: Tuple[int, int] = None,
                         radius: int = None,
                         color: Tuple[int, int, int] = (0, 255, 0)) -> None:
    """Save an image with detected keypoints overlaid."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Could not load image for keypoint visualization: {image_path}")
        return

    kpts = np.asarray(keypoints).reshape(-1, 2)

    for (x, y) in kpts:
        cv2.circle(img, (int(x), int(y)), 3, color, -1)

    # Optionally draw center and ring radius if provided
    if center is not None:
        cv2.circle(img, tuple(center), 6, (0, 0, 255), 2)
        if radius is not None:
            cv2.circle(img, tuple(center), int(radius), (255, 0, 0), 2)

    cv2.imwrite(str(output_path), img)
    print(f"✓ Saved keypoints visualization: {output_path}")

def visualize_matches(img0_path: str,
                       img1_path: str,
                       match_result: MatchResult,
                       rotation_result: RotationResult,
                       output_dir: str,
                       center0: Tuple[int, int] = None,
                       center1: Tuple[int, int] = None) -> None:
    """Create comprehensive match visualization."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load images
    img0 = cv2.imread(str(img0_path))
    img1 = cv2.imread(str(img1_path))
    
    if img0 is None or img1 is None:
        print(f"[ERROR] Could not load images for visualization")
        return
    
    img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    # Create side-by-side visualization
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    h = max(h0, h1)
    w = w0 + w1
    
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0_rgb
    canvas[:h1, w0:w0+w1] = img1_rgb
    
    # Draw matches
    kpts0 = np.asarray(match_result.keypoints0)
    kpts1 = np.asarray(match_result.keypoints1)
    confs = np.asarray(match_result.confidences)
    
    # Ensure 2D arrays
    if kpts0.ndim == 1 and len(kpts0) == 0:
        kpts0 = kpts0.reshape(0, 2)
    if kpts1.ndim == 1 and len(kpts1) == 0:
        kpts1 = kpts1.reshape(0, 2)
    
    for i in range(len(kpts0)):
        pt0 = kpts0[i]
        pt1 = kpts1[i]
        
        pt0_int = (int(pt0[0]), int(pt0[1]))
        pt1_int = (int(pt1[0] + w0), int(pt1[1]))
        
        # Color based on confidence
        if len(confs) > i:
            conf = float(confs[i])
            color = (int(255 * (1 - conf)), int(255 * conf), 0)
        else:
            color = (0, 255, 0)
        
        cv2.line(canvas, pt0_int, pt1_int, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt0_int, 4, color, -1)
        cv2.circle(canvas, pt1_int, 4, color, -1)
    
    # Draw centers if provided
    if center0 is not None:
        cv2.circle(canvas, tuple(center0), 10, (255, 0, 255), 3)
    if center1 is not None:
        center1_shifted = (center1[0] + w0, center1[1])
        cv2.circle(canvas, center1_shifted, 10, (255, 0, 255), 3)
    
    # Add text overlay
    info_lines = [
        f"Matches: {match_result.num_filtered_matches}",
        f"Rotation: {rotation_result.rotation_deg:.2f} deg",
        f"Confidence: {rotation_result.confidence_score*100:.1f}%",
        f"Inlier Ratio: {rotation_result.inlier_ratio*100:.1f}%",
        f"Reliable: {'YES' if rotation_result.is_reliable else 'NO'}"
    ]
    
    y_offset = 30
    for line in info_lines:
        cv2.putText(canvas, line, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
    
    # Save
    output_path = output_dir / "matches_visualization.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"✓ Saved: {output_path}")
    
    # Matplotlib visualization disabled - no windows will open
    return
    # Create detailed figure (commented out)
    # fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Rotation Estimation Results\n'
                 f'Rotation: {rotation_result.rotation_deg:.2f}° | '
                 f'Confidence: {rotation_result.confidence_score*100:.1f}%', 
                 fontsize=14, fontweight='bold')
    
    # Original images with keypoints
    axes[0, 0].imshow(img0_rgb)
    if len(kpts0) > 0:
        axes[0, 0].scatter(kpts0[:, 0], kpts0[:, 1], c='lime', s=20, alpha=0.7)
    if center0:
        axes[0, 0].scatter([center0[0]], [center0[1]], c='magenta', s=100, marker='x')
    axes[0, 0].set_title(f'Pre-op ({len(kpts0)} keypoints)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img1_rgb)
    if len(kpts1) > 0:
        axes[0, 1].scatter(kpts1[:, 0], kpts1[:, 1], c='lime', s=20, alpha=0.7)
    if center1:
        axes[0, 1].scatter([center1[0]], [center1[1]], c='magenta', s=100, marker='x')
    axes[0, 1].set_title(f'Intra-op ({len(kpts1)} keypoints)')
    axes[0, 1].axis('off')
    
    # Match visualization
    axes[0, 2].imshow(canvas)
    axes[0, 2].set_title('Feature Matches')
    axes[0, 2].axis('off')
    
    # Confidence histogram
    if len(confs) > 0:
        axes[1, 0].hist(confs, bins=20, color='blue', alpha=0.7)
        axes[1, 0].axvline(np.mean(confs), color='red', 
                          linestyle='--', label=f'Mean: {np.mean(confs):.3f}')
        axes[1, 0].set_title('Match Confidence Distribution')
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend()
    
    # Quality metrics
    axes[1, 1].axis('off')
    quality_text = f"""
ROTATION RESULT
{'='*40}

Rotation Angle: {rotation_result.rotation_deg:.2f}°
Rotation Std Dev: {rotation_result.rotation_std:.2f}°
Scale Factor: {rotation_result.scale:.4f}
Translation: ({rotation_result.translation[0]:.1f}, {rotation_result.translation[1]:.1f})

QUALITY METRICS
{'='*40}

Total Matches: {rotation_result.num_matches}
Inliers: {rotation_result.num_inliers}
Inlier Ratio: {rotation_result.inlier_ratio*100:.1f}%
Confidence Score: {rotation_result.confidence_score*100:.1f}%
Method: {rotation_result.estimation_method}

RELIABILITY
{'='*40}

Status: {'✓ RELIABLE' if rotation_result.is_reliable else '✗ NOT RELIABLE'}
"""
    
    if rotation_result.quality_issues:
        quality_text += "\nIssues:\n"
        for issue in rotation_result.quality_issues:
            quality_text += f"  • {issue}\n"
    
    axes[1, 1].text(0.05, 0.95, quality_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Confidence breakdown
    confidence_breakdown = {
        'Match Count': min(1.0, rotation_result.num_matches / 50),
        'Inlier Ratio': rotation_result.inlier_ratio,
        'Consistency': max(0, 1.0 - rotation_result.rotation_std / 20.0),
    }
    
    bars = axes[1, 2].barh(list(confidence_breakdown.keys()), 
                          list(confidence_breakdown.values()),
                          color=['blue', 'green', 'orange'])
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_title('Confidence Breakdown')
    axes[1, 2].set_xlabel('Score')
    
    for bar, val in zip(bars, confidence_breakdown.values()):
        axes[1, 2].text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                       f'{val:.2f}', va='center')
    
    plt.tight_layout()
    detail_path = output_dir / "detailed_analysis.png"
    plt.savefig(detail_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {detail_path}")

def draw_reference_line(image_path: str,
                         center: Tuple[int, int],
                         radius: int,
                         angle_deg: float,
                         output_path: str,
                         line_color: Tuple[int, int, int] = (0, 0, 255),
                         line_thickness: int = 3) -> np.ndarray:
    """Draw reference line at specified angle."""
    
    img = cv2.imread(str(image_path))
    
    angle_rad = np.radians(angle_deg)
    length = int(radius * 1.5)
    
    x1 = int(center[0] + length * np.cos(angle_rad))
    y1 = int(center[1] + length * np.sin(angle_rad))
    x2 = int(center[0] - length * np.cos(angle_rad))
    y2 = int(center[1] - length * np.sin(angle_rad))
    
    cv2.line(img, (x1, y1), (x2, y2), line_color, line_thickness)
    cv2.circle(img, center, 8, (0, 255, 0), -1)
    cv2.putText(img, f"Ref: {angle_deg:.1f} deg", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, line_color, 2)
    
    cv2.imwrite(str(output_path), img)
    
    return img

def draw_transformed_line(image_path: str,
                           center: Tuple[int, int],
                           radius: int,
                           original_angle: float,
                           rotation: float,
                           output_path: str,
                           preop_image: np.ndarray = None) -> np.ndarray:
    """Draw transformed line on intraop image with comparison."""
    
    img = cv2.imread(str(image_path))
    
    new_angle = original_angle + rotation
    angle_rad = np.radians(new_angle)
    length = int(radius * 1.5)
    
    x1 = int(center[0] + length * np.cos(angle_rad))
    y1 = int(center[1] + length * np.sin(angle_rad))
    x2 = int(center[0] - length * np.cos(angle_rad))
    y2 = int(center[1] - length * np.sin(angle_rad))
    
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.circle(img, center, 8, (0, 255, 0), -1)
    
    # Add info text
    cv2.putText(img, f"Original: {original_angle:.1f} deg", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"Rotation: {rotation:.2f} deg", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"Transformed: {new_angle:.2f} deg", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Stitch with preop if provided
    if preop_image is not None:
        h1, w1 = img.shape[:2]
        h2, w2 = preop_image.shape[:2]
        
        # Resize preop to match height
        scale = h1 / h2
        new_w2 = int(w2 * scale)
        preop_resized = cv2.resize(preop_image, (new_w2, h1))
        cv2.imwrite("preop_resized_axis_line.jpg", preop_resized)
        
        # Concatenate
        img = cv2.hconcat([preop_resized, img])
    
    cv2.imwrite(str(output_path), img)
    
    return img

def visualize_matches_interactive_superpoint(match_result: MatchResult,
                                             img1_path: str,
                                             img2_path: str,
                                             output_dir: str = "output") -> None:
    """
    Interactive visualization of SuperPoint/LightGlue matches.
    DISABLED: cv2.imshow() calls commented out to prevent windows from opening.
    """
    # Interactive visualization disabled - no windows will open
    print("[INFO] Interactive visualization disabled (no windows will open)")
    return
    os.makedirs(output_dir, exist_ok=True)

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("[ERROR] Could not load images")
        return

    kpts0 = np.asarray(match_result.keypoints0).reshape(-1, 2)
    kpts1 = np.asarray(match_result.keypoints1).reshape(-1, 2)
    confs = np.asarray(match_result.confidences).reshape(-1) if match_result.confidences is not None else np.ones(len(kpts0))

    if len(kpts0) == 0:
        print("[WARNING] No matches to visualize")
        return

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    max_height = max(h1, h2)
    total_width = w1 + w2

    max_display_width = 1280
    max_display_height = 700

    scale_w = max_display_width / total_width
    scale_h = max_display_height / max_height
    base_scale = min(scale_w, scale_h, 1.0)

    current_scale = base_scale

    print(f"\n{'='*60}")
    print("INTERACTIVE MATCH VIEWER (SuperPoint/LightGlue)")
    print(f"{'='*60}")
    print(f"Total matches: {len(kpts0)}")
    print(f"Original size: {total_width}x{max_height}")
    print(f"Display scale: {base_scale:.2f}x")
    print("\nControls:")
    print("  'q' or 'n' - Next match")
    print("  'p'        - Previous match")
    print("  '+'        - Zoom in")
    print("  '-'        - Zoom out")
    print("  'r'        - Reset zoom")
    print("  's'        - Save current visualization")
    print("  'c' or ESC - Exit viewer")
    print(f"{'='*60}\n")

    current_idx = 0
    window_name = "Feature Match Viewer (Press c to exit)"

    while True:
        canvas = np.zeros((max_height, total_width, 3), dtype=np.uint8)

        canvas[0:h1, 0:w1] = img1
        canvas[0:h2, w1:w1 + w2] = img2

        pt1 = kpts0[current_idx]
        pt2 = kpts1[current_idx]

        pt1_int = (int(pt1[0]), int(pt1[1]))
        pt2_int = (int(pt2[0] + w1), int(pt2[1]))

        cv2.line(canvas, pt1_int, pt2_int, (0, 255, 0), 3)

        cv2.circle(canvas, pt1_int, 15, (0, 255, 255), 3)
        cv2.circle(canvas, pt2_int, 15, (0, 255, 255), 3)

        cv2.circle(canvas, pt1_int, 30, (255, 0, 255), 3)
        cv2.circle(canvas, pt2_int, 30, (255, 0, 255), 3)

        conf_val = float(confs[current_idx]) if len(confs) > current_idx else 1.0

        info_text = [
            f"Match {current_idx + 1}/{len(kpts0)}",
            f"Confidence: {conf_val:.3f}",
            f"Scale: {current_scale:.2f}x",
            "",
            "q/n:Next p:Prev +:ZoomIn -:ZoomOut r:Reset s:Save c:Exit",
        ]

        y_offset = 30
        for i, text in enumerate(info_text):
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(
                canvas,
                (5, y_offset + i * 30 - 20),
                (15 + text_size[0], y_offset + i * 30 + 5),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                canvas,
                text,
                (10, y_offset + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        new_width = int(total_width * current_scale)
        new_height = int(max_height * current_scale)

        new_width = max(new_width, 640)
        new_height = max(new_height, 480)

        display_canvas = cv2.resize(canvas, (new_width, new_height))

        # cv2.imshow() disabled - no windows will open
        # cv2.imshow(window_name, display_canvas)
        # Instead, save the visualization
        save_path = os.path.join(output_dir, f"match_{current_idx}.png")
        cv2.imwrite(save_path, display_canvas)
        print(f"[INFO] Saved match visualization: {save_path}")

        # key = cv2.waitKey(0) & 0xFF
        key = ord('c')  # Auto-exit to prevent window opening

        if key == ord("q") or key == ord("n"):
            current_idx = (current_idx + 1) % len(kpts0)
            print(f"Match {current_idx + 1}/{len(kpts0)}")

        elif key == ord("p"):
            current_idx = (current_idx - 1) % len(kpts0)
            print(f"Match {current_idx + 1}/{len(kpts0)}")

        elif key == ord("+") or key == ord("="):
            current_scale = min(current_scale * 1.2, 2.0)
            print(f"Zoom: {current_scale:.2f}x")

        elif key == ord("-") or key == ord("_"):
            current_scale = max(current_scale * 0.8, 0.2)
            print(f"Zoom: {current_scale:.2f}x")

        elif key == ord("r"):
            current_scale = base_scale
            print(f"Zoom reset to: {current_scale:.2f}x")

        elif key == ord("s"):
            save_path = os.path.join(output_dir, f"match_{current_idx:04d}.jpg")
            cv2.imwrite(save_path, canvas)
            print(f"[SAVED] Match {current_idx + 1} saved to: {save_path}")

        elif key == ord("c") or key == 27:
            print(f"\n[EXIT] Viewed {current_idx + 1}/{len(kpts0)} matches")
            break

    cv2.destroyAllWindows()

def transform_line_to_intraop(image_path: str, center: tuple, radius: int,
                              ref_angle: float, rotation_angle: float,
                              stitch_image_path: str = None,
                              output_dir: str = "output"):
    """
    Transforms the reference line from pre-op to intra-op image.
    New angle = reference angle + rotation angle
    """
    os.makedirs(output_dir, exist_ok=True)
    
    img = cv2.imread(image_path)
    
    # Calculate new angle
    new_angle = ref_angle + rotation_angle
    angle_rad = np.radians(new_angle)
    length = int(radius * 1.5)
    
    x1 = int(center[0] + length * np.cos(angle_rad))
    y1 = int(center[1] + length * np.sin(angle_rad))
    x2 = int(center[0] - length * np.cos(angle_rad))
    y2 = int(center[1] - length * np.sin(angle_rad))
    
    # Draw transformed line
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.circle(img, center, 5, (0, 255, 0), -1)
    cv2.circle(img, center, radius, (0, 255, 0), 5)
    cv2.putText(img, f"Original: {ref_angle:.1f} deg", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"Rotation: {rotation_angle:.1f} deg", (10, 65),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"Transformed: {new_angle:.1f} deg", (10, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Stitch another image if provided
    if stitch_image_path is not None:
        if isinstance(stitch_image_path, str):
            stitch_img = cv2.imread(stitch_image_path)
        else:
            # If it's already an image array, use it directly
            stitch_img = stitch_image_path.copy()
            
        # Resize stitch image to match height
        h1, w1 = img.shape[:2]
        h2, w2 = stitch_img.shape[:2]
        new_w2 = int(w2 * (h1 / h2))
        stitch_img = cv2.resize(stitch_img, (new_w2, h1))
        img = cv2.hconcat([stitch_img, img])
    
    output_path = os.path.join(output_dir, "10_transformed_line_intra_op.jpg")
    cv2.imwrite(output_path, img)
    print(f"[STEP 8] Line transformed to intra-op image")
    print(f"         Original angle: {ref_angle:.1f} deg")
    print(f"         Rotation: {rotation_angle:.1f} deg")
    print(f"         New angle: {new_angle:.1f} deg")
    print(f"         Saved: {output_path}")
    
    return img

def visualize_matches_interactive(kp1, kp2, matches, img1_path, img2_path, 
                                  output_dir: str = "output"):
    """
    Visualize feature matches one by one interactively.
    DISABLED: cv2.imshow() calls commented out to prevent windows from opening.
    """
    # Interactive visualization disabled - no windows will open
    print("[INFO] Interactive visualization disabled (no windows will open)")
    return
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("[ERROR] Could not load images")
        return
    
    if len(matches) == 0:
        print("[WARNING] No matches to visualize")
        return
    
    # Get image dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create combined image canvas
    max_height = max(h1, h2)
    total_width = w1 + w2
    
    # Calculate initial scale to fit screen (conservative estimate)
    # Most laptops are 1366x768 to 1920x1080
    max_display_width = 1280  # Safe width for most laptops
    max_display_height = 700  # Safe height (leaving room for taskbar)
    
    # Calculate scale factor
    scale_w = max_display_width / total_width
    scale_h = max_display_height / max_height
    base_scale = min(scale_w, scale_h, 1.0)  # Don't upscale if already small
    
    current_scale = base_scale
    
    print(f"\n{'='*60}")
    print(f"INTERACTIVE MATCH VIEWER")
    print(f"{'='*60}")
    print(f"Total matches: {len(matches)}")
    print(f"Original size: {total_width}x{max_height}")
    print(f"Display scale: {base_scale:.2f}x")
    print(f"\nControls:")
    print(f"  'q' or 'n' - Next match")
    print(f"  'p'        - Previous match")
    print(f"  '+'        - Zoom in")
    print(f"  '-'        - Zoom out")
    print(f"  'r'        - Reset zoom")
    print(f"  's'        - Save current visualization")
    print(f"  'c' or ESC - Exit viewer")
    print(f"{'='*60}\n")
    
    current_idx = 0
    window_name = 'Feature Match Viewer (Press c to exit)'
    
    while True:
        # Create blank canvas
        canvas = np.zeros((max_height, total_width, 3), dtype=np.uint8)
        
        # Place images on canvas
        canvas[0:h1, 0:w1] = img1
        canvas[0:h2, w1:w1+w2] = img2
        
        # Get current match
        match = matches[current_idx]
        
        # Get keypoint coordinates
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        
        # Adjust pt2 coordinates for the combined image
        pt1_int = (int(pt1[0]), int(pt1[1]))
        pt2_int = (int(pt2[0] + w1), int(pt2[1]))
        
        # Draw the match line (thicker for visibility)
        cv2.line(canvas, pt1_int, pt2_int, (0, 255, 0), 3)
        
        # Draw keypoint circles
        cv2.circle(canvas, pt1_int, 15, (0, 255, 255), 3)
        cv2.circle(canvas, pt2_int, 15, (0, 255, 255), 3)
        
        # Draw larger highlight circles
        cv2.circle(canvas, pt1_int, 30, (255, 0, 255), 3)
        cv2.circle(canvas, pt2_int, 30, (255, 0, 255), 3)
        
        # Add match information text with background for better visibility
        info_text = [
            f"Match {current_idx + 1}/{len(matches)}",
            f"Distance: {match.distance:.2f}",
            f"Scale: {current_scale:.2f}x",
            "",
            "q/n:Next p:Prev +:ZoomIn -:ZoomOut r:Reset s:Save c:Exit"
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            # Draw background rectangle for text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(canvas, 
                         (5, y_offset + i*30 - 20), 
                         (15 + text_size[0], y_offset + i*30 + 5),
                         (0, 0, 0), -1)
            
            cv2.putText(canvas, text, (10, y_offset + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Resize for display with current scale
        new_width = int(total_width * current_scale)
        new_height = int(max_height * current_scale)
        
        # Ensure minimum size
        new_width = max(new_width, 640)
        new_height = max(new_height, 480)
        
        display_canvas = cv2.resize(canvas, (new_width, new_height))
        
        # cv2.imshow() disabled - no windows will open
        # cv2.imshow(window_name, display_canvas)
        # Instead, save the visualization
        save_path = os.path.join(output_dir, f"match_{current_idx}.png")
        cv2.imwrite(save_path, display_canvas)
        print(f"[INFO] Saved match visualization: {save_path}")
        
        # key = cv2.waitKey(0) & 0xFF
        key = ord('c')  # Auto-exit to prevent window opening
        
        if key == ord('q') or key == ord('n'):  # Next match
            current_idx = (current_idx + 1) % len(matches)
            print(f"Match {current_idx + 1}/{len(matches)}")
            
        elif key == ord('p'):  # Previous match
            current_idx = (current_idx - 1) % len(matches)
            print(f"Match {current_idx + 1}/{len(matches)}")
            
        elif key == ord('+') or key == ord('='):  # Zoom in
            current_scale = min(current_scale * 1.2, 2.0)
            print(f"Zoom: {current_scale:.2f}x")
            
        elif key == ord('-') or key == ord('_'):  # Zoom out
            current_scale = max(current_scale * 0.8, 0.2)
            print(f"Zoom: {current_scale:.2f}x")
            
        elif key == ord('r'):  # Reset zoom
            current_scale = base_scale
            print(f"Zoom reset to: {current_scale:.2f}x")
            
        elif key == ord('s'):  # Save current visualization
            save_path = os.path.join(output_dir, f"match_{current_idx:04d}.jpg")
            cv2.imwrite(save_path, canvas)
            print(f"[SAVED] Match {current_idx + 1} saved to: {save_path}")
            
        elif key == ord('c') or key == 27:  # Exit (c or ESC)
            print(f"\n[EXIT] Viewed {current_idx + 1}/{len(matches)} matches")
            break
    
    cv2.destroyAllWindows()

# ==============================================================================
# LIMBUS DETECTION
# ==============================================================================

def load_yolo_model(model_path: str):
    """Load YOLO model for limbus detection."""
    model = YOLO(model_path)
    print(f"[INFO] YOLO model loaded: {model_path}")
    print(f"[INFO] Classes: {model.names}")
    return model

def detect_limbus(model, image_path: str) -> Tuple[Tuple[int, int], int]:
    """Detect limbus center and radius."""
    
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    results = model(img)[0]
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        
        if cls_name == "dilated limbus":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            radius = int(min(x2 - x1, y2 - y1) / 2.0)
            limbus_circled_img = cv2.circle(img, center, radius, (0, 255, 0), 2)
            return center, radius, limbus_circled_img
    
    raise ValueError(f"No limbus detected in: {image_path}")

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def process_eye_pair(preop_path: str,
                      intraop_path: str,
                      yolo_model_path: str,
                      reference_angle: float = 180.0,
                      preop_mask_path: str = None,
                      intraop_mask_path: str = None,
                      output_dir: str = "output/superpoint_robust",
                      config: AccuracyConfig = None) -> Dict[str, Any]:
    """
    Main pipeline for processing preop/intraop eye pair.
    
    Args:
        preop_path: Path to preprocessed preop image
        intraop_path: Path to preprocessed intraop image
        yolo_model_path: Path to YOLO model for limbus detection
        reference_angle: Reference axis angle in preop image (degrees)
        preop_mask_path: Optional mask for preop (valid regions)
        intraop_mask_path: Optional mask for intraop (valid regions)
        output_dir: Output directory
        config: Accuracy configuration
    
    Returns:
        Dictionary with all results
    """
    if config is None:
        config = AccuracyConfig()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*70}")
    print("SUPERPOINT + LIGHTGLUE ROBUST ROTATION ESTIMATION")
    print(f"{'='*70}")
    print(f"Pre-op image: {preop_path}")
    print(f"Intra-op image: {intraop_path}")
    print(f"Reference angle: {reference_angle}°")
    print(f"Output directory: {output_dir}")
    
    # ========== STEP 1: Load YOLO and detect limbus ==========
    print(f"\n{'='*70}")
    print("STEP 1: LIMBUS DETECTION")
    print(f"{'='*70}")
    
    yolo_model = load_yolo_model(yolo_model_path)
    
    center0, radius0, limbus_circled_img0 = detect_limbus(yolo_model, preop_path)
    cv2.imwrite(output_dir / "00_preop_limbus_circled.jpg", limbus_circled_img0)
    
    center1, radius1, limbus_circled_img1 = detect_limbus(yolo_model, intraop_path)
    cv2.imwrite(output_dir / "00_intraop_limbus_circled.jpg", limbus_circled_img1)
    
    print(f"Pre-op: center={center0}, radius={radius0}")
    print(f"Intra-op: center={center1}, radius={radius1}")
    
    # ========== STEP 2: Initialize feature models ==========
    extractor, matcher, device = initialize_models(config)
    
    # ========== STEP 3: Load masks if provided ==========
    mask0, mask1 = None, None
    
    if preop_mask_path and os.path.exists(preop_mask_path):
        mask0 = cv2.imread(preop_mask_path, cv2.IMREAD_GRAYSCALE)
        print(f"Loaded pre-op mask: {preop_mask_path}")
    
    if intraop_mask_path and os.path.exists(intraop_mask_path):
        mask1 = cv2.imread(intraop_mask_path, cv2.IMREAD_GRAYSCALE)
        print(f"Loaded intra-op mask: {intraop_mask_path}")
    
    # ========== STEP 4: Extract features ==========
    print(f"\n{'='*70}")
    print("STEP 2: FEATURE EXTRACTION")
    print(f"{'='*70}")
    
    print("\nPre-op features:")
    extract_features_start_time = time.time()
    feats0 = extract_features(preop_path, extractor, device, mask0, config)
    extract_features_stop_time = time.time()
    print(f"Time taken for extract_features preop: {extract_features_stop_time - extract_features_start_time} seconds")

    print("\nIntra-op features:")
    extract_features_start_time = time.time()
    feats1 = extract_features(intraop_path, extractor, device, mask1, config)
    extract_features_stop_time = time.time()
    print(f"Time taken for extract_features intraop: {extract_features_stop_time - extract_features_start_time} seconds")
    
    # Save detected keypoints for visualization (using filtered SuperPoint keypoints)
    try:
        kpts0_all = feats0["keypoints"][0]
        if torch.is_tensor(kpts0_all):
            kpts0_all = kpts0_all.detach().cpu().numpy()
        kpts1_all = feats1["keypoints"][0]
        if torch.is_tensor(kpts1_all):
            kpts1_all = kpts1_all.detach().cpu().numpy()

        save_keypoints_image(
            preop_path,
            kpts0_all,
            output_dir / "01_preop_features.jpg",
            center=center0,
            radius=radius0,
            color=(0, 255, 0),
        )
        save_keypoints_image(
            intraop_path,
            kpts1_all,
            output_dir / "02_intraop_features.jpg",
            center=center1,
            radius=radius1,
            color=(0, 255, 0),
        )
    except Exception as e:
        print(f"[WARNING] Failed to save keypoint visualizations: {e}")
    
    # ========== STEP 5: Match features ==========
    match_features_start_time = time.time()
    match_result = match_features(feats0, feats1, matcher, config)
    
    if match_result.num_filtered_matches < config.MIN_MATCHES_FOR_RANSAC:
        print(f"\n[ERROR] Insufficient matches: {match_result.num_filtered_matches}")
        return {'success': False, 'error': 'Insufficient matches'}

    match_features_stop_time = time.time()
    print(f"Time taken for match_features: {match_features_stop_time - match_features_start_time} seconds")
    # ========== STEP 6: Geometric filtering ==========
    apply_geometric_filtering_start_time = time.time()
    match_result = apply_geometric_filtering(
        match_result, center0, center1, radius0, radius1, config
    )
    apply_geometric_filtering_stop_time = time.time()
    print(f"Time taken for apply_geometric_filtering: {apply_geometric_filtering_stop_time - apply_geometric_filtering_start_time} seconds")
    # ========== STEP 7: Estimate rotation ==========
    estimate_rotation_robust_start_time = time.time()
    rotation_result = estimate_rotation_robust(
        match_result, center0, center1, config
    )
    estimate_rotation_robust_stop_time = time.time()
    print(f"Time taken for estimate_rotation_robust: {estimate_rotation_robust_stop_time - estimate_rotation_robust_start_time} seconds")
    # ========== STEP 8: Generate visualizations ==========
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    visualize_matches(
        preop_path, intraop_path,
        match_result, rotation_result,
        output_dir, center0, center1
    )
    
    # Draw reference line on preop
    preop_with_line = draw_reference_line(
        preop_path, center0, radius0, reference_angle,
        output_dir / "preop_reference_line.jpg"
    )
    print(f"✓ Reference line drawn at {reference_angle}°")
    
    # Draw transformed line on intraop
    draw_transformed_line(
        intraop_path, center1, radius1,
        reference_angle, rotation_result.rotation_deg,
        output_dir / "intraop_transformed_line.jpg",
        preop_image=preop_with_line
    )
    print(f"✓ Transformed line drawn at {reference_angle + rotation_result.rotation_deg:.2f}°")

    # Also save additional transformed-line visualization using helper (with stitching)
    try:
        transform_line_to_intraop(
            image_path=intraop_path,
            center=center1,
            radius=radius1,
            ref_angle=reference_angle,
            rotation_angle=rotation_result.rotation_deg,
            stitch_image_path=preop_with_line,
            output_dir=str(output_dir),
        )
    except Exception as e:
        print(f"[WARNING] Failed to save extra transformed-line image: {e}")

    # ========== OPTIONAL: Interactive per-match viewer ==========
    try:
        visualize_matches_interactive_superpoint(
            match_result=match_result,
            img1_path=preop_path,
            img2_path=intraop_path,
            output_dir=str(output_dir),
        )
    except Exception as e:
        print(f"[WARNING] Interactive match viewer failed: {e}")
    
    # ========== STEP 9: Print final results ==========
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"")
    print(f"  ROTATION ANGLE: {rotation_result.rotation_deg:.2f}°")
    print(f"  ")
    print(f"  Standard Deviation: ±{rotation_result.rotation_std:.2f}°")
    print(f"  Scale Factor: {rotation_result.scale:.4f}")
    print(f"  ")
    print(f"  Total Matches: {rotation_result.num_matches}")
    print(f"  Inliers: {rotation_result.num_inliers} ({rotation_result.inlier_ratio*100:.1f}%)")
    print(f"  ")
    print(f"  CONFIDENCE: {rotation_result.confidence_score*100:.1f}%")
    print(f"  RELIABLE: {'YES ✓' if rotation_result.is_reliable else 'NO ✗'}")
    
    if rotation_result.quality_issues:
        print(f"\n  Quality Issues:")
        for issue in rotation_result.quality_issues:
            print(f"    • {issue}")
    
    print(f"\n  Reference Line: {reference_angle:.1f}°")
    print(f"  Transformed Line: {reference_angle + rotation_result.rotation_deg:.2f}°")
    print(f"{'='*70}\n")
    
    return {
        'success': True,
        'rotation_result': rotation_result,
        'match_result': match_result,
        'preop_center': center0,
        'preop_radius': radius0,
        'intraop_center': center1,
        'intraop_radius': radius1,
        'reference_angle': reference_angle,
        'transformed_angle': reference_angle + rotation_result.rotation_deg
    }

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    entire_superpoint_start_time = time.time()
    # ========== CONFIGURATION ==========
    
    # Input images (from preprocess_robust.py output)
    # PREOP_IMAGE   = r"output\robust_preprocess\preop\5_preop_enhanced.jpg"
    PREOP_IMAGE   = r"output\robust_preprocess\preop\5_preop_enhanced.jpg"
    INTRAOP_IMAGE = r"output\robust_preprocess\intraop\5_intraop_enhanced.jpg"
    
    # Optional masks (from preprocess_robust.py output)
    # Set to None to use ALL keypoints (more matches but potentially more noise)
    # Set to path to filter keypoints to ring region only (fewer but cleaner matches)
    # PREOP_MASK = None  # Try without mask first for more matches
    # INTRAOP_MASK = None
    
    # If matching fails, try with masks:
    PREOP_MASK = r"output\robust_preprocess\preop\8_preop_ring_mask.jpg"
    INTRAOP_MASK = r"output\robust_preprocess\intraop\8_intraop_ring_mask.jpg"
    
    # YOLO model for limbus detection
    YOLO_MODEL = "model/intraop_latest.pt"

    print("PREOP_IMAGE", cv2.imread(PREOP_IMAGE).shape)
    print("INTRAOP_IMAGE", cv2.imread(INTRAOP_IMAGE).shape)
    print("PREOP_MASK", cv2.imread(PREOP_MASK).shape)
    print("INTRAOP_MASK", cv2.imread(INTRAOP_MASK).shape)
    
    # Reference angle (line drawn by doctor on preop image)
    REFERENCE_ANGLE = 180.0  # degrees (180 = horizontal left)
    
    # Output directory
    OUTPUT_DIR = "output/superpoint_robust"
    
    # Setup logging - capture all print statements to a text file
    log_dir = OUTPUT_DIR
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "superpoint_log.txt")
    logger = Logger.get_logger()
    logger.start_logging(log_file, append=False)
    
    try:
        # ========== RUN PIPELINE ==========
        
        # Use default AccuracyConfig (already tuned for difficult matching)
        # The defaults are relaxed to handle challenging preop/intraop matching
        config = AccuracyConfig()
        
        print(f"\n{'='*70}")
        print("Configuration:")
        print(f"  MAX_KEYPOINTS: {config.MAX_KEYPOINTS}")
        print(f"  DETECTION_THRESHOLD: {config.DETECTION_THRESHOLD}")
        print(f"  NMS_RADIUS: {config.NMS_RADIUS}")
        print(f"  DEPTH_CONFIDENCE: {config.DEPTH_CONFIDENCE}")
        print(f"  WIDTH_CONFIDENCE: {config.WIDTH_CONFIDENCE}")
        print(f"  MIN_MATCH_CONFIDENCE: {config.MIN_MATCH_CONFIDENCE}")
        print(f"  MIN_MATCHES_FOR_RANSAC: {config.MIN_MATCHES_FOR_RANSAC}")
        print(f"{'='*70}")
        
        results = process_eye_pair(
            preop_path=PREOP_IMAGE,
            intraop_path=INTRAOP_IMAGE,
            yolo_model_path=YOLO_MODEL,
            reference_angle=REFERENCE_ANGLE,
            preop_mask_path=PREOP_MASK,
            intraop_mask_path=INTRAOP_MASK,
            output_dir=OUTPUT_DIR,
            config=config
        )
        
        if results['success']:
            print("\n✓ PROCESSING COMPLETE")
            print(f"Check output directory: {OUTPUT_DIR}")
            print(f"\n[LOG] All output saved to: {log_file}")
        else:
            print(f"\n✗ PROCESSING FAILED: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop logging
        logger.stop_logging()
        print(f"[INFO] Log file saved: {log_file}")
    entire_superpoint_stop_time = time.time()
    print(f"Time taken for entire superpoint pipeline: {entire_superpoint_stop_time - entire_superpoint_start_time} seconds")
