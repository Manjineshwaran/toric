"""
Robust Preprocessing Pipeline for Eye Image Feature Matching
=============================================================

This module provides preprocessing for preoperative and intraoperative eye images
to enable accurate feature matching (SuperPoint, SIFT, ORB) for:
- Rotation angle computation
- Scale factor estimation  
- Toric lens axis alignment

Key Design Principles:
----------------------
1. PRESERVE FEATURES: Minimal texture distortion for reliable matching
2. NORMALIZE APPEARANCE: Handle different cameras, lighting, glare
3. HANDLE ARTIFACTS: Surgical instruments, fluids, reflections
4. QUALITY ASSESSMENT: Detect blur, occlusions, poor images

Pipeline Steps:
---------------
1. Limbus Detection (YOLO) → Center, radius extraction
2. Scale Normalization → Both images to same limbus radius
3. Glare Detection & Inpainting → Remove specular reflections
4. Conservative Enhancement → CLAHE in LAB space (preserve texture)
5. Histogram Matching → Match intraop to preop appearance
6. Artifact Detection → Mask surgical instruments/fluids
7. Quality Assessment → Blur, occlusion, feature density scoring
8. Ring Mask Generation → Focus on limbus region for matching

Author: Preprocessing for Toric Lens Surgery
"""

import os
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List, Any

# Matplotlib visualization disabled - no windows will open
# import matplotlib
# matplotlib.use('Agg')  # Non-interactive backend (no GUI, no windows)
# import matplotlib.pyplot as plt
plt = None  # Placeholder to avoid import errors
from enum import Enum
import time
import sys
from datetime import datetime


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
# CONFIGURATION
# ==============================================================================

class Config:
    """Central configuration for preprocessing parameters"""
    
    # Scale normalization
    STANDARD_RADIUS = 400  # Normalized limbus radius in pixels
    
    # Cropping ratios (relative to limbus radius)
    INNER_EXCLUDE_RATIO = 0.8   # Inner circle exclusion (0.7 = include from 70% of limbus radius inward)
    OUTER_INCLUDE_RATIO = 1.3  # Outer circle boundary
    CROP_WIDTH_RATIO = 4.0       # Crop width = ratio * radius
    CROP_HEIGHT_RATIO = 3.0      # Crop height = ratio * radius
    
    # Glare detection
    GLARE_INTENSITY_THRESHOLD = 235  # High intensity = glare
    GLARE_KERNEL_SIZE = 5
    GLARE_DILATE_ITERATIONS = 2
    GLARE_INPAINT_RADIUS = 5
    
    # Conservative CLAHE (feature-preserving)
    CLAHE_CLIP_LIMIT = 1.5       # 1.2-2.0 (lower = safer)
    CLAHE_TILE_GRID = (16, 16)   # Larger = more global/conservative
    
    # Blur detection
    BLUR_LAPLACIAN_THRESHOLD = 100  # Below this = blurry
    
    # Artifact detection (surgical instruments)
    ARTIFACT_DARK_THRESHOLD = 30    # Very dark = instrument
    ARTIFACT_MIN_AREA = 500         # Minimum artifact size
    
    # Feature preservation
    BILATERAL_D = 5                 # Bilateral filter diameter
    BILATERAL_SIGMA_COLOR = 50      # Color sigma (higher = more smoothing)
    BILATERAL_SIGMA_SPACE = 50      # Space sigma
    
    # Quality thresholds
    MIN_ACCEPTABLE_BLUR_SCORE = 50
    MAX_ACCEPTABLE_GLARE_PERCENT = 15
    MAX_ACCEPTABLE_ARTIFACT_PERCENT = 20

    # Eyelid / eyelash trimming (vertical crop around limbus)
    # Keep only a horizontal band around limbus center:
    #   [center_y - upper_ratio*radius, center_y + lower_ratio*radius]
    EYELID_TRIM_UPPER_RATIO = 0.1   # Cut more above limbus (smaller value = more cut)
    EYELID_TRIM_LOWER_RATIO = 1.0   # Cut less below limbus (larger value = keep more)


class ImageType(Enum):
    """Image type enumeration"""
    PREOP = "preop"
    INTRAOP = "intraop"


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class LimbusInfo:
    """Limbus detection results"""
    center: Tuple[int, int]          # (x, y) center coordinates
    radius: int                       # Radius in pixels
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    confidence: float = 1.0           # Detection confidence


@dataclass
class QualityMetrics:
    """Image quality assessment metrics"""
    blur_score: float           # Laplacian variance (higher = sharper)
    glare_percentage: float     # Percentage of glare pixels
    artifact_percentage: float  # Percentage of artifact pixels
    mean_intensity: float       # Average brightness
    std_intensity: float        # Intensity standard deviation
    is_acceptable: bool = True  # Overall quality flag
    issues: List[str] = field(default_factory=list)


@dataclass
class PreprocessResult:
    """Complete preprocessing result"""
    original_image: np.ndarray
    processed_image: np.ndarray
    limbus_info: LimbusInfo
    scale_factor: float
    quality_metrics: QualityMetrics
    glare_mask: np.ndarray
    artifact_mask: np.ndarray
    ring_mask: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# YOLO MODEL LOADING
# ==============================================================================

def load_yolo_model(model_path: str):
    """Load YOLO model for limbus detection"""
    from ultralytics import YOLO
    model = YOLO(model_path)
    print(f"[INFO] Loaded YOLO model: {model_path}")
    print(f"[INFO] Model classes: {model.names}")
    return model


# ==============================================================================
# LIMBUS DETECTION
# ==============================================================================

def detect_limbus(model, image: np.ndarray, 
                  class_name: str = "dilated limbus") -> Optional[LimbusInfo]:
    """
    Detect limbus using YOLO model.
    
    Args:
        model: YOLO model
        image: BGR image array
        class_name: Target class name to detect
    
    Returns:
        LimbusInfo or None if not detected
    """
    results = model(image)[0]
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        
        if cls_name == class_name:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Calculate center and radius
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Use min dimension for radius (handle elliptical detections)
            radius = int(min(x2 - x1, y2 - y1) / 2.0)
            
            # Get confidence
            confidence = float(box.conf[0]) if hasattr(box, 'conf') else 1.0
            
            return LimbusInfo(
                center=(center_x, center_y),
                radius=radius,
                bbox=(x1, y1, x2, y2),
                confidence=confidence
            )
    
    return None


def detect_limbus_from_path(model, image_path: str) -> Optional[LimbusInfo]:
    """Detect limbus from image path"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return detect_limbus(model, img)


# ==============================================================================
# SCALE NORMALIZATION
# ==============================================================================

def normalize_scale(image: np.ndarray, 
                    limbus: LimbusInfo,
                    target_radius: int = Config.STANDARD_RADIUS
                    ) -> Tuple[np.ndarray, LimbusInfo, float]:
    """
    Normalize image so limbus radius becomes target_radius.
    
    CRITICAL: Uses appropriate interpolation to preserve features:
    - INTER_CUBIC for upscaling (smoother)
    - INTER_AREA for downscaling (better feature preservation)
    
    Args:
        image: Input BGR image
        limbus: Detected limbus info
        target_radius: Target radius after normalization
    
    Returns:
        (normalized_image, updated_limbus_info, scale_factor)
    """
    scale = target_radius / limbus.radius
    
    # Choose interpolation based on scaling direction
    if scale > 1:
        interp = cv2.INTER_CUBIC  # Upscaling
    else:
        interp = cv2.INTER_AREA   # Downscaling (best for feature preservation)
    
    # Resize image
    new_w = int(image.shape[1] * scale)
    new_h = int(image.shape[0] * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    
    # Update limbus info
    new_center = (int(limbus.center[0] * scale), int(limbus.center[1] * scale))
    new_bbox = tuple(int(v * scale) for v in limbus.bbox)
    
    new_limbus = LimbusInfo(
        center=new_center,
        radius=target_radius,
        bbox=new_bbox,
        confidence=limbus.confidence
    )
    
    return resized, new_limbus, scale


def crop_around_limbus(image: np.ndarray, 
                       limbus: LimbusInfo,
                       width_ratio: float = Config.CROP_WIDTH_RATIO,
                       height_ratio: float = Config.CROP_HEIGHT_RATIO
                       ) -> Tuple[np.ndarray, LimbusInfo]:
    """
    Crop fixed-size region around limbus center.
    
    Args:
        image: Input image
        limbus: Limbus info with center
        width_ratio: Crop width = ratio * radius
        height_ratio: Crop height = ratio * radius
    
    Returns:
        (cropped_image, updated_limbus_info)
    """
    cx, cy = limbus.center
    radius = limbus.radius
    
    crop_w = int(width_ratio * radius)
    crop_h = int(height_ratio * radius)
    
    # Calculate crop boundaries
    x1 = cx - crop_w // 2
    x2 = cx + crop_w // 2
    y1 = cy - crop_h // 2
    y2 = cy + crop_h // 2
    
    # Clamp to image boundaries
    h, w = image.shape[:2]
    x1_clamped = max(0, x1)
    x2_clamped = min(w, x2)
    y1_clamped = max(0, y1)
    y2_clamped = min(h, y2)
    
    # Crop
    cropped = image[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
    
    # Update limbus center relative to crop
    new_center = (cx - x1_clamped, cy - y1_clamped)
    
    new_limbus = LimbusInfo(
        center=new_center,
        radius=radius,
        bbox=(0, 0, cropped.shape[1], cropped.shape[0]),
        confidence=limbus.confidence
    )
    
    return cropped, new_limbus


# ==============================================================================
# GLARE DETECTION AND REMOVAL
# ==============================================================================

def detect_glare(image: np.ndarray,
                 threshold: int = Config.GLARE_INTENSITY_THRESHOLD,
                 kernel_size: int = Config.GLARE_KERNEL_SIZE,
                 dilate_iters: int = Config.GLARE_DILATE_ITERATIONS
                 ) -> Tuple[np.ndarray, float]:
    """
    Detect glare/specular reflections in image.
    
    Args:
        image: Input BGR image
        threshold: Intensity threshold for glare detection
        kernel_size: Morphological kernel size
        dilate_iters: Dilation iterations to expand mask
    
    Returns:
        (glare_mask, glare_percentage)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Threshold high-intensity regions
    _, glare_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_CLOSE, kernel)
    glare_mask = cv2.dilate(glare_mask, kernel, iterations=dilate_iters)
    
    # Calculate percentage
    glare_pixels = np.sum(glare_mask > 0)
    total_pixels = glare_mask.shape[0] * glare_mask.shape[1]
    glare_percentage = (glare_pixels / total_pixels) * 100
    
    return glare_mask, glare_percentage


def inpaint_glare(image: np.ndarray,
                  glare_mask: np.ndarray,
                  radius: int = Config.GLARE_INPAINT_RADIUS,
                  method: str = 'telea') -> np.ndarray:
    """
    Remove glare using inpainting.
    
    IMPORTANT: Inpainting can affect feature matching!
    Only use for small glare regions (<5% of image).
    For larger regions, consider masking instead.
    
    Args:
        image: Input BGR image
        glare_mask: Binary mask of glare regions
        radius: Inpainting neighborhood radius
        method: 'telea' (fast) or 'ns' (Navier-Stokes, slower but better)
    
    Returns:
        Inpainted image
    """
    if method == 'telea':
        flags = cv2.INPAINT_TELEA
    else:
        flags = cv2.INPAINT_NS
    
    # Ensure mask is uint8
    mask = glare_mask.astype(np.uint8)
    
    # Inpaint
    inpainted = cv2.inpaint(image, mask, radius, flags)
    
    return inpainted


def remove_glare_safe(image: np.ndarray,
                      threshold: int = Config.GLARE_INTENSITY_THRESHOLD,
                      max_inpaint_percent: float = 5.0
                      ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """
    Safely remove glare with feature preservation.
    
    Strategy:
    - Small glare (<5%): Inpaint (safe for features)
    - Large glare (>5%): Return mask only, don't inpaint
    
    Args:
        image: Input BGR image
        threshold: Glare detection threshold
        max_inpaint_percent: Maximum glare % to inpaint
    
    Returns:
        (processed_image, glare_mask, glare_percentage, was_inpainted)
    """
    glare_mask, glare_pct = detect_glare(image, threshold)
    
    if glare_pct > 0 and glare_pct <= max_inpaint_percent:
        # Safe to inpaint
        processed = inpaint_glare(image, glare_mask)
        return processed, glare_mask, glare_pct, True
    elif glare_pct > max_inpaint_percent:
        # Too much glare - don't inpaint, just return mask
        print(f"[WARN] High glare ({glare_pct:.1f}%) - skipping inpaint to preserve features")
        return image.copy(), glare_mask, glare_pct, False
    else:
        # No glare
        return image.copy(), glare_mask, glare_pct, False


# ==============================================================================
# CONSERVATIVE IMAGE ENHANCEMENT (FEATURE PRESERVING)
# ==============================================================================

def enhance_conservative(image: np.ndarray,
                         clip_limit: float = Config.CLAHE_CLIP_LIMIT,
                         tile_grid: Tuple[int, int] = Config.CLAHE_TILE_GRID
                         ) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Conservative CLAHE enhancement in LAB color space.
    
    CRITICAL FOR FEATURE MATCHING:
    - Only enhances luminance (L channel)
    - Preserves color/texture structure
    - Low clip limit prevents over-enhancement
    - Large tile grid provides smooth enhancement
    
    Args:
        image: Input BGR image
        clip_limit: CLAHE clip limit (1.2-2.0 recommended)
        tile_grid: CLAHE tile grid size (larger = more conservative)
    
    Returns:
        (enhanced_image, stats_dict)
    """
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel only
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l)
    
    # Merge and convert back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # Calculate change statistics
    gray_orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray_orig, gray_enh)
    
    stats = {
        'mean_change': float(np.mean(diff)),
        'max_change': float(np.max(diff)),
        'contrast_ratio': float(np.std(gray_enh) / (np.std(gray_orig) + 1e-6)),
        'original_std': float(np.std(gray_orig)),
        'enhanced_std': float(np.std(gray_enh))
    }
    
    # Warning for aggressive enhancement
    if stats['mean_change'] > 20:
        print(f"[WARN] Enhancement may affect features (mean change: {stats['mean_change']:.1f})")
    
    return enhanced, stats


def apply_bilateral_filter(image: np.ndarray,
                           d: int = Config.BILATERAL_D,
                           sigma_color: float = Config.BILATERAL_SIGMA_COLOR,
                           sigma_space: float = Config.BILATERAL_SIGMA_SPACE
                           ) -> np.ndarray:
    """
    Apply bilateral filter for noise reduction while preserving edges.
    
    FEATURE PRESERVATION:
    - Smooths homogeneous regions
    - Preserves edges (important for feature detection)
    - Conservative parameters to avoid texture loss
    
    Args:
        image: Input BGR image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space
    
    Returns:
        Filtered image
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


# ==============================================================================
# HISTOGRAM MATCHING
# ==============================================================================

def match_histograms(source: np.ndarray, 
                     reference: np.ndarray,
                     match_luminance_only: bool = True) -> np.ndarray:
    """
    Match histogram of source image to reference image.
    
    IMPORTANT FOR FEATURE MATCHING:
    - Normalizes appearance between preop and intraop
    - Handles different camera/lighting conditions
    - Matching luminance only is safer for color preservation
    
    Args:
        source: Source image (intraop) to be matched
        reference: Reference image (preop) to match to
        match_luminance_only: If True, only match L channel in LAB
    
    Returns:
        Histogram-matched source image
    """
    if match_luminance_only:
        # Convert both to LAB
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
        
        # Split channels
        source_l, source_a, source_b = cv2.split(source_lab)
        ref_l, _, _ = cv2.split(ref_lab)
        
        # Match L channel histogram
        matched_l = _match_histogram_channel(source_l, ref_l)
        
        # Merge back
        matched_lab = cv2.merge([matched_l, source_a, source_b])
        matched = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)
    else:
        # Match all channels independently
        matched_channels = []
        for i in range(3):
            matched_ch = _match_histogram_channel(source[:,:,i], reference[:,:,i])
            matched_channels.append(matched_ch)
        matched = cv2.merge(matched_channels)
    
    return matched


def _match_histogram_channel(source: np.ndarray, 
                              reference: np.ndarray) -> np.ndarray:
    """Match histogram of single channel"""
    # Get histograms
    src_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])
    
    # Compute CDFs
    src_cdf = src_hist.cumsum()
    ref_cdf = ref_hist.cumsum()
    
    # Normalize CDFs
    src_cdf = src_cdf / src_cdf[-1]
    ref_cdf = ref_cdf / ref_cdf[-1]
    
    # Create lookup table
    lookup = np.zeros(256, dtype=np.uint8)
    ref_idx = 0
    for src_val in range(256):
        while ref_idx < 255 and ref_cdf[ref_idx] < src_cdf[src_val]:
            ref_idx += 1
        lookup[src_val] = ref_idx
    
    # Apply lookup
    matched = lookup[source]
    
    return matched.astype(np.uint8)


# ==============================================================================
# ARTIFACT DETECTION (SURGICAL INSTRUMENTS)
# ==============================================================================

def detect_artifacts(image: np.ndarray,
                     dark_threshold: int = Config.ARTIFACT_DARK_THRESHOLD,
                     min_area: int = Config.ARTIFACT_MIN_AREA,
                     limbus: Optional[LimbusInfo] = None
                     ) -> Tuple[np.ndarray, float]:
    """
    Detect surgical artifacts (instruments, forceps, etc).
    
    Strategy:
    - Very dark regions are likely metal instruments
    - Exclude pupil area (legitimately dark)
    - Filter by minimum area (ignore small specs)
    
    Args:
        image: Input BGR image
        dark_threshold: Threshold for dark regions
        min_area: Minimum artifact area in pixels
        limbus: If provided, exclude pupil region from artifact detection
    
    Returns:
        (artifact_mask, artifact_percentage)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect very dark regions
    _, dark_mask = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Exclude pupil region if limbus provided
    if limbus is not None:
        pupil_mask = np.zeros_like(dark_mask)
        pupil_radius = int(limbus.radius * 0.35)  # Approximate pupil size
        cv2.circle(pupil_mask, limbus.center, pupil_radius, 255, -1)
        dark_mask[pupil_mask > 0] = 0
    
    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    
    # Filter by minimum area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask)
    artifact_mask = np.zeros_like(dark_mask)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            artifact_mask[labels == i] = 255
    
    # Calculate percentage
    artifact_pixels = np.sum(artifact_mask > 0)
    total_pixels = artifact_mask.shape[0] * artifact_mask.shape[1]
    artifact_pct = (artifact_pixels / total_pixels) * 100
    
    return artifact_mask, artifact_pct


def detect_fluid_regions(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Detect surgical fluids (oil, water, BSS) on eye surface.
    
    Fluids often appear as:
    - Unusual color tints
    - High saturation regions
    - Irregular reflections
    
    Args:
        image: Input BGR image
    
    Returns:
        (fluid_mask, fluid_percentage)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    
    # High saturation with medium value might indicate fluid
    high_sat = s > 100
    med_val = (v > 50) & (v < 200)
    
    fluid_mask = (high_sat & med_val).astype(np.uint8) * 255
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fluid_mask = cv2.morphologyEx(fluid_mask, cv2.MORPH_OPEN, kernel)
    
    fluid_pct = (np.sum(fluid_mask > 0) / fluid_mask.size) * 100
    
    return fluid_mask, fluid_pct


# ==============================================================================
# QUALITY ASSESSMENT
# ==============================================================================

def assess_blur(image: np.ndarray) -> float:
    """
    Assess image blur using Laplacian variance.
    
    Higher value = sharper image.
    
    Args:
        image: Input image
    
    Returns:
        Blur score (Laplacian variance)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(laplacian))


def assess_quality(image: np.ndarray,
                   glare_mask: np.ndarray,
                   artifact_mask: np.ndarray,
                   config: Config = Config) -> QualityMetrics:
    """
    Comprehensive quality assessment of preprocessed image.
    
    Args:
        image: Preprocessed image
        glare_mask: Detected glare regions
        artifact_mask: Detected artifact regions
        config: Configuration parameters
    
    Returns:
        QualityMetrics with scores and issues
    """
    # Calculate metrics
    blur_score = assess_blur(image)
    
    total_pixels = image.shape[0] * image.shape[1]
    glare_pct = (np.sum(glare_mask > 0) / total_pixels) * 100
    artifact_pct = (np.sum(artifact_mask > 0) / total_pixels) * 100
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))
    
    # Identify issues
    issues = []
    is_acceptable = True
    
    if blur_score < config.MIN_ACCEPTABLE_BLUR_SCORE:
        issues.append(f"Blurry image (score: {blur_score:.1f})")
        is_acceptable = False
    
    if glare_pct > config.MAX_ACCEPTABLE_GLARE_PERCENT:
        issues.append(f"High glare ({glare_pct:.1f}%)")
        is_acceptable = False
    
    if artifact_pct > config.MAX_ACCEPTABLE_ARTIFACT_PERCENT:
        issues.append(f"Heavy occlusion ({artifact_pct:.1f}%)")
        is_acceptable = False
    
    if mean_intensity < 30:
        issues.append("Too dark")
        is_acceptable = False
    elif mean_intensity > 225:
        issues.append("Too bright/overexposed")
        is_acceptable = False
    
    return QualityMetrics(
        blur_score=blur_score,
        glare_percentage=glare_pct,
        artifact_percentage=artifact_pct,
        mean_intensity=mean_intensity,
        std_intensity=std_intensity,
        is_acceptable=is_acceptable,
        issues=issues
    )


# ==============================================================================
# RING MASK GENERATION
# ==============================================================================

def create_ring_mask(image_shape: Tuple[int, int],
                     center: Tuple[int, int],
                     radius: int,
                     inner_ratio: float = Config.INNER_EXCLUDE_RATIO,
                     outer_ratio: float = Config.OUTER_INCLUDE_RATIO
                     ) -> np.ndarray:
    """
    Create donut-shaped mask for limbus region.
    
    This focuses feature matching on the limbus area where:
    - Blood vessels are visible
    - Texture is stable across preop/intraop
    - Inner region (pupil) is partially excluded based on inner_ratio
    
    The mask includes a ring from inner_ratio * radius to outer_ratio * radius.
    For example, if inner_ratio=0.7, it includes from 70% of limbus radius inward
    (excluding only the very center 70% of the limbus).
    
    Args:
        image_shape: (height, width) of image
        center: (x, y) limbus center
        radius: Limbus radius
        inner_ratio: Inner exclusion ratio (0.7 = exclude center 70%, include from 70% outward)
        outer_ratio: Outer inclusion ratio (sclera boundary)
    
    Returns:
        Binary ring mask
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    inner_r = int(radius * inner_ratio)
    outer_r = int(radius * outer_ratio)
    
    # Draw outer circle
    cv2.circle(mask, center, outer_r, 255, -1)
    # Remove inner circle
    cv2.circle(mask, center, inner_r, 0, -1)
    
    return mask


def trim_eyelids_from_mask(mask: np.ndarray,
                           center: Tuple[int, int],
                           radius: int,
                           upper_ratio: float = Config.EYELID_TRIM_UPPER_RATIO,
                           lower_ratio: float = Config.EYELID_TRIM_LOWER_RATIO
                           ) -> np.ndarray:
    """
    Remove eyelid / eyelash regions above and below the limbus.
    
    Keeps only an asymmetric horizontal band around the limbus center:
        [center_y - upper_ratio * radius, center_y + lower_ratio * radius]
    Everything outside this band is set to 0 in the mask. Use a **smaller**
    upper_ratio to cut more eyelid/eyelash region above the limbus.
    """
    h, w = mask.shape[:2]
    cx, cy = center
    
    upper_band = int(radius * upper_ratio)
    lower_band = int(radius * lower_ratio)
    y_min = max(0, cy - upper_band)
    y_max = min(h, cy + lower_band)
    
    trimmed = np.zeros_like(mask)
    trimmed[y_min:y_max, :] = mask[y_min:y_max, :]
    
    return trimmed


def apply_mask(image: np.ndarray, 
               mask: np.ndarray, 
               background: int = 0) -> np.ndarray:
    """Apply binary mask to image"""
    result = np.full_like(image, background)
    if len(image.shape) == 3:
        mask_3ch = cv2.merge([mask, mask, mask])
        result = np.where(mask_3ch > 0, image, result)
    else:
        result = np.where(mask > 0, image, result)
    return result


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def visualize_preprocessing_comparison(original: np.ndarray,
                                        processed: np.ndarray,
                                        glare_mask: np.ndarray,
                                        artifact_mask: np.ndarray,
                                        ring_mask: np.ndarray,
                                        quality: QualityMetrics,
                                        title: str = "Preprocessing Results",
                                        save_path: Optional[str] = None):
    """
    Create comprehensive visualization of preprocessing steps.
    DISABLED: Matplotlib visualization commented out to avoid opening windows.
    """
    # Matplotlib visualization disabled - no windows will open
    return None
    # Original matplotlib code commented out:
    # fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    # fig.suptitle(title, fontsize=14, fontweight='bold')
    # ... (rest of visualization code commented out)


def visualize_pair_comparison(preop_result: PreprocessResult,
                               intraop_result: PreprocessResult,
                               save_path: Optional[str] = None):
    """
    Side-by-side comparison of preop and intraop preprocessing.
    DISABLED: Matplotlib visualization commented out to avoid opening windows.
    """
    # Matplotlib visualization disabled - no windows will open
    return None
    # Original matplotlib code commented out:
    # fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    # fig.suptitle('PreOp vs IntraOp Preprocessing Comparison', fontsize=14, fontweight='bold')
    # ... (rest of visualization code commented out)


# ==============================================================================
# MAIN PREPROCESSING PIPELINE
# ==============================================================================

def preprocess_single(image_path: str,
                      model,
                      image_type: ImageType = ImageType.PREOP,
                      output_dir: str = "output",
                      reference_image: Optional[np.ndarray] = None,
                      apply_histogram_match: bool = False,
                      trim_eyelids: bool = False,
                      eyelid_upper_ratio: float = Config.EYELID_TRIM_UPPER_RATIO,
                      eyelid_lower_ratio: float = Config.EYELID_TRIM_LOWER_RATIO,
                      inner_exclude_ratio: float = Config.INNER_EXCLUDE_RATIO,
                      verbose: bool = True) -> PreprocessResult:
    """
    Complete preprocessing pipeline for a single image.
    
    Pipeline Steps:
    1. Load image
    2. Detect limbus (YOLO)
    3. Normalize scale (limbus → standard radius)
    4. Crop around limbus
    5. Detect and remove glare (if small)
    6. Conservative CLAHE enhancement
    7. Optional: Histogram matching to reference
    8. Detect artifacts
    9. Create ring mask
    10. Quality assessment
    
    Args:
        image_path: Path to input image
        model: YOLO model for limbus detection
        image_type: PREOP or INTRAOP
        output_dir: Output directory for results
        reference_image: Reference for histogram matching (preop for intraop)
        apply_histogram_match: Whether to match histogram to reference
        trim_eyelids: Whether to trim eyelids from mask
        eyelid_upper_ratio: Upper eyelid trim ratio
        eyelid_lower_ratio: Lower eyelid trim ratio
        inner_exclude_ratio: Inner exclusion ratio for ring mask (0.7 = include from 70% of limbus radius)
        verbose: Print progress messages
    
    Returns:
        PreprocessResult with all outputs
    """
    single_time_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    type_str = image_type.value
    if verbose:
        print(f"\n{'='*60}")
        print(f"PREPROCESSING: {type_str.upper()}")
        print(f"Image: {image_path}")
        print(f"{'='*60}")
    
    # Step 1: Load image
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if verbose:
        print(f"[1/10] Loaded image: {original.shape}")
    
    # Step 2: Detect limbus
    limbus = detect_limbus(model, original)
    print("limbus", limbus)
    if limbus is None:
        raise ValueError(f"No limbus detected in: {image_path}")
    
    if verbose:
        print(f"[2/10] Limbus detected: center={limbus.center}, radius={limbus.radius}")
    
    # Save limbus visualization
    vis_img = original.copy()
    cv2.circle(vis_img, limbus.center, limbus.radius, (255, 0, 0), 2)
    cv2.circle(vis_img, limbus.center, 5, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(output_dir, f"1_{type_str}_limbus_detected.jpg"), vis_img)
    
    # Step 3: Normalize scale
    normalized, norm_limbus, scale = normalize_scale(original, limbus, Config.STANDARD_RADIUS)
    
    if verbose:
        print(f"[3/10] Scale normalized: {scale:.3f}x (radius: {limbus.radius} → {norm_limbus.radius})")
    
    cv2.imwrite(os.path.join(output_dir, f"2_{type_str}_normalized.jpg"), normalized)
    
    # Step 4: Crop around limbus
    cropped, crop_limbus = crop_around_limbus(normalized, norm_limbus)
    
    if verbose:
        print(f"[4/10] Cropped: {cropped.shape}")
    
    cv2.imwrite(os.path.join(output_dir, f"3_{type_str}_cropped.jpg"), cropped)
    
    # Step 5: Re-detect limbus in cropped image (more accurate)
    crop_limbus_detected = detect_limbus(model, cropped)
    if crop_limbus_detected is not None:
        crop_limbus = crop_limbus_detected
        if verbose:
            print(f"[5/10] Limbus re-detected in crop: center={crop_limbus.center}")
    else:
        if verbose:
            print(f"[5/10] Using estimated limbus center in crop")
    
    # Step 6: Detect and remove glare
    deglared, glare_mask, glare_pct, was_inpainted = remove_glare_safe(cropped)
    
    if verbose:
        status = "inpainted" if was_inpainted else "masked only"
        print(f"[6/10] Glare detection: {glare_pct:.1f}% ({status})")
    
    cv2.imwrite(os.path.join(output_dir, f"4_{type_str}_deglared.jpg"), deglared)
    cv2.imwrite(os.path.join(output_dir, f"4_{type_str}_glare_mask.jpg"), glare_mask)
    
    # Step 7: Conservative enhancement
    enhanced, enhance_stats = enhance_conservative(deglared)
    
    if verbose:
        print(f"[7/10] Enhancement: mean_change={enhance_stats['mean_change']:.1f}, "
              f"contrast_ratio={enhance_stats['contrast_ratio']:.3f}")
    
    cv2.imwrite(os.path.join(output_dir, f"5_{type_str}_enhanced.jpg"), enhanced)
    
    # Step 8: Optional histogram matching
    if apply_histogram_match and reference_image is not None:
        enhanced = match_histograms(enhanced, reference_image, match_luminance_only=True)
        if verbose:
            print(f"[8/10] Histogram matched to reference")
        cv2.imwrite(os.path.join(output_dir, f"6_{type_str}_hist_matched.jpg"), enhanced)
    else:
        if verbose:
            print(f"[8/10] Histogram matching: skipped")
    
    # Step 9: Detect artifacts
    artifact_mask, artifact_pct = detect_artifacts(enhanced, limbus=crop_limbus)
    
    if verbose:
        print(f"[9/10] Artifacts detected: {artifact_pct:.1f}%")
    
    cv2.imwrite(os.path.join(output_dir, f"7_{type_str}_artifact_mask.jpg"), artifact_mask)
    
    # Step 10: Create ring mask
    ring_mask = create_ring_mask(
        enhanced.shape[:2], 
        crop_limbus.center, 
        crop_limbus.radius,
        inner_ratio=inner_exclude_ratio
    )
    # Optional: trim eyelids (top and bottom) using vertical band around limbus
    if trim_eyelids:
        if verbose:
            print(f"[10/10] Trimming eyelids (upper_ratio={eyelid_upper_ratio:.2f}, "
                  f"lower_ratio={eyelid_lower_ratio:.2f})")
        ring_mask = trim_eyelids_from_mask(
            ring_mask,
            crop_limbus.center,
            crop_limbus.radius,
            upper_ratio=eyelid_upper_ratio,
            lower_ratio=eyelid_lower_ratio,
        )
    
    if verbose:
        print(f"[10/10] Ring mask created")
    
    cv2.imwrite(os.path.join(output_dir, f"8_{type_str}_ring_mask.jpg"), ring_mask)
    
    # Apply ring mask for final output
    final_masked = apply_mask(enhanced, ring_mask)
    cv2.imwrite(os.path.join(output_dir, f"9_{type_str}_final_masked.jpg"), final_masked)
    
    # Quality assessment
    quality = assess_quality(enhanced, glare_mask, artifact_mask)
    
    if verbose:
        print(f"\n[QUALITY] Blur={quality.blur_score:.1f}, Glare={quality.glare_percentage:.1f}%, "
              f"Artifacts={quality.artifact_percentage:.1f}%")
        if quality.is_acceptable:
            print(f"[QUALITY] ✓ Image acceptable for feature matching")
        else:
            print(f"[QUALITY] ✗ Issues: {', '.join(quality.issues)}")
    
    # Build result
    single_time_stop = time.time()
    print(f"Time taken for single preprocess: {single_time_stop - single_time_start} seconds")
    result = PreprocessResult(
        original_image=cropped,  # Use cropped as "original" for comparison
        processed_image=enhanced,
        limbus_info=crop_limbus,
        scale_factor=scale,
        quality_metrics=quality,
        glare_mask=glare_mask,
        artifact_mask=artifact_mask,
        ring_mask=ring_mask,
        metadata={
            'image_path': image_path,
            'image_type': type_str,
            'output_dir': output_dir,
            'enhance_stats': enhance_stats,
            'was_inpainted': was_inpainted
        }
    )
    
    # Save visualization (disabled - matplotlib commented out)
    # fig = visualize_preprocessing_comparison(
    #     cropped, enhanced, glare_mask, artifact_mask, ring_mask, quality,
    #     title=f"{type_str.upper()} Preprocessing Results",
    #     save_path=os.path.join(output_dir, f"10_{type_str}_comparison.png")
    # )
    # plt.close(fig)
    
    return result


def preprocess_pair(preop_path: str,
                    intraop_path: str,
                    model,
                    preop_output_dir: str = "output/preop",
                    intraop_output_dir: str = "output/intraop",
                    apply_histogram_match: bool = True,
                    trim_eyelids_preop: bool = False,
                    trim_eyelids_intraop: bool = False,
                    eyelid_upper_ratio: float = Config.EYELID_TRIM_UPPER_RATIO,
                    eyelid_lower_ratio: float = Config.EYELID_TRIM_LOWER_RATIO,
                    inner_exclude_ratio: float = Config.INNER_EXCLUDE_RATIO,
                    verbose: bool = True) -> Tuple[PreprocessResult, PreprocessResult]:
    """
    Preprocess preop and intraop image pair.
    
    The intraop image is histogram-matched to the preop image for
    consistent appearance before feature matching.
    
    Args:
        preop_path: Path to preoperative image
        intraop_path: Path to intraoperative image
        model: YOLO model for limbus detection
        preop_output_dir: Output directory for preop results
        intraop_output_dir: Output directory for intraop results
        apply_histogram_match: Match intraop histogram to preop
        trim_eyelids_preop: Whether to trim eyelids in preop mask
        trim_eyelids_intraop: Whether to trim eyelids in intraop mask
        eyelid_upper_ratio: Relative band size above limbus (smaller = cut more)
        eyelid_lower_ratio: Relative band size below limbus (larger = keep more)
        inner_exclude_ratio: Inner exclusion ratio for ring mask (0.7 = include from 70% of limbus radius)
        verbose: Print progress
    
    Returns:
        (preop_result, intraop_result)
    """
    # Process preop first (reference)
    preop_result = preprocess_single(
        preop_path, model, 
        ImageType.PREOP, 
        preop_output_dir,
        trim_eyelids=trim_eyelids_preop,
        eyelid_upper_ratio=eyelid_upper_ratio,
        eyelid_lower_ratio=eyelid_lower_ratio,
        inner_exclude_ratio=inner_exclude_ratio,
        verbose=verbose,
    )
    
    # Process intraop with histogram matching to preop
    intraop_result = preprocess_single(
        intraop_path, model,
        ImageType.INTRAOP,
        intraop_output_dir,
        reference_image=preop_result.processed_image if apply_histogram_match else None,
        apply_histogram_match=apply_histogram_match,
        trim_eyelids=trim_eyelids_intraop,
        eyelid_upper_ratio=eyelid_upper_ratio,
        eyelid_lower_ratio=eyelid_lower_ratio,
        inner_exclude_ratio=inner_exclude_ratio,
        verbose=verbose,
    )
    
    # Generate pair comparison (disabled - matplotlib commented out)
    # comparison_dir = os.path.dirname(preop_output_dir)
    # os.makedirs(comparison_dir, exist_ok=True)
    # 
    # fig = visualize_pair_comparison(
    #     preop_result, intraop_result,
    #     save_path=os.path.join(comparison_dir, "pair_comparison.png")
    # )
    # plt.close(fig)
    
    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("PREPROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"PreOp:  Scale={preop_result.scale_factor:.3f}, "
              f"Quality={'✓' if preop_result.quality_metrics.is_acceptable else '✗'}")
        print(f"IntraOp: Scale={intraop_result.scale_factor:.3f}, "
              f"Quality={'✓' if intraop_result.quality_metrics.is_acceptable else '✗'}")
        print(f"\nOutputs saved to:")
        print(f"  PreOp:  {preop_output_dir}")
        print(f"  IntraOp: {intraop_output_dir}")
        print(f"{'='*60}\n")
    
    return preop_result, intraop_result


# ==============================================================================
# UTILITY FUNCTIONS FOR FEATURE MATCHING
# ==============================================================================

def get_feature_matching_images(preop_result: PreprocessResult,
                                 intraop_result: PreprocessResult,
                                 use_masked: bool = True
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get images ready for feature matching (SuperPoint/SIFT/ORB).
    
    Args:
        preop_result: Preprocessed preop result
        intraop_result: Preprocessed intraop result
        use_masked: If True, return ring-masked images
    
    Returns:
        (preop_image, intraop_image, preop_mask, intraop_mask)
        
    The masks can be used to filter features:
    - Only detect features where mask > 0
    - Ignore features in glare/artifact regions
    """
    if use_masked:
        preop_img = apply_mask(preop_result.processed_image, preop_result.ring_mask)
        intraop_img = apply_mask(intraop_result.processed_image, intraop_result.ring_mask)
    else:
        preop_img = preop_result.processed_image
        intraop_img = intraop_result.processed_image
    
    # Combined valid region mask (ring + no glare + no artifacts)
    preop_valid = preop_result.ring_mask.copy()
    preop_valid[preop_result.glare_mask > 0] = 0
    preop_valid[preop_result.artifact_mask > 0] = 0
    
    intraop_valid = intraop_result.ring_mask.copy()
    intraop_valid[intraop_result.glare_mask > 0] = 0
    intraop_valid[intraop_result.artifact_mask > 0] = 0
    
    return preop_img, intraop_img, preop_valid, intraop_valid


def get_limbus_centers(preop_result: PreprocessResult,
                        intraop_result: PreprocessResult
                        ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Get limbus centers for rotation computation.
    
    Returns:
        (preop_center, intraop_center)
    """
    return preop_result.limbus_info.center, intraop_result.limbus_info.center


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    time_start = time.time()
    # Example usage
    
    # Configuration
    # PRE_OP_IMAGE = r"_inputs\IMG_4484\IMG_4484.jpeg"
    # INTRA_OP_IMAGE = r"input_frames\IMG_4484_frames\rotated_210_frame_00707.jpg"

    # PRE_OP_IMAGE = r"_inputs\IMG_4139\IMG_4139.jpeg"
    # INTRA_OP_IMAGE = r"input_frames\IMG_4139_frames\rotated_80_frame_01859.jpg"
    PRE_OP_IMAGE = r"D:\AIDS\wetransfer_os-2025-11-26_161143-jpg_2025-12-04_0550\OS-2025-11-26_161143.jpg"
    INTRA_OP_IMAGE = r"D:\AIDS\wetransfer_os-2025-11-26_161143-jpg_2025-12-04_0550\OS-2025-11-26_161143.jpg"

    YOLO_MODEL = "model\intraop_latest.pt"
    
    # Output directories
    preop_dir = "output/robust_preprocess/preop"
    intraop_dir = "output/robust_preprocess/intraop"
    log_dir = "output/robust_preprocess"
    
    # Create directories
    os.makedirs(preop_dir, exist_ok=True)
    os.makedirs(intraop_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging - capture all print statements to a text file
    log_file = os.path.join(log_dir, "preprocessing_log.txt")
    logger = Logger.get_logger()
    logger.start_logging(log_file, append=False)
    
    try:
        # Load model
        model = load_yolo_model(YOLO_MODEL)
        
        # Ask user whether to trim eyelids / eyelashes from ring mask
        try:
            preop_inp = "yes" #input("Trim upper and lower eyelids for PREOP mask? [y/N]: ").strip().lower()
            intraop_inp = "no" #input("Trim upper and lower eyelids for INTRAOP mask? [y/N]: ").strip().lower()
            trim_eyelids_preop = preop_inp in ("y", "yes")
            trim_eyelids_intraop = intraop_inp in ("y", "yes")
        except Exception:
            trim_eyelids_preop = False
            trim_eyelids_intraop = False
        
        # Process pair
        preop_result, intraop_result = preprocess_pair(
            PRE_OP_IMAGE,
            INTRA_OP_IMAGE,
            model,
            preop_dir,
            intraop_dir,
            apply_histogram_match=True,
            trim_eyelids_preop=trim_eyelids_preop,
            trim_eyelids_intraop=trim_eyelids_intraop,
            eyelid_upper_ratio=Config.EYELID_TRIM_UPPER_RATIO,
            eyelid_lower_ratio=Config.EYELID_TRIM_LOWER_RATIO,
            verbose=True
        )
        
        # Get images ready for feature matching
        preop_img, intraop_img, preop_mask, intraop_mask = get_feature_matching_images(
            preop_result, intraop_result, use_masked=True
        )
        
        print("\n[READY FOR FEATURE MATCHING]")
        print(f"PreOp image shape: {preop_img.shape}")
        print(f"IntraOp image shape: {intraop_img.shape}")
        print(f"Use preop_mask and intraop_mask to filter features in valid regions only")
        
        # Save feature-matching-ready images
        cv2.imwrite(os.path.join(preop_dir, "FEATURE_MATCH_preop.jpg"), preop_img)
        cv2.imwrite(os.path.join(intraop_dir, "FEATURE_MATCH_intraop.jpg"), intraop_img)
        cv2.imwrite(os.path.join(preop_dir, "FEATURE_MATCH_preop_mask.jpg"), preop_mask)
        cv2.imwrite(os.path.join(intraop_dir, "FEATURE_MATCH_intraop_mask.jpg"), intraop_mask)
        
        print(f"\n[LOG] All output saved to: {log_file}")
        print(f"[INFO] Log file location: {log_file}")
    
    finally:
        # Stop logging (this will also print a closing message)
        logger.stop_logging()
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")
