"""
Merged Pipeline: Preprocessing + SuperPoint Feature Matching (Image to Video)
=============================================================================

This script combines preprocessing and feature matching pipelines:
1. Preprocess preop image and extract features (store them)
2. Extract frames from intraop video
3. For each frame:
   - Preprocess frame and extract features
   - Match with preop features
   - Estimate rotation angle
   - Draw transformed line on frame
4. Convert processed frames back to video

Press 'q' during processing to quit early and save video.

Author: Merged Pipeline for Toric Lens Surgery
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set matplotlib backend to non-interactive before any imports
# This prevents tkinter threading issues when matplotlib is used in background threads
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI, no tkinter)

# Import from preprocessing module
from preprocess_robust import (
    Logger,
    load_yolo_model,
    preprocess_single,
    ImageType,
    Config,
    detect_limbus,
    detect_limbus_from_path
)

# Import handler for freeze confidence threshold
from handler import get_config_handler

# Import from superpoint module
from superpoint_6_log_saved import (
    initialize_models,
    extract_features,
    match_features,
    apply_geometric_filtering,
    estimate_rotation_robust,
    save_keypoints_image,
    visualize_matches,
    draw_reference_line,
    draw_transformed_line,
    AccuracyConfig,
    MatchResult,
    RotationResult
)


# Global flag for early exit (now handled by cv2.waitKey in main loop)


class SharedRotationState:
    """Thread-safe shared state for rotation data between video playback and analysis threads."""
    def __init__(self):
        self.lock = threading.Lock()
        self.rotation_angle = 0.0
        self.limbus_center = None
        self.limbus_radius = None
        self.confidence = 0.0
        self.frame_num = 0
        self.skip_analysis_frames = 0  # Counter to skip analysis for N frames
    
    def update(self, rotation_angle: float, limbus_center: Tuple[int, int], 
               limbus_radius: int, confidence: float, frame_num: int):
        """Update rotation data thread-safely."""
        with self.lock:
            self.rotation_angle = rotation_angle
            self.limbus_center = limbus_center
            self.limbus_radius = limbus_radius
            self.confidence = confidence
            self.frame_num = frame_num
    
    def get(self):
        """Get current rotation data thread-safely."""
        with self.lock:
            return (self.rotation_angle, self.limbus_center, self.limbus_radius, 
                   self.confidence, self.frame_num)
    
    def set_skip_analysis(self, frames: int):
        """Set number of frames to skip analysis."""
        with self.lock:
            self.skip_analysis_frames = frames
    
    def should_skip_analysis(self) -> bool:
        """Check if analysis should be skipped and decrement counter."""
        with self.lock:
            if self.skip_analysis_frames > 0:
                self.skip_analysis_frames -= 1
                return True
            return False


class RotationStatsLogger:
    """Simple thread-safe logger for rotation statistics."""
    def __init__(self):
        self.lock = threading.Lock()
        self.logs: List[Dict] = []
    
    def log(self, frame_num: int, rotation_deg: float, confidence: float, num_matches: int):
        """Log rotation statistics thread-safely."""
        with self.lock:
            self.logs.append({
                'frame_num': frame_num,
                'rotation_deg': rotation_deg,
                'confidence': confidence,
                'num_matches': num_matches,
                'is_reliable': confidence >= 0.50
            })
            
            # Live append to log_stat.txt after each analyzed frame
            try:
                video_output_dir = os.path.join("output", "video_output")
                os.makedirs(video_output_dir, exist_ok=True)
                log_stat_path = os.path.join(video_output_dir, "log_stat.txt")
                
                # If file newly created, add header once
                file_exists = os.path.exists(log_stat_path)
                with open(log_stat_path, 'a') as f:
                    if not file_exists:
                        f.write("Frame-by-Frame Rotation Statistics\n\n")
                        f.write("==============================================\n\n")
                    f.write(f"Frame {frame_num}:\n")
                    f.write(f"  Rotation: {rotation_deg:.2f}°\n")
                    f.write(f"  Confidence: {confidence*100:.1f}%\n")
                    f.write(f"  Matches: {num_matches}\n")
                    f.write(f"  Reliable: {confidence >= 0.50}\n\n")
            except Exception as e:
                # Do not block pipeline on logging errors
                print(f"[LOG WARNING] Failed to write live log: {e}")
    
    def get_all(self):
        """Get all logs thread-safely."""
        with self.lock:
            return sorted(self.logs, key=lambda x: x['frame_num'])


def rotate_image(image: np.ndarray, angle_deg: float, center: Tuple[int, int] = None) -> np.ndarray:
    """
    Rotate an image by specified angle in degrees.
    
    Args:
        image: Input image (BGR or grayscale)
        angle_deg: Rotation angle in degrees (positive = counterclockwise)
        center: Rotation center (x, y). If None, uses image center.
    
    Returns:
        Rotated image
    """
    h, w = image.shape[:2]
    
    if center is None:
        center = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    
    # Calculate new dimensions to avoid cropping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new center
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Rotate image
    rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)
    
    return rotated


def test_single_rotation_angle(test_angle: int,
                                preprocessed_frame: np.ndarray,
                                ring_mask: np.ndarray,
                                preop_feats,
                                extractor,
                                matcher,
                                device,
                                config,
                                preop_result,
                                yolo_model,
                                temp_frame_dir: str,
                                frame_num: int) -> Optional[Dict]:
    """
    Test a single rotation angle. This function will be called in parallel.
    
    Returns:
        Result dictionary with image_rotation_angle, rotation_angle, and confidence, or None
    """
    try:
        h, w = preprocessed_frame.shape[:2]
        rotation_center = (w // 2, h // 2)
        
        # Rotate the preprocessed frame
        rotated_frame = rotate_image(preprocessed_frame, test_angle, rotation_center)
        
        # Rotate the mask as well
        rotated_mask = None
        if ring_mask is not None:
            rotated_mask = rotate_image(ring_mask, test_angle, rotation_center)
        
        # Save rotated frame temporarily
        rotated_path = os.path.join(temp_frame_dir, f"temp_rotated_{frame_num:06d}_{test_angle:03d}.jpg")
        cv2.imwrite(rotated_path, rotated_frame)
        
        # Detect limbus on rotated frame
        try:
            # Get YOLO confidence from handler
            config_handler = get_config_handler()
            yolo_confidence = config_handler.get_yolo_confidence()
            limbus_detected = detect_limbus_from_path(yolo_model, rotated_path, confidence_threshold=yolo_confidence)
            if not limbus_detected:
                try:
                    os.remove(rotated_path)
                except:
                    pass
                return None
            rotated_center = limbus_detected.center
            rotated_radius = limbus_detected.radius
        except:
            try:
                os.remove(rotated_path)
            except:
                pass
            return None
        
        # Extract features from rotated frame with rotated mask
        rotated_feats = extract_features(
            rotated_path,
            extractor,
            device,
            mask=rotated_mask,
            config=config
        )
        
        # Match features
        match_result = match_features(preop_feats, rotated_feats, matcher, config)
        
        if match_result.num_filtered_matches < config.MIN_MATCHES_FOR_RANSAC:
            try:
                os.remove(rotated_path)
            except:
                pass
            return None
        
        # Geometric filtering
        match_result = apply_geometric_filtering(
            match_result,
            preop_result.limbus_info.center,
            rotated_center,
            preop_result.limbus_info.radius,
            rotated_radius,
            config
        )
        
        if match_result.num_filtered_matches < config.MIN_MATCHES_FOR_RANSAC:
            try:
                os.remove(rotated_path)
            except:
                pass
            return None
        
        # Estimate rotation
        rotation_result = estimate_rotation_robust(
            match_result,
            preop_result.limbus_info.center,
            rotated_center,
            config
        )
        
        # Calculate final rotation
        final_rotation = test_angle + rotation_result.rotation_deg
        
        # Normalize to [-180, 180]
        while final_rotation > 180:
            final_rotation -= 360
        while final_rotation < -180:
            final_rotation += 360
        
        confidence = rotation_result.confidence_score
        
        # Store result with all required information
        result = {
            'image_rotation_angle': test_angle,  # The angle we rotated the image
            'rotation_angle': final_rotation,    # The calculated rotation angle
            'confidence': confidence,
            'estimated_rotation': rotation_result.rotation_deg,
            'num_matches': rotation_result.num_matches,
            'is_reliable': rotation_result.is_reliable,
            'limbus_center': rotated_center,
            'limbus_radius': rotated_radius,
            'rotation_result': rotation_result
        }
        
        # Clean up rotated frame
        try:
            os.remove(rotated_path)
        except:
            pass
        
        return result
        
    except Exception as e:
        # Clean up on error
        try:
            rotated_path = os.path.join(temp_frame_dir, f"temp_rotated_{frame_num:06d}_{test_angle:03d}.jpg")
            os.remove(rotated_path)
        except:
            pass
        return None


def test_frame_at_multiple_angles(preprocessed_frame: np.ndarray,
                                   ring_mask: np.ndarray,
                                   preprocessed_frame_path: str,
                                   preop_feats,
                                   extractor,
                                   matcher,
                                   device,
                                   config,
                                   preop_result,
                                   yolo_model,
                                   temp_frame_dir: str,
                                   frame_num: int,
                                   rotation_step: int = 20) -> Optional[Dict]:
    """
    Test frame at multiple rotation angles in parallel using multi-threading and return the best result.
    
    Args:
        preprocessed_frame: Preprocessed intraop frame (numpy array)
        ring_mask: Ring mask for feature extraction
        preprocessed_frame_path: Path to save rotated versions
        preop_feats: Preop features
        extractor: Feature extractor
        matcher: Feature matcher
        device: Computing device
        config: Configuration
        preop_result: Preop preprocessing result
        yolo_model: YOLO model for limbus detection
        temp_frame_dir: Temporary directory for rotated frames
        frame_num: Frame number
        rotation_step: Step size for rotation angles (degrees)
    
    Returns:
        Best result dictionary with rotation info, or None if no good match
    """
    # Define rotation angles to test
    rotation_angles = list(range(0, 360, rotation_step))  # [0, 20, 40, ..., 340]
    
    print(f"[ANALYSIS] Frame {frame_num}: Testing {len(rotation_angles)} rotation angles in parallel...")
    
    # Test all angles in parallel using ThreadPoolExecutor
    all_results = []
    
    with ThreadPoolExecutor(max_workers=min(len(rotation_angles), 18)) as executor:
        # Submit all tasks
        future_to_angle = {
            executor.submit(
                test_single_rotation_angle,
                test_angle,
                preprocessed_frame.copy(),  # Make a copy for each thread
                ring_mask.copy() if ring_mask is not None else None,
                preop_feats,
                extractor,
                matcher,
                device,
                config,
                preop_result,
                yolo_model,
                temp_frame_dir,
                frame_num
            ): test_angle
            for test_angle in rotation_angles
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_angle):
            test_angle = future_to_angle[future]
            try:
                result = future.result()
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                continue
    
    # Find the result with highest confidence
    if not all_results:
        print(f"[ANALYSIS] Frame {frame_num}: No valid results from any angle")
        return None
    
    # Sort by confidence (highest first), prefer reliable results
    sorted_results = sorted(all_results, key=lambda x: (x['is_reliable'], x['confidence']), reverse=True)
    best_result = sorted_results[0]
    
    print(f"[ANALYSIS] Frame {frame_num}: Best result - Image rotation: {best_result['image_rotation_angle']}°, "
          f"Rotation angle: {best_result['rotation_angle']:.2f}°, "
          f"Confidence: {best_result['confidence']*100:.1f}%")
    
    return best_result


def analyze_frame_async(frame_queue: queue.Queue,
                        shared_state: SharedRotationState,
                        stats_logger: RotationStatsLogger,
                        yolo_model,
                        preop_result,
                        preop_feats,
                        extractor,
                        matcher,
                        device,
                        config,
                        temp_frame_dir: str,
                        reference_angle: float,
                        quit_flag_ref):
    """Background thread function to analyze frames asynchronously."""
    # Set matplotlib to non-interactive backend to avoid tkinter threading issues
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    while True:
        try:
            # Get frame from queue (with timeout to check quit flag)
            try:
                item = frame_queue.get(timeout=0.1)
                if item is None:  # Poison pill to stop
                    break
                frame, frame_num = item
            except queue.Empty:
                if quit_flag_ref[0]:
                    break
                continue
            
            if quit_flag_ref[0]:
                break
            
            # Check if analysis should be skipped (high confidence mode)
            if shared_state.should_skip_analysis():
                print(f"[ANALYSIS] Frame {frame_num}: Skipping analysis (high confidence mode, smooth playback)")
                continue  # Skip entire analysis process
            
            # Analyze this frame
            print(f"[ANALYSIS] Frame {frame_num}: Starting analysis...")
            
            try:
                # Save frame temporarily for preprocessing
                temp_frame_path = os.path.join(temp_frame_dir, f"temp_frame_{frame_num:06d}.jpg")
                cv2.imwrite(temp_frame_path, frame)
                
                # Preprocess frame
                # Get YOLO confidence from handler
                config_handler = get_config_handler()
                yolo_confidence = config_handler.get_yolo_confidence()
                intraop_result = preprocess_single(
                    temp_frame_path,
                    yolo_model,
                    ImageType.INTRAOP,
                    temp_frame_dir,
                    reference_image=preop_result.processed_image,
                    apply_histogram_match=True,
                    trim_eyelids=False,
                    eyelid_upper_ratio=Config.EYELID_TRIM_UPPER_RATIO,
                    eyelid_lower_ratio=Config.EYELID_TRIM_LOWER_RATIO,
                    inner_exclude_ratio=Config.INNER_EXCLUDE_RATIO,
                    confidence_threshold=yolo_confidence,
                    verbose=False
                )
                
                # Save preprocessed frame temporarily
                temp_processed_path = os.path.join(temp_frame_dir, f"temp_processed_{frame_num:06d}.jpg")
                cv2.imwrite(temp_processed_path, intraop_result.processed_image)
                
                # Test frame at multiple rotation angles
                best_result = test_frame_at_multiple_angles(
                    intraop_result.processed_image,
                    intraop_result.ring_mask,
                    temp_processed_path,
                    preop_feats,
                    extractor,
                    matcher,
                    device,
                    config,
                    preop_result,
                    yolo_model,
                    temp_frame_dir,
                    frame_num,
                    rotation_step=20  # Test every 20 degrees
                )
                
                if best_result:
                    # Detect limbus on original frame for drawing coordinates
                    # Save frame temporarily for limbus detection
                    try:
                        temp_frame_for_limbus = os.path.join(temp_frame_dir, f"temp_limbus_{frame_num:06d}.jpg")
                        cv2.imwrite(temp_frame_for_limbus, frame)
                        # Get YOLO confidence from handler
                        config_handler = get_config_handler()
                        yolo_confidence = config_handler.get_yolo_confidence()
                        limbus_detected = detect_limbus_from_path(yolo_model, temp_frame_for_limbus, confidence_threshold=yolo_confidence)
                        if limbus_detected:
                            limbus_center = limbus_detected.center
                            limbus_radius = limbus_detected.radius
                        else:
                            h, w = frame.shape[:2]
                            limbus_center = (w // 2, h // 2)
                            limbus_radius = min(w, h) // 4
                        # Clean up
                        try:
                            os.remove(temp_frame_for_limbus)
                        except:
                            pass
                    except Exception as e:
                        # Fallback if detection fails
                        h, w = frame.shape[:2]
                        limbus_center = (w // 2, h // 2)
                        limbus_radius = min(w, h) // 4
                        print(f"[ANALYSIS] Frame {frame_num}: Limbus detection failed: {e}")
                    
                    # Update shared state with best rotation data
                    # Use rotation_angle from best result (this is the final calculated rotation)
                    confidence_value = best_result['confidence']
                    
                    # Get freeze confidence threshold from handler
                    config_handler = get_config_handler()
                    freeze_threshold = config_handler.get_freeze_confidence_threshold()
                    
                    # If confidence exceeds freeze threshold, skip analysis for next 5000 frames for smooth playback
                    if confidence_value > freeze_threshold:
                        shared_state.set_skip_analysis(5000)
                        print(f"[ANALYSIS] Frame {frame_num}: High confidence ({confidence_value*100:.1f}%) > freeze threshold ({freeze_threshold*100:.1f}%) - Skipping analysis for next 5000 frames for smooth playback")
                    
                    shared_state.update(
                        best_result['rotation_angle'],
                        limbus_center,
                        limbus_radius,
                        confidence_value,
                        frame_num
                    )
                    
                    # Log rotation statistics
                    stats_logger.log(
                        frame_num,
                        best_result['rotation_angle'],
                        best_result['confidence'],
                        best_result['num_matches']
                    )
                    
                    print(f"[ANALYSIS] Frame {frame_num}: ✓ Best result - "
                          f"Image rotation: {best_result['image_rotation_angle']}°, "
                          f"Rotation angle: {best_result['rotation_angle']:.2f}°, "
                          f"Conf={best_result['confidence']*100:.1f}%")
                else:
                    print(f"[ANALYSIS] Frame {frame_num}: ✗ No valid match found at any angle")
                
                # Clean up temp files
                try:
                    os.remove(temp_frame_path)
                    os.remove(temp_processed_path)
                except:
                    pass
                    
            except Exception as e:
                print(f"[ANALYSIS] Frame {frame_num}: Error - {e}")
            
            frame_queue.task_done()
            
        except Exception as e:
            print(f"[ANALYSIS] Thread error: {e}")
            if quit_flag_ref[0]:
                break


def draw_line_on_frame(frame: np.ndarray,
                       center: Tuple[int, int],
                       radius: int,
                       transformed_angle: float,
                       reference_angle: float,
                       rotation_angle: float,
                       toric_angle: float = None,
                       incision_angle: float = None,
                       line_color: Tuple[int, int, int] = (0, 255, 255),
                       line_thickness: int = 2,
                       preop_limbus_radius: int = None,
                       show_reference: bool = True,
                       show_limbus: bool = True) -> np.ndarray:
    """
    Draw lines on a frame at specified angles from center.
    Matches Tab 3 (preop axis setup) visual style.
    
    Args:
        frame: Input frame (BGR)
        center: Center point (x, y)
        radius: Radius for line length calculation (current frame limbus radius)
        transformed_angle: Final transformed reference angle in degrees
        reference_angle: Reference angle from preop
        rotation_angle: Rotation angle detected
        toric_angle: Toric angle (reference + 30)
        incision_angle: Incision angle (reference + 60)
        line_color: BGR color tuple for reference line (default: cyan)
        line_thickness: Line thickness (default: 2 to match Tab 3)
        preop_limbus_radius: Preop limbus radius for offset calculation (for toric parallel lines)
    
    Returns:
        Frame with lines drawn
    """
    frame_copy = frame.copy()
    
    # Use preop limbus radius for offset calculation if provided, otherwise use current radius
    offset_base_radius = preop_limbus_radius if preop_limbus_radius is not None else radius
    length = int(radius * 1.5)
    
    # Draw reference line (broken yellow line) - matches Tab 3
    if show_reference:
        angle_rad = np.radians(transformed_angle)
        x1 = int(center[0] + length * np.cos(angle_rad))
        y1 = int(center[1] + length * np.sin(angle_rad))
        x2 = int(center[0] - length * np.cos(angle_rad))
        y2 = int(center[1] - length * np.sin(angle_rad))
        # Draw broken/dotted yellow line (simulate with segments)
        num_segments = 20
        dx = (x2 - x1) / num_segments
        dy = (y2 - y1) / num_segments
        for i in range(0, num_segments, 2):  # Draw every other segment for dotted effect
            seg_x1 = int(x1 + i * dx)
            seg_y1 = int(y1 + i * dy)
            seg_x2 = int(x1 + (i + 1) * dx)
            seg_y2 = int(y1 + (i + 1) * dy)
            cv2.line(frame_copy, (seg_x1, seg_y1), (seg_x2, seg_y2), (0, 255, 255), line_thickness)  # Yellow in BGR
    
    # Draw toric line (if provided) - blue solid line with parallel offset lines
    # Calculate relative to transformed reference angle (base = reference)
    if toric_angle is not None:
        # Offset between toric and reference is constant; keep offset but rotate with reference
        toric_offset = toric_angle - reference_angle
        # Use same rotation as reference, then apply offset in the same frame (subtract to keep relative CW/CCW)
        transformed_toric = transformed_angle - toric_offset
        # Normalize to [0, 360)
        while transformed_toric >= 360:
            transformed_toric -= 360
        while transformed_toric < 0:
            transformed_toric += 360
        angle_rad = np.radians(transformed_toric)
        x1 = int(center[0] + length * np.cos(angle_rad))
        y1 = int(center[1] + length * np.sin(angle_rad))
        x2 = int(center[0] - length * np.cos(angle_rad))
        y2 = int(center[1] - length * np.sin(angle_rad))
        
        # Draw main toric line (solid blue)
        # Blue: BGR(255, 0, 0)
        cv2.line(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), line_thickness)  # Blue for toric
        
        # Draw two parallel offset lines on both sides (solid blue)
        # Offset distance is 5% of preop limbus radius
        offset_distance = max(1, int(offset_base_radius * 0.05))
        
        # Calculate perpendicular direction (rotate by 90 degrees)
        # Perpendicular vector: (-sin(angle), cos(angle))
        perp_x = -np.sin(angle_rad)
        perp_y = np.cos(angle_rad)
        
        # Offset line 1 (one side)
        # In OpenCV, y increases downward, so we add perp_y (opposite of Qt which subtracts)
        x1_offset1 = int(x1 + offset_distance * perp_x)
        y1_offset1 = int(y1 + offset_distance * perp_y)
        x2_offset1 = int(x2 + offset_distance * perp_x)
        y2_offset1 = int(y2 + offset_distance * perp_y)
        # Draw offset line 1 (solid blue)
        cv2.line(frame_copy, (x1_offset1, y1_offset1), (x2_offset1, y2_offset1), (255, 0, 0), line_thickness)
        
        # Offset line 2 (other side)
        x1_offset2 = int(x1 - offset_distance * perp_x)
        y1_offset2 = int(y1 - offset_distance * perp_y)
        x2_offset2 = int(x2 - offset_distance * perp_x)
        y2_offset2 = int(y2 - offset_distance * perp_y)
        # Draw offset line 2 (solid blue)
        cv2.line(frame_copy, (x1_offset2, y1_offset2), (x2_offset2, y2_offset2), (255, 0, 0), line_thickness)
    
    # Draw incision line (if provided) - red solid line, thickness 2
    # Calculate relative to transformed reference angle (base = reference)
    if incision_angle is not None:
        # Offset between incision and reference is constant; keep offset but rotate with reference
        incision_offset = incision_angle - reference_angle
        # Use same rotation as reference, then apply offset in the same frame (subtract to keep relative CW/CCW)
        transformed_incision = transformed_angle - incision_offset
        # Normalize to [0, 360)
        while transformed_incision >= 360:
            transformed_incision -= 360
        while transformed_incision < 0:
            transformed_incision += 360
        angle_rad = np.radians(transformed_incision)
        x1 = int(center[0] + length * np.cos(angle_rad))
        y1 = int(center[1] + length * np.sin(angle_rad))
        x2 = int(center[0] - length * np.cos(angle_rad))
        y2 = int(center[1] - length * np.sin(angle_rad))
        cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), line_thickness)  # Red for incision (BGR format)
    
    cv2.circle(frame_copy, center, 8, (0, 255, 0), -1)
    
    # Draw limbus circle (broken green line) - matches Tab 3
    if show_limbus:
        # Draw broken/dotted green circle by drawing arc segments
        num_circle_segments = 60
        for i in range(0, num_circle_segments, 2):  # Draw every other segment for dotted effect
            angle1 = 2 * np.pi * i / num_circle_segments
            angle2 = 2 * np.pi * (i + 1) / num_circle_segments
            x1_circle = int(center[0] + radius * np.cos(angle1))
            y1_circle = int(center[1] + radius * np.sin(angle1))
            x2_circle = int(center[0] + radius * np.cos(angle2))
            y2_circle = int(center[1] + radius * np.sin(angle2))
            cv2.line(frame_copy, (x1_circle, y1_circle), (x2_circle, y2_circle), (0, 255, 0), line_thickness)
    
    # Add text overlay
    text_y = 30

    # cv2.putText(frame_copy, f"Ref: {reference_angle:.1f} deg", (10, text_y),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # cv2.putText(frame_copy, f"Rotation: {rotation_angle:.2f} deg", (10, text_y + 35),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # cv2.putText(frame_copy, f"Transformed: {transformed_angle:.2f} deg", (10, text_y + 70),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2)
    
    # Add toric and incision angle text if provided
    if toric_angle is not None:
        toric_offset = toric_angle - reference_angle
        transformed_toric = transformed_angle - toric_offset
        while transformed_toric >= 360:
            transformed_toric -= 360
        while transformed_toric < 0:
            transformed_toric += 360
        # cv2.putText(frame_copy, f"Toric: {transformed_toric:.2f} deg", (10, text_y + 105),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    if incision_angle is not None:
        incision_offset = incision_angle - reference_angle
        transformed_incision = transformed_angle - incision_offset
        while transformed_incision >= 360:
            transformed_incision -= 360
        while transformed_incision < 0:
            transformed_incision += 360
        # cv2.putText(frame_copy, f"Incision: {transformed_incision:.2f} deg", (10, text_y + 140),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    return frame_copy


def process_video_frame(frame: np.ndarray,
                        frame_num: int,
                        yolo_model,
                        preop_result,
                        preop_feats,
                        extractor,
                        matcher,
                        device,
                        config,
                        temp_frame_dir: str,
                        reference_angle: float) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Process a single video frame: preprocess, extract features, match, estimate rotation.
    
    Returns:
        (processed_frame_with_line, rotation_info_dict) or (None, None) if failed
    """
    global quit_flag
    
    if quit_flag:
        return None, None
    
    try:
        # Save frame temporarily for preprocessing
        temp_frame_path = os.path.join(temp_frame_dir, f"temp_frame_{frame_num:06d}.jpg")
        cv2.imwrite(temp_frame_path, frame)
        
        # Preprocess frame
        # Get YOLO confidence from handler
        config_handler = get_config_handler()
        yolo_confidence = config_handler.get_yolo_confidence()
        intraop_result = preprocess_single(
            temp_frame_path,
            yolo_model,
            ImageType.INTRAOP,
            temp_frame_dir,
            reference_image=preop_result.processed_image,
            apply_histogram_match=True,
            trim_eyelids=False,  # Can be configured
            confidence_threshold=yolo_confidence,
            eyelid_upper_ratio=Config.EYELID_TRIM_UPPER_RATIO,
            eyelid_lower_ratio=Config.EYELID_TRIM_LOWER_RATIO,
            inner_exclude_ratio=Config.INNER_EXCLUDE_RATIO,
            verbose=False  # Less verbose for frame-by-frame
        )
        
        if quit_flag:
            return None, None
        
        # Save preprocessed frame temporarily
        temp_processed_path = os.path.join(temp_frame_dir, f"temp_processed_{frame_num:06d}.jpg")
        cv2.imwrite(temp_processed_path, intraop_result.processed_image)
        
        # Extract features from frame
        intraop_feats = extract_features(
            temp_processed_path,
            extractor,
            device,
            mask=intraop_result.ring_mask,
            config=config
        )
        
        if quit_flag:
            return None, None
        
        # Match features
        match_result = match_features(preop_feats, intraop_feats, matcher, config)
        
        if match_result.num_filtered_matches < config.MIN_MATCHES_FOR_RANSAC:
            print(f"  Frame {frame_num}: Insufficient matches ({match_result.num_filtered_matches})")
            # Return original frame without line
            return frame.copy(), None
        
        # Geometric filtering
        match_result = apply_geometric_filtering(
            match_result,
            preop_result.limbus_info.center,
            intraop_result.limbus_info.center,
            preop_result.limbus_info.radius,
            intraop_result.limbus_info.radius,
            config
        )
        
        if match_result.num_filtered_matches < config.MIN_MATCHES_FOR_RANSAC:
            print(f"  Frame {frame_num}: Insufficient matches after filtering ({match_result.num_filtered_matches})")
            return frame.copy(), None
        
        # Estimate rotation
        rotation_result = estimate_rotation_robust(
            match_result,
            preop_result.limbus_info.center,
            intraop_result.limbus_info.center,
            config
        )
        
        # Draw line on original frame (not preprocessed)
        transformed_angle = reference_angle + rotation_result.rotation_deg
        frame_with_line = draw_line_on_frame(
            frame,
            intraop_result.limbus_info.center,
            intraop_result.limbus_info.radius,
            transformed_angle,
            reference_angle,
            rotation_result.rotation_deg
        )
        
        rotation_info = {
            'frame_num': frame_num,
            'rotation_deg': rotation_result.rotation_deg,
            'confidence': rotation_result.confidence_score,
            'num_matches': rotation_result.num_matches,
            'is_reliable': rotation_result.is_reliable
        }
        
        print(f"  Frame {frame_num}: Rotation={rotation_result.rotation_deg:.2f}°, "
              f"Confidence={rotation_result.confidence_score*100:.1f}%, "
              f"Matches={rotation_result.num_matches}")
        
        # Clean up temp files
        try:
            os.remove(temp_frame_path)
            os.remove(temp_processed_path)
        except:
            pass
        
        return frame_with_line, rotation_info
        
    except Exception as e:
        print(f"  Frame {frame_num}: Error - {e}")
        return frame.copy(), None


# def main():
#     """
#     Main pipeline: 
#     1. Preprocess preop → extract and store features
#     2. Extract frames from intraop video
#     3. For each frame: preprocess, extract features, match, estimate rotation, draw line
#     4. Convert processed frames back to video
#     """
#     global quit_flag, REFERENCE_ANGLE
    
#     # ========== CONFIGURATION ==========
    
#     # Input: preop is image, intraop is video
#     PRE_OP_IMAGE = r"D:\AIDS\cv\orb_eye\_inputs\IMG_4484\IMG_4483.jpeg"
#     INTRA_OP_VIDEO = r"D:\AIDS\cv\orb_eye\_inputs\IMG_4484\OD-2025-11-19_171152_fixed.mp4" # Change to your video path
    
#     # YOLO model for limbus detection
#     YOLO_MODEL = "model\intraop_latest.pt"
    
#     # Reference angle (line drawn by doctor on preop image)
#     REFERENCE_ANGLE = 0  # degrees (180 = horizontal left)
#     TORIC_ANGLE = 0
#     INCISION_ANGLE = 0

#     # Output directories
#     preprocess_output_dir = "output/robust_preprocess"
#     preop_dir = os.path.join(preprocess_output_dir, "preop")
#     intraop_dir = os.path.join(preprocess_output_dir, "intraop")
#     superpoint_output_dir = "output/superpoint_robust"
#     video_output_dir = "output/video_output"
#     temp_frame_dir = os.path.join(video_output_dir, "temp_frames")
#     frames_output_dir = os.path.join(video_output_dir, "frames_with_lines")
    
#     # Create directories
#     os.makedirs(preop_dir, exist_ok=True)
#     os.makedirs(intraop_dir, exist_ok=True)
#     os.makedirs(superpoint_output_dir, exist_ok=True)
#     os.makedirs(video_output_dir, exist_ok=True)
#     os.makedirs(temp_frame_dir, exist_ok=True)
#     os.makedirs(frames_output_dir, exist_ok=True)
    
#     # Setup logging - capture all print statements to a text file
#     log_file = os.path.join(preprocess_output_dir, "merged_pipeline_log.txt")
#     logger = Logger.get_logger()
#     logger.start_logging(log_file, append=False)
    
#     try:
#         print(f"\n{'='*70}")
#         print("MERGED PIPELINE: PREPROCESSING + FEATURE MATCHING (IMAGE TO VIDEO)")
#         print(f"{'='*70}")
#         print(f"Pre-op image: {PRE_OP_IMAGE}")
#         print(f"Intra-op video: {INTRA_OP_VIDEO}")
#         print(f"Reference angle: {REFERENCE_ANGLE}°")
#         print(f"{'='*70}\n")
        
#         # ========== STEP 1: LOAD YOLO MODELS ==========
#         print(f"\n{'='*70}")
#         print("STEP 1: LOADING YOLO MODELS")
#         print(f"{'='*70}")
#         # Load separate YOLO model instances for each thread to avoid blocking
#         # Main thread model: for limbus detection during live video playback
#         # Analysis thread model: for limbus detection during frame analysis
#         yolo_model = load_yolo_model(YOLO_MODEL)  # For analysis thread
#         yolo_model_main = load_yolo_model(YOLO_MODEL)  # For main thread (live video)
#         print("  ✓ Loaded YOLO model for analysis thread")
#         print("  ✓ Loaded YOLO model for main thread (live video)")
        
#         # Ask user whether to trim eyelids / eyelashes from ring mask
#         try:
#             preop_inp = input("Trim upper and lower eyelids for PREOP mask? [y/N]: ").strip().lower()
#             intraop_inp = input("Trim upper and lower eyelids for INTRAOP mask? [y/N]: ").strip().lower()
#             trim_eyelids_preop = preop_inp in ("y", "yes")
#             trim_eyelids_intraop = intraop_inp in ("y", "yes")
#         except Exception:
#             trim_eyelids_preop = False
#             trim_eyelids_intraop = False
        
#         # ========== STEP 2: PREPROCESS PREOP ==========
#         print(f"\n{'='*70}")
#         print("STEP 2: PREPROCESSING PREOP IMAGE")
#         print(f"{'='*70}")
        
#         preop_result = preprocess_single(
#             PRE_OP_IMAGE,
#             yolo_model,
#             ImageType.PREOP,
#             preop_dir,
#             reference_image=None,
#             apply_histogram_match=False,
#             trim_eyelids=trim_eyelids_preop,
#             eyelid_upper_ratio=Config.EYELID_TRIM_UPPER_RATIO,
#             eyelid_lower_ratio=Config.EYELID_TRIM_LOWER_RATIO,
#             inner_exclude_ratio=Config.INNER_EXCLUDE_RATIO,
#             verbose=True
#         )
        
#         print(f"\n[PREOP PREPROCESSING COMPLETE]")
#         print(f"  Processed image shape: {preop_result.processed_image.shape}")
#         print(f"  Limbus center: {preop_result.limbus_info.center}")
#         print(f"  Limbus radius: {preop_result.limbus_info.radius}")
        
#         # Save preprocessed preop image path
#         preop_processed_path = os.path.join(preop_dir, "5_preop_enhanced.jpg")
#         if not os.path.exists(preop_processed_path):
#             preop_processed_path = os.path.join(preop_dir, "3_preop_cropped.jpg")
#         cv2.imwrite(preop_processed_path, preop_result.processed_image)
        
#         # Save preop mask path
#         preop_mask_path = os.path.join(preop_dir, "8_preop_ring_mask.jpg")
#         cv2.imwrite(preop_mask_path, preop_result.ring_mask)
        
#         # ========== STEP 3: INITIALIZE FEATURE MODELS ==========
#         print(f"\n{'='*70}")
#         print("STEP 3: INITIALIZING FEATURE MODELS")
#         print(f"{'='*70}")
        
#         config = AccuracyConfig()
#         extractor, matcher, device = initialize_models(config)
        
#         print(f"\nConfiguration:")
#         print(f"  MAX_KEYPOINTS: {config.MAX_KEYPOINTS}")
#         print(f"  DETECTION_THRESHOLD: {config.DETECTION_THRESHOLD}")
#         print(f"  NMS_RADIUS: {config.NMS_RADIUS}")
#         print(f"  DEPTH_CONFIDENCE: {config.DEPTH_CONFIDENCE}")
#         print(f"  WIDTH_CONFIDENCE: {config.WIDTH_CONFIDENCE}")
#         print(f"  MIN_MATCH_CONFIDENCE: {config.MIN_MATCH_CONFIDENCE}")
#         print(f"  MIN_MATCHES_FOR_RANSAC: {config.MIN_MATCHES_FOR_RANSAC}")
        
#         # ========== STEP 4: EXTRACT FEATURES FROM PREOP ==========
#         print(f"\n{'='*70}")
#         print("STEP 4: EXTRACTING FEATURES FROM PREOP")
#         print(f"{'='*70}")
        
#         print(f"\nPre-op feature extraction:")
#         preop_feats = extract_features(
#             preop_processed_path,
#             extractor,
#             device,
#             mask=preop_result.ring_mask,
#             config=config
#         )
        
#         # Store preop features
#         print(f"  ✓ Preop features extracted and stored")
#         print(f"  Keypoints: {preop_feats['keypoints'].shape[1]}")
        
#         # Save preop keypoints visualization
#         try:
#             kpts0_all = preop_feats["keypoints"][0]
#             if torch.is_tensor(kpts0_all):
#                 kpts0_all = kpts0_all.detach().cpu().numpy()
            
#             save_keypoints_image(
#                 preop_processed_path,
#                 kpts0_all,
#                 os.path.join(superpoint_output_dir, "01_preop_features.jpg"),
#                 center=preop_result.limbus_info.center,
#                 radius=preop_result.limbus_info.radius,
#                 color=(0, 255, 0),
#             )
#         except Exception as e:
#             print(f"[WARNING] Failed to save preop keypoint visualization: {e}")
        
#         # ========== STEP 5: LOAD VIDEO ==========
#         print(f"\n{'='*70}")
#         print("STEP 5: LOADING INTRAOP VIDEO")
#         print(f"{'='*70}")
        
#         if not os.path.exists(INTRA_OP_VIDEO):
#             print(f"[ERROR] Video not found: {INTRA_OP_VIDEO}")
#             return
        
#         cap = cv2.VideoCapture(INTRA_OP_VIDEO)
#         if not cap.isOpened():
#             print(f"[ERROR] Could not open video: {INTRA_OP_VIDEO}")
#             return
        
#         # Get video properties
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         print(f"  Video properties:")
#         print(f"    FPS: {fps}")
#         print(f"    Resolution: {width}x{height}")
#         print(f"    Total frames: {total_frames}")
#         print(f"\n  Live preview window will open")
#         print(f"  Press 'q' in the preview window to quit early and save video")
        
#         # ========== STEP 6: PROCESS VIDEO FRAMES WITH ASYNC ANALYSIS ==========
#         print(f"\n{'='*70}")
#         print("STEP 6: PROCESSING VIDEO FRAMES (INDEPENDENT PLAYBACK + ASYNC ANALYSIS)")
#         print(f"{'='*70}")
        
#         # Create preview window
#         window_name = "Live Video Processing - Press 'q' to quit"
#         cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#         cv2.resizeWindow(window_name, min(1280, width), min(720, height))
        
#         rotation_info_list: List[Dict] = []
#         frame_num = 0
#         frames_written_count = 0
        
#         # Thread-safe shared state for rotation data
#         shared_state = SharedRotationState()
        
#         # Rotation statistics logger
#         stats_logger = RotationStatsLogger()
        
#         # Queue for sending frames to analysis thread (unbounded for smooth video)
#         frame_queue = queue.Queue()  # Unbounded queue - analysis thread handles at its own pace
        
#         # Quit flag as list for thread-safe reference
#         quit_flag_ref = [False]
        
#         # Frame analysis settings: analyze 1 frame every N frames
#         FRAME_SKIP_INTERVAL = 60
#         MIN_CONFIDENCE_THRESHOLD = 0.60
        
#         # Calculate frame delay to maintain video FPS timing
#         frame_delay_ms = max(1, int(1000 / fps)) if fps > 0 else 33
        
#         print(f"\nVideo playback configuration:")
#         print(f"  Window: {window_name}")
#         print(f"  Main thread: Detects limbus on each frame (separate YOLO model, never blocks)")
#         print(f"  Analysis thread: Analyzes 1 frame every {FRAME_SKIP_INTERVAL} frames (separate YOLO model, background)")
#         print(f"  Frame delay: {frame_delay_ms}ms (for {fps} FPS)")
#         print(f"  Video plays smoothly - limbus detection runs simultaneously with analysis")
#         print(f"  Rotation angle from analysis thread, limbus center from current frame")
#         print(f"  Press 'q' in the window to quit early\n")
        
#         # Initialize video writer to write frames directly (prevents memory accumulation)
#         output_video_path = os.path.join(video_output_dir, "output_with_lines.mp4")
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
#         if not out.isOpened():
#             print(f"[ERROR] Could not open video writer: {output_video_path}")
#             return
#         print(f"[VIDEO] Output video writer initialized: {output_video_path}")
        
#         # Start background analysis thread
#         analysis_thread = threading.Thread(
#             target=analyze_frame_async,
#             args=(frame_queue, shared_state, stats_logger, yolo_model, preop_result, preop_feats,
#                   extractor, matcher, device, config, temp_frame_dir, REFERENCE_ANGLE, quit_flag_ref),
#             daemon=True
#         )
#         analysis_thread.start()
#         print("[THREAD] Background analysis thread started\n")
        
#         # Store last good rotation data (with confidence >= threshold)
#         last_good_rotation_data = None  # (rotation_angle, limbus_center, limbus_radius, confidence, analyzed_frame_num)
        
#         while cap.isOpened() and not quit_flag_ref[0]:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame_num += 1
#             frame_start_time = time.time()
            
#             # Get latest rotation data from shared state (non-blocking)
#             rotation_angle, limbus_center, limbus_radius, confidence, analyzed_frame_num = shared_state.get()
            
#             # Determine if this frame should be sent for analysis
#             should_analyze = (frame_num == 1) or ((frame_num - 1) % FRAME_SKIP_INTERVAL == 0)
            
#             # Send frame to analysis queue (non-blocking)
#             # Check queue size to prevent memory buildup - if analysis is slow, skip this frame
#             if should_analyze:
#                 try:
#                     # Check if queue is getting too large (analysis thread is falling behind)
#                     # If queue has more than 3 frames waiting, skip this frame to keep video smooth
#                     if frame_queue.qsize() < 10:
#                         # Make a copy of the frame for analysis thread (non-blocking)
#                         frame_copy = frame.copy()
#                         frame_queue.put_nowait((frame_copy, frame_num))
#                     else:
#                         # Analysis thread is busy, skip this frame to maintain smooth playback
#                         pass
#                 except queue.Full:
#                     # Queue full, skip sending this frame for analysis (shouldn't happen with unbounded queue)
#                     pass
#                 except Exception as e:
#                     # Any other error, just continue - don't block video playback
#                     pass
            
#             # Draw line on current frame using latest rotation data
#             # Use current data if confidence >= threshold, otherwise use last good rotation data
#             rotation_angle_to_use = None
#             limbus_center_to_use = None
#             limbus_radius_to_use = None
#             confidence_to_use = 0.0
#             analyzed_frame_num_to_use = 0
            
#             if confidence >= MIN_CONFIDENCE_THRESHOLD and analyzed_frame_num > 0:
#                 # Current confidence is good, use it and update last good rotation data
#                 rotation_angle_to_use = rotation_angle
#                 limbus_center_to_use = limbus_center
#                 limbus_radius_to_use = limbus_radius
#                 confidence_to_use = confidence
#                 analyzed_frame_num_to_use = analyzed_frame_num
#                 # Update last good rotation data
#                 last_good_rotation_data = (rotation_angle, limbus_center, limbus_radius, confidence, analyzed_frame_num)
#             elif last_good_rotation_data is not None:
#                 # Current confidence is low, use last good rotation data
#                 rotation_angle_to_use, limbus_center_to_use, limbus_radius_to_use, confidence_to_use, analyzed_frame_num_to_use = last_good_rotation_data
            
#             # Draw line if we have valid rotation data to use
#             if rotation_angle_to_use is not None and analyzed_frame_num_to_use > 0:
#                 # Detect limbus on current frame for accurate drawing (fast YOLO inference)
#                 # This ensures the line is drawn from the correct limbus center for each frame
#                 # Main thread: limbus detection for drawing (never stops/lags)
#                 # Uses separate YOLO model instance to avoid blocking by analysis thread
#                 try:
#                     limbus_detected = detect_limbus(yolo_model_main, frame)
#                     if limbus_detected:
#                         current_limbus_center = limbus_detected.center
#                         current_limbus_radius = limbus_detected.radius
#                     else:
#                         # Fallback to analysis thread's limbus data or frame center
#                         current_limbus_center = limbus_center_to_use if limbus_center_to_use else (width // 2, height // 2)
#                         current_limbus_radius = limbus_radius_to_use if limbus_radius_to_use else min(width, height) // 4
#                 except Exception as e:
#                     # Fallback if detection fails (use analysis thread's limbus data)
#                     current_limbus_center = limbus_center_to_use if limbus_center_to_use else (width // 2, height // 2)
#                     current_limbus_radius = limbus_radius_to_use if limbus_radius_to_use else min(width, height) // 4
                
#                 # Use rotation angle from analysis thread, draw on current frame with detected limbus center
#                 transformed_angle = REFERENCE_ANGLE + rotation_angle_to_use
#                 final_frame = draw_line_on_frame(
#                     frame,
#                     current_limbus_center,
#                     current_limbus_radius,
#                     transformed_angle,
#                     REFERENCE_ANGLE,
#                     rotation_angle_to_use,
#                     toric_angle=TORIC_ANGLE,
#                     incision_angle=INCISION_ANGLE
#                 )
                
#                 if frame_num == analyzed_frame_num_to_use:
#                     analysis_status = f"ANALYZED (Conf: {confidence_to_use*100:.1f}%)"
#                 else:
#                     analysis_status = f"Using rotation from frame {analyzed_frame_num_to_use}"
#             else:
#                 # No rotation data available yet
#                 final_frame = frame.copy()
#                 analysis_status = "Waiting for analysis..."
            
#             # Add frame info overlay for display
#             display_frame = final_frame.copy()
#             info_text = f"Frame: {frame_num}/{total_frames} - {analysis_status}"
#             cv2.putText(display_frame, info_text, (10, height - 20),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
#             if confidence_to_use > 0 and analyzed_frame_num_to_use > 0:
#                 rot_text = f"Rotation: {rotation_angle_to_use:.2f}deg"
#                 conf_text = f"Confidence: {confidence_to_use*100:.1f}%"
#                 cv2.putText(display_frame, conf_text, (10, height - 80),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#                 cv2.putText(display_frame, rot_text, (10, height - 50),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
#             # Display frame in live preview window
#             cv2.imshow(window_name, display_frame)
            
#             # Calculate time to wait to maintain FPS
#             elapsed_ms = int((time.time() - frame_start_time) * 1000)
#             wait_time = max(1, frame_delay_ms - elapsed_ms)
            
#             # Check for 'q' key press
#             key = cv2.waitKey(wait_time) & 0xFF
#             if key == ord('q'):
#                 quit_flag_ref[0] = True
#                 print(f"\n[QUIT] 'q' pressed - Processing stopped at frame {frame_num}")
#                 break
            
#             # Write frame directly to video file (prevents memory accumulation)
#             out.write(final_frame)
#             frames_written_count += 1
            
#             # Store rotation info if this frame was just analyzed
#             if frame_num == analyzed_frame_num and rotation_angle != 0.0:
#                 rotation_info_list.append({
#                     'frame_num': frame_num,
#                     'rotation_deg': rotation_angle,
#                     'confidence': confidence,
#                     'is_reliable': confidence >= MIN_CONFIDENCE_THRESHOLD
#                 })
            
#             # Save frame to folder
#             frame_filename = os.path.join(frames_output_dir, f"frame_{frame_num:06d}.jpg")
#             cv2.imwrite(frame_filename, final_frame)
            
#             # Print progress every 30 frames
#             if frame_num % 30 == 0:
#                 print(f"  Frame {frame_num}/{total_frames}: {analysis_status} (Video playing smoothly)")
        
#         # Signal analysis thread to stop
#         quit_flag_ref[0] = True
#         frame_queue.put(None)  # Poison pill
        
#         # Wait for analysis thread to finish (with timeout)
#         analysis_thread.join(timeout=5.0)
        
#         # Close preview window
#         cv2.destroyAllWindows()
        
#         cap.release()
        
#         # Close video writer
#         out.release()
        
#         if frames_written_count == 0:
#             print(f"\n[ERROR] No frames processed")
#             return
        
#         print(f"\n✓ Processed {frames_written_count} frames")
#         print(f"  Successful rotations: {len(rotation_info_list)}")
#         print(f"  Frames saved to: {frames_output_dir}")
#         print(f"✓ Video saved: {output_video_path}")
        
#         # Save rotation statistics from logger
#         all_stats = stats_logger.get_all()
#         if all_stats:
#             stats_path = os.path.join(video_output_dir, "rotation_stats.txt")
#             with open(stats_path, 'w') as f:
#                 f.write("Frame-by-Frame Rotation Statistics\n")
#                 f.write("=" * 50 + "\n\n")
#                 for info in all_stats:
#                     f.write(f"Frame {info['frame_num']}:\n")
#                     f.write(f"  Rotation: {info['rotation_deg']:.2f}°\n")
#                     f.write(f"  Confidence: {info['confidence']*100:.1f}%\n")
#                     f.write(f"  Matches: {info['num_matches']}\n")
#                     f.write(f"  Reliable: {info['is_reliable']}\n\n")
            
#             print(f"\nRotation Statistics:")
#             print(f"  Total analyzed frames: {len(all_stats)}")
#             print(f"  Stats saved: {stats_path}")
        
#         # Clean up temp directory
#         try:
#             import shutil
#             shutil.rmtree(temp_frame_dir)
#             os.makedirs(temp_frame_dir, exist_ok=True)
#         except:
#             pass
        
#         # ========== FINAL RESULTS ==========
#         print(f"\n{'='*70}")
#         print("PIPELINE COMPLETE")
#         print(f"{'='*70}")
        
#         print(f"\n✓ SUCCESS")
#         print(f"  Processed frames: {frames_written_count}")
#         print(f"  Output video: {output_video_path}")
#         print(f"  Reference angle: {REFERENCE_ANGLE:.1f}°")
        
#         print(f"\nOutput directories:")
#         print(f"  Preprocessing: {preprocess_output_dir}")
#         print(f"  Video output: {video_output_dir}")
#         print(f"\n[LOG] All output saved to: {log_file}")
#         print(f"{'='*70}\n")
        
#     except Exception as e:
#         print(f"\n✗ ERROR: {e}")
#         import traceback
#         traceback.print_exc()
    
#     finally:
#         # Stop logging
#         logger.stop_logging()
#         print(f"[INFO] Log file saved: {log_file}")


# if __name__ == "__main__":
#     main()



