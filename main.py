"""
Main Pipeline Entry Point for UI Integration
=============================================

This module bridges the PyQt5 UI (app.py) with the main pipeline logic (merged_pipeline.py).
It provides the process_video_for_ui function that the UI calls to process video with live tracking.

Author: Toric Lens Surgery Pipeline
"""

import os
import cv2
import numpy as np
import time
import queue
import threading
import sys
import contextlib
from pathlib import Path
from typing import Callable, Optional, Tuple, List

# Import handler for configuration
from handler import get_config_handler

# Import preprocessing functions
from preprocess_robust import (
    load_yolo_model,
    preprocess_single,
    ImageType,
    Config,
    detect_limbus
)

# Import SuperPoint feature detection functions
from superpoint_6_log_saved import (
    initialize_models,
    extract_features,
    match_features,
    apply_geometric_filtering,
    estimate_rotation_robust,
    AccuracyConfig
)

# Import functions from merged_pipeline
from merged_pipeline import (
    SharedRotationState,
    RotationStatsLogger,
    test_frame_at_multiple_angles,
    draw_line_on_frame,
    analyze_frame_async
)


def detect_available_cameras(max_cameras: int = 10) -> List[int]:
    """
    Detect available cameras by trying to open them.
    Suppresses OpenCV error messages during detection.
    
    Args:
        max_cameras: Maximum number of camera indices to check (default: 10)
    
    Returns:
        List of available camera indices
    """
    available_cameras = []
    
    # Suppress OpenCV error messages during camera detection
    # Try to set OpenCV log level to suppress errors (may not be available in all versions)
    original_log_level = None
    try:
        if hasattr(cv2, 'getLogLevel'):
            original_log_level = cv2.getLogLevel()
        if hasattr(cv2, 'setLogLevel'):
            # Try to set to silent - use different constants based on OpenCV version
            try:
                cv2.setLogLevel(cv2.LOG_LEVEL_SILENT)
            except AttributeError:
                try:
                    cv2.setLogLevel(0)  # Some versions use numeric levels
                except:
                    pass
    except:
        pass
    
    # Suppress stderr for OpenCV errors (works regardless of OpenCV version)
    @contextlib.contextmanager
    def suppress_stderr():
        with open(os.devnull, 'w') as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stderr = old_stderr
    
    try:
        with suppress_stderr():
            for camera_index in range(max_cameras):
                try:
                    cap = cv2.VideoCapture(camera_index)
                    if cap.isOpened():
                        # Try to read a frame to confirm camera works
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            available_cameras.append(camera_index)
                        cap.release()
                    else:
                        # If we fail to open this camera, assume no more cameras
                        break
                except Exception as e:
                    # Error opening camera, skip it
                    continue
    finally:
        # Restore original log level if we changed it
        if original_log_level is not None:
            try:
                if hasattr(cv2, 'setLogLevel'):
                    cv2.setLogLevel(original_log_level)
            except:
                pass
    
    return available_cameras


def open_camera_preview(camera_index: int) -> bool:
    """
    Open a camera preview window to verify camera works.
    Suppresses OpenCV error messages during camera access.
    
    Args:
        camera_index: Camera index to open (0, 1, 2, etc.)
    
    Returns:
        True if camera opened successfully, False otherwise
    """
    # Suppress OpenCV error messages
    original_log_level = None
    try:
        if hasattr(cv2, 'getLogLevel'):
            original_log_level = cv2.getLogLevel()
        if hasattr(cv2, 'setLogLevel'):
            # Try to set to silent - use different constants based on OpenCV version
            try:
                cv2.setLogLevel(cv2.LOG_LEVEL_SILENT)
            except AttributeError:
                try:
                    cv2.setLogLevel(0)  # Some versions use numeric levels
                except:
                    pass
    except:
        pass
    
    @contextlib.contextmanager
    def suppress_stderr():
        with open(os.devnull, 'w') as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stderr = old_stderr
    
    try:
        with suppress_stderr():
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                return False
            
            # Try to read a frame to verify camera works
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                print(f"[CAMERA] Camera {camera_index} opened successfully")
                return True
            else:
                print(f"[CAMERA] Camera {camera_index} failed to read frame")
                return False
    except Exception as e:
        print(f"[CAMERA] Error opening camera {camera_index}: {e}")
        return False
    finally:
        # Restore original log level if we changed it
        if original_log_level is not None:
            try:
                if hasattr(cv2, 'setLogLevel'):
                    cv2.setLogLevel(original_log_level)
            except:
                pass


def process_video_for_ui_simple(frame_callback: Callable[[np.ndarray, float, float], None],
                                quit_flag_ref: list,
                                pause_flag_ref: list,
                                show_reference: bool = True,
                                show_toric: bool = True,
                                show_incision: bool = True):
    """
    Simple fast video processing - just limbus detection and angle drawing.
    Based on limbus_detector_70deg.py - no heavy feature matching!
    
    Args:
        frame_callback: Callback function(frame, rotation_angle, confidence) to update UI
        quit_flag_ref: List with single bool element [False] for quit flag
        pause_flag_ref: List with single bool element [False] for pause flag
        show_reference: Whether to show reference line
        show_toric: Whether to show toric line
        show_incision: Whether to show incision line
    
    Returns:
        None (no stats in simple mode)
    """
    
    # Get configuration from handler
    config_handler = get_config_handler()
    
    # Get paths and parameters from handler
    intraop_video_path = config_handler.get_intraop_video_path()
    reference_angle = config_handler.get_reference_angle()
    toric_angle = config_handler.get_toric_angle()
    incision_angle = config_handler.get_incision_angle()
    yolo_model_path = config_handler.get_intraop_model_path()
    preop_result = config_handler.get_preop_result()  # Get preop result for limbus radius
    
    # Validate inputs
    if intraop_video_path is None:
        raise ValueError("Intraop video/camera path not set")
    
    # Check if camera mode
    is_camera_mode = False
    camera_index = 0
    if "camera:" in intraop_video_path.lower():
        is_camera_mode = True
        try:
            camera_index = int(intraop_video_path.split(":")[-1])
        except:
            camera_index = 0
        print(f"[VIDEO] Using camera mode: Camera {camera_index}")
    elif not os.path.exists(intraop_video_path):
        raise ValueError(f"Video file not found: {intraop_video_path}")
    
    print(f"\n{'='*70}")
    print("VIDEO PROCESSING - FAST LIMBUS TRACKING MODE")
    print(f"{'='*70}")
    print(f"Video source: {intraop_video_path}")
    print(f"Reference angle: {reference_angle}°")
    print(f"Toric angle: {toric_angle}°")
    print(f"Incision angle: {incision_angle}°")
    print(f"Mode: Real-time limbus detection (no feature matching)")
    print(f"{'='*70}\n")
    
    # Load YOLO model
    print("[INIT] Loading YOLO model...")
    yolo_model = load_yolo_model(yolo_model_path)
    print("  ✓ YOLO model loaded")
    
    # Open video capture
    if is_camera_mode:
        cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(intraop_video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {intraop_video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # Default for camera
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n[VIDEO] Video properties:")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    if not is_camera_mode:
        print(f"  Total frames: {total_frames}")
    print(f"  Press 'q' to quit, 'p' to pause/resume")
    
    # Create output directories
    video_output_dir = "output/video_output"
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Initialize video writer
    # output_video_path = os.path.join(video_output_dir, "output_with_lines.mp4")
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    # if not out.isOpened():
    #     raise ValueError(f"Could not open video writer: {output_video_path}")
    # print(f"[VIDEO] Output video: {output_video_path}")
    
    # Frame processing loop
    frame_num = 0
    frames_written_count = 0
    last_limbus = None  # Store last good detection
    
    # Frame processing settings
    FRAME_PROCESS_INTERVAL = 4  # Process 1 frame out of every N frames (1 out of 4)
    
    # Calculate frame delay for video timing
    # Since we process 1/4 frames, show each processed frame for 4x duration to maintain FPS
    base_frame_delay_ms = max(1, int(1000 / fps)) if fps > 0 else 33
    frame_delay_ms = base_frame_delay_ms * FRAME_PROCESS_INTERVAL
    
    print("\n[PROCESSING] Starting video processing...")
    print(f"[INFO] Fast mode: Processing 1 out of {FRAME_PROCESS_INTERVAL} frames")
    print(f"[INFO] Frame delay: {frame_delay_ms}ms (adjusted for {FRAME_PROCESS_INTERVAL}x frame skipping)")
    
    while cap.isOpened() and not quit_flag_ref[0]:
        # Check pause flag
        while pause_flag_ref[0] and not quit_flag_ref[0]:
            time.sleep(0.1)
        
        if quit_flag_ref[0]:
            break
        
        frame_num += 1
        
        # Skip frames: only process every 4th frame (1, 5, 9, 13, ...)
        if (frame_num - 1) % FRAME_PROCESS_INTERVAL != 0:
            # Skip this frame - read and discard without processing
            ret = cap.read()[0]
            if not ret:
                break
            continue  # Skip all processing for this frame
        
        # Process this frame (every 4th frame)
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start_time = time.time()
        
        # Detect limbus on current frame (fast ~200ms)
        try:
            limbus_detected = detect_limbus(yolo_model, frame)
            if limbus_detected:
                last_limbus = limbus_detected
                current_center = limbus_detected.center
                current_radius = limbus_detected.radius
            elif last_limbus:
                current_center = last_limbus.center
                current_radius = last_limbus.radius
            else:
                current_center = (width // 2, height // 2)
                current_radius = min(width, height) // 4
        except:
            if last_limbus:
                current_center = last_limbus.center
                current_radius = last_limbus.radius
            else:
                current_center = (width // 2, height // 2)
                current_radius = min(width, height) // 4
        
        # Draw lines at configured angles (no rotation calculation - just draw the angles!)
        # Get preop limbus radius for offset calculation (matches Tab 3)
        preop_limbus_radius = preop_result.limbus_info.radius if preop_result and preop_result.limbus_info else None
        final_frame = draw_line_on_frame(
            frame,
            current_center,
            current_radius,
            reference_angle,  # Draw at reference angle
            reference_angle,
            0.0,  # No rotation offset (simple mode)
            toric_angle=toric_angle if show_toric else None,
            incision_angle=incision_angle if show_incision else None,
            preop_limbus_radius=preop_limbus_radius,
            show_reference=show_reference,
            show_limbus=True  # Simple mode always shows limbus
        )
        
        # Write frame to video (non-blocking)
        # try:
        #     out.write(final_frame)
        #     frames_written_count += 1
        # except:
            # pass
        
        # Call UI callback
        try:
            frame_callback(final_frame, 0.0, 1.0)  # No rotation angle, 100% confidence
        except:
            pass
        
        # Maintain FPS: wait appropriate time for processed frame
        if not is_camera_mode:
            elapsed_ms = int((time.time() - frame_start_time) * 1000)
            if elapsed_ms < (frame_delay_ms - 2):
                wait_time = frame_delay_ms - elapsed_ms
                time.sleep(wait_time / 1000.0)
        
        # Print progress
        if frame_num % (120 * FRAME_PROCESS_INTERVAL) == 0:
            print(f"  Frame {frame_num}: Limbus detected, angles drawn")
    
    # Close video
    cap.release()
    # out.release()
    
    # print(f"\n[COMPLETE] Video processing finished")
    # print(f"  Processed frames: {frames_written_count}")
    # print(f"  Output video: {output_video_path}")
    
    return None  # No stats in simple mode


def process_video_for_ui(frame_callback: Callable[[np.ndarray, float, float], None],
                         quit_flag_ref: list,
                         pause_flag_ref: list,
                         show_reference: bool = True,
                         show_toric: bool = True,
                         show_incision: bool = True,
                         show_limbus: bool = True,
                         show_reference_ref: list = None,
                         show_toric_ref: list = None,
                         show_incision_ref: list = None,
                         show_limbus_ref: list = None) -> RotationStatsLogger:
    """
    Process video with live tracking and UI callback integration.
    
    Args:
        frame_callback: Callback function(frame, rotation_angle, confidence) to update UI
        quit_flag_ref: List with single bool element [False] for quit flag
        pause_flag_ref: List with single bool element [False] for pause flag
        show_reference: Whether to show reference line
        show_toric: Whether to show toric line
        show_incision: Whether to show incision line
    
    Returns:
        RotationStatsLogger instance with collected statistics
    """
    
    # Get configuration from handler
    config_handler = get_config_handler()
    
    # Get paths and parameters from handler
    intraop_video_path = config_handler.get_intraop_video_path()
    preop_result = config_handler.get_preop_result()
    reference_angle = config_handler.get_reference_angle()
    toric_angle = config_handler.get_toric_angle()
    incision_angle = config_handler.get_incision_angle()
    yolo_model_path = config_handler.get_intraop_model_path()
    matching_confidence_threshold = config_handler.get_matching_confidence_threshold()
    
    # Validate inputs
    if preop_result is None:
        raise ValueError("Pre-op image must be preprocessed before starting tracking")
    
    if intraop_video_path is None:
        raise ValueError("Intraop video/camera path not set")
    
    # Check if camera mode
    is_camera_mode = False
    camera_index = 0
    if "camera:" in intraop_video_path.lower():
        is_camera_mode = True
        try:
            camera_index = int(intraop_video_path.split(":")[-1])
        except:
            camera_index = 0
        print(f"[VIDEO] Using camera mode: Camera {camera_index}")
    elif not os.path.exists(intraop_video_path):
        raise ValueError(f"Video file not found: {intraop_video_path}")
    
    # Create output directories
    preprocess_output_dir = "output/robust_preprocess"
    intraop_dir = os.path.join(preprocess_output_dir, "intraop")
    video_output_dir = "output/video_output"
    temp_frame_dir = os.path.join(video_output_dir, "temp_frames")
    frames_output_dir = os.path.join(video_output_dir, "frames_with_lines")
    
    os.makedirs(intraop_dir, exist_ok=True)
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(temp_frame_dir, exist_ok=True)
    os.makedirs(frames_output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("VIDEO PROCESSING WITH LIVE TRACKING")
    print(f"{'='*70}")
    print(f"Video source: {intraop_video_path}")
    print(f"Reference angle: {reference_angle}°")
    print(f"Toric angle: {toric_angle}°")
    print(f"Incision angle: {incision_angle}°")
    print(f"Matching confidence threshold: {matching_confidence_threshold}")
    print(f"{'='*70}\n")
    
    # Load YOLO models (separate instances for main thread and analysis thread)
    print("[INIT] Loading YOLO models...")
    yolo_model_main = load_yolo_model(yolo_model_path)  # For main thread (live video)
    yolo_model_analysis = load_yolo_model(yolo_model_path)  # For analysis thread
    print("  ✓ Loaded YOLO models")
    
    # Initialize feature extraction models
    print("[INIT] Initializing feature models...")
    config = AccuracyConfig()
    extractor, matcher, device = initialize_models(config)
    print("  ✓ Feature models initialized")
    
    # Extract features from preprocessed preop
    print("[INIT] Extracting features from pre-op image...")
    preop_processed_path = os.path.join("output/robust_preprocess/preop", "5_preop_enhanced.jpg")
    if not os.path.exists(preop_processed_path):
        preop_processed_path = os.path.join("output/robust_preprocess/preop", "3_preop_cropped.jpg")
    
    # Save preprocessed preop if not already saved
    if not os.path.exists(preop_processed_path):
        os.makedirs(os.path.dirname(preop_processed_path), exist_ok=True)
        cv2.imwrite(preop_processed_path, preop_result.processed_image)
    
    preop_feats = extract_features(
        preop_processed_path,
        extractor,
        device,
        mask=preop_result.ring_mask,
        config=config
    )
    print(f"  ✓ Pre-op features extracted: {preop_feats['keypoints'].shape[1]} keypoints")
    
    # Open video capture
    if is_camera_mode:
        cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(intraop_video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {intraop_video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # Default for camera
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n[VIDEO] Video properties:")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    if not is_camera_mode:
        print(f"  Total frames: {total_frames}")
    print(f"  Press 'q' to quit, 'p' to pause/resume")
    
    # Create shared state and logger
    shared_state = SharedRotationState()
    stats_logger = RotationStatsLogger()
    
    # Frame queue for analysis thread
    frame_queue = queue.Queue()
    
    # Frame processing settings
    FRAME_PROCESS_INTERVAL = 4  # Process 1 frame out of every N frames (1 out of 4)
    FRAME_SKIP_INTERVAL = 60  # Analyze 1 frame every N frames (for analysis thread)
    MIN_CONFIDENCE_THRESHOLD = matching_confidence_threshold
    
    # Calculate frame delay for video timing
    # Since we process 1/4 frames, show each processed frame for 4x duration to maintain FPS
    base_frame_delay_ms = max(1, int(1000 / fps)) if fps > 0 else 33
    frame_delay_ms = base_frame_delay_ms * FRAME_PROCESS_INTERVAL
    
    print(f"\n[CONFIG] Processing configuration:")
    print(f"  Frame process interval: {FRAME_PROCESS_INTERVAL} (process 1 out of {FRAME_PROCESS_INTERVAL} frames)")
    print(f"  Analysis skip interval: {FRAME_SKIP_INTERVAL} (analyze 1 out of {FRAME_SKIP_INTERVAL} frames)")
    print(f"  Min confidence threshold: {MIN_CONFIDENCE_THRESHOLD}")
    print(f"  Frame delay: {frame_delay_ms}ms (adjusted for {FRAME_PROCESS_INTERVAL}x frame skipping)")
    
    # Initialize video writer
    # output_video_path = os.path.join(video_output_dir, "output_with_lines.mp4")
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    # if not out.isOpened():
    #     raise ValueError(f"Could not open video writer: {output_video_path}")
    # print(f"[VIDEO] Output video: {output_video_path}")
    
    # Start background analysis thread
    analysis_thread = threading.Thread(
        target=analyze_frame_async,
        args=(frame_queue, shared_state, stats_logger, yolo_model_analysis, preop_result, preop_feats,
              extractor, matcher, device, config, temp_frame_dir, reference_angle, quit_flag_ref),
        daemon=True
    )
    analysis_thread.start()
    print("[THREAD] Background analysis thread started\n")
    
    # Store last good rotation data
    last_good_rotation_data = None
    
    # Frame processing loop
    frame_num = 0
    frames_written_count = 0
    
    print("[PROCESSING] Starting video processing loop...")
    
    while cap.isOpened() and not quit_flag_ref[0]:
        # Check pause flag
        while pause_flag_ref[0] and not quit_flag_ref[0]:
            time.sleep(0.1)
        
        if quit_flag_ref[0]:
            break
        
        frame_num += 1
        
        # Skip frames: only process every 4th frame (1, 5, 9, 13, ...)
        if (frame_num - 1) % FRAME_PROCESS_INTERVAL != 0:
            # Skip this frame - read and discard without processing
            ret = cap.read()[0]
            if not ret:
                break
            continue  # Skip all processing for this frame
        
        # Process this frame (every 4th frame)
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start_time = time.time()
        
        # Get latest rotation data from shared state
        rotation_angle, limbus_center, limbus_radius, confidence, analyzed_frame_num = shared_state.get()
        
        # Determine if this frame should be analyzed (for analysis thread)
        should_analyze = (frame_num == 1) or ((frame_num - 1) % FRAME_SKIP_INTERVAL == 0)
        
        # Send frame to analysis queue
        # Note: Analysis thread will automatically skip processing if confidence > 90% (handled in analyze_frame_async)
        if should_analyze and frame_queue.qsize() < 10:
            try:
                frame_copy = frame.copy()
                frame_queue.put_nowait((frame_copy, frame_num))
            except:
                pass
        
        # Determine rotation data to use for drawing
        rotation_angle_to_use = None
        limbus_center_to_use = None
        limbus_radius_to_use = None
        confidence_to_use = 0.0
        
        if confidence >= MIN_CONFIDENCE_THRESHOLD and analyzed_frame_num > 0:
            # Current confidence is good
            rotation_angle_to_use = rotation_angle
            limbus_center_to_use = limbus_center
            limbus_radius_to_use = limbus_radius
            confidence_to_use = confidence
            last_good_rotation_data = (rotation_angle, limbus_center, limbus_radius, confidence, analyzed_frame_num)
        elif last_good_rotation_data is not None:
            # Use last good rotation data
            rotation_angle_to_use, limbus_center_to_use, limbus_radius_to_use, confidence_to_use, analyzed_frame_num = last_good_rotation_data
        
        # Draw lines on frame if we have rotation data
        if rotation_angle_to_use is not None and analyzed_frame_num > 0:
            # Detect limbus on CURRENT frame for accurate center position
            # (YOLO is fast ~200ms, won't cause lag. Heavy processing is in background thread)
            try:
                limbus_detected = detect_limbus(yolo_model_main, frame)
                if limbus_detected:
                    current_limbus_center = limbus_detected.center
                    current_limbus_radius = limbus_detected.radius
                else:
                    # Fallback to analysis thread data if detection fails
                    current_limbus_center = limbus_center_to_use if limbus_center_to_use else (width // 2, height // 2)
                    current_limbus_radius = limbus_radius_to_use if limbus_radius_to_use else min(width, height) // 4
            except:
                # Fallback on error
                current_limbus_center = limbus_center_to_use if limbus_center_to_use else (width // 2, height // 2)
                current_limbus_radius = limbus_radius_to_use if limbus_radius_to_use else min(width, height) // 4
            
            # Draw lines using CURRENT frame's limbus + analysis thread's rotation angle
            transformed_angle = reference_angle + rotation_angle_to_use
            # Get preop limbus radius for offset calculation (matches Tab 3)
            preop_limbus_radius = preop_result.limbus_info.radius if preop_result and preop_result.limbus_info else None
            
            # Read checkbox states dynamically if references provided, otherwise use defaults
            current_show_reference = show_reference_ref[0] if show_reference_ref is not None else show_reference
            current_show_toric = show_toric_ref[0] if show_toric_ref is not None else show_toric
            current_show_incision = show_incision_ref[0] if show_incision_ref is not None else show_incision
            current_show_limbus = show_limbus_ref[0] if show_limbus_ref is not None else show_limbus
            
            final_frame = draw_line_on_frame(
                frame,
                current_limbus_center,
                current_limbus_radius,
                transformed_angle,
                reference_angle,
                rotation_angle_to_use,
                toric_angle=toric_angle if current_show_toric else None,
                incision_angle=incision_angle if current_show_incision else None,
                preop_limbus_radius=preop_limbus_radius,
                show_reference=current_show_reference,
                show_limbus=current_show_limbus
            )
        else:
            # No rotation data available yet
            final_frame = frame.copy()
        
        # Write frame to video in background (non-blocking)
        # try:
        #     out.write(final_frame)
        #     frames_written_count += 1
        # except:
        #     pass  # Don't block if write fails
        
        # Skip saving individual frames - they cause lag! Only save final video.
        # If you need frames, extract them from the final video later.
        
        # Call UI callback to update display (non-blocking)
        try:
            frame_callback(final_frame, rotation_angle_to_use if rotation_angle_to_use else 0.0, confidence_to_use)
        except:
            pass  # Never block on UI callback errors
        
        # Print progress (very infrequently - console I/O can cause lag!)
        if frame_num % 120 == 0:  # Every 120 frames (~5 seconds at 25fps)
            status = f"Frame {analyzed_frame_num} analyzed" if analyzed_frame_num > 0 else "Analysis pending"
            rotation_str = f"{rotation_angle_to_use:.2f}" if rotation_angle_to_use is not None else "N/A"
            print(f"  Frame {frame_num}: {status} | Rotation: {rotation_str}° | Conf: {confidence_to_use*100:.1f}%")
        
        # SMOOTH PLAYBACK: Maintain target framerate without blocking
        if not is_camera_mode:
            elapsed_ms = int((time.time() - frame_start_time) * 1000)
            # Only sleep if significantly ahead of schedule
            if elapsed_ms < (frame_delay_ms - 2):  # 2ms buffer for processing overhead
                wait_time = frame_delay_ms - elapsed_ms
                time.sleep(wait_time / 1000.0)
            # If behind schedule, continue immediately (smooth playback priority!)
    
    # Signal analysis thread to stop
    quit_flag_ref[0] = True
    frame_queue.put(None)  # Poison pill
    
    # Wait for analysis thread
    analysis_thread.join(timeout=5.0)
    
    # Close video
    cap.release()
    # out.release()
    
    # print(f"\n[COMPLETE] Video processing finished")
    # print(f"  Processed frames: {frames_written_count}")
    # print(f"  Output video: {output_video_path}")
    
    # Clean up temp files
    try:
        import shutil
        shutil.rmtree(temp_frame_dir)
        os.makedirs(temp_frame_dir, exist_ok=True)
    except:
        pass
    
    # Return stats logger for UI to save
    return stats_logger

