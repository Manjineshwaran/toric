"""
Limbus Detector with 70-Degree Line
====================================
Standalone script that detects the limbus (iris boundary), finds its center,
and draws a line at 70 degrees from the horizontal. Plays video output live.

QUICK START:
    1. Edit the CONFIGURATION section (around line 350) to set your video source
    2. Run: python limbus_detector_70deg.py
    3. Press 'q' to quit, 'p' to pause/resume

Configuration Options:
    - VIDEO_SOURCE: Path to video file or camera index (0 for webcam)
    - YOLO_MODEL_PATH: Path to YOLO model weights (default: model\intraop_latest.pt)
    - OUTPUT_VIDEO_PATH: Optional path to save output (None to skip saving)
    - ANGLE_TO_DRAW: Angle in degrees to draw (default: 70)
    - SHOW_LIVE_WINDOW: Show live video window (default: True)

Features:
    ✓ Real-time limbus detection
    ✓ Center point identification
    ✓ 70-degree line visualization
    ✓ Reference line at 0 degrees (horizontal)
    ✓ Live FPS counter
    ✓ Detection status overlay
    ✓ Optional video output saving
"""

import cv2
import numpy as np
import sys
import os
from typing import Optional, Tuple
from dataclasses import dataclass
from video_utils import open_video


@dataclass
class LimbusInfo:
    """Information about detected limbus"""
    center: Tuple[int, int]  # (x, y) center coordinates
    radius: int              # radius in pixels
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) bounding box
    confidence: float        # detection confidence


class LimbusDetector70Deg:
    """Detects limbus and draws 70-degree reference line"""
    
    def __init__(self, model_path: str):
        """
        Initialize detector with YOLO model
        
        Args:
            model_path: Path to YOLO model weights
        """
        print(f"[INFO] Loading YOLO model from: {model_path}")
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"[INFO] Model loaded successfully. Classes: {self.model.names}")
        except ImportError:
            print("[ERROR] ultralytics package not found. Install with: pip install ultralytics")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            sys.exit(1)
    
    def detect_limbus(self, frame: np.ndarray, 
                     class_name: str = "dilated limbus") -> Optional[LimbusInfo]:
        """
        Detect limbus in a frame
        
        Args:
            frame: BGR image frame
            class_name: Target class name to detect
            
        Returns:
            LimbusInfo if detected, None otherwise
        """
        try:
            results = self.model(frame, verbose=False)[0]
            
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                
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
        except Exception as e:
            print(f"[WARNING] Detection failed: {e}")
            return None
        
        return None
    
    def draw_70_degree_line(self, frame: np.ndarray, 
                           center: Tuple[int, int], 
                           radius: int,
                           angle_deg: float = 70.0,
                           draw_circle: bool = True,
                           draw_center: bool = True,
                           draw_reference: bool = True) -> np.ndarray:
        """
        Draw 70-degree line and limbus visualization
        
        Args:
            frame: Input BGR image
            center: (x, y) center coordinates
            radius: Limbus radius
            angle_deg: Angle in degrees (default 70)
            draw_circle: Whether to draw limbus circle
            draw_center: Whether to draw center point
            draw_reference: Whether to draw horizontal reference line (0 degrees)
            
        Returns:
            Frame with overlays drawn
        """
        output = frame.copy()
        
        # Draw limbus circle (cyan)
        if draw_circle:
            cv2.circle(output, center, radius, (255, 255, 0), 2)
        
        # Draw center point (red)
        if draw_center:
            cv2.circle(output, center, 5, (0, 0, 255), -1)
        
        # Draw horizontal reference line (0 degrees) - green
        if draw_reference:
            ref_x_end = int(center[0] + radius * np.cos(np.radians(0)))
            ref_y_end = int(center[1] - radius * np.sin(np.radians(0)))
            cv2.line(output, center, (ref_x_end, ref_y_end), (0, 255, 0), 2)
            cv2.putText(output, "0°", (ref_x_end + 10, ref_y_end), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw 70-degree line (blue)
        # Note: In image coordinates, y increases downward, so we negate the sin component
        angle_rad = np.radians(angle_deg)
        x_end = int(center[0] + radius * np.cos(angle_rad))
        y_end = int(center[1] - radius * np.sin(angle_rad))
        
        # Draw line from center to edge
        cv2.line(output, center, (x_end, y_end), (255, 0, 0), 3)

        #Draw Reference angle line (yellow)
        cv2.line(output, center, (x_end, y_end), (0, 255, 255), 3)

        #Draw toric angle line (green)
        cv2.line(output, center, (x_end, y_end), (0, 255, 0), 3)

        #Draw incision angle line (red)
        cv2.line(output, center, (x_end, y_end), (0, 0, 255), 3)

        # Draw line label
        cv2.putText(output, f"{angle_deg:.0f}°", (x_end + 10, y_end), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        return output
    
    def add_info_overlay(self, frame: np.ndarray, 
                        limbus_info: Optional[LimbusInfo],
                        fps: float = 0.0) -> np.ndarray:
        """
        Add information overlay to frame
        
        Args:
            frame: Input frame
            limbus_info: Detected limbus information
            fps: Current FPS
            
        Returns:
            Frame with info overlay
        """
        output = frame.copy()
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay area
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (350, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
        
        # Add text information
        y_offset = 35
        line_height = 25
        
        # Title
        cv2.putText(output, "Limbus Detector - 70° Line", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height
        
        # FPS
        cv2.putText(output, f"FPS: {fps:.1f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += line_height
        
        # Limbus detection status
        if limbus_info:
            cv2.putText(output, f"Status: DETECTED", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += line_height
            
            cv2.putText(output, f"Center: ({limbus_info.center[0]}, {limbus_info.center[1]})", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            cv2.putText(output, f"Radius: {limbus_info.radius}px", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(output, f"Status: NOT DETECTED", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return output
    
    def process_video(self, video_source, 
                     output_path: Optional[str] = None,
                     show_live: bool = True) -> None:
        """
        Process video source and display/save results
        
        Args:
            video_source: Video file path or camera index (0 for default webcam)
            output_path: Optional path to save output video
            show_live: Whether to display live window
        """
        # Open video source (use FFmpeg for video files to handle corrupted metadata)
        if isinstance(video_source, int):
            cap = open_video("", is_camera=True, camera_index=video_source)
            print(f"[INFO] Opening camera {video_source}...")
        else:
            cap = open_video(video_source, is_camera=False)
            print(f"[INFO] Opening video: {video_source}")
        
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video source: {video_source}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Default for webcam
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Video properties: {width}x{height} @ {fps:.1f} FPS")
        if total_frames > 0:
            print(f"[INFO] Total frames: {total_frames}")
        
        # Setup video writer if output path specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"[INFO] Saving output to: {output_path}")
        
        # Processing loop
        frame_count = 0
        last_limbus = None  # Store last good detection
        
        # For FPS calculation
        import time
        prev_time = time.time()
        
        print("\n[INFO] Starting processing...")
        print("[INFO] Press 'q' to quit, 'p' to pause/resume")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    print("\n[INFO] End of video or read error")
                    break
                
                frame_count += 1
                
                # Detect limbus
                limbus_info = self.detect_limbus(frame)
                
                # Use last good detection if current frame fails
                if limbus_info is not None:
                    last_limbus = limbus_info
                elif last_limbus is not None:
                    limbus_info = last_limbus
                
                # Draw 70-degree line if limbus detected
                if limbus_info:
                    frame = self.draw_70_degree_line(
                        frame, 
                        limbus_info.center, 
                        limbus_info.radius,
                        angle_deg=70.0,
                        draw_circle=True,
                        draw_center=True,
                        draw_reference=True
                    )
                
                # Calculate FPS
                curr_time = time.time()
                fps_actual = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                
                # Add info overlay
                frame = self.add_info_overlay(frame, limbus_info, fps_actual)
                
                # Write to output video
                if writer:
                    writer.write(frame)
                
                # Show progress
                if frame_count % 30 == 0:
                    if total_frames > 0:
                        progress = (frame_count / total_frames) * 100
                        print(f"\r[INFO] Processing: Frame {frame_count}/{total_frames} ({progress:.1f}%) | FPS: {fps_actual:.1f}", end='')
                    else:
                        print(f"\r[INFO] Processing: Frame {frame_count} | FPS: {fps_actual:.1f}", end='')
            
            # Display live window
            if show_live:
                if paused:
                    # Show paused indicator
                    paused_frame = frame.copy()
                    cv2.putText(paused_frame, "PAUSED - Press 'p' to resume", 
                               (width // 2 - 150, height // 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow('Limbus Detector - 70° Line', paused_frame)
                else:
                    cv2.imshow('Limbus Detector - 70° Line', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[INFO] Quit requested by user")
                    break
                elif key == ord('p'):
                    paused = not paused
                    if paused:
                        print("\n[INFO] Paused")
                    else:
                        print("\n[INFO] Resumed")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"\n[INFO] Processed {frame_count} frames")
        print("[INFO] Done!")


# ==============================================================================
# CONFIGURATION - Edit these settings before running
# ==============================================================================

# ----- VIDEO SOURCE -----
# Option 1: Use webcam (default)
VIDEO_SOURCE = r"D:\AIDS\cv\orb_eye\_inputs\IMG_4484\OD-2025-11-19_171152_fixed.mp4"  # 0 = default camera, 1 = second camera, etc.

# Option 2: Use video file (uncomment and set path)
# VIDEO_SOURCE = r"D:\AIDS\cv\orb_eye\videos\eye_surgery.mp4"
# VIDEO_SOURCE = r"path\to\your\video.mp4"

# ----- YOLO MODEL PATH -----
# Default path used in the project
YOLO_MODEL_PATH = r"model\intraop_latest.pt"

# ----- OUTPUT VIDEO -----
# Save processed video (set to None to disable saving)
OUTPUT_VIDEO_PATH = None  # No saving by default

# To save output, uncomment and set path:
# OUTPUT_VIDEO_PATH = r"output\limbus_70deg_output.mp4"

# ----- ANGLE SETTINGS -----
# The angle to draw from horizontal (in degrees)
ANGLE_TO_DRAW = 70.0

# ----- DISPLAY SETTINGS -----
# Show live window while processing
SHOW_LIVE_WINDOW = True  # Set to False to run headless (only save to file)

# ==============================================================================


def main():
    """Main entry point"""
    
    # Print configuration
    print("\n" + "="*70)
    print("LIMBUS DETECTOR - 70° LINE")
    print("="*70)
    print(f"Video Source: {VIDEO_SOURCE}")
    print(f"YOLO Model: {YOLO_MODEL_PATH}")
    print(f"Angle: {ANGLE_TO_DRAW}°")
    if OUTPUT_VIDEO_PATH:
        print(f"Output Video: {OUTPUT_VIDEO_PATH}")
    else:
        print("Output Video: Not saving")
    print(f"Live Display: {'Enabled' if SHOW_LIVE_WINDOW else 'Disabled'}")
    print("="*70 + "\n")
    
    # Check if model exists
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"[ERROR] YOLO model not found: {YOLO_MODEL_PATH}")
        print("[INFO] Please update YOLO_MODEL_PATH in the configuration section")
        return
    
    # Check if video file exists (if not camera)
    if not isinstance(VIDEO_SOURCE, int):
        if not os.path.exists(VIDEO_SOURCE):
            print(f"[ERROR] Video file not found: {VIDEO_SOURCE}")
            print("[INFO] Please update VIDEO_SOURCE in the configuration section")
            return
    
    # Create output directory if saving video
    if OUTPUT_VIDEO_PATH:
        output_dir = os.path.dirname(OUTPUT_VIDEO_PATH)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # Create detector
    detector = LimbusDetector70Deg(YOLO_MODEL_PATH)
    
    # Update the draw method to use configured angle
    original_draw = detector.draw_70_degree_line
    def custom_draw(frame, center, radius, angle_deg=ANGLE_TO_DRAW, **kwargs):
        return original_draw(frame, center, radius, angle_deg, **kwargs)
    detector.draw_70_degree_line = custom_draw
    
    # Process video
    try:
        detector.process_video(
            VIDEO_SOURCE,
            output_path=OUTPUT_VIDEO_PATH,
            show_live=SHOW_LIVE_WINDOW
        )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

