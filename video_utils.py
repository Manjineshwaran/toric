"""
Video Utilities using FFmpeg
============================
Handles video reading using FFmpeg to correctly process videos with incorrect metadata.
Always uses FFmpeg for video files to ensure full video is processed.
"""

import subprocess
import sys
import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import Tuple, Optional, Iterator


def _find_ffmpeg_executable(name: str) -> str:
    """
    Find FFmpeg executable (ffmpeg or ffprobe).
    Checks bundled location first, then system PATH.
    
    Args:
        name: Executable name ('ffmpeg' or 'ffprobe')
    
    Returns:
        Path to executable or just name if found in PATH
    """
    # Check if running as PyInstaller exe
    if getattr(sys, 'frozen', False):
        # Running as compiled exe
        exe_dir = Path(sys.executable).parent
        
        # Check for bundled FFmpeg in exe directory
        bundled_path = exe_dir / f"{name}.exe"
        if bundled_path.exists():
            return str(bundled_path)
        
        # Check in ffmpeg subdirectory (common bundling location)
        ffmpeg_dir = exe_dir / "ffmpeg"
        if ffmpeg_dir.exists():
            bundled_path = ffmpeg_dir / f"{name}.exe"
            if bundled_path.exists():
                return str(bundled_path)
    
    # Fall back to system PATH
    # On Windows, try to find in common locations
    if sys.platform == 'win32':
        common_paths = [
            r"C:\ffmpeg\bin",
            r"C:\Program Files\ffmpeg\bin",
            r"C:\Program Files (x86)\ffmpeg\bin",
        ]
        
        for base_path in common_paths:
            exe_path = Path(base_path) / f"{name}.exe"
            if exe_path.exists():
                return str(exe_path)
    
    # Return just the name - will use system PATH
    return name


class FFmpegVideoReader:
    """Video reader using FFmpeg to handle videos with incorrect metadata."""
    
    def __init__(self, video_path: str):
        """
        Initialize FFmpeg video reader.
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.process = None
        self.width = None
        self.height = None
        self.fps = None
        self.duration = None
        self.total_frames = None
        self.frame_size = None
        
        # Get video metadata using ffprobe
        self._get_metadata()
        
        # Start FFmpeg process
        self._start_ffmpeg()
    
    def _get_metadata(self):
        """Get video metadata using ffprobe."""
        try:
            # Find ffprobe executable (bundled or in PATH)
            ffprobe_exe = _find_ffmpeg_executable("ffprobe")
            
            probe_cmd = [
                ffprobe_exe,
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                self.video_path
            ]
            
            # Suppress console window on Windows
            kwargs = {}
            if sys.platform == 'win32':
                kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
            
            probe_out = subprocess.check_output(
                probe_cmd, 
                stderr=subprocess.DEVNULL,
                **kwargs
            )
            info = json.loads(probe_out)
            
            # Get video stream
            video_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
            
            self.width = int(video_stream["width"])
            self.height = int(video_stream["height"])
            
            # Get FPS
            fps_eval = video_stream.get("r_frame_rate", "25/1")
            if "/" in fps_eval:
                num, den = map(int, fps_eval.split("/"))
                self.fps = num / den if den > 0 else 25.0
            else:
                self.fps = float(fps_eval)
            
            # Get duration (this is the correct duration from FFmpeg)
            self.duration = float(info["format"].get("duration", 0))
            
            # Calculate total frames
            if self.duration > 0 and self.fps > 0:
                self.total_frames = int(self.duration * self.fps)
            else:
                self.total_frames = 0
            
            self.frame_size = self.width * self.height * 3
            
            print(f"[FFMPEG] Video metadata:")
            print(f"  Resolution: {self.width} x {self.height}")
            print(f"  FPS: {self.fps:.2f}")
            print(f"  Duration: {self.duration/60:.2f} min ({self.duration:.2f} sec)")
            print(f"  Total frames: {self.total_frames}")
            
        except Exception as e:
            raise ValueError(f"Failed to get video metadata with ffprobe: {e}")
    
    def _start_ffmpeg(self):
        """Start FFmpeg process to read video frames."""
        try:
            # Find ffmpeg executable (bundled or in PATH)
            ffmpeg_exe = _find_ffmpeg_executable("ffmpeg")
            
            cmd = [
                ffmpeg_exe,
                "-i", self.video_path,
                "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-f", "rawvideo",
                "pipe:1"
            ]
            
            # Start FFmpeg process, redirect stderr to DEVNULL to prevent freeze
            # Suppress console window on Windows
            kwargs = {'stdout': subprocess.PIPE, 'stderr': subprocess.DEVNULL, 
                     'bufsize': self.frame_size * 10}  # Buffer for 10 frames
            if sys.platform == 'win32':
                kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
            
            self.process = subprocess.Popen(cmd, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to start FFmpeg process: {e}")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame from video.
        Compatible with cv2.VideoCapture.read() interface.
        
        Returns:
            (success, frame) tuple. success is False when video ends.
        """
        if self.process is None:
            return False, None
        
        try:
            raw = self.process.stdout.read(self.frame_size)
            if not raw or len(raw) < self.frame_size:
                # Video ended
                return False, None
            
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
            return True, frame
        except Exception as e:
            # Error reading frame (likely EOF or process ended)
            return False, None
    
    def release(self):
        """Release video resources."""
        if self.process:
            try:
                self.process.stdout.close()
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                try:
                    self.process.kill()
                except:
                    pass
            self.process = None
    
    def isOpened(self) -> bool:
        """Check if video is opened."""
        return self.process is not None
    
    def get(self, prop: int) -> float:
        """
        Get video property (compatible with cv2.VideoCapture interface).
        
        Args:
            prop: Property ID (cv2.CAP_PROP_*)
        
        Returns:
            Property value
        """
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        elif prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        elif prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.total_frames
        elif prop == cv2.CAP_PROP_POS_FRAMES:
            # Not easily trackable with FFmpeg, return 0
            return 0
        else:
            return 0.0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def open_video(video_path: str, is_camera: bool = False, camera_index: int = 0):
    """
    Open video file or camera.
    
    For video files: Always uses FFmpeg to handle incorrect metadata.
    For cameras: Uses cv2.VideoCapture (cameras don't have metadata issues).
    
    Args:
        video_path: Path to video file (ignored if is_camera=True)
        is_camera: If True, open camera instead of video file
        camera_index: Camera index (if is_camera=True)
    
    Returns:
        Video reader object (FFmpegVideoReader or cv2.VideoCapture)
    """
    if is_camera:
        # Use OpenCV for cameras (no metadata issues)
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_index}")
        return cap
    else:
        # Always use FFmpeg for video files to handle incorrect metadata
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        return FFmpegVideoReader(video_path)

