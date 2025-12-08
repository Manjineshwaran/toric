# Limbus Detector with 70° Line

A standalone script that detects the limbus (iris boundary), finds its center, and draws a line at 70 degrees from the horizontal. Displays live video output.

## Quick Start

### Step 1: Configure the Script

Open `limbus_detector_70deg.py` and edit the **CONFIGURATION** section (around line 350):

```python
# For webcam (default)
VIDEO_SOURCE = 0

# Or for video file
# VIDEO_SOURCE = r"path\to\your\video.mp4"

# Model path (default)
YOLO_MODEL_PATH = r"model\intraop_latest.pt"

# Optional: Save output
# OUTPUT_VIDEO_PATH = r"output\limbus_70deg_output.mp4"

# Angle to draw
ANGLE_TO_DRAW = 70.0
```

### Step 2: Run the Script

```bash
python limbus_detector_70deg.py
```

### Step 3: Controls

- Press **`q`** to quit
- Press **`p`** to pause/resume

## Features

- ✅ **Real-time limbus detection** using YOLO
- ✅ **Center point identification** with precise coordinates
- ✅ **70-degree line visualization** (or any custom angle)
- ✅ **Reference line** at 0 degrees (horizontal)
- ✅ **Live FPS counter** for performance monitoring
- ✅ **Detection status overlay** showing center and radius
- ✅ **Video output saving** (optional)
- ✅ **Pause/Resume** functionality
- ✅ **Fallback mechanism** - uses last good detection if current frame fails

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `VIDEO_SOURCE` | Video file path or camera index | `0` (webcam) |
| `YOLO_MODEL_PATH` | Path to YOLO model weights | `model\intraop_latest.pt` |
| `OUTPUT_VIDEO_PATH` | Path to save output video | `None` (no saving) |
| `ANGLE_TO_DRAW` | Angle in degrees to draw | `70.0` |
| `SHOW_LIVE_WINDOW` | Show live display window | `True` |

## Visual Elements

The script draws the following on each frame:

1. **Cyan Circle** - Detected limbus boundary
2. **Red Dot** - Limbus center point
3. **Green Line** - 0° reference (horizontal)
4. **Blue Line** - 70° angle line (or custom angle)
5. **Info Overlay** - FPS, status, coordinates, radius

## Examples

### Example 1: Use Webcam
```python
VIDEO_SOURCE = 0  # Default camera
YOLO_MODEL_PATH = r"model\intraop_latest.pt"
OUTPUT_VIDEO_PATH = None
```

### Example 2: Process Video File
```python
VIDEO_SOURCE = r"D:\videos\eye_surgery.mp4"
YOLO_MODEL_PATH = r"model\intraop_latest.pt"
OUTPUT_VIDEO_PATH = r"output\processed_video.mp4"
```

### Example 3: Different Angle
```python
VIDEO_SOURCE = 0
YOLO_MODEL_PATH = r"model\intraop_latest.pt"
ANGLE_TO_DRAW = 45.0  # Draw 45-degree line instead
```

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- NumPy
- Ultralytics YOLO (`ultralytics`)

Install dependencies:
```bash
pip install opencv-python numpy ultralytics
```

## Troubleshooting

**Model not found:**
- Make sure `model\intraop_latest.pt` exists in your project folder
- Update `YOLO_MODEL_PATH` to the correct path

**Video not found:**
- Check that the video file path is correct
- Use raw string: `r"path\to\video.mp4"`

**Camera not working:**
- Try different camera indices: 0, 1, 2, etc.
- Make sure no other application is using the camera

**No limbus detected:**
- Check video quality and lighting
- Verify the YOLO model is trained for limbus detection
- The script will use the last good detection as fallback

## Integration with Main Project

This script uses the same:
- YOLO model path as the main project (`model\intraop_latest.pt`)
- Detection methods from `preprocess_robust.py`
- Configuration structure from `handler.py`

You can easily integrate this into the main project or use it standalone for quick testing and visualization.

