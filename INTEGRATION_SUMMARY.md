# UI + Main Logic Integration Summary

## Overview
The UI (`app.py`) has been successfully integrated with the main pipeline logic. The integration is handled through three key files:

## File Structure

### 1. **app.py** - PyQt5 UI Application
- Provides the graphical user interface for the toric lens surgery tracking system
- Contains tabs for:
  - Configuration (YOLO models, confidence thresholds)
  - Pre-op image preprocessing
  - Axis setup (reference, toric, incision angles)
  - Live tracking (video processing)
- Handles all user interactions and displays

### 2. **handler.py** - Configuration Bridge
- `PipelineConfigHandler` class manages all configuration between UI and pipeline
- Thread-safe singleton pattern
- Stores:
  - Model paths (pre-op and intra-op YOLO models)
  - Confidence thresholds
  - Preprocessing parameters
  - Angle configurations (reference, toric, incision)
  - Pre-op preprocessing results
  - Manual offsets for limbus center adjustment

### 3. **main.py** - Pipeline Entry Point (NEW)
- **This is the newly created bridge file**
- Contains `process_video_for_ui()` function that the UI calls
- Responsibilities:
  - Gets configuration from handler
  - Loads YOLO models
  - Initializes feature extraction models
  - Processes video frames in real-time
  - Calls UI callback to update display
  - Manages background analysis thread
  - Returns statistics for saving

### 4. **merged_pipeline.py** - Core Logic
- Contains all the main pipeline functions:
  - `SharedRotationState` - Thread-safe rotation data sharing
  - `RotationStatsLogger` - Statistics collection
  - `test_frame_at_multiple_angles()` - Multi-angle frame testing
  - `draw_line_on_frame()` - Drawing reference/toric/incision lines
  - `analyze_frame_async()` - Background frame analysis
  - Frame rotation and matching logic
- These functions are imported and used by `main.py`

## How It Works

### Workflow:
```
User Interaction (app.py)
    ↓
Configuration Storage (handler.py)
    ↓
Pipeline Execution (main.py)
    ↓
Core Processing (merged_pipeline.py)
    ↓
Results back to UI (app.py via callback)
```

### Key Integration Points:

1. **Configuration Flow:**
   - User sets parameters in UI → `handler.py` stores them
   - `main.py` reads from `handler.py` → Uses in pipeline

2. **Pre-op Processing:**
   - User loads image in UI → UI preprocesses using `preprocess_robust.py`
   - Result stored in `handler.py` → `main.py` retrieves it for feature extraction

3. **Video Processing:**
   - User starts tracking → UI calls `main.py.process_video_for_ui()`
   - `main.py` uses `merged_pipeline.py` functions for processing
   - Each frame is sent back to UI via `frame_callback()`
   - Statistics collected and returned to UI for saving

4. **Thread Safety:**
   - UI runs in main thread
   - Video processing runs in separate thread
   - Background analysis runs in another thread
   - Shared state protected by locks in `SharedRotationState`

## Running the Application

```bash
python app.py
```

The application will:
1. Launch the PyQt5 UI
2. Allow configuration of models and parameters
3. Process pre-op images
4. Track rotation in real-time on video/camera feed
5. Display results live in the UI
6. Save processed video and statistics

## Important Notes

- **Do not modify `handler.py`** - It's the bridge between UI and pipeline
- **All inputs are managed by `handler.py`** - Direct file I/O minimized
- **UI remains responsive** - Processing runs in background threads
- **Press 'q' to quit tracking** - 'p' to pause/resume

## Dependencies

- PyQt5 (for UI)
- OpenCV (for video/image processing)
- PyTorch (for deep learning models)
- NumPy (for numerical operations)
- YOLO model files in `model/` directory
- All preprocessing and feature extraction modules

## Output Structure

```
output/
├── robust_preprocess/
│   ├── preop/          # Pre-op preprocessing results
│   └── intraop/        # Intra-op preprocessing results
├── superpoint_robust/  # Feature extraction visualizations
└── video_output/
    ├── frames_with_lines/     # Individual processed frames
    ├── output_with_lines.mp4  # Final output video
    └── rotation_stats.txt     # Frame-by-frame statistics
```

## Success! 🎉

The UI and main logic are now fully integrated and ready to use.

