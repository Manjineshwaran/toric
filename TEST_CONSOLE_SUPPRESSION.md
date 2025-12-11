# Testing Console Window Suppression Before Building EXE

## Quick Test Methods

### Method 1: Simple Test Script (Recommended)
Run the test script I created:

```bash
python test_console_suppression.py
```

**What to check:**
- No console windows should appear during execution
- The script tests all subprocess calls that might spawn windows

### Method 2: Test with Actual Video (Most Accurate)
Run the video_utils test:

```bash
python test_video_utils_suppression.py
```

**What to check:**
- Place a test video file (.mp4, .avi, etc.) in the directory
- Run the script
- **Watch carefully** - no console windows should appear when FFmpegVideoReader is created
- This tests the actual code path your app uses

### Method 3: Temporarily Modify app.py (Advanced)

1. **Temporarily** modify `app.py` around line 2974 to always suppress console:

```python
# Change this:
if getattr(sys, 'frozen', False):

# To this (for testing):
if True:  # Temporarily always suppress for testing
```

2. Run your app normally:
```bash
python app.py
```

3. Try loading/processing a video
4. **Watch for console windows** - none should appear
5. **IMPORTANT:** Change it back after testing!

### Method 4: Manual Visual Test

1. Run your app:
```bash
python app.py
```

2. Perform these actions and watch for console windows:
   - Load a video file
   - Start video processing
   - Navigate through the UI

3. **If you see ANY console windows**, the suppression isn't working
4. **If NO windows appear**, you're good to go!

## What Each Test Checks

- ✅ **Subprocess CREATE_NO_WINDOW flag** - Prevents FFmpeg/ffprobe from spawning windows
- ✅ **Stdout/Stderr redirection** - Prevents Python print statements from showing windows
- ✅ **Video processing** - Tests the actual code path used by your application

## Expected Behavior

**✅ GOOD (Suppression Working):**
- App runs normally
- UI appears and works
- No console windows pop up
- Video processing works silently

**❌ BAD (Suppression Not Working):**
- Console windows flash open and close
- Black command prompt windows appear briefly
- Multiple windows appear (one for each subprocess)

## After Testing

If all tests pass (no console windows):
1. ✅ You're ready to build the exe
2. Run PyInstaller with your spec file
3. The exe should run without console windows

If tests fail (windows still appear):
1. Check that all changes were saved
2. Verify `video_utils.py` has the `CREATE_NO_WINDOW` flags
3. Make sure `sys` is imported in `video_utils.py`
4. Re-run the tests

