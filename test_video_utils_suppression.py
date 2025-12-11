"""
Test script to verify video_utils.py subprocess calls don't spawn console windows.
This tests the actual FFmpeg subprocess calls used in the application.
"""

import sys
import os

# Simulate frozen exe environment (like PyInstaller)
sys.frozen = True

# Suppress stdout/stderr like app.py does
class NullWriter:
    def write(self, text):
        pass
    def flush(self):
        pass
    def close(self):
        pass

# Store original (for restoring)
original_stdout = sys.stdout
original_stderr = sys.stderr
sys._original_stdout_exe = sys.stdout
sys._original_stderr_exe = sys.stderr

# Redirect to null (simulating exe mode)
sys.stdout = NullWriter()
sys.stderr = NullWriter()

print("Testing video_utils.py console suppression...")
print("(This should NOT appear - we're simulating exe mode)")

try:
    # Import video_utils after setting up suppression
    from video_utils import FFmpegVideoReader, open_video
    
    # Restore stdout to show test results
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    
    print("\n" + "="*60)
    print("TESTING VIDEO_UTILS CONSOLE SUPPRESSION")
    print("="*60)
    
    # Test 1: Check if the module has the right imports
    print("\n[TEST 1] Checking video_utils.py imports...")
    import video_utils
    if hasattr(video_utils, 'sys') and hasattr(video_utils, 'subprocess'):
        print("✓ video_utils.py has sys and subprocess imported")
    else:
        print("✗ Missing imports in video_utils.py")
    
    # Test 2: Try to create a video reader (this will call ffprobe and ffmpeg)
    print("\n[TEST 2] Testing FFmpegVideoReader subprocess calls...")
    print("Looking for a test video file...")
    
    # Look for common video files in the directory
    test_video = None
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.m4v']
    
    for ext in video_extensions:
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.lower().endswith(ext):
                    test_video = os.path.join(root, file)
                    break
            if test_video:
                break
        if test_video:
            break
    
    if test_video and os.path.exists(test_video):
        print(f"Found test video: {test_video}")
        print("Attempting to open with FFmpegVideoReader...")
        print("WATCH FOR CONSOLE WINDOWS - None should appear!")
        
        try:
            reader = FFmpegVideoReader(test_video)
            print("✓ FFmpegVideoReader created successfully")
            print("✓ No console windows should have appeared during creation")
            
            # Try reading a frame
            success, frame = reader.read()
            if success:
                print("✓ Successfully read a frame (no console windows)")
            else:
                print("⚠ Frame read returned False (video may be at end)")
            
            reader.release()
            print("✓ Reader released successfully")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            print("(This might be expected if ffmpeg is not in PATH)")
    else:
        print("⚠ No test video found - skipping actual FFmpeg test")
        print("  To fully test:")
        print("  1. Place a video file (mp4, avi, etc.) in this directory")
        print("  2. Run this test again")
        print("  3. Watch for console windows when FFmpegVideoReader is created")
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("\nIf you saw ANY console windows during this test:")
    print("  ✗ Console suppression is NOT working properly")
    print("\nIf NO console windows appeared:")
    print("  ✓ Console suppression is working correctly!")
    print("  ✓ You're ready to build the exe")
    print("\nNote: Make sure ffmpeg and ffprobe are in your system PATH")
    
except ImportError as e:
    # Restore stdout before showing error
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    print(f"\n✗ Import error: {e}")
    print("Make sure video_utils.py is in the same directory")
except Exception as e:
    # Restore stdout before showing error
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    print(f"\n✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

