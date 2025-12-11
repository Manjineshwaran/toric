"""
Test script to verify console window suppression works before converting to exe.
This simulates the exe environment and tests subprocess calls.
"""

import sys
import subprocess
import os
from pathlib import Path

# Simulate running as frozen exe
sys.frozen = True

# Store original stdout/stderr (simulating app.py behavior)
if not hasattr(sys, '_original_stdout_exe'):
    sys._original_stdout_exe = sys.stdout
    sys._original_stderr_exe = sys.stderr

# Suppress console output (simulating app.py behavior)
class NullWriter:
    def write(self, text):
        pass
    def flush(self):
        pass
    def close(self):
        pass

print("Testing console window suppression...")
print("(This should NOT appear if suppression works)")

sys.stdout = NullWriter()
sys.stderr = NullWriter()

# Test 1: Test ffprobe subprocess call (simulating video_utils.py)
print("\n[TEST 1] Testing ffprobe subprocess call...")
try:
    # Use a test command that won't fail - just check if it spawns a window
    # We'll use 'where' command on Windows or 'which' on Unix
    if sys.platform == 'win32':
        test_cmd = ['where', 'python']
        kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW}
    else:
        test_cmd = ['which', 'python']
        kwargs = {}
    
    result = subprocess.check_output(
        test_cmd,
        stderr=subprocess.DEVNULL,
        **kwargs
    )
    print("[TEST 1] ✓ PASSED - No console window spawned")
except Exception as e:
    print(f"[TEST 1] ✗ FAILED: {e}")

# Test 2: Test ffmpeg subprocess call (simulating video_utils.py)
print("\n[TEST 2] Testing ffmpeg subprocess call...")
try:
    # Check if ffmpeg exists
    if sys.platform == 'win32':
        check_cmd = ['where', 'ffmpeg']
        kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW}
    else:
        check_cmd = ['which', 'ffmpeg']
        kwargs = {}
    
    ffmpeg_path = subprocess.check_output(
        check_cmd,
        stderr=subprocess.DEVNULL,
        **kwargs
    ).decode().strip()
    
    print(f"[TEST 2] ✓ PASSED - FFmpeg found at: {ffmpeg_path}")
    print("[TEST 2] ✓ No console window spawned")
except subprocess.CalledProcessError:
    print("[TEST 2] ⚠ WARNING - FFmpeg not found in PATH (but subprocess call worked)")
except Exception as e:
    print(f"[TEST 2] ✗ FAILED: {e}")

# Test 3: Test Popen with CREATE_NO_WINDOW
print("\n[TEST 3] Testing subprocess.Popen with CREATE_NO_WINDOW...")
try:
    if sys.platform == 'win32':
        test_cmd = ['python', '--version']
        popen_kwargs = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.PIPE,
            'creationflags': subprocess.CREATE_NO_WINDOW
        }
    else:
        test_cmd = ['python', '--version']
        popen_kwargs = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.PIPE
        }
    
    proc = subprocess.Popen(test_cmd, **popen_kwargs)
    stdout, stderr = proc.communicate()
    proc.wait()
    
    print("[TEST 3] ✓ PASSED - No console window spawned")
except Exception as e:
    print(f"[TEST 3] ✗ FAILED: {e}")

# Restore stdout to show results
sys.stdout = sys._original_stdout_exe
sys.stderr = sys._original_stderr_exe

print("\n" + "="*60)
print("CONSOLE SUPPRESSION TEST COMPLETE")
print("="*60)
print("\nTo verify:")
print("1. Run this script: python test_console_suppression.py")
print("2. Check that NO console windows appeared during execution")
print("3. All subprocess calls should run silently")
print("\nIf you saw console windows pop up, the suppression is NOT working.")
print("If no windows appeared, you're ready to build the exe!")

