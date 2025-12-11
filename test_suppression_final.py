"""
Final comprehensive test for console window suppression.
Tests all subprocess calls and provides clear pass/fail results.
"""

import sys
import subprocess
import os

# Simulate frozen exe environment
sys.frozen = True
sys._original_stdout_exe = sys.stdout
sys._original_stderr_exe = sys.stderr

# Suppress stdout/stderr
class NullWriter:
    def write(self, text):
        pass
    def flush(self):
        pass

sys.stdout = NullWriter()
sys.stderr = NullWriter()

print("Starting tests (this won't show - simulating exe mode)...")

# Restore stdout for results
sys.stdout = sys._original_stdout_exe

print("="*70)
print("CONSOLE WINDOW SUPPRESSION TEST")
print("="*70)
print("\n⚠️  IMPORTANT: Watch your screen during this test!")
print("   If you see ANY black console windows pop up, suppression is NOT working.\n")

results = []

# Test 1: Basic subprocess.check_output with CREATE_NO_WINDOW
print("[TEST 1] Testing subprocess.check_output with CREATE_NO_WINDOW...")
try:
    if sys.platform == 'win32':
        result = subprocess.check_output(
            ['python', '--version'],
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        results.append(("TEST 1", True, "subprocess.check_output with CREATE_NO_WINDOW"))
        print("✓ PASSED - Command executed, no console window should have appeared")
    else:
        print("⏭ SKIPPED - Windows-only test")
        results.append(("TEST 1", True, "Skipped (not Windows)"))
except Exception as e:
    results.append(("TEST 1", False, str(e)))
    print(f"✗ FAILED: {e}")

# Test 2: subprocess.Popen with CREATE_NO_WINDOW
print("\n[TEST 2] Testing subprocess.Popen with CREATE_NO_WINDOW...")
try:
    if sys.platform == 'win32':
        proc = subprocess.Popen(
            ['python', '--version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        proc.wait()
        results.append(("TEST 2", True, "subprocess.Popen with CREATE_NO_WINDOW"))
        print("✓ PASSED - Process executed, no console window should have appeared")
    else:
        print("⏭ SKIPPED - Windows-only test")
        results.append(("TEST 2", True, "Skipped (not Windows)"))
except Exception as e:
    results.append(("TEST 2", False, str(e)))
    print(f"✗ FAILED: {e}")

# Test 3: Test ffprobe (if available)
print("\n[TEST 3] Testing ffprobe subprocess call...")
try:
    if sys.platform == 'win32':
        # Check if ffprobe exists
        try:
            subprocess.check_output(['where', 'ffprobe'], 
                                  stderr=subprocess.DEVNULL,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
            
            # Try a simple ffprobe command (just version, doesn't need a file)
            result = subprocess.check_output(
                ['ffprobe', '-version'],
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            results.append(("TEST 3", True, "ffprobe with CREATE_NO_WINDOW"))
            print("✓ PASSED - FFprobe executed, no console window should have appeared")
        except subprocess.CalledProcessError:
            print("⚠ SKIPPED - FFprobe not found in PATH")
            results.append(("TEST 3", True, "Skipped (ffprobe not in PATH)"))
        except FileNotFoundError:
            print("⚠ SKIPPED - FFprobe not found in PATH")
            results.append(("TEST 3", True, "Skipped (ffprobe not in PATH)"))
    else:
        print("⏭ SKIPPED - Windows-only test")
        results.append(("TEST 3", True, "Skipped (not Windows)"))
except Exception as e:
    results.append(("TEST 3", False, str(e)))
    print(f"✗ FAILED: {e}")

# Test 4: Test video_utils import and structure
print("\n[TEST 4] Testing video_utils.py code structure...")
try:
    import video_utils
    import inspect
    
    # Check _get_metadata method has CREATE_NO_WINDOW
    source = inspect.getsource(video_utils.FFmpegVideoReader._get_metadata)
    has_create_no_window = 'CREATE_NO_WINDOW' in source
    
    # Check _start_ffmpeg method has CREATE_NO_WINDOW
    source2 = inspect.getsource(video_utils.FFmpegVideoReader._start_ffmpeg)
    has_create_no_window2 = 'CREATE_NO_WINDOW' in source2
    
    if has_create_no_window and has_create_no_window2:
        results.append(("TEST 4", True, "video_utils.py has CREATE_NO_WINDOW flags"))
        print("✓ PASSED - video_utils.py contains CREATE_NO_WINDOW flags")
        print("  - _get_metadata method: ✓")
        print("  - _start_ffmpeg method: ✓")
    else:
        results.append(("TEST 4", False, "Missing CREATE_NO_WINDOW flags"))
        print("✗ FAILED - CREATE_NO_WINDOW flags not found in video_utils.py")
        if not has_create_no_window:
            print("  - _get_metadata method: ✗ Missing")
        if not has_create_no_window2:
            print("  - _start_ffmpeg method: ✗ Missing")
except Exception as e:
    results.append(("TEST 4", False, str(e)))
    print(f"✗ FAILED: {e}")

# Final Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

passed = sum(1 for _, success, _ in results if success)
total = len(results)

for test_name, success, detail in results:
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"{status} - {test_name}: {detail}")

print(f"\nResults: {passed}/{total} tests passed")

print("\n" + "="*70)
print("FINAL VERIFICATION")
print("="*70)
print("\n⚠️  CRITICAL QUESTION: Did you see ANY console windows during this test?")
print("\nIf NO windows appeared:")
print("  ✅ Console suppression is WORKING!")
print("  ✅ You can safely build your exe")
print("  ✅ The CREATE_NO_WINDOW flags are effective")
print("\nIf windows DID appear:")
print("  ❌ Console suppression is NOT working")
print("  ❌ Check that video_utils.py changes were saved")
print("  ❌ Verify CREATE_NO_WINDOW flags are present")
print("\n" + "="*70)

