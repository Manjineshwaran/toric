"""
Global subprocess patch to suppress console windows on Windows.
This should be imported at the very start of the application, before any other imports.
"""

import sys
import subprocess

if sys.platform == 'win32':
    # Patch subprocess.Popen to always use CREATE_NO_WINDOW on Windows
    # Use a class wrapper to preserve inheritance capability
    _original_popen = subprocess.Popen
    
    class _PatchedPopen(_original_popen):
        """Popen wrapper class that adds CREATE_NO_WINDOW flag on Windows"""
        def __init__(self, *args, **kwargs):
            if 'creationflags' not in kwargs:
                kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
            super().__init__(*args, **kwargs)
    
    _patched_popen = _PatchedPopen
    
    # Also patch other subprocess functions
    _original_call = subprocess.call
    _original_check_call = subprocess.check_call
    _original_check_output = subprocess.check_output
    _original_run = subprocess.run
    
    def _patched_call(*args, **kwargs):
        if 'creationflags' not in kwargs:
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        return _original_call(*args, **kwargs)
    
    def _patched_check_call(*args, **kwargs):
        if 'creationflags' not in kwargs:
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        return _original_check_call(*args, **kwargs)
    
    def _patched_check_output(*args, **kwargs):
        if 'creationflags' not in kwargs:
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        return _original_check_output(*args, **kwargs)
    
    def _patched_run(*args, **kwargs):
        if 'creationflags' not in kwargs:
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        return _original_run(*args, **kwargs)
    
    # Apply patches
    subprocess.Popen = _patched_popen
    subprocess.call = _patched_call
    subprocess.check_call = _patched_check_call
    subprocess.check_output = _patched_check_output
    subprocess.run = _patched_run
    
    # Don't print - this would show a console window if console=False
    # print("[INFO] Subprocess console window suppression enabled")

