import os
import sys
import platform

# Only needed on macOS
if platform.system() == 'Darwin':
    try:
        import torch
        # Get the directory where torch libraries are located
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        
        # Add to DYLD_LIBRARY_PATH
        if 'DYLD_LIBRARY_PATH' in os.environ:
            os.environ['DYLD_LIBRARY_PATH'] = f"{torch_lib_path}:{os.environ['DYLD_LIBRARY_PATH']}"
        else:
            os.environ['DYLD_LIBRARY_PATH'] = torch_lib_path
            
        # For macOS 10.11+, you might need to use DYLD_FALLBACK_LIBRARY_PATH
        if 'DYLD_FALLBACK_LIBRARY_PATH' in os.environ:
            os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = f"{torch_lib_path}:{os.environ['DYLD_FALLBACK_LIBRARY_PATH']}"
        else:
            os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = torch_lib_path
            
    except ImportError:
        raise ImportError("PyTorch must be installed before importing torchrdit_rs")
    
from .torchrdit_rs import *