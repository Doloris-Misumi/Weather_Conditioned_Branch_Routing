
import sys
import os
import importlib.util

print(f"Python executable: {sys.executable}")

try:
    import nms
    print(f"nms imported successfully.")
    print(f"nms package path: {os.path.dirname(nms.__file__)}")
    
    # Check nms.py in the package directory
    package_dir = os.path.dirname(nms.__file__)
    nms_py_path = os.path.join(package_dir, 'nms.py')
    
    if os.path.exists(nms_py_path):
        print(f"Found nms.py at: {nms_py_path}")
        with open(nms_py_path, 'r') as f:
            content = f.read()
            
            if "DEBUG_MARKER" in content:
                print("STATUS: DEBUG_MARKER found in nms.py (My changes are present)")
            else:
                print("STATUS: DEBUG_MARKER NOT found in nms.py (Using different file/version?)")

            if 'print(r)' in content:
                # check if it is commented
                lines = content.split('\n')
                active_print = False
                for line in lines:
                    if 'print(r)' in line and not line.strip().startswith('#'):
                        active_print = True
                        print(f"ALERT: Found active print: {line.strip()}")
                
                if not active_print:
                    print("STATUS: 'print(r)' is commented out in nms.py")
            else:
                print("STATUS: 'print(r)' not found in nms.py")
    else:
        print(f"ERROR: nms.py not found at {nms_py_path}")

except ImportError:
    print("ERROR: Could not import nms")
