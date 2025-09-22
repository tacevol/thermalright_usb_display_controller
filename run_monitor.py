#!/usr/bin/env python3
"""
Convenience script to run the system monitor from the project root.
This script can be run from anywhere and will always find the correct paths.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import and run the main monitor
from monitor.thermalright_system_monitor import main

if __name__ == "__main__":
    main()
