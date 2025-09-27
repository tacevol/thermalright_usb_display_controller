#!/usr/bin/env python3
"""
Convenience script to run the system monitor from the project root.
This script can be run from anywhere and will always find the correct paths.
"""

import sys
import argparse
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import and run the main monitor
from monitor.thermalright_system_monitor import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Thermalright system monitor')
    parser.add_argument('--preview', action='store_true', 
                       help='Show the monitor image in a preview window instead of sending to device')
    args = parser.parse_args()
    
    # Pass preview flag to main function
    main(preview=args.preview)
