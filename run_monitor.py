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
from thermalright_system_monitor import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Thermalright system monitor')
    parser.add_argument('--preview', action='store_true', 
                       help='Show the monitor image in a preview window instead of sending to device')
    parser.add_argument('--plot-x', type=int, default=30,
                       help='X position of plot bottom-left corner (default: 30)')
    parser.add_argument('--plot-y', type=int, default=40,
                       help='Y position of plot bottom-left corner (default: 40)')
    parser.add_argument('--plot-w', type=int, default=250,
                       help='Plot area width (default: 250)')
    parser.add_argument('--plot-h', type=int, default=400,
                       help='Plot area height (default: 400)')
    args = parser.parse_args()
    
    # Pass all arguments to main function
    main(preview=args.preview, plot_x=args.plot_x, plot_y=args.plot_y, 
         plot_w=args.plot_w, plot_h=args.plot_h)
