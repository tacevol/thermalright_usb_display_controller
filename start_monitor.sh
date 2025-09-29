#!/bin/bash
# Simple script to start the Thermalright monitor

# Navigate to the project directory
cd /home/turbo/Documents/python/thermalright_usb_display_controller

# Activate virtual environment
source venv/bin/activate

# Start the monitor (device mode by default)
python run_monitor.py

# Keep terminal open to see any errors
read -p "Press Enter to close..."
