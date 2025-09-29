#!/usr/bin/env python3
"""
Data Simulator for Thermalright System Monitor

Simulates CPU and GPU data for testing the monitoring system.
Provides various test scenarios including corner cases, edge cases, and realistic data patterns.
"""

import sys
import os
import time
import math
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thermalright_system_monitor import (
    create_monitoring_overlay, 
    update_cpu_history,
    get_cpu_info,
    get_gpu_info,
    main
)

def mock_psutil_cpu_freq():
    """Mock psutil.cpu_freq"""
    class MockFreq:
        current = 3000.0
        max = 4000.0
    return MockFreq()

def mock_subprocess_sensors():
    """Mock subprocess for sensors command - corner test"""
    class MockResult:
        returncode = 0
        stdout = """Core 0: +30.0°C
Core 1: +30.0°C  
Core 2: +105.0°C
Core 3: +105.0°C"""
    return MockResult()

def mock_subprocess_sensors_edges():
    """Mock subprocess for sensors command - edge test"""
    class MockResult:
        returncode = 0
        stdout = """Core 0: +70°C
Core 1: +30.0°C
Core 2: +80°C
Core 3: +105.0°C
Core 4: +90°C
Core 5: +30.0°C
Core 6: +105.0°C"""
    return MockResult()

def mock_subprocess_sensors_realistic():
    """Mock subprocess for sensors command - realistic test"""
    class MockResult:
        returncode = 0
        stdout = """Core 0: +45°C
Core 1: +42°C
Core 2: +48°C
Core 3: +44°C
Core 4: +52°C
Core 5: +46°C
Core 6: +49°C
Core 7: +43°C
Core 8: +47°C
Core 9: +45°C
Core 10: +51°C
Core 11: +44°C
Core 12: +48°C
Core 13: +46°C
Core 14: +50°C
Core 15: +43°C
Core 16: +47°C
Core 17: +45°C
Core 18: +49°C
Core 19: +44°C"""
    return MockResult()

def mock_cpu_percent_corners(interval=None, percpu=False):
    """Mock psutil.cpu_percent for corner test"""
    if percpu:
        return [0, 100, 0, 100]  # Four corners: 0%, 100%, 0%, 100%
    else:
        return 50.0

def mock_cpu_percent_edges(interval=None, percpu=False):
    """Mock psutil.cpu_percent for edge test"""
    if percpu:
        return [0, 25, 50, 75, 100, 50, 50]  # Various positions
    else:
        return 50.0

def mock_cpu_percent_realistic(interval=None, percpu=False):
    """Mock psutil.cpu_percent for realistic test"""
    if percpu:
        # Simulate realistic CPU usage across 20 cores
        return [15, 8, 22, 12, 35, 18, 28, 14, 31, 16, 42, 19, 38, 21, 45, 17, 33, 20, 40, 23]
    else:
        return 25.5  # Average of the above

def mock_cpu_percent_drifting(interval=None, percpu=False):
    """Mock psutil.cpu_percent with drifting data for color gradient testing"""
    current_time = time.time()
    
    if percpu:
        # Create drifting CPU usage across 20 cores with different patterns
        cores = []
        for i in range(20):
            # Each core has a different frequency and phase for variety
            base_usage = 30 + 20 * math.sin(current_time * 0.5 + i * 0.3)
            # Add some noise and different patterns
            if i % 4 == 0:  # Every 4th core has higher variation
                base_usage += 15 * math.sin(current_time * 1.2 + i * 0.5)
            elif i % 3 == 0:  # Every 3rd core has medium variation
                base_usage += 10 * math.cos(current_time * 0.8 + i * 0.4)
            
            # Clamp to 0-100%
            cores.append(max(0, min(100, base_usage)))
        return cores
    else:
        # Average usage also drifts
        return 40 + 20 * math.sin(current_time * 0.3)

def mock_subprocess_sensors_drifting():
    """Mock subprocess for sensors command with drifting temperatures"""
    current_time = time.time()
    
    class MockResult:
        returncode = 0
        stdout = ""
        
        # Generate drifting temperatures for 20 cores
        for i in range(20):
            # Base temperature with drift
            base_temp = 45 + 15 * math.sin(current_time * 0.4 + i * 0.2)
            # Add some variation based on core index
            if i % 5 == 0:  # Every 5th core runs hotter
                base_temp += 10 + 5 * math.sin(current_time * 0.6 + i * 0.3)
            elif i % 7 == 0:  # Every 7th core runs cooler
                base_temp -= 5 + 3 * math.cos(current_time * 0.7 + i * 0.2)
            
            # Clamp to reasonable range (30-85°C)
            temp = max(30, min(85, base_temp))
            stdout += f"Core {i}: +{temp:.1f}°C\n"
    
    return MockResult()

def mock_nvidia_smi(test_type="corners"):
    """Mock nvidia-smi output for GPU data simulation."""
    class MockResult:
        returncode = 0
        if test_type == "corners":
            # High usage, high temp
            stdout = "NVIDIA GeForce RTX 4090, 95, 85, 20480, 24576, 2500, 450"
        elif test_type == "edges":
            # Medium usage, medium temp
            stdout = "NVIDIA GeForce RTX 4090, 60, 65, 12288, 24576, 1800, 280"
        else:  # realistic
            # Realistic gaming load
            stdout = "NVIDIA GeForce RTX 4090, 78, 72, 16384, 24576, 2200, 320"
    return MockResult()

def get_drifting_gpu_info():
    """Get GPU info that drifts over time - called each frame"""
    return get_mocked_gpu_info("drifting")

def run_monitor_with_mocked_data(test_type="corners"):
    """Run the actual monitoring system with mocked CPU and GPU data."""
    print(f"Running thermalright_system_monitor.py with mocked {test_type} data...")
    print("Press Ctrl+C to stop")
    
    # Choose the appropriate mock functions based on test type
    if test_type == "corners":
        mock_cpu_func = mock_cpu_percent_corners
        mock_sensors_func = mock_subprocess_sensors
        model_name = "Test CPU - Corners"
        gpu_info = get_mocked_gpu_info("corners")
    elif test_type == "edges":
        mock_cpu_func = mock_cpu_percent_edges
        mock_sensors_func = mock_subprocess_sensors_edges
        model_name = "Test CPU - Edges"
        gpu_info = get_mocked_gpu_info("edges")
    elif test_type == "drifting":
        mock_cpu_func = mock_cpu_percent_drifting
        mock_sensors_func = mock_subprocess_sensors_drifting
        model_name = "Test CPU - Drifting"
        # For drifting, we need to call the function each time, not use a static value
        gpu_info = get_drifting_gpu_info
    else:  # realistic
        mock_cpu_func = mock_cpu_percent_realistic
        mock_sensors_func = mock_subprocess_sensors_realistic
        model_name = "Test CPU - Realistic"
        gpu_info = get_mocked_gpu_info("realistic")
    
    with patch('thermalright_system_monitor.psutil.cpu_percent', side_effect=mock_cpu_func), \
         patch('thermalright_system_monitor.psutil.cpu_freq', return_value=mock_psutil_cpu_freq()), \
         patch('thermalright_system_monitor.subprocess.run', return_value=mock_sensors_func()), \
         patch('thermalright_system_monitor.get_cpu_model', return_value=model_name), \
         patch('thermalright_system_monitor.get_gpu_info', side_effect=gpu_info if callable(gpu_info) else lambda: gpu_info):
        
        # Run the actual main function with mocked data
        try:
            main(preview=True)  # Use preview mode so it shows on screen
        except KeyboardInterrupt:
            print("\nStopped by user")

def get_mocked_gpu_info(test_type="corners"):
    """Generate mocked GPU info based on test type."""
    current_time = time.time()
    
    if test_type == "corners":
        return {
            'available': True,
            'name': 'NVIDIA GeForce RTX 4090',
            'usage_percent': 95,
            'temperature': 85,
            'memory_used': 20 * 1024 * 1024 * 1024,  # 20GB
            'memory_total': 24 * 1024 * 1024 * 1024,  # 24GB
            'memory_percent': 83.3,
            'fan_speed': 2500,
            'power_usage': 450
        }
    elif test_type == "edges":
        return {
            'available': True,
            'name': 'NVIDIA GeForce RTX 4090',
            'usage_percent': 60,
            'temperature': 65,
            'memory_used': 12 * 1024 * 1024 * 1024,  # 12GB
            'memory_total': 24 * 1024 * 1024 * 1024,  # 24GB
            'memory_percent': 50.0,
            'fan_speed': 1800,
            'power_usage': 280
        }
    elif test_type == "drifting":
        # GPU data that drifts over time
        base_usage = 50 + 30 * math.sin(current_time * 0.3)
        base_temp = 60 + 15 * math.sin(current_time * 0.4 + 1.5)
        base_memory = 12 + 8 * math.sin(current_time * 0.2 + 2.0)
        base_power = 250 + 100 * math.sin(current_time * 0.35 + 0.8)
        base_fan = 1500 + 800 * math.sin(current_time * 0.5 + 1.2)
        
        return {
            'available': True,
            'name': 'NVIDIA GeForce RTX 4090',
            'usage_percent': max(0, min(100, base_usage)),
            'temperature': max(30, min(85, base_temp)),
            'memory_used': max(0, min(24, base_memory)) * 1024 * 1024 * 1024,  # GB to bytes
            'memory_total': 24 * 1024 * 1024 * 1024,  # 24GB
            'memory_percent': (max(0, min(24, base_memory)) / 24) * 100,
            'fan_speed': max(0, min(3000, base_fan)),
            'power_usage': max(0, min(450, base_power))
        }
    else:  # realistic
        return {
            'available': True,
            'name': 'NVIDIA GeForce RTX 4090',
            'usage_percent': 78,
            'temperature': 72,
            'memory_used': 16 * 1024 * 1024 * 1024,  # 16GB
            'memory_total': 24 * 1024 * 1024 * 1024,  # 24GB
            'memory_percent': 66.7,
            'fan_speed': 2200,
            'power_usage': 320
        }

def test_data_simulation():
    """Test the data simulation with various scenarios."""
    print("Choose test type:")
    print("1. Four corners test (0%, 100% utilization at 30°C, 105°C)")
    print("2. Edges test (various positions along grid lines)")
    print("3. Realistic test (realistic CPU/GPU usage patterns)")
    print("4. Drifting test (data drifts to show color gradients)")
    
    choice = input("Enter choice (1/2/3/4): ").strip()
    
    if choice == "1":
        run_monitor_with_mocked_data("corners")
    elif choice == "2":
        run_monitor_with_mocked_data("edges")
    elif choice == "3":
        run_monitor_with_mocked_data("realistic")
    elif choice == "4":
        run_monitor_with_mocked_data("drifting")
    else:
        print("Invalid choice. Please run the script again and choose 1, 2, 3, or 4.")

if __name__ == "__main__":
    test_data_simulation()
