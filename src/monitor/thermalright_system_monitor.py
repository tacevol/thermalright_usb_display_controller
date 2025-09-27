#!/usr/bin/env python3
"""
Thermalright USB Display System Monitor

A real-time system monitor that displays CPU usage data on Thermalright USB displays.
Features BTOP-style themes, timeseries heatmaps, and runtime theme cycling.

Usage:
    python thermalright_monitor.py [options]
    
Options:
    --refresh-rate FPS    Refresh rate in FPS (default: 15)
    --theme INDEX         Initial theme 0-29 (default: 0)
    --quality 1-100       JPEG quality (default: 80)

Runtime Controls:
    Press 't' to cycle themes
    Press 'q' to quit
"""

import io
import time
import sys
import struct
import os
import re
import subprocess
import threading
import select
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import usb.core
import usb.util
import psutil

# Optional NVIDIA NVML support
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

# ============================================================================
# Configuration
# ============================================================================

# Display settings
DISPLAY_WIDTH = 480
DISPLAY_HEIGHT = 480
MARGIN = 0

# Background image settings
BACKGROUND_IMAGE_PATH = None  # Will be set to assets/images/ if available
BLUR_LEVELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 blur levels
GRADIENT_LEVELS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]  # 10 opacity levels
current_blur_level = 0  # Index into BLUR_LEVELS
current_gradient_level = 0  # Index into GRADIENT_LEVELS
background_image = None
stats_enabled = True  # Toggle for showing/hiding all stats

# Colors (RGB format for JPEG compatibility)
WHITE = (240, 240, 240)
BACKGROUND = (0, 0, 0)
BAR_BACKGROUND = (45, 45, 55)

# Font paths (Linux)
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

# Device settings
VENDOR_ID = 0x87ad
PRODUCT_ID = 0x70db
CHUNK_SIZE = 4096
HEADER_LENGTH = 64
LENGTH_FIELD_OFFSET = 60

# Get project root for asset paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
FRAME_BIN_PATH = PROJECT_ROOT / "assets" / "data" / "frame.bin"

# Get actual CPU core count at startup
NUM_CPU_CORES = psutil.cpu_count(logical=True)  # Include hyperthreading
print(f"Detected {NUM_CPU_CORES} CPU cores")

# Debug: Show temperature sensor detection at startup (same as BTOP)
try:
    import subprocess
    result = subprocess.run(['sensors', 'coretemp-isa-0000'], 
                          capture_output=True, text=True, timeout=1)
    if result.returncode == 0:
        print("Using lm-sensors (same as BTOP) for temperature readings")
        # Count available cores
        core_count = sum(1 for line in result.stdout.split('\n') if 'Core ' in line and '°C' in line)
        print(f"Found {core_count} CPU core temperature sensors")
    else:
        print("lm-sensors not available, falling back to hwmon")
except Exception as e:
    print(f"Could not detect temperature sensors: {e}")

# ============================================================================
# Theme System
# ============================================================================

class Theme:
    """BTOP-style color theme for CPU usage visualization."""
    
    def __init__(self, name: str, cpu_start: Tuple[int, int, int], 
                 cpu_mid: Tuple[int, int, int], cpu_end: Tuple[int, int, int],
                 temp_start: Tuple[int, int, int] = None, 
                 temp_mid: Tuple[int, int, int] = None, 
                 temp_end: Tuple[int, int, int] = None):
        self.name = name
        self.cpu_start = cpu_start
        self.cpu_mid = cpu_mid
        self.cpu_end = cpu_end
        
        # Temperature colors - use CPU colors as fallback if not specified
        self.temp_start = temp_start if temp_start else cpu_start
        self.temp_mid = temp_mid if temp_mid else cpu_mid
        self.temp_end = temp_end if temp_end else cpu_end
        
        # Legacy support for old code
        self.start_color = cpu_start
        self.mid_color = cpu_mid
        self.end_color = cpu_end
    
    def get_color_for_usage(self, usage_pct: float) -> Tuple[int, int, int]:
        """Map CPU usage percentage to color using this theme's gradient."""
        if usage_pct is None or usage_pct < 0:
            return (80, 80, 80)
        
        usage = max(0, min(100, usage_pct))
        
        if usage < 50:
            # Interpolate between start and mid colors
            t = usage / 50.0
            r = int(self.start_color[0] + (self.mid_color[0] - self.start_color[0]) * t)
            g = int(self.start_color[1] + (self.mid_color[1] - self.start_color[1]) * t)
            b = int(self.start_color[2] + (self.mid_color[2] - self.start_color[2]) * t)
        else:
            # Interpolate between mid and end colors
            t = (usage - 50) / 50.0
            r = int(self.cpu_mid[0] + (self.cpu_end[0] - self.cpu_mid[0]) * t)
            g = int(self.cpu_mid[1] + (self.cpu_end[1] - self.cpu_mid[1]) * t)
            b = int(self.cpu_mid[2] + (self.cpu_end[2] - self.cpu_mid[2]) * t)
        
        return (r, g, b)
    
    def get_color_for_temperature(self, temp_c: float) -> Tuple[int, int, int]:
        """Map temperature in Celsius to color using this theme's temperature gradient."""
        if temp_c is None or temp_c < 0:
            return (80, 80, 80)
        
        # Temperature range: 30°C to 85°C (typical CPU range)
        # Map to 0-100% for color interpolation
        temp_normalized = max(0, min(100, ((temp_c - 30) / 55) * 100))
        
        if temp_normalized < 50:
            # Interpolate between temperature start and mid colors
            t = temp_normalized / 50.0
            r = int(self.temp_start[0] + (self.temp_mid[0] - self.temp_start[0]) * t)
            g = int(self.temp_start[1] + (self.temp_mid[1] - self.temp_start[1]) * t)
            b = int(self.temp_start[2] + (self.temp_mid[2] - self.temp_start[2]) * t)
        else:
            # Interpolate between temperature mid and end colors
            t = (temp_normalized - 50) / 50.0
            r = int(self.temp_mid[0] + (self.temp_end[0] - self.temp_mid[0]) * t)
            g = int(self.temp_mid[1] + (self.temp_end[1] - self.temp_mid[1]) * t)
            b = int(self.temp_mid[2] + (self.temp_end[2] - self.temp_mid[2]) * t)
        
        return (r, g, b)

# BTOP-style themes
THEMES = [
    # Official BTOP Themes (extracted from https://github.com/aristocratos/btop/tree/main/themes)
    Theme("Default", 
          cpu_start=(0, 255, 0), cpu_mid=(255, 255, 0), cpu_end=(255, 0, 0),
          temp_start=(0, 200, 255), temp_mid=(255, 200, 0), temp_end=(255, 100, 100)),  # BTOP built-in default theme
    Theme("HotPurpleTrafficLight",
          cpu_start=(0, 255, 0), cpu_mid=(204, 255, 102), cpu_end=(255, 0, 0),
          temp_start=(0, 255, 0), temp_mid=(255, 153, 51), temp_end=(255, 0, 0)),
    Theme("Adapta",
          cpu_start=(0, 188, 212), cpu_mid=(212, 212, 0), cpu_end=(255, 0, 64),
          temp_start=(0, 188, 212), temp_mid=(212, 212, 0), temp_end=(255, 0, 64)),
    Theme("Adwaita Dark",
          cpu_start=(98, 160, 234), cpu_mid=(28, 113, 216), cpu_end=(224, 27, 36),
          temp_start=(98, 160, 234), temp_mid=(28, 113, 216), temp_end=(224, 27, 36)),
    Theme("Adwaita",
          cpu_start=(26, 95, 180), cpu_mid=(26, 95, 180), cpu_end=(192, 28, 40),
          temp_start=(26, 95, 180), temp_mid=(26, 95, 180), temp_end=(192, 28, 40)),
    Theme("Ayu",
          cpu_start=(223, 191, 255), cpu_mid=(210, 166, 255), cpu_end=(163, 122, 204),
          temp_start=(223, 191, 255), temp_mid=(210, 166, 255), temp_end=(163, 122, 204)),
    Theme("Dracula",
          cpu_start=(189, 147, 249), cpu_mid=(139, 233, 253), cpu_end=(80, 250, 123),
          temp_start=(189, 147, 249), temp_mid=(255, 121, 198), temp_end=(255, 51, 168)),
    Theme("Dusklight",
          cpu_start=(0, 212, 255), cpu_mid=(255, 248, 107), cpu_end=(255, 127, 0),
          temp_start=(0, 173, 255), temp_mid=(0, 255, 255), temp_end=(255, 248, 107)),
    Theme("Elementarish",
          cpu_start=(133, 153, 0), cpu_mid=(178, 134, 2), cpu_end=(209, 48, 44),
          temp_start=(133, 153, 0), temp_mid=(178, 134, 2), temp_end=(209, 48, 44)),
    Theme("Everforest Dark Hard",
          cpu_start=(167, 192, 128), cpu_mid=(219, 188, 127), cpu_end=(248, 85, 82),
          temp_start=(167, 192, 128), temp_mid=(219, 188, 127), temp_end=(248, 85, 82)),
    Theme("Flat Remix Light",
          cpu_start=(54, 123, 240), cpu_mid=(74, 174, 230), cpu_end=(84, 189, 142),
          temp_start=(54, 123, 240), temp_mid=(184, 23, 76), temp_end=(212, 25, 25)),
    Theme("Flat Remix",
          cpu_start=(54, 123, 240), cpu_mid=(74, 174, 230), cpu_end=(84, 189, 142),
          temp_start=(54, 123, 240), temp_mid=(184, 23, 76), temp_end=(212, 25, 25)),
    Theme("Greyscale",
          cpu_start=(80, 80, 80), cpu_mid=(128, 128, 128), cpu_end=(255, 255, 255),
          temp_start=(80, 80, 80), temp_mid=(128, 128, 128), temp_end=(255, 255, 255)),
    Theme("Gruvbox Dark",
          cpu_start=(184, 187, 38), cpu_mid=(215, 153, 33), cpu_end=(251, 73, 52),
          temp_start=(69, 133, 136), temp_mid=(211, 134, 155), temp_end=(251, 67, 148)),
    Theme("Gruvbox Dark V2",
          cpu_start=(142, 192, 124), cpu_mid=(215, 153, 33), cpu_end=(204, 36, 29),
          temp_start=(142, 192, 124), temp_mid=(215, 153, 33), temp_end=(204, 36, 29)),
    Theme("Gruvbox Material Dark",
          cpu_start=(169, 182, 101), cpu_mid=(216, 166, 87), cpu_end=(234, 105, 98),
          temp_start=(125, 174, 163), temp_mid=(231, 138, 78), temp_end=(234, 105, 98)),
    Theme("Horizon",
          cpu_start=(39, 215, 150), cpu_mid=(250, 194, 154), cpu_end=(233, 86, 120),
          temp_start=(39, 215, 150), temp_mid=(250, 194, 154), temp_end=(233, 86, 120)),
    Theme("Kyli0X",
          cpu_start=(33, 214, 201), cpu_mid=(26, 171, 160), cpu_end=(94, 196, 188),
          temp_start=(33, 214, 201), temp_mid=(26, 171, 160), temp_end=(94, 196, 188)),
    Theme("Matcha Dark Sea",
          cpu_start=(51, 177, 101), cpu_mid=(248, 248, 242), cpu_end=(46, 179, 152),
          temp_start=(121, 118, 183), temp_mid=(216, 184, 178), temp_end=(51, 177, 101)),
    Theme("Monokai",
          cpu_start=(166, 226, 46), cpu_mid=(248, 248, 242), cpu_end=(249, 38, 114),
          temp_start=(121, 118, 183), temp_mid=(216, 184, 178), temp_end=(249, 38, 114)),
    Theme("Night Owl",
          cpu_start=(34, 218, 110), cpu_mid=(173, 219, 103), cpu_end=(239, 83, 80),
          temp_start=(130, 170, 255), temp_mid=(199, 146, 234), temp_end=(251, 67, 148)),
    Theme("Nord",
          cpu_start=(129, 161, 193), cpu_mid=(136, 192, 208), cpu_end=(236, 239, 244),
          temp_start=(129, 161, 193), temp_mid=(136, 192, 208), temp_end=(236, 239, 244)),
    Theme("Onedark",
          cpu_start=(152, 195, 121), cpu_mid=(229, 192, 123), cpu_end=(224, 108, 117),
          temp_start=(152, 195, 121), temp_mid=(229, 192, 123), temp_end=(224, 108, 117)),
    Theme("Paper",
          cpu_start=(85, 85, 85), cpu_mid=(0, 0, 0), cpu_end=(204, 62, 40),
          temp_start=(85, 85, 85), temp_mid=(0, 0, 0), temp_end=(204, 62, 40)),
    Theme("Solarized Dark",
          cpu_start=(173, 199, 0), cpu_mid=(214, 162, 0), cpu_end=(230, 83, 23),
          temp_start=(38, 139, 210), temp_mid=(204, 181, 247), temp_end=(252, 83, 120)),
    Theme("Solarized Light",
          cpu_start=(173, 199, 0), cpu_mid=(214, 162, 0), cpu_end=(230, 83, 23),
          temp_start=(38, 139, 210), temp_mid=(204, 181, 247), temp_end=(252, 83, 120)),
    Theme("Tokyo Night",
          cpu_start=(158, 206, 106), cpu_mid=(224, 175, 104), cpu_end=(247, 118, 142),
          temp_start=(158, 206, 106), temp_mid=(224, 175, 104), temp_end=(247, 118, 142)),
    Theme("Tokyo Storm",
          cpu_start=(158, 206, 106), cpu_mid=(224, 175, 104), cpu_end=(247, 118, 142),
          temp_start=(158, 206, 106), temp_mid=(224, 175, 104), temp_end=(247, 118, 142)),
    Theme("Tomorrow Night",
          cpu_start=(181, 189, 104), cpu_mid=(240, 198, 116), cpu_end=(204, 102, 102),
          temp_start=(181, 189, 104), temp_mid=(240, 198, 116), temp_end=(204, 102, 102)),
    Theme("Whiteout",
          cpu_start=(11, 142, 68), cpu_mid=(164, 145, 4), cpu_end=(141, 2, 2),
          temp_start=(24, 69, 103), temp_mid=(18, 44, 135), temp_end=(158, 0, 97)),
]

# Global theme state (set by set_theme() in main())
_current_theme_index = 0

# ============================================================================
# Utility Functions
# ============================================================================

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))

def load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a font with the specified size."""
    for font_path in FONT_PATHS:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size)
            except OSError:
                continue
    
    # Fallback to default font
    return ImageFont.load_default()

def get_current_theme() -> Theme:
    """Get the currently active theme."""
    return THEMES[_current_theme_index]

def get_current_theme_name() -> str:
    """Get the name of the currently active theme."""
    return get_current_theme().name

def cycle_theme():
    """Cycle to the next theme."""
    global _current_theme_index
    _current_theme_index = (_current_theme_index + 1) % len(THEMES)
    print(f"Switched to theme: {get_current_theme_name()}")

def set_theme(index: int):
    """Set the theme by index."""
    global _current_theme_index
    if 0 <= index < len(THEMES):
        _current_theme_index = index
        print(f"Set theme to: {get_current_theme_name()}")
    else:
        print(f"Invalid theme index: {index}. Using default theme.")

# ============================================================================
# Keyboard Input Handler
# ============================================================================

def keyboard_listener():
    """Listen for keyboard input to cycle themes, blur, gradient, toggle stats, and quit."""
    print("Press 't' to cycle themes, 'b' to cycle blur, 'g' to cycle gradient, 's' to toggle stats, 'q' to quit")
    while True:
        try:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1).lower()
                if key == 't':
                    cycle_theme()
                elif key == 'b':
                    cycle_blur_level()
                elif key == 'g':
                    cycle_gradient_level()
                elif key == 's':
                    toggle_stats()
                elif key == 'q':
                    print("Quitting...")
                    os._exit(0)
        except (EOFError, KeyboardInterrupt):
            break
        except Exception:
            pass

# ============================================================================
# CPU Data Collection and History
# ============================================================================

# Global CPU history storage
_cpu_history = []
_core_histories = [[] for _ in range(NUM_CPU_CORES)]  # Support all detected cores
_buffer_size = 30

# Rolling averages for smooth display (separate from history)
_core_usage_averages = [[] for _ in range(NUM_CPU_CORES)]  # Rolling averages for display
_temp_averages = [[] for _ in range(NUM_CPU_CORES)]  # Rolling averages for temperatures
_avg_window_size = 20  # Average over last 10 values

# Cache for thermal zones to avoid repeated file system access
_thermal_zones_cache = None

def update_cpu_history(cpu_percent: float, cores_pct: List[float], temps: List[float]):
    """Update the rolling history for overall CPU and individual cores."""
    global _cpu_history, _core_histories, _core_usage_averages, _temp_averages
    
    # Update overall CPU history
    _cpu_history.append(cpu_percent)
    if len(_cpu_history) > _buffer_size:
        _cpu_history.pop(0)
    
    # Update individual core histories and rolling averages
    for i, core_pct in enumerate(cores_pct):
        if i < len(_core_histories):
            # Update history for heatmap
            _core_histories[i].append(core_pct)
            if len(_core_histories[i]) > _buffer_size:
                _core_histories[i].pop(0)
            
            # Update rolling average for smooth display
            _core_usage_averages[i].append(core_pct)
            if len(_core_usage_averages[i]) > _avg_window_size:
                _core_usage_averages[i].pop(0)
    
    # Update temperature rolling averages
    for i, temp in enumerate(temps):
        if i < len(_temp_averages):
            _temp_averages[i].append(temp)
            if len(_temp_averages[i]) > _avg_window_size:
                _temp_averages[i].pop(0)

def get_core_history(core_index: int) -> List[float]:
    """Get the rolling history for a specific core."""
    if 0 <= core_index < len(_core_histories):
        return _core_histories[core_index]
    return []

def _get_smoothed_value(values_list: List[List[float]], index: int) -> float:
    """Get smoothed value from a list of rolling averages."""
    if 0 <= index < len(values_list):
        values = values_list[index]
        if values:
            return sum(values) / len(values)
    return 0.0

def get_smoothed_usage(core_index: int) -> float:
    """Get smoothed CPU usage for display."""
    return _get_smoothed_value(_core_usage_averages, core_index)

def get_smoothed_temp(core_index: int) -> float:
    """Get smoothed temperature for display."""
    return _get_smoothed_value(_temp_averages, core_index)

def get_cpu_info(sampling_interval: float = 0.05) -> Dict:
    """Collect comprehensive CPU information using BPYTOP's proven methods."""
    # Use BPYTOP's exact CPU collection method (lines 3062, 3075)
    # For high refresh rates, use non-blocking CPU sampling
    if sampling_interval < 0.1:
        # Use non-blocking calls for fast updates (BPYTOP compatible)
        cpu_percent = psutil.cpu_percent(interval=None)  # Non-blocking
        cores_pct = psutil.cpu_percent(interval=None, percpu=True)  # Non-blocking
    else:
        # Use blocking calls for slower updates (BPYTOP compatible)
        cpu_percent = psutil.cpu_percent(interval=sampling_interval)
        cores_pct = psutil.cpu_percent(interval=sampling_interval, percpu=True)
    
    # Get CPU frequencies (skip for high refresh rates to improve performance)
    freq_current = freq_max = None
    if sampling_interval >= 0.1:  # Only check frequencies for slower refresh rates
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                freq_current = cpu_freq.current
                freq_max = cpu_freq.max
        except Exception:
            pass
    
    # Get CPU temperatures (Linux) - using same method as BTOP (lm-sensors)
    temps = []
    try:
        # Use the same data source as BTOP: lm-sensors via sensors command
        import subprocess
        result = subprocess.run(['sensors', 'coretemp-isa-0000'], 
                              capture_output=True, text=True, timeout=1)
        
        if result.returncode == 0:
            # Parse sensors output to extract core temperatures
            temp_sensors = {}
            for line in result.stdout.split('\n'):
                if 'Core ' in line and '°C' in line:
                    # Parse line like "Core 0:        +36.0°C  (high = +85.0°C, crit = +105.0°C)"
                    parts = line.split(':')
                    if len(parts) >= 2:
                        core_part = parts[0].strip()
                        temp_part = parts[1].strip()
                        
                        # Extract core number
                        if 'Core ' in core_part:
                            core_num = int(core_part.replace('Core ', ''))
                            
                            # Extract temperature (remove °C and any additional info)
                            temp_str = temp_part.split()[0].replace('+', '').replace('°C', '')
                            try:
                                temp_c = float(temp_str)
                                temp_sensors[core_num] = temp_c
                            except ValueError:
                                continue
            
            if temp_sensors:
                # Map physical cores to logical cores (0-19) - same as BTOP
                # Sort physical core numbers and map to logical indices
                physical_cores = sorted([k for k in temp_sensors.keys() if isinstance(k, int)])
                
                for i in range(len(cores_pct)):
                    if i < len(physical_cores):
                        # Use actual core temperature
                        temps.append(temp_sensors[physical_cores[i]])
                    else:
                        # Use the first available core temperature as fallback
                        temps.append(list(temp_sensors.values())[0] if temp_sensors else 35.0)
        
        # Fallback to hwmon if sensors command fails
        if not temps:
            hwmon_path = Path("/sys/class/hwmon")
            if hwmon_path.exists():
                for hwmon_dir in hwmon_path.iterdir():
                    try:
                        name_file = hwmon_dir / "name"
                        if name_file.exists() and "coretemp" in name_file.read_text().strip():
                            temp_sensors = {}
                            for temp_file in hwmon_dir.glob("temp*_input"):
                                try:
                                    temp_id = temp_file.name.replace("temp", "").replace("_input", "")
                                    label_file = hwmon_dir / f"temp{temp_id}_label"
                                    if label_file.exists():
                                        label = label_file.read_text().strip()
                                        temp_millic = int(temp_file.read_text().strip())
                                        temp_c = temp_millic / 1000.0
                                        
                                        if label.startswith("Core "):
                                            core_num = int(label.split()[1])
                                            temp_sensors[core_num] = temp_c
                                except Exception:
                                    continue
                            
                            if temp_sensors:
                                physical_cores = sorted([k for k in temp_sensors.keys() if isinstance(k, int)])
                                for i in range(len(cores_pct)):
                                    if i < len(physical_cores):
                                        temps.append(temp_sensors[physical_cores[i]])
                                    else:
                                        temps.append(list(temp_sensors.values())[0] if temp_sensors else 35.0)
                                break
                    except Exception:
                        continue
        
        # Final fallback: estimate from CPU usage
        if len(temps) < len(cores_pct):
            base_temp = 30.0
            for i in range(len(cores_pct)):
                if i >= len(temps):
                    usage_factor = cores_pct[i] / 100.0
                    estimated_temp = base_temp + (usage_factor * 25.0)
                    temps.append(estimated_temp)
    except Exception:
        # Fallback: estimate all temperatures
        base_temp = 30.0
        temps = [base_temp + (usage / 100.0) * 25.0 for usage in cores_pct]
    
    # Update history
    update_cpu_history(cpu_percent, cores_pct, temps)
    
    return {
        'cpu_percent': cpu_percent,
        'cores_pct': cores_pct,
        'freq_current': freq_current,
        'freq_max': freq_max,
        'temps': temps,
        'num_cores': NUM_CPU_CORES
    }

def get_gpu_info() -> Dict:
    """Get GPU information using NVIDIA NVML (similar to BPYTOP's approach)."""
    gpu_info = {
        'available': False,
        'usage_percent': 0,
        'temperature': 0,
        'memory_used': 0,
        'memory_total': 0,
        'memory_percent': 0
    }
    
    if not HAS_NVML:
        return gpu_info
    
    try:
        # Initialize NVML
        pynvml.nvmlInit()
        
        # Get GPU count
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            return gpu_info
        
        # Get first GPU (index 0)
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Get GPU utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_info['usage_percent'] = util.gpu
        
        # Get GPU temperature
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        gpu_info['temperature'] = temp
        
        # Get GPU memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_info['memory_used'] = mem_info.used
        gpu_info['memory_total'] = mem_info.total
        gpu_info['memory_percent'] = (mem_info.used / mem_info.total) * 100
        
        gpu_info['available'] = True
        
    except Exception as e:
        # GPU not available or error occurred
        pass
    
    return gpu_info

# ============================================================================
# Image Rendering
# ============================================================================

def create_monitoring_overlay(cpu_info: Dict, gpu_info: Dict = None, width: int = DISPLAY_WIDTH, 
                            height: int = DISPLAY_HEIGHT) -> Image.Image:
    """Create the system monitoring overlay image."""
    # Start with background image if available, otherwise use solid background
    processed_bg = get_processed_background()
    if processed_bg is not None:
        overlay = processed_bg.copy()
    else:
        overlay = Image.new('RGB', (width, height), BACKGROUND)
    
    draw = ImageDraw.Draw(overlay)
    
    # If stats are disabled, return just the background image
    if not stats_enabled:
        return overlay
    
    # Load fonts
    f_big = load_font(40)
    f_main = load_font(20)
    f_small = load_font(16)
    f_tiny = load_font(14)
    
    # Layout parameters - optimized to fit all 20 cores
    bar_width = 130  # Reduced from 200
    bar_height = 18  # Reduced from 22
    bar_spacing = 2  # Reduced from 3
    temp_bar_width = 25  # Reduced from 40
    temp_bar_height = 10  # Reduced from 12
    label_width = 30  # Reduced from 35
    
    # BTOP-style layout
    btop_start_x = MARGIN
    btop_start_y = MARGIN
    
    # Show all detected cores, but limit to what fits on display
    cores_pct = cpu_info['cores_pct']
    temps = cpu_info['temps']
    
    # Calculate how many cores can fit on the display
    available_height = height - (2 * MARGIN)
    max_cores_fit = available_height // (bar_height + bar_spacing)
    ncores = min(NUM_CPU_CORES, len(cores_pct), max_cores_fit)
    
    if ncores < NUM_CPU_CORES:
        print(f"Warning: Only showing {ncores} of {NUM_CPU_CORES} cores due to display size limits")
    
    current_theme = get_current_theme()
    
    for i in range(ncores):
        y_pos = btop_start_y + i * (bar_height + bar_spacing)
        
        # Core label (C0, C1, etc.)
        core_label = f"C{i}"
        draw.text((btop_start_x, y_pos), core_label, font=f_small, fill=WHITE)
        
        # CPU usage rolling timeseries heatmap (BTOP style)
        heatmap_x = btop_start_x + label_width + 5
        heatmap_y = y_pos + 1
        
        # Background for heatmap
        draw.rectangle([heatmap_x, heatmap_y, heatmap_x + bar_width, heatmap_y + bar_height], 
                      fill=BAR_BACKGROUND)
        
        # Draw rolling timeseries heatmap for this core
        core_history = get_core_history(i)
        if len(core_history) > 0:
            # Calculate box width for each time sample
            num_samples = len(core_history)
            box_width = bar_width // num_samples if num_samples > 0 else 1
            
            # Draw each time sample as a colored box
            for j, usage_value in enumerate(core_history):
                box_x = heatmap_x + j * box_width
                box_x2 = min(heatmap_x + bar_width, box_x + box_width)
                
                # Color based on CPU usage value using current theme
                color = current_theme.get_color_for_usage(usage_value)
                
                # Draw filled box for this time sample
                draw.rectangle([box_x, heatmap_y, box_x2, heatmap_y + bar_height], fill=color)
        
        # Current usage value for percentage display (smoothed)
        usage = get_smoothed_usage(i)
        
        # Usage percentage text
        usage_text = f"{int(usage)}%"
        usage_text_x = heatmap_x + bar_width + 5
        draw.text((usage_text_x, heatmap_y), usage_text, font=f_small, fill=WHITE)
        
        # Temperature bar and text
        temp_bar_x = usage_text_x + 50
        temp_bar_y = y_pos + 5
        
        # Temperature background
        draw.rectangle([temp_bar_x, temp_bar_y, temp_bar_x + temp_bar_width, 
                       temp_bar_y + temp_bar_height], fill=BAR_BACKGROUND)
        
        # Temperature value (smoothed)
        temp = get_smoothed_temp(i)
        
        # Draw temperature bar with color based on temperature
        if temp > 30:  # Only show colored bar if temperature is above 30°C
            temp_color = current_theme.get_color_for_temperature(temp)
            # Fill temperature bar with color proportional to temperature
            temp_fill_width = int((temp - 30) / 55 * temp_bar_width)  # 30-85°C range
            temp_fill_width = max(0, min(temp_bar_width, temp_fill_width))
            
            if temp_fill_width > 0:
                draw.rectangle([temp_bar_x, temp_bar_y, temp_bar_x + temp_fill_width, 
                               temp_bar_y + temp_bar_height], fill=temp_color)
        
        temp_text = f"{int(temp)}°C"
        temp_text_x = temp_bar_x + temp_bar_width + 5
        draw.text((temp_text_x, temp_bar_y), temp_text, font=f_small, fill=WHITE)
    
    # Display GPU information (if available)
    if gpu_info and gpu_info.get('available', False):
        gpu_y = height - 60  # Bottom area for GPU info
        
        # GPU title
        draw.text((MARGIN, gpu_y), "GPU", font=f_main, fill=WHITE)
        
        # GPU usage
        gpu_usage = gpu_info.get('usage_percent', 0)
        gpu_usage_text = f"{gpu_usage}%"
        draw.text((MARGIN + 50, gpu_y), gpu_usage_text, font=f_main, fill=WHITE)
        
        # GPU temperature
        gpu_temp = gpu_info.get('temperature', 0)
        gpu_temp_text = f"{gpu_temp}°C"
        draw.text((MARGIN + 120, gpu_y), gpu_temp_text, font=f_main, fill=WHITE)
        
        # GPU memory
        gpu_mem_percent = gpu_info.get('memory_percent', 0)
        gpu_mem_text = f"VRAM: {gpu_mem_percent:.1f}%"
        draw.text((MARGIN + 200, gpu_y), gpu_mem_text, font=f_main, fill=WHITE)
    
    return overlay

# ============================================================================
# USB Device Communication
# ============================================================================

def split_frame(frame_data: bytes) -> Tuple[bytes, bytes, bytes]:
    """Split frame into header, JPEG, and tail."""
    soi = frame_data.find(b"\xff\xd8")
    eoi = frame_data.rfind(b"\xff\xd9")
    if soi < 0 or eoi < 0:
        raise RuntimeError("SOI/EOI not found in frame.bin")
    return frame_data[:soi], frame_data[soi:eoi+2], frame_data[eoi+2:]

def make_jpeg(image: Image.Image, quality: int = 80) -> bytes:
    """Convert PIL image to JPEG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality, optimize=True)
    return buffer.getvalue()

def load_background_image():
    """Load background image from assets/images/ directory."""
    global background_image, BACKGROUND_IMAGE_PATH
    
    # Try to find a background image in assets/images/
    assets_dir = Path(__file__).parent.parent.parent / "assets" / "images"
    if assets_dir.exists():
        # Look for common image formats
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            images = list(assets_dir.glob(ext))
            if images:
                BACKGROUND_IMAGE_PATH = str(images[0])
                try:
                    background_image = Image.open(BACKGROUND_IMAGE_PATH).convert("RGB")
                    # Resize to display dimensions
                    background_image = background_image.resize((DISPLAY_WIDTH, DISPLAY_HEIGHT), Image.LANCZOS)
                    print(f"Loaded background image: {BACKGROUND_IMAGE_PATH}")
                    return True
                except Exception as e:
                    print(f"Failed to load background image {BACKGROUND_IMAGE_PATH}: {e}")
                    background_image = None
                    BACKGROUND_IMAGE_PATH = None
                    return False
    
    print("No background image found in assets/images/")
    return False

def get_processed_background():
    """Get the background image with current blur and gradient effects applied."""
    global background_image, current_blur_level, current_gradient_level
    
    if background_image is None:
        return None
    
    # Create a copy to avoid modifying the original
    processed = background_image.copy()
    
    # Apply blur effect
    blur_radius = BLUR_LEVELS[current_blur_level]
    if blur_radius > 0:
        processed = processed.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Apply gradient/opacity effect
    opacity = GRADIENT_LEVELS[current_gradient_level]
    if opacity < 1.0:
        # Create a black overlay and blend
        overlay = Image.new("RGB", (DISPLAY_WIDTH, DISPLAY_HEIGHT), (0, 0, 0))
        processed = Image.blend(processed, overlay, 1.0 - opacity)
    
    return processed

def cycle_blur_level():
    """Cycle through blur levels."""
    global current_blur_level
    current_blur_level = (current_blur_level + 1) % len(BLUR_LEVELS)
    blur_radius = BLUR_LEVELS[current_blur_level]
    print(f"Blur level: {current_blur_level + 1}/10 (radius: {blur_radius})")

def cycle_gradient_level():
    """Cycle through gradient/opacity levels."""
    global current_gradient_level
    current_gradient_level = (current_gradient_level + 1) % len(GRADIENT_LEVELS)
    opacity = GRADIENT_LEVELS[current_gradient_level]
    print(f"Gradient level: {current_gradient_level + 1}/10 (opacity: {opacity:.1f})")

def toggle_stats():
    """Toggle stats display on/off."""
    global stats_enabled
    stats_enabled = not stats_enabled
    status = "ON" if stats_enabled else "OFF"
    print(f"Stats display: {status}")

def show_preview(image: Image.Image):
    """Show the image in a preview window."""
    try:
        import tkinter as tk
        from PIL import ImageTk
        
        # Create or update preview window
        if not hasattr(show_preview, 'window') or not show_preview.window.winfo_exists():
            show_preview.window = tk.Tk()
            show_preview.window.title("Thermalright Monitor Preview")
            show_preview.window.geometry("500x500")
            show_preview.window.configure(bg='black')
            
            # Create label for image
            show_preview.label = tk.Label(show_preview.window, bg='black')
            show_preview.label.pack(expand=True, fill='both')
        
        # Convert PIL image to PhotoImage and display
        photo = ImageTk.PhotoImage(image)
        show_preview.label.configure(image=photo)
        show_preview.label.image = photo  # Keep a reference
        
        # Update window
        show_preview.window.update_idletasks()
        show_preview.window.update()
        
    except ImportError:
        # Fallback: save image to file
        image.save("preview.png")
        print("Preview saved as preview.png (tkinter not available)")
    except Exception as e:
        print(f"Preview error: {e}")

def open_device():
    """Open and configure the USB device."""
    device = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
    if device is None:
        raise RuntimeError(f"Device {VENDOR_ID:04x}:{PRODUCT_ID:04x} not found")
    
    try:
        device.set_configuration()
        cfg = device.get_active_configuration()
        
        # Find OUT endpoint
        for interface in cfg:
            for endpoint in interface:
                if usb.util.endpoint_direction(endpoint.bEndpointAddress) == usb.util.ENDPOINT_OUT:
                    return device, interface, endpoint
        
        raise RuntimeError("No OUT endpoint found")
    except Exception as e:
        raise RuntimeError(f"Failed to configure device: {e}")

def send_payload(endpoint, payload: bytes):
    """Send payload to device in chunks."""
    offset = 0
    while offset < len(payload):
        chunk_size = min(CHUNK_SIZE, len(payload) - offset)
        bytes_written = endpoint.write(payload[offset:offset + chunk_size], timeout=3000)
        offset += bytes_written

# ============================================================================
# Main Application
# ============================================================================

def main(preview=False):
    """Main application entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="System monitor for thermalright USB display")
    parser.add_argument("--refresh-rate", type=float, default=15.0, help="Refresh rate in FPS (default: 15)")
    parser.add_argument("--theme", type=int, default=2, help="Theme index (0-29). Press 't' to cycle themes at runtime.")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality (1-100)")
    parser.add_argument("--preview", action='store_true', help="Show preview window instead of sending to device")
    
    args = parser.parse_args()
    
    # Use preview parameter if passed directly, otherwise use command line arg
    if preview:
        args.preview = True
    
    # Convert FPS to interval and ensure reasonable limits
    args.refresh_rate = max(1.0, min(20.0, args.refresh_rate))  # Limit to 1-20 FPS
    args.interval = 1.0 / args.refresh_rate  # Convert FPS to seconds
    args.cpu_sampling = args.interval  # Use same rate for CPU sampling
    
    # Optimize for 100ms refresh rates (10 FPS) like BTOP
    if args.refresh_rate >= 10:
        print(f"Optimized for high refresh rate: {args.refresh_rate:.1f} FPS (BPYTOP-style)")
    
    # Auto-adjust quality for high refresh rates to improve performance
    if args.refresh_rate > 15 and args.quality > 60:
        original_quality = args.quality
        args.quality = 60  # Reduce quality for high FPS
        print(f"Auto-reduced JPEG quality from {original_quality} to {args.quality} for {args.refresh_rate} FPS")
    
    # Set initial theme
    set_theme(args.theme)
    
    # Load background image
    load_background_image()
    
    # Start keyboard listener thread
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()
    
    try:
        # Load reference frame
        if not FRAME_BIN_PATH.exists():
            raise RuntimeError(f"Reference frame not found: {FRAME_BIN_PATH}")
        
        frame_data = FRAME_BIN_PATH.read_bytes()
        header, _, tail = split_frame(frame_data)
        print(f"Loaded reference frame, header length: {len(header)}")
        
        # Open device
        device, interface, endpoint = open_device()
        print("Device opened successfully")
        
        print(f"Starting system monitor on device (refresh rate: {args.refresh_rate:.1f} FPS, quality: {args.quality})")
        print(f"Monitoring {NUM_CPU_CORES} CPU cores")
        
        # Check GPU availability
        gpu_test = get_gpu_info()
        if gpu_test.get('available', False):
            print("GPU monitoring: ENABLED (NVIDIA)")
        else:
            print("GPU monitoring: DISABLED (no NVIDIA GPU or pynvml)")
        
        print("Press Ctrl+C to stop")
        print(f"Available themes: {len(THEMES)}")
        
        # Performance monitoring
        frame_count = 0
        start_time = time.time()
        
        # Main loop
        while True:
            loop_start = time.time()
            
            # Collect CPU data (BPYTOP-style)
            cpu_start = time.time()
            cpu_info = get_cpu_info(args.cpu_sampling)
            cpu_time = time.time() - cpu_start
            
            # Collect GPU data (if available)
            gpu_start = time.time()
            gpu_info = get_gpu_info()
            gpu_time = time.time() - gpu_start
            
            # Create monitoring overlay
            render_start = time.time()
            overlay = create_monitoring_overlay(cpu_info, gpu_info)
            render_time = time.time() - render_start
            
            # Convert to JPEG
            jpeg_start = time.time()
            jpeg_data = make_jpeg(overlay, quality=args.quality)
            jpeg_time = time.time() - jpeg_start
            
            # Patch header with JPEG length
            header_copy = bytearray(header)
            struct.pack_into("<I", header_copy, LENGTH_FIELD_OFFSET, len(jpeg_data))
            
            # Create payload
            payload = bytes(header_copy) + jpeg_data + tail
            
            # Send to device or show preview
            send_start = time.time()
            if args.preview:
                # Show preview window
                show_preview(overlay)
            else:
                # Send to device
                send_payload(endpoint, payload)
            send_time = time.time() - send_start
            
            total_loop_time = time.time() - loop_start
            
            # Performance monitoring
            frame_count += 1
            if frame_count % 50 == 0:  # Show FPS every 50 frames
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed
                gpu_status = "GPU: ON" if gpu_info.get('available', False) else "GPU: OFF"
                print(f"Actual FPS: {actual_fps:.1f} (target: {args.refresh_rate:.1f}) - {gpu_status}")
                print(f"  CPU: {cpu_time*1000:.1f}ms, GPU: {gpu_time*1000:.1f}ms, Render: {render_time*1000:.1f}ms, JPEG: {jpeg_time*1000:.1f}ms, Send: {send_time*1000:.1f}ms, Total: {total_loop_time*1000:.1f}ms")
            
            # Wait for next update
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        try:
            usb.util.release_interface(device, interface)
        except:
            pass

if __name__ == "__main__":
    main()
