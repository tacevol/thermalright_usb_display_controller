# Thermalright USB Display Controller

System monitor and utilities for Thermalright USB display devices.

## Overview

This project provides a complete system monitoring solution for Thermalright USB display devices. The main functionality includes:

- **System Monitoring**: Real-time CPU usage monitoring with BTOP-style themes
- **Device Communication**: USB protocol handling for Thermalright displays
- **Image Streaming**: Stream system monitor data to the display in real-time
- **Theme Support**: 30 authentic BTOP color themes with runtime cycling

## Project Structure

```
thermalright_usb_display_controller/
├── src/                    # Organized Python scripts
│   ├── monitor/           # System monitoring and display
│   │   └── thermalright_system_monitor.py  # Main production script
│   ├── device/            # Device communication and protocols
│   │   ├── analyze_headers.py           # Protocol analysis
│   │   ├── send_image_patched.py        # Send single image
│   │   ├── replay_loop.py               # Replay captured payloads
│   │   └── tiny_monitor_stream.py       # Stream screen content
│   └── tools/             # Development and testing tools
│       └── simple_stress.py             # CPU stress testing
├── assets/                 # Images, binaries, and data files
│   ├── images/            # Test images
│   └── data/              # Binary payloads and captures
├── run_monitor.py         # Convenience script to run monitor
├── requirements.txt       # Python dependencies
├── CHANGELOG.md          # Project changelog
└── venv/                  # Python virtual environment
```

## Main Scripts

### `src/monitor/thermalright_system_monitor.py` (Main Production Script)
The primary system monitor that:
- Automatically detects and monitors all CPU cores using `psutil`
- Collects real-time CPU usage data with rolling averages for smooth display
- Uses BTOP-compatible temperature readings via `lm-sensors`
- Renders BTOP-style timeseries heatmaps for all CPU cores
- Supports 30 authentic BTOP color themes with runtime cycling
- Includes NVIDIA GPU monitoring (usage, temperature, VRAM) via `pynvml`
- Sends data directly to the Thermalright USB display
- Features keyboard controls (press 't' to cycle themes, 'q' to quit)
- Optimized for high refresh rates (up to 20 FPS)

### `run_monitor.py` (Convenience Script)
Easy-to-use wrapper that can be run from anywhere:
```bash
python run_monitor.py --refresh-rate 15 --theme 0
```

### Device Communication Scripts

#### `src/device/send_image_patched.py`
Sends a single image to the display:
- Takes an image file as input
- Encodes as JPEG with size matching the original payload
- Patches the header with correct length information
- Sends to device once

#### `src/device/replay_loop.py`
Replays a captured payload (`patched_payload.bin`) to the device in a continuous loop. Useful for testing and demonstration.

#### `src/device/tiny_monitor_stream.py`
Real-time screen mirroring tool that:
- Captures a specific window (configurable by title)
- Encodes frames as JPEG to match the original payload size
- Streams to the Thermalright display at configurable FPS

#### `src/device/analyze_headers.py`
Protocol analysis tool that:
- Analyzes multiple captured frames
- Identifies length fields, frame counters, and checksums
- Helps understand the USB communication protocol

## Device Information

- **Vendor ID**: 0x87ad
- **Product ID**: 0x70db
- **Protocol**: USB HID with custom payload format

## Requirements

- Python 3.8+
- pyusb (USB device communication)
- Pillow (PIL) (image processing)
- psutil (system monitoring)
- mss (screen capture)
- pynvml (NVIDIA GPU monitoring)
- lm-sensors (CPU temperature readings)

## Setup

1. Create virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install system dependencies (Ubuntu/Debian):
   ```bash
   sudo apt install lm-sensors
   sudo sensors-detect --auto
   ```

4. Set up USB permissions (Linux):
   ```bash
   sudo usermod -a -G plugdev $USER
   # Add udev rule for the device
   ```

## Usage

### Run the main system monitor:
```bash
# Using the convenience script (recommended) - 15 FPS default
python run_monitor.py

# For high performance (20 FPS)
python run_monitor.py --refresh-rate 20

# For slower updates (10 FPS)
python run_monitor.py --refresh-rate 10

# Or directly
python src/monitor/thermalright_system_monitor.py --refresh-rate 15 --theme 0
```

### Available options:
- `--refresh-rate`: Refresh rate in FPS (default: 15, max: 20)
- `--theme`: Initial theme index 0-29 (default: 0)
- `--quality`: JPEG quality 1-100 (default: 80)

### Runtime controls:
- Press `t` to cycle through themes
- Press `q` to quit

### Other utilities:

#### Stream screen content:
```bash
python src/device/tiny_monitor_stream.py
```

#### Send a single image:
```bash
python src/device/send_image_patched.py assets/images/moose.png
```

#### Analyze captured frames:
```bash
python src/device/analyze_headers.py assets/data/frame.bin
```

#### Replay captured payload:
```bash
python src/device/replay_loop.py
```

#### CPU stress testing:
```bash
python src/tools/simple_stress.py
```

## Features

- **Automatic CPU Detection**: Automatically detects and monitors all CPU cores on any system
- **Real-time CPU Monitoring**: Shows all CPU cores with usage percentages and temperatures
- **Smooth Display Values**: Rolling averages eliminate noisy readings while preserving real-time heatmaps
- **BTOP-style Themes**: 30 authentic color themes matching BTOP's visual style
- **Timeseries Heatmaps**: Rolling window visualization for each CPU core
- **Runtime Theme Cycling**: Press 't' to cycle through themes while running
- **Robust Path Handling**: All scripts use proper path resolution and work from any directory
- **USB Device Communication**: Direct communication with Thermalright USB displays

## Notes

- The device appears to have a ~3 second watchdog that reverts to default display
- Frame rate should be kept reasonable to avoid overwhelming the USB connection
- JPEG encoding is optimized to match the exact size of captured frames
- Some scripts require the device to be connected and accessible via USB
- All paths are resolved relative to the project root, making scripts portable

## License

This project is for educational and research purposes. Use responsibly and in accordance with applicable laws and regulations.
