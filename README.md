# Thermalright USB Display System Monitor

A real-time system monitor that displays CPU and GPU usage data on Thermalright USB displays. Features BTOP-style themes, scatter plot visualization, and runtime controls.

## Overview

This project provides a complete system monitoring solution for Thermalright USB display devices with the following key features:

- **Real-time CPU Monitoring**: Scatter plot visualization showing CPU usage vs temperature for all cores
- **GPU Monitoring**: NVIDIA GPU metrics including usage, temperature, VRAM, power, and fan speed
- **30 BTOP-style Themes**: Authentic color themes with runtime cycling
- **Background Image Support**: Customizable background images with blur and gradient effects
- **Runtime Controls**: Interactive theme cycling, blur/gradient adjustment, and stats toggling
- **Data Simulation**: Comprehensive testing tools with drifting data patterns

## Project Structure

```
thermalright_usb_display_controller/
├── src/                           # Main source code
│   ├── thermalright_system_monitor.py  # Main production script
│   └── utils/                     # Utility tools
│       └── analyze_headers.py     # Protocol analysis
├── test/                          # Testing and development scripts
│   ├── data_simulator.py          # CPU/GPU data simulation for testing
│   ├── send_image_patched.py      # Send single image to display
│   ├── simple_stress.py           # CPU stress testing
│   └── replay_loop.py             # Replay captured payloads
├── assets/                        # Images, binaries, and data files
│   ├── images/                    # Background images
│   └── data/                      # Binary payloads and captures
├── run_monitor.py                 # Convenience script to run monitor
├── requirements.txt               # Python dependencies
└── CHANGELOG.md                   # Project changelog
```

## Main Scripts

### `src/thermalright_system_monitor.py` (Main Production Script)

The primary system monitor featuring:

- **CPU Scatter Plot**: Visualization of CPU cores as colored circles positioned by usage (x-axis) and temperature (y-axis)
- **Temperature Zones**: Transparent overlay showing critical (≥90°C) and warning (80-90°C) temperature ranges
- **GPU Metrics**: Real-time display of GPU usage, temperature, VRAM, power consumption, and fan speed
- **Theme System**: 30 authentic BTOP color themes with smooth color gradients
- **Background Images**: Support for custom background images with configurable blur and gradient effects
- **Runtime Controls**: Interactive keyboard controls for theme cycling and display adjustments

**Usage:**
```bash
python src/thermalright_system_monitor.py [options]

Options:
  --refresh-rate FPS    Refresh rate in FPS (default: 15.0)
  --theme INDEX         Initial theme 0-29 (default: 2)
  --quality 1-100       JPEG quality (default: 80)
  --preview             Show preview window instead of sending to device

Runtime Controls:
  Press 't' to cycle themes
  Press 'b' to cycle blur levels
  Press 'g' to cycle gradient levels
  Press 's' to toggle stats display
  Press 'q' to quit
```

### `run_monitor.py` (Convenience Script)

Easy-to-use wrapper that can be run from anywhere:
```bash
python run_monitor.py --preview  # Show preview window
python run_monitor.py            # Send to device
```

### Testing and Development Scripts

#### `test/data_simulator.py`
Comprehensive data simulation tool for testing the monitoring system:

- **Corner Test**: CPU cores at extreme positions (0%, 100% utilization at 30°C, 105°C)
- **Edge Test**: Various positions along grid lines
- **Realistic Test**: Realistic CPU/GPU usage patterns
- **Drifting Test**: Data that drifts over time to demonstrate color gradients

```bash
python test/data_simulator.py
# Choose from 4 test scenarios
```

#### `test/send_image_patched.py`
Send a single image to the display:
```bash
python test/send_image_patched.py assets/images/moose.png
```

#### `test/simple_stress.py`
CPU stress testing tool:
```bash
python test/simple_stress.py 4 30  # Stress 4 cores for 30 seconds
```

#### `test/replay_loop.py`
Replay captured payloads in a continuous loop:
```bash
python test/replay_loop.py
```

#### `src/utils/analyze_headers.py`
Protocol analysis tool for understanding USB communication:
```bash
python src/utils/analyze_headers.py assets/data/frame.bin
```

## Device Information

- **Vendor ID**: 0x87ad
- **Product ID**: 0x70db
- **Protocol**: USB HID with custom payload format
- **Display**: 480x480 pixels
- **Communication**: Chunked USB transfers with JPEG encoding

## Requirements

### System Requirements
- Python 3.8+
- Linux (tested on Ubuntu/Debian)
- lm-sensors for CPU temperature readings
- NVIDIA drivers and nvidia-smi for GPU monitoring (optional)

### Python Dependencies
See `requirements.txt` for complete list:
- `pyusb>=1.3.1` - USB device communication
- `Pillow>=9.0.0` - Image processing
- `psutil>=5.8.0` - System monitoring
- `nvidia-ml-py>=11.0.0` - NVIDIA GPU monitoring (optional)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd thermalright_usb_display_controller
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install system dependencies (Ubuntu/Debian):**
   ```bash
   sudo apt install lm-sensors
   sudo sensors-detect --auto
   ```

5. **Set up USB permissions (Linux):**
   ```bash
   sudo usermod -a -G plugdev $USER
   # Add udev rule for the device if needed
   ```

## Usage

### Quick Start
```bash
# Run with preview window (recommended for testing)
python run_monitor.py --preview

# Run on actual device
python run_monitor.py
```

### Advanced Usage
```bash
# High performance mode (20 FPS)
python src/thermalright_system_monitor.py --refresh-rate 20

# Custom theme and quality
python src/thermalright_system_monitor.py --theme 5 --quality 90

# Preview mode for development
python src/thermalright_system_monitor.py --preview
```

### Testing and Development
```bash
# Test with simulated data
python test/data_simulator.py

# Stress test CPU cores
python test/simple_stress.py 4 30

# Send single image
python test/send_image_patched.py assets/images/moose.png
```

## Features

### CPU Monitoring
- **Scatter Plot Visualization**: CPU cores displayed as colored circles
- **Real-time Data**: Usage and temperature data updated continuously
- **Temperature Zones**: Visual indicators for critical and warning temperatures
- **Smooth Display**: Rolling averages for stable visualization
- **Multi-core Support**: Automatically detects and monitors all CPU cores

### GPU Monitoring
- **NVIDIA Support**: Full GPU monitoring via nvidia-smi
- **Comprehensive Metrics**: Usage, temperature, VRAM, power, and fan speed
- **Theme Integration**: GPU metrics use the same color themes as CPU data
- **Graceful Fallback**: Works without GPU if nvidia-smi is unavailable

### Theme System
- **30 Authentic Themes**: Based on popular BTOP color schemes
- **Runtime Cycling**: Press 't' to cycle through themes while running
- **Color Gradients**: Smooth color transitions based on usage and temperature
- **Consistent Styling**: All UI elements use the same theme colors

### Background Images
- **Custom Backgrounds**: Support for custom background images
- **Blur Effects**: 5 levels of Gaussian blur (0, 2, 4, 6, 8 radius)
- **Gradient Overlays**: 5 levels of opacity (1.0, 0.8, 0.6, 0.4, 0.2)
- **Runtime Adjustment**: Press 'b' for blur, 'g' for gradient levels

### Performance
- **High Refresh Rates**: Support up to 20 FPS
- **Optimized Rendering**: Efficient image processing and USB communication
- **Caching**: Font and background processing caching for performance
- **Error Handling**: Robust error handling with logging

## Configuration

### Display Settings
- **Resolution**: 480x480 pixels
- **Format**: JPEG encoding
- **Quality**: Configurable 1-100 (default: 80)

### Performance Settings
- **Refresh Rate**: 1-20 FPS (default: 15)
- **CPU Sampling**: Optimized for high refresh rates
- **Memory Management**: Efficient rolling averages and caching

### Theme Settings
- **Default Theme**: Index 2 (Adapta)
- **Theme Range**: 0-29 (30 total themes)
- **Color Mapping**: Usage-based and temperature-based gradients

## Troubleshooting

### Common Issues

1. **Device not found**: Ensure USB device is connected and permissions are set
2. **No temperature data**: Install and configure lm-sensors
3. **No GPU data**: Install NVIDIA drivers and nvidia-smi
4. **Permission errors**: Add user to plugdev group and check udev rules

### Debug Mode
```bash
# Run with preview to see output without device
python run_monitor.py --preview

# Check system sensors
sensors

# Check GPU status
nvidia-smi
```

## Development

### Code Structure
- **Modular Design**: Separated into logical classes and functions
- **Error Handling**: Comprehensive error handling with logging
- **Performance Optimized**: Caching and efficient algorithms
- **Well Documented**: Extensive comments and docstrings

### Testing
- **Data Simulation**: Comprehensive test scenarios
- **Live Testing**: Preview mode for development
- **Stress Testing**: CPU stress testing tools
- **Protocol Analysis**: USB communication analysis tools

## License

This project is for educational and research purposes. Use responsibly and in accordance with applicable laws and regulations.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Changelog

See `CHANGELOG.md` for detailed version history and changes.