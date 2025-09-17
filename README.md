# Thermalright Display Hack

Reverse engineering and utilities for Thermalright display devices.

## Overview

This project contains tools for analyzing and controlling Thermalright display devices via USB communication. The main functionality includes:

- **Frame Analysis**: Analyze captured USB frames to understand the communication protocol
- **Image Streaming**: Stream images to the display in real-time
- **Screen Mirroring**: Mirror a specific window or screen region to the display
- **Payload Replay**: Replay captured payloads to the device

## Project Structure

```
thermalright_display_hack/
├── src/                    # Main Python scripts
│   ├── replay_loop.py      # Replay captured payloads in a loop
│   ├── tiny_monitor_stream.py  # Stream screen content to display
│   ├── analyze_headers.py  # Analyze USB frame headers and protocol
│   └── send_image_patched.py   # Send single image to display
├── assets/                 # Images, binaries, and data files
│   ├── images/            # Test images
│   ├── data/              # Binary payloads and captures
│   └── old/               # Legacy experiments
└── venv/                  # Python virtual environment
```

## Main Scripts

### `src/replay_loop.py`
Replays a captured payload (`patched_payload.bin`) to the device in a continuous loop. Useful for testing and demonstration.

### `src/tiny_monitor_stream.py`
Real-time screen mirroring tool that:
- Captures a specific window (configurable by title)
- Encodes frames as JPEG to match the original payload size
- Streams to the Thermalright display at configurable FPS

### `src/analyze_headers.py`
Protocol analysis tool that:
- Analyzes multiple captured frames
- Identifies length fields, frame counters, and checksums
- Helps understand the USB communication protocol

### `src/send_image_patched.py`
Sends a single image to the display:
- Takes an image file as input
- Encodes as JPEG with size matching the original payload
- Patches the header with correct length information
- Sends to device once

## Device Information

- **Vendor ID**: 0x87ad
- **Product ID**: 0x70db
- **Protocol**: USB HID with custom payload format

## Requirements

- Python 3.6+
- pyusb
- Pillow (PIL)
- mss (for screen capture)
- xdotool (for window detection on Linux)

## Setup

1. Create virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install pyusb pillow mss
   ```

3. Set up USB permissions (Linux):
   ```bash
   sudo usermod -a -G plugdev $USER
   # Add udev rule for the device
   ```

## Usage

### Stream screen content:
```bash
python src/tiny_monitor_stream.py
```

### Send a single image:
```bash
python src/send_image_patched.py assets/images/moose.png
```

### Analyze captured frames:
```bash
python src/analyze_headers.py assets/data/frame.bin
```

### Replay captured payload:
```bash
python src/replay_loop.py
```

## Notes

- The device appears to have a ~3 second watchdog that reverts to default display
- Frame rate should be kept reasonable to avoid overwhelming the USB connection
- JPEG encoding is optimized to match the exact size of captured frames
- Some scripts require the device to be connected and accessible via USB

## License

This project is for educational and research purposes. Use responsibly and in accordance with applicable laws and regulations.
