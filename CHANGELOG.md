# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2024-12-19

### Added
- **BTOP-compatible temperature readings**: Now uses `lm-sensors` (same as BTOP) for accurate CPU temperature readings
- **NVIDIA GPU monitoring**: Added GPU usage, temperature, and VRAM monitoring via `pynvml`
- **30 authentic BTOP themes**: Integrated all official BTOP color themes including the built-in "Default" theme
- **Dynamic CPU core detection**: Automatically detects and monitors all CPU cores using `psutil`
- **Rolling averages**: Smooth out noisy CPU usage and temperature readings
- **High refresh rate optimization**: Support for up to 20 FPS with performance monitoring
- **Robust path handling**: All scripts use proper path resolution and work from any directory

### Changed
- **Simplified refresh rate control**: Replaced confusing `interval` and `cpu-sampling` arguments with single `--refresh-rate` parameter
- **Improved theme system**: Now uses authentic BTOP theme colors with proper gradient mapping
- **Enhanced display layout**: Optimized for 480x480 display with all 20 CPU cores visible
- **Better error handling**: More robust USB device communication and fallback mechanisms

### Fixed
- **Temperature accuracy**: CPU temperatures now match BTOP readings exactly
- **Theme count discrepancy**: Fixed mismatch between BTOP's 30 themes and our previous count
- **Performance issues**: Optimized for high refresh rates with proper timing controls
- **Path resolution**: Fixed broken links to `frame.bin` and other assets after file reorganization

### Removed
- **Duplicate themes**: Removed 6 duplicate theme entries that had identical color schemes
- **Legacy files**: Cleaned up old test files and unused scripts from archive directory
- **Deprecated arguments**: Removed confusing `--interval` and `--cpu-sampling` parameters

### Technical Details
- **CPU Monitoring**: Uses `psutil.cpu_percent()` with configurable sampling intervals
- **Temperature Reading**: Primary method via `sensors coretemp-isa-0000`, fallback to `hwmon`, then estimation
- **GPU Monitoring**: NVIDIA NVML integration for real-time GPU metrics
- **Theme System**: 30 authentic BTOP themes with 3-color gradient mapping (start → mid → end)
- **Display Protocol**: USB HID communication with JPEG encoding and header patching
- **Performance**: Optimized for 15-20 FPS with sub-100ms loop times

## [Previous Versions]

### Initial Implementation
- Basic system monitoring with CPU usage visualization
- USB device communication for Thermalright displays
- Simple theme system with basic color mapping
- Image streaming and protocol analysis tools
