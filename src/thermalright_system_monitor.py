"""
Thermalright System Monitor (recreated)

Features implemented per spec:
- 480x480 canvas; CPU plot on left 2/3 (320px), GPU panel on right 1/3 (160px)
- Background image from assets/images (first PNG); rescale+top-left crop; cycle with 'i'
- Blur levels (b): [0,2,4,6,8]; Gray levels (g): [0,0.2,0.4,0.6,0.8]
- Stats toggle (s): hide/show data; when hidden background is unblurred/ungrayed
- Themes (t): cycle; temp-based color gradient for CPU dots and bars
- Dashed grid every 25% (x) and 10°C (y), axes box with separate opacity
- Warning/Critical bands (80–90°C, ≥90°C) as translucent rectangles aligned to grid
- GPU panel (usage,temp,VRAM,power,fan) using pynvml or nvidia-smi fallback
- Key handling via terminal stdin (no GUI dependency) in both preview/device modes

Exports expected by tests:
- TARGET_W, TARGET_H
- update_cpu_history, get_cpu_model, get_cpu_info, get_gpu_info
- create_monitoring_overlay, main
"""

from __future__ import annotations

# Plot area configuration
PLOT_X = 25      # X position of plot bottom-left corner
PLOT_Y = 50      # Y position of plot bottom-left corner  
PLOT_W = 250     # Plot area width
PLOT_H = 400     # Plot area height

import io
import logging
import math
import os
import select
import struct
import sys
import termios
import time
import tty
import subprocess
import signal
from dataclasses import dataclass
import contextlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import psutil
import usb.core
import usb.util
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageTk


TARGET_W, TARGET_H = 480, 480

# USB Device constants
VENDOR_ID = 0x87ad
PRODUCT_ID = 0x70db
CHUNK_SIZE = 4096
HEADER_LEN = 64
LEN_FIELD_OFF = 60


# ------------------------------ Utility ------------------------------
def _load_first_background_image(root: Path) -> Image.Image:
    images_dir = root / "assets" / "images"
    if not images_dir.exists():
        return Image.new("RGB", (TARGET_W, TARGET_H), (16, 16, 18))
    candidates = sorted([p for p in images_dir.glob("*.png")])
    if not candidates:
        return Image.new("RGB", (TARGET_W, TARGET_H), (16, 16, 18))
    img = Image.open(candidates[0]).convert("RGB")
    return _resize_crop_topleft(img, TARGET_W, TARGET_H)


def _resize_crop_topleft(img: Image.Image, width: int, height: int) -> Image.Image:
    # Scale to cover, then crop top-left
    w, h = img.size
    scale = max(width / w, height / h)
    new_size = (int(w * scale + 0.5), int(h * scale + 0.5))
    img_resized = img.resize(new_size, Image.LANCZOS)
    return img_resized.crop((0, 0, width, height))


def _list_backgrounds(root: Path) -> List[Path]:
    images_dir = root / "assets" / "images"
    return sorted([p for p in images_dir.glob("*.png")]) if images_dir.exists() else []


def _apply_gray_overlay(img: Image.Image, opacity: float) -> Image.Image:
    if opacity <= 0:
        return img
    overlay = Image.new("RGB", img.size, (0, 0, 0))
    return Image.blend(img, overlay, alpha=min(1.0, max(0.0, opacity)))


def _draw_dashed_line(draw: ImageDraw.ImageDraw, xy: Tuple[int, int, int, int], color: Tuple[int, int, int] | Tuple[int, int, int, int], width: int, dash: Tuple[int, int]):
    x1, y1, x2, y2 = xy
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    on, off = dash
    dist = 0.0
    while dist < length:
        start = dist
        end = min(dist + on, length)
        sx = int(x1 + ux * start)
        sy = int(y1 + uy * start)
        ex = int(x1 + ux * end)
        ey = int(y1 + uy * end)
        draw.line((sx, sy, ex, ey), fill=color, width=width)
        dist += on + off


def _load_font_mono(size: int, bold: bool = False) -> ImageFont.ImageFont:
    # Prefer bold variants first when bold=True
    if bold:
        font_names = [
            "LiberationMono-Bold.ttf",
            "DejaVuSansMono-Bold.ttf",
            "Courier New Bold.ttf",
            "LiberationMono-Regular.ttf",
            "DejaVuSansMono.ttf",
            "Courier New.ttf",
            "DejaVuSansMono",
            "DejaVuSans.ttf",
        ]
    else:
        font_names = [
            "LiberationMono-Regular.ttf",
            "DejaVuSansMono.ttf",
            "Courier New.ttf",
            "DejaVuSansMono",
            "DejaVuSans.ttf",
        ]

    for name in font_names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


# ------------------------------ Device Communication ------------------------------
def _split_frame(frame_data: bytes) -> Tuple[bytes, bytes, bytes]:
    """Split frame data into header, JPEG, and tail."""
    soi = frame_data.find(b"\xff\xd8")
    eoi = frame_data.rfind(b"\xff\xd9")
    if soi < 0 or eoi < 0:
        raise RuntimeError("SOI/EOI not found in frame data")
    return frame_data[:soi], frame_data[soi:eoi+2], frame_data[eoi+2:]


def _open_device():
    """Open and configure the USB device."""
    device = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
    if device is None:
        raise RuntimeError(f"Device {VENDOR_ID:04x}:{PRODUCT_ID:04x} not found")
    
    try:
        device.set_configuration()
        cfg = device.get_active_configuration()
        
        # Find OUT endpoint
        for interface in cfg:
            try:
                if device.is_kernel_driver_active(interface.bInterfaceNumber):
                    device.detach_kernel_driver(interface.bInterfaceNumber)
            except Exception:
                pass
            usb.util.claim_interface(device, interface.bInterfaceNumber)
            for endpoint in interface:
                if usb.util.endpoint_direction(endpoint.bEndpointAddress) == usb.util.ENDPOINT_OUT:
                    return device, interface, endpoint
        
        raise RuntimeError("No OUT endpoint found")
    except Exception as e:
        raise RuntimeError(f"Failed to configure device: {e}")


def _send_payload(endpoint, payload: bytes):
    """Send payload to device in chunks."""
    offset = 0
    while offset < len(payload):
        chunk_size = min(CHUNK_SIZE, len(payload) - offset)
        endpoint.write(payload[offset:offset+chunk_size], timeout=3000)
        offset += chunk_size


def _create_device_payload(image: Image.Image, header: bytes, tail: bytes, quality: int = 80) -> bytes:
    """Create device payload from image."""
    # Convert to JPEG
    bio = io.BytesIO()
    image.save(bio, format="JPEG", quality=quality, optimize=False, progressive=False, subsampling="4:2:0")
    jpeg_data = bio.getvalue()
    
    # Create header copy and patch length
    header_copy = bytearray(header[:HEADER_LEN])
    struct.pack_into("<I", header_copy, LEN_FIELD_OFF, len(jpeg_data))
    
    return bytes(header_copy) + jpeg_data + tail


# ------------------------------ Themes ------------------------------
@dataclass
class Theme:
    background: Tuple[int, int, int]
    grid: Tuple[int, int, int]
    axes: Tuple[int, int, int]
    warn_band: Tuple[int, int, int]
    crit_band: Tuple[int, int, int]
    text: Tuple[int, int, int]
    bar_usage: Tuple[int, int, int]
    bar_temp: Tuple[int, int, int]
    bar_vram: Tuple[int, int, int]
    bar_power: Tuple[int, int, int]
    bar_fan: Tuple[int, int, int]


THEMES: List[Theme] = [
    # Adapta theme (from btop-legacy branch)
    Theme(
        background=(46, 52, 54), grid=(136, 138, 133), axes=(211, 215, 207),
        warn_band=(252, 175, 62), crit_band=(204, 0, 0), text=(211, 215, 207),
        bar_usage=(0, 188, 212), bar_temp=(255, 0, 64), bar_vram=(138, 226, 52), bar_power=(212, 212, 0), bar_fan=(173, 127, 168),
    ),
    Theme(
        background=(12, 12, 14), grid=(70, 70, 80), axes=(220, 220, 230),
        warn_band=(180, 120, 20), crit_band=(150, 30, 30), text=(230, 230, 235),
        bar_usage=(60, 140, 255), bar_temp=(255, 0, 64), bar_vram=(20, 200, 220), bar_power=(255, 255, 0), bar_fan=(200, 180, 40),
    ),
    Theme(
        background=(10, 10, 12), grid=(60, 60, 70), axes=(240, 240, 245),
        warn_band=(170, 110, 25), crit_band=(160, 40, 40), text=(230, 230, 230),
        bar_usage=(80, 180, 255), bar_temp=(255, 0, 64), bar_vram=(40, 210, 230), bar_power=(255, 255, 0), bar_fan=(210, 190, 50),
    ),
]

_current_theme_index = 0


def _temp_to_color(t: float, theme: Theme) -> Tuple[int, int, int]:
    # Adapta theme temperature gradient (from btop-legacy)
    # cpu_start=(0, 188, 212), cpu_mid=(212, 212, 0), cpu_end=(255, 0, 64)
    if t <= 40:
        # Cool cyan (Adapta start color)
        return theme.bar_usage  # (0, 188, 212)
    if t <= 70:
        # Cyan to yellow transition
        a = (t - 40) / 30.0
        cyan = theme.bar_usage  # (0, 188, 212)
        yellow = theme.bar_temp  # (212, 212, 0)
        return (
            int(cyan[0] * (1 - a) + yellow[0] * a),
            int(cyan[1] * (1 - a) + yellow[1] * a),
            int(cyan[2] * (1 - a) + yellow[2] * a),
        )
    if t <= 85:
        # Yellow to red transition
        a = (t - 70) / 15.0
        yellow = theme.bar_temp  # (212, 212, 0)
        red = (255, 0, 64)  # Adapta end color
        return (
            int(yellow[0] * (1 - a) + red[0] * a),
            int(yellow[1] * (1 - a) + red[1] * a),
            int(yellow[2] * (1 - a) + red[2] * a),
        )
    # Red for high temperatures
    return (255, 0, 64)


# ------------------------------ CPU history ------------------------------
def update_cpu_history(history: List[List[float]], values: Sequence[float], max_len: int) -> List[List[float]]:
    while len(history) < len(values):
        history.append([])
    if len(history) > len(values):
        del history[len(values):]
    for i, v in enumerate(values):
        series = history[i]
        series.append(float(v))
        if len(series) > max_len:
            del series[: len(series) - max_len]
    return history


# ------------------------------ CPU/GPU info ------------------------------
def get_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    try:
        freq = psutil.cpu_freq()
        return f"CPU • {psutil.cpu_count(logical=True)} Cores @ {freq.max:.0f} MHz" if freq else f"CPU • {psutil.cpu_count(logical=True)} Cores"
    except Exception:
        return f"CPU • {psutil.cpu_count(logical=True) or 0} Cores"


def _parse_sensors(text: str) -> List[float]:
    temps: List[float] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw.lower().startswith("core "):
            continue
        plus = raw.find("+")
        deg = raw.find("°C")
        if plus != -1 and deg != -1 and deg > plus:
            try:
                temps.append(float(raw[plus + 1 : deg]))
            except Exception:
                continue
    return temps


def get_cpu_info(sampling_interval: Optional[float] = 0.1) -> Dict[str, object]:
    try:
        per_core = psutil.cpu_percent(interval=sampling_interval or 0.0, percpu=True)
        usage_percent = float(sum(per_core) / max(1, len(per_core)))
    except Exception:
        per_core = []
        usage_percent = 0.0

    temps: List[float] = []
    try:
        res = subprocess.run(["sensors"], capture_output=True, text=True, check=False)
        if res.returncode == 0 and res.stdout:
            temps = _parse_sensors(res.stdout)
    except Exception:
        pass

    try:
        freq = psutil.cpu_freq()
        freq_mhz = float(freq.current) if freq and freq.current else None
    except Exception:
        freq_mhz = None

    return {
        "model": get_cpu_model(),
        "usage_percent": usage_percent,
        "per_core": per_core,
        "temps": temps,
        "freq_mhz": freq_mhz,
    }


def _get_gpu_info_pynvml() -> Optional[Dict[str, object]]:
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h).decode("utf-8", "ignore")
        util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        fan = None
        try:
            fan = pynvml.nvmlDeviceGetFanSpeed(h)
        except Exception:
            fan = None
        power = None
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        except Exception:
            power = None
        pynvml.nvmlShutdown()
        return {
            "available": True,
            "name": name,
            "usage_percent": float(util),
            "temperature": float(temp),
            "memory_used": float(mem.used) / (1024 * 1024),  # MiB
            "memory_total": float(mem.total) / (1024 * 1024),
            "memory_percent": float(mem.used) / float(mem.total) * 100.0 if mem.total else 0.0,
            "fan_speed": None if fan is None else float(fan),
            "power_usage": None if power is None else float(power),
        }
    except Exception:
        return None


def _get_gpu_info_nvidia_smi() -> Optional[Dict[str, object]]:
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw,fan.speed",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0 or not res.stdout.strip():
            return None
        line = res.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            return None
        name = parts[0]
        util = float(parts[1])
        temp = float(parts[2])
        mem_used = float(parts[3])  # MiB
        mem_total = float(parts[4])
        power = None if parts[5] in {"N/A", ""} else float(parts[5])
        fan = None if parts[6] in {"N/A", ""} else float(parts[6])
        return {
            "available": True,
            "name": name,
            "usage_percent": util,
            "temperature": temp,
            "memory_used": mem_used,
            "memory_total": mem_total,
            "memory_percent": (mem_used / mem_total * 100.0) if mem_total else 0.0,
            "fan_speed": fan,
            "power_usage": power,
        }
    except Exception:
        return None


def get_gpu_info() -> Dict[str, object]:
    info = _get_gpu_info_pynvml()
    if info:
        return info
    info = _get_gpu_info_nvidia_smi()
    if info:
        return info
    return {"available": False}


# ------------------------------ Rendering ------------------------------
def _map(value: float, a: float, b: float, c: float, d: float) -> float:
    if b == a:
        return (c + d) / 2
    return c + (value - a) * (d - c) / (b - a)


def _draw_cpu_plot(draw: ImageDraw.ImageDraw, img: Image.Image, cpu: Dict[str, object], theme: Theme, *, origin: Tuple[int, int], size: Tuple[int, int], grid_alpha: int, axes_alpha: int):
    ox, oy = origin
    w, h = size
    # Background transparent since full bg is already drawn
    # Critical and Warning Bands
    warn_top = int(_map(90.0, 30.0, 105.0, oy + h, oy))
    warn_bottom = int(_map(80.0, 30.0, 105.0, oy + h, oy))
    crit_top = oy
    crit_bottom = warn_top

    warn_col = (*theme.warn_band, max(0, min(255, int(255 * 0.35))))
    crit_col = (*theme.crit_band, max(0, min(255, int(255 * 0.25))))
    band = Image.new("RGBA", img.size, (0, 0, 0, 0))
    bd = ImageDraw.Draw(band)
    bd.rectangle([ox, warn_top, ox + w, warn_bottom], fill=warn_col)
    bd.rectangle([ox, crit_top, ox + w, crit_bottom], fill=crit_col)
    img.alpha_composite(band)
    
    # Grid (dashed) with fixed semi-transparent color
    grid_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(grid_layer)
    grid_color = (200, 200, 200, 110)  # light gray, semi-transparent
    dash = (4, 4)
    # vertical: x every 25%
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = ox + int(w * frac)
        _draw_dashed_line(gd, (x, oy, x, oy + h), grid_color, 1, dash)
    # horizontal: y every 10°C
    for t in range(30, 106, 10):
        y = int(_map(float(t), 30.0, 105.0, oy + h, oy))
        _draw_dashed_line(gd, (ox, y, ox + w, y), grid_color, 1, dash)
    img.alpha_composite(grid_layer)

    # Axes box
    axes_col = (*theme.axes, axes_alpha)
    box = Image.new("RGBA", img.size, (0, 0, 0, 0))
    bx = ImageDraw.Draw(box)
    bx.rectangle([ox, oy, ox + w, oy + h], outline=theme.axes, width=1)
    img.alpha_composite(box)

    # Add CRITICAL and WARNING labels on top of grid/axes
    labels_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(labels_layer)
    f_label = _load_font_mono(14, bold=True)
    # CRITICAL label (red zone)
    crit_label_y = crit_top + 3
    ld.text((ox + 3, crit_label_y), "CRITICAL ≥90°C", font=f_label, fill=(254, 202, 202))
    # WARNING label (orange zone)
    warn_label_y = warn_top + 3
    ld.text((ox + 3, warn_label_y), "WARNING 80-90°C", font=f_label, fill=(254, 215, 170))
    img.alpha_composite(labels_layer)

    # Axis labels
    f_axis = _load_font_mono(16)  # 2pts larger
    # Center the x-axis label properly
    label_w = draw.textbbox((0, 0), "% Utilization", font=f_axis)[2] - draw.textbbox((0, 0), "% Utilization", font=f_axis)[0]
    draw.text((ox + w // 2 - label_w // 2, oy + h + 8), "% Utilization", font=f_axis, fill=theme.text)
    # Y-axis label: simpler, unrotated "°C" centered vertically next to axis
    y_label = "°C"
    f_axis_bold = _load_font_mono(16, bold=False)
    tb = draw.textbbox((0, 0), y_label, font=f_axis_bold)
    text_w = tb[2] - tb[0]
    text_h = tb[3] - tb[1]
    margin = 6
    tx = ox - margin - text_w
    if tx < 0:
        tx = 0
    ty = oy + (h - text_h) // 2
    draw.text((tx, ty), y_label, font=f_axis_bold, fill=theme.text)

    # CPU dots
    temps = (cpu.get("temps") or []) if isinstance(cpu, dict) else []
    per_core = (cpu.get("per_core") or []) if isinstance(cpu, dict) else []
    n = max(len(temps), len(per_core))
    if len(temps) < n:
        temps = list(temps) + [temps[-1] if temps else 50.0] * (n - len(temps))
    if len(per_core) < n:
        per_core = list(per_core) + [0.0] * (n - len(per_core))

    r = 8 # circle radius
    for i in range(n):
        t = float(temps[i])
        u = float(per_core[i])
        x = int(_map(u, 0.0, 100.0, ox, ox + w))
        y = int(_map(t, 30.0, 105.0, oy + h, oy))
        color = _temp_to_color(t, THEMES[_current_theme_index])
        # Draw circle with white outline
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(255, 255, 255), width=1)


def _draw_gpu_panel(draw: ImageDraw.ImageDraw, img: Image.Image, gpu: Dict[str, object], theme: Theme, *, origin: Tuple[int, int], size: Tuple[int, int], panel_gray_opacity: float):
    ox, oy = origin
    w, h = size
    # Panel gray overlay
    panel = Image.new("RGBA", img.size, (0, 0, 0, 0))
    pd = ImageDraw.Draw(panel)
    pd.rectangle([ox, oy, ox + w, oy + h], fill=(0, 0, 0, int(255 * panel_gray_opacity)))
    img.alpha_composite(panel)

    f_title = _load_font_mono(20, bold=True)
    f_label = _load_font_mono(16, bold=True)  
    f_value = _load_font_mono(16)  

    bar_margin = 10  # Margin on both left and right sides for bars and title
    x = ox + bar_margin  # Use bar_margin for consistent positioning
    
    y = oy +  120 # Vertical shift of GPU panel

    if not gpu.get("available"):
        draw.text((x, y), "GPU not available", font=f_label, fill=theme.text)
        return

    raw_name = str(gpu.get("name", "GPU"))
    # Strip common prefixes per spec
    name = raw_name
    for prefix in ("NVIDIA ", "NVIDIA GeForce ", "GeForce "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    draw.text((x, y), name, font=f_title, fill=theme.text)
    y += 50

    def bar(label: str, value_str: str, pct: Optional[float], color: Tuple[int, int, int]):
        nonlocal y
        bar_w = w - (2 * bar_margin)  # Total width minus margins on both sides
        bar_h = 8
        # Label left-aligned to bar start, value right-aligned to bar end
        bar_x = ox + bar_margin  # Same as bar start position
        bar_end_x = bar_x + bar_w  # Bar end position
        draw.text((bar_x, y - 4), label, font=f_label, fill=theme.text)
        # Measure value text width using textbbox for compatibility
        bbox = draw.textbbox((0, 0), value_str, font=f_value)
        vw = bbox[2] - bbox[0]
        draw.text((bar_end_x - vw, y - 4), value_str, font=f_value, fill=theme.text)
        by = y + 14
        # Track - centered within GPU panel with equal margins
        bar_x = ox + bar_margin  # Start at GPU panel origin + margin
        draw.rectangle([bar_x, by, bar_x + bar_w, by + bar_h], outline=(200, 200, 200), width=1)
        if pct is not None:
            fill_w = int(bar_w * max(0.0, min(1.0, pct / 100.0)))
            if fill_w > 0:
                draw.rectangle([bar_x, by, bar_x + fill_w, by + bar_h], fill=color)
        y = by + bar_h + 24  # Increased from 16 to 24 for more vertical space

    # GPU %
    bar("Utilization", f"{gpu.get('usage_percent', 0):.0f}%", float(gpu.get("usage_percent", 0.0)), theme.bar_usage)
    # Temp (scale 30-85°C to 0-100%)
    temp = float(gpu.get("temperature", 0))
    temp_pct = max(0, min(100, (temp - 30) / (85 - 30) * 100)) if temp > 0 else 0
    bar("Temp", f"{temp:.0f}°C", temp_pct, theme.bar_temp)
    # VRAM
    used_mib = float(gpu.get("memory_used", 0.0))
    tot_mib = float(gpu.get("memory_total", 0.0)) or 1.0
    pct_mem = used_mib / tot_mib * 100.0
    bar("VRAM", f"{used_mib/1024:.1f}/{tot_mib/1024:.0f} GB", pct_mem, theme.bar_vram)
    # Power (scale 0-450W to 0-100%)
    power = gpu.get("power_usage")
    power_pct = max(0, min(100, (float(power) / 450 * 100))) if power is not None else 0
    bar("Power", f"{power:.0f} W" if power is not None else "N/A", power_pct, theme.bar_power)
    # Fan (scale 0-3000 RPM to 0-100%)
    fan = gpu.get("fan_speed")
    fan_pct = max(0, min(100, (float(fan) / 3000 * 100))) if fan is not None else 0
    bar("Fan", f"{fan:.0f} RPM" if fan is not None else "N/A", fan_pct, theme.bar_fan)


def create_monitoring_overlay(
    cpu_info: Dict[str, object],
    gpu_info: Optional[Dict[str, object]],
    width: int = TARGET_W,
    height: int = TARGET_H,
) -> Image.Image:
    theme = THEMES[_current_theme_index]
    # Create fully transparent overlay - no background
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Title aligned with left edge of plot area, bold, with core count
    title = "Ultra 7 265K • 20 Cores"
    draw.text((PLOT_X, 10), title, font=_load_font_mono(18, bold=True), fill=theme.text)

    # Regions
    # CPU plot rectangle: configurable position and size
    cpu_origin = (PLOT_X, PLOT_Y)
    cpu_plot_w = PLOT_W
    cpu_plot_h = PLOT_H
    _draw_cpu_plot(draw, img, cpu_info, theme, origin=cpu_origin, size=(cpu_plot_w, cpu_plot_h), grid_alpha=120, axes_alpha=220)

    # GPU panel occupies right 1/3 full height
    gpu_origin = (300, 0)  # Start at 300px to make CPU area exactly 300px wide
    gpu_size = (180, height)  # Adjust width to fill remaining space
    _draw_gpu_panel(draw, img, gpu_info or {"available": False}, theme, origin=gpu_origin, size=gpu_size, panel_gray_opacity=0.5)

    return img  # Keep as RGBA for proper transparency


# ------------------------------ Main loop with key handling ------------------------------
def _stdin_key_nonblocking() -> Optional[str]:
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if not dr:
        return None
    ch = sys.stdin.read(1)
    return ch


@contextlib.contextmanager
def _with_raw_stdin():
    fd = sys.stdin.fileno()
    old = None
    try:
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except Exception:
        old = None
    try:
        yield
    finally:
        if old is not None:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass


def main(*, preview: bool = False, refresh_rate: float = 15.0) -> None:
    global _current_theme_index
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = _get_project_root()

    # Backgrounds
    bg_paths = _list_backgrounds(root)
    bg_index = 0
    if bg_paths:
        try:
            base_bg = _resize_crop_topleft(Image.open(bg_paths[bg_index]).convert("RGB"), TARGET_W, TARGET_H)
            print(f"Loaded background: {bg_paths[bg_index].name}")
        except Exception as e:
            print(f"Failed to load background {bg_paths[bg_index].name}: {e}")
            base_bg = Image.new("RGB", (TARGET_W, TARGET_H), THEMES[_current_theme_index].background)
    else:
        print("No background images found in assets/images/")
        base_bg = Image.new("RGB", (TARGET_W, TARGET_H), THEMES[_current_theme_index].background)

    blur_levels = [0, 2, 4, 6, 8]
    gray_levels = [0.0, 0.2, 0.4, 0.6, 0.8]
    blur_idx = 2  # Default to 4px blur
    gray_idx = 2  # Default to 0.4 gray
    show_stats = True

    # Cache processed backgrounds per (bg_index, blur_idx, gray_idx, show_stats)
    bg_cache: Dict[Tuple[int, int, int, bool], Image.Image] = {}

    # EMA smoothing state for CPU - moderate smoothing for balanced drifting
    prev_per_core: List[float] = []
    prev_temps: List[float] = []
    alpha = 0.05  # Gentle smoothing (10% new, 90% previous) - less jittery

    period = 1.0 / max(1e-6, refresh_rate)
    logging.info("Monitor started (preview=%s, target_fps=%.1f)", preview, refresh_rate)
    
    # Frame rate tracking
    frame_count = 0
    fps_start_time = time.time()

    def get_processed_bg() -> Image.Image:
        key = (bg_index, blur_idx, gray_idx, show_stats)
        if key in bg_cache:
            return bg_cache[key]
        # Always start from the current base background
        bg = base_bg.copy()
        # When stats shown, apply blur+gray per current indices
        if show_stats:
            if blur_levels[blur_idx] > 0:
                bg = bg.filter(ImageFilter.GaussianBlur(blur_levels[blur_idx]))
            if gray_levels[gray_idx] > 0:
                bg = _apply_gray_overlay(bg, gray_levels[gray_idx])
        bg_cache[key] = bg
        return bg

    # Raw stdin for key handling
    if sys.stdin.isatty():
        pass

    # Device mode: initialize USB device and load reference frame
    device = None
    interface = None
    endpoint = None
    header = None
    tail = None
    
    if not preview:
        try:
            # Load reference frame data
            frame_bin_path = root / "assets" / "data" / "frame.bin"
            if not frame_bin_path.exists():
                raise RuntimeError(f"Reference frame not found: {frame_bin_path}")
            
            frame_data = frame_bin_path.read_bytes()
            header, _, tail = _split_frame(frame_data)
            print(f"Loaded reference frame, header length: {len(header)}")
            
            # Open device
            device, interface, endpoint = _open_device()
            print("Device opened successfully")
            print(f"Starting system monitor on device (refresh rate: {refresh_rate:.1f} FPS)")
            
        except Exception as e:
            logging.error("Failed to initialize device: %s", e)
            print(f"Device initialization failed: {e}")
            print("Falling back to preview mode...")
            preview = True

    # Preview mode: use Tkinter window for efficient updates
    if preview:
        try:
            import tkinter as tk
        except Exception as exc:
            logging.warning("Tkinter not available for preview: %s", exc)
            # Fallback to single-frame show
            frame = get_processed_bg().copy()
            overlay = create_monitoring_overlay(get_cpu_info(0.05), get_gpu_info(), TARGET_W, TARGET_H)
            frame.paste(overlay, (0, 0), None)
            frame.show()
            return

        root = tk.Tk()
        root.title("Thermalright Monitor Preview")
        root.geometry(f"{TARGET_W}x{TARGET_H}")
        label = tk.Label(root)
        label.pack()

        img_holder = {"img": None}  # prevent GC
        stopping = [False]

        def _on_sigint(signum, frame):
            # Mark stopping; update_frame will quit the loop ASAP
            stopping[0] = True

        try:
            signal.signal(signal.SIGINT, _on_sigint)
        except Exception:
            pass

        def update_frame():
            nonlocal base_bg, bg_index, blur_idx, gray_idx, show_stats, prev_per_core, prev_temps, frame_count
            if stopping[0]:
                try:
                    root.quit()
                finally:
                    return
            # Handle terminal keys
            ch = _stdin_key_nonblocking()
            if ch:
                if ch == 't':
                    _current_theme_index = (_current_theme_index + 1) % len(THEMES)
                    bg_cache.clear()
                    print(f"Key 't': Theme changed to {_current_theme_index}")
                elif ch == 'b':
                    blur_idx = (blur_idx + 1) % len(blur_levels)
                    bg_cache.clear()
                    print(f"Key 'b': Blur level {blur_idx} ({blur_levels[blur_idx]}px)")
                elif ch == 'g':
                    gray_idx = (gray_idx + 1) % len(gray_levels)
                    bg_cache.clear()
                    print(f"Key 'g': Gray level {gray_idx} ({gray_levels[gray_idx]:.1f})")
                elif ch == 's':
                    show_stats = not show_stats
                    print(f"Key 's': Stats {'shown' if show_stats else 'hidden'}")
                elif ch == 'i':
                    if bg_paths:
                        bg_index = (bg_index + 1) % len(bg_paths)
                        base_bg = _resize_crop_topleft(Image.open(bg_paths[bg_index]).convert("RGB"), TARGET_W, TARGET_H)
                        bg_cache.clear()
                        print(f"Key 'i': Background changed to {bg_paths[bg_index].name}")

            cpu = get_cpu_info(0.0)  # No sampling delay for better FPS
            gpu = get_gpu_info()
            # Moderate smoothing for balanced drifting
            cur_pc = list(cpu.get("per_core", []))
            cur_t = list(cpu.get("temps", []))
            
            # Initialize arrays if needed
            if len(prev_per_core) < len(cur_pc):
                prev_per_core.extend([cur_pc[len(prev_per_core)]] * (len(cur_pc) - len(prev_per_core)))
            if len(prev_temps) < len(cur_t):
                prev_temps.extend([cur_t[len(prev_temps)]] * (len(cur_t) - len(prev_temps)))
            
            if cur_pc:
                sm_pc = [alpha * float(v) + (1 - alpha) * float(prev_per_core[i] if i < len(prev_per_core) else v) 
                        for i, v in enumerate(cur_pc)]
                prev_per_core = sm_pc
                cpu["per_core"] = sm_pc
                
            if cur_t:
                sm_t = [alpha * float(v) + (1 - alpha) * float(prev_temps[i] if i < len(prev_temps) else v) 
                       for i, v in enumerate(cur_t)]
                prev_temps = sm_t
                cpu["temps"] = sm_t

            # Use cached background directly, no copy/conversion
            bg = get_processed_bg()
            if show_stats:
                overlay = create_monitoring_overlay(cpu, gpu, TARGET_W, TARGET_H)
                # Composite transparent overlay onto background
                bg_rgba = bg.convert("RGBA")
                bg_rgba.alpha_composite(overlay)
                frame_img = bg_rgba.convert("RGB")
            else:
                frame_img = bg
            tk_img = ImageTk.PhotoImage(frame_img)
            img_holder["img"] = tk_img
            label.configure(image=tk_img)
            
            # Frame rate tracking
            frame_count += 1
            if frame_count % 30 == 0:  # Print every 30 frames (~2 seconds at 15fps)
                elapsed = time.time() - fps_start_time
                actual_fps = frame_count / elapsed
                print(f"FPS: target={refresh_rate:.1f}, actual={actual_fps:.1f}")
            
            root.after(int(1000 / refresh_rate), update_frame)

        update_frame()
        try:
            with _with_raw_stdin():
                root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                root.destroy()
            except Exception:
                pass
        return

    # Device mode (no preview window)
    start = time.time()
    next_frame = start
    try:
        with _with_raw_stdin():
            while True:
                frame_start = time.time()

                # Handle keys
                ch = _stdin_key_nonblocking()
                if ch:
                    if ch == 't':
                        _current_theme_index = (_current_theme_index + 1) % len(THEMES)
                        bg_cache.clear()
                        print(f"Key 't': Theme changed to {_current_theme_index}")
                    elif ch == 'b':
                        blur_idx = (blur_idx + 1) % len(blur_levels)
                        bg_cache.clear()
                        print(f"Key 'b': Blur level {blur_idx} ({blur_levels[blur_idx]}px)")
                    elif ch == 'g':
                        gray_idx = (gray_idx + 1) % len(gray_levels)
                        bg_cache.clear()
                        print(f"Key 'g': Gray level {gray_idx} ({gray_levels[gray_idx]:.1f})")
                    elif ch == 's':
                        show_stats = not show_stats
                        print(f"Key 's': Stats {'shown' if show_stats else 'hidden'}")
                    elif ch == 'i':
                        if bg_paths:
                            bg_index = (bg_index + 1) % len(bg_paths)
                            base_bg = _resize_crop_topleft(Image.open(bg_paths[bg_index]).convert("RGB"), TARGET_W, TARGET_H)
                            bg_cache.clear()
                            print(f"Key 'i': Background changed to {bg_paths[bg_index].name}")

                # Data
                cpu = get_cpu_info(0.0)  # No sampling delay for better FPS
                gpu = get_gpu_info()
                # Moderate smoothing for balanced drifting
                cur_pc = list(cpu.get("per_core", []))
                cur_t = list(cpu.get("temps", []))
                
                # Initialize arrays if needed
                if len(prev_per_core) < len(cur_pc):
                    prev_per_core.extend([cur_pc[len(prev_per_core)]] * (len(cur_pc) - len(prev_per_core)))
                if len(prev_temps) < len(cur_t):
                    prev_temps.extend([cur_t[len(prev_temps)]] * (len(cur_t) - len(prev_temps)))
                
                if cur_pc:
                    sm_pc = [alpha * float(v) + (1 - alpha) * float(prev_per_core[i] if i < len(prev_per_core) else v) 
                            for i, v in enumerate(cur_pc)]
                    prev_per_core = sm_pc
                    cpu["per_core"] = sm_pc
                    
                if cur_t:
                    sm_t = [alpha * float(v) + (1 - alpha) * float(prev_temps[i] if i < len(prev_temps) else v) 
                           for i, v in enumerate(cur_t)]
                    prev_temps = sm_t
                    cpu["temps"] = sm_t

                # Compose
                bg = get_processed_bg()
                if show_stats:
                    overlay = create_monitoring_overlay(cpu, gpu, TARGET_W, TARGET_H)
                    bg_rgba = bg.convert("RGBA")
                    bg_rgba.alpha_composite(overlay)
                    frame_img = bg_rgba.convert("RGB")
                else:
                    frame_img = bg

                # Send to device
                if not preview and device and endpoint:
                    try:
                        payload = _create_device_payload(frame_img, header, tail, quality=80)
                        _send_payload(endpoint, payload)
                    except Exception as e:
                        logging.error("Failed to send to device: %s", e)
                        print(f"Device send failed: {e}")
                        # Continue running even if send fails

                # Frame pacing and FPS tracking
                frame_count += 1
                if frame_count % 30 == 0:  # Print every 30 frames (~2 seconds at 15fps)
                    elapsed = time.time() - fps_start_time
                    actual_fps = frame_count / elapsed
                    print(f"FPS: target={refresh_rate:.1f}, actual={actual_fps:.1f}")
                
                elapsed = time.time() - frame_start
                sleep_time = period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        # Cleanup device resources
        if device and interface:
            try:
                usb.util.release_interface(device, interface.bInterfaceNumber)
                device.attach_kernel_driver(interface.bInterfaceNumber)
            except Exception:
                pass


__all__ = [
    "TARGET_W",
    "TARGET_H",
    "update_cpu_history",
    "get_cpu_model",
    "get_cpu_info",
    "get_gpu_info",
    "create_monitoring_overlay",
    "main",
]


