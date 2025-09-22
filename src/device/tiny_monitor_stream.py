#!/usr/bin/env python3
import io, time, subprocess, re, sys, signal
from pathlib import Path
from PIL import Image
import mss
import usb.core, usb.util
import statistics

# ====== user settings ======
WINDOW_TITLE   = "btop"     # substring to find the window to mirror
FPS            = 20          # try 5 or 10, keep comfortably under USB and CPU limits
TARGET_W       = 480
TARGET_H       = 480

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
FRAME_BIN = PROJECT_ROOT / "assets" / "data" / "frame.bin"  # the capture you used earlier
# ===========================


VENDOR = 0x87ad
PRODUCT = 0x70db
CHUNK   = 4096

def split_frame(path):
    data = Path(path).read_bytes()
    soi = data.find(b"\xff\xd8")
    eoi = data.rfind(b"\xff\xd9")
    if soi < 0 or eoi < 0:
        raise RuntimeError("SOI or EOI not found in frame.bin")
    header = data[:soi]
    jpg    = data[soi:eoi+2]
    tail   = data[eoi+2:]
    return header, jpg, tail

def find_window_geometry(title_substring):
    # returns x, y, w, h for the best window match
    try:
        ids = subprocess.check_output(["xdotool", "search", "--name", title_substring], text=True).strip().splitlines()
        if not ids:
            return None
        wid = ids[-1].strip()
        geom = subprocess.check_output(["xdotool", "getwindowgeometry", "--shell", wid], text=True)
        # output like:
        #  X=123
        #  Y=456
        #  WIDTH=800
        #  HEIGHT=600
        vals = dict(re.findall(r"(?m)^([A-Z]+)=(\d+)$", geom))
        x = int(vals["X"]); y = int(vals["Y"]); w = int(vals["WIDTH"]); h = int(vals["HEIGHT"])
        return x, y, w, h
    except subprocess.CalledProcessError:
        return None

def square_crop_box(x, y, w, h):
    s = min(w, h)
    left = x + (w - s)//2
    top  = y + (h - s)//2
    return left, top, s, s

def encode_jpeg_fit(im_rgb, target_len, q_start=80):
    # encode, never exceed target_len, then pad to exact length
    def try_jpeg(q):
        bio = io.BytesIO()
        im_rgb.save(bio, format="JPEG", quality=q, optimize=False, progressive=False, subsampling="4:2:0")
        return bio.getvalue()

    q = q_start
    data = try_jpeg(q)
    # reduce quality if we overshoot
    while len(data) > target_len and q > 5:
        q = max(5, q - 5)
        data = try_jpeg(q)

    # if still too large at q=5, drop subsampling quality further
    while len(data) > target_len and q > 1:
        q -= 1
        data = try_jpeg(q)

    # pad up to exact length
    if len(data) <= target_len:
        return data + b"\x00" * (target_len - len(data))
    # as a last resort, hard trim after EOI if target allows
    eoi = data.rfind(b"\xff\xd9")
    if eoi >= 0 and eoi + 2 <= target_len:
        out = data[:target_len]
        if not out.endswith(b"\xff\xd9"):
            out = out[:-2] + b"\xff\xd9"
        return out + b"\x00" * (target_len - len(out))
    # if we get here the target_len is smaller than a minimal JPEG
    raise RuntimeError("Target length is too small for an encodable JPEG")

def open_device():
    dev = usb.core.find(idVendor=VENDOR, idProduct=PRODUCT)
    if dev is None:
        raise RuntimeError("Device 87ad:70db not found")
    dev.set_configuration()
    cfg = dev.get_active_configuration()
    chosen = None
    for intf in cfg:
        try:
            if dev.is_kernel_driver_active(intf.bInterfaceNumber):
                dev.detach_kernel_driver(intf.bInterfaceNumber)
        except Exception:
            pass
        usb.util.claim_interface(dev, intf.bInterfaceNumber)
        for ep in intf:
            if usb.util.endpoint_direction(ep.bEndpointAddress) == usb.util.ENDPOINT_OUT:
                chosen = (intf, ep)
                break
        if chosen:
            break
    if not chosen:
        raise RuntimeError("No OUT endpoint found")
    return dev, chosen[0], chosen[1]

def send_payload(ep_out, payload):
    off = 0
    ln = len(payload)
    while off < ln:
        n = ep_out.write(payload[off:off+CHUNK], timeout=3000)
        off += n

def main():
    header, orig_jpg, tail = split_frame(FRAME_BIN)
    target_len = len(orig_jpg)
    print(f"Target JPEG length from capture: {target_len}")

    dev, intf, ep_out = open_device()
    mps = ep_out.wMaxPacketSize
    print(f"Using OUT 0x{ep_out.bEndpointAddress:02x} maxpkt {mps}")

    sct = mss.mss()
    q_guess = 80
    last_geom = None
    last_geom_check = 0

    def cleanup(*_):
        usb.util.release_interface(dev, intf.bInterfaceNumber)
        try:
            dev.attach_kernel_driver(intf.bInterfaceNumber)
        except Exception:
            pass
        print("\nReleased USB interface")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    frame_time = 1.0 / max(1, FPS)
    tick = 0
    ema =  None
    while True:
        # refresh window geometry once per second
        now = time.time()
        if last_geom is None or now - last_geom_check > 1.0:
            g = find_window_geometry(WINDOW_TITLE)
            if g is not None:
                last_geom = square_crop_box(*g)
            last_geom_check = now

        # fallback if window not found: center of primary monitor
        if last_geom is None:
            mon = sct.monitors[1]
            cx = mon["left"] + mon["width"] // 2
            cy = mon["top"] + mon["height"] // 2
            left = cx - TARGET_W // 2
            top  = cy - TARGET_H // 2
            bbox = {"left": left, "top": top, "width": TARGET_W, "height": TARGET_H}
        else:
            left, top, size, _ = last_geom
            bbox = {"left": left, "top": top, "width": size, "height": size}

        # capture
        shot = sct.grab(bbox)
        im = Image.frombytes("RGB", shot.size, shot.rgb)
        if im.size != (TARGET_W, TARGET_H):
            im = im.resize((TARGET_W, TARGET_H), Image.LANCZOS)

        # encode JPEG to the exact captured length
        jpg = encode_jpeg_fit(im, target_len, q_start=q_guess)
        # simple adaptive tweak for next frame
        if len(jpg) < target_len * 0.8 and q_guess < 95:
            q_guess += 2
        elif len(jpg) > target_len and q_guess > 10:
            q_guess -= 2

        payload = header + jpg + tail
        send_payload(ep_out, payload)

        tick += 1
        # basic pacing
        elapsed = time.time() - now
        sleep_for = frame_time - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)
        # tiny heartbeat print
        if tick % FPS == 0:
            print(f"sent {tick} frames")

if __name__ == "__main__":
    main()
