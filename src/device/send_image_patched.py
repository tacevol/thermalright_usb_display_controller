#!/usr/bin/env python3
import sys, io, struct
from pathlib import Path
from PIL import Image
import usb.core, usb.util

VENDOR = 0x87ad
PRODUCT = 0x70db
CHUNK  = 4096

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
FRAME_BIN = PROJECT_ROOT / "assets" / "data" / "frame.bin"     # reference capture you used before (to steal header & tail)
HEADER_LEN = 64             # analyzer showed common header length = 64
LEN_FIELD_OFF = 60          # u32 little-endian = JPEG length (bytes)
TARGET_W, TARGET_H = 480, 480

def split_frame(b: bytes):
    soi = b.find(b"\xff\xd8")
    eoi = b.rfind(b"\xff\xd9")
    if soi < 0 or eoi < 0:
        raise RuntimeError("SOI/EOI not found in frame.bin")
    return b[:soi], b[soi:eoi+2], b[eoi+2:]

def make_jpeg(path: str, quality=80) -> bytes:
    im = Image.open(path).convert("RGB")
    # center-crop to square, then resize to 480×480
    w, h = im.size
    if w != h:
        s = min(w, h)
        left = (w - s)//2
        top  = (h - s)//2
        im = im.crop((left, top, left+s, top+s))
    if im.size != (TARGET_W, TARGET_H):
        im = im.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    bio = io.BytesIO()
    im.save(bio, format="JPEG", quality=quality, optimize=False, progressive=False, subsampling="4:2:0")
    return bio.getvalue()

def open_dev():
    dev = usb.core.find(idVendor=VENDOR, idProduct=PRODUCT)
    if dev is None:
        raise RuntimeError("Device 87ad:70db not found")
    dev.set_configuration()
    cfg = dev.get_active_configuration()
    intf = None
    ep_out = None
    for ifc in cfg:
        try:
            if dev.is_kernel_driver_active(ifc.bInterfaceNumber):
                dev.detach_kernel_driver(ifc.bInterfaceNumber)
        except Exception:
            pass
        usb.util.claim_interface(dev, ifc.bInterfaceNumber)
        for ep in ifc:
            if usb.util.endpoint_direction(ep.bEndpointAddress) == usb.util.ENDPOINT_OUT:
                intf, ep_out = ifc, ep
                break
        if ep_out:
            break
    if not ep_out:
        raise RuntimeError("No OUT endpoint found")
    return dev, intf, ep_out

def send(ep_out, payload: bytes):
    off = 0
    ln = len(payload)
    while off < ln:
        n = ep_out.write(payload[off:off+CHUNK], timeout=3000)
        off += n

def main():
    img_path = PROJECT_ROOT / "assets" / "images" / "moose02.png" if len(sys.argv) < 2 else sys.argv[1]
    quality  = 100
    if len(sys.argv) >= 3:
        try: quality = int(sys.argv[2])
        except: pass

    ref = Path(FRAME_BIN).read_bytes()
    header, _refjpg, tail = split_frame(ref)
    if len(header) < HEADER_LEN:
        raise RuntimeError(f"Header too short: {len(header)} bytes")
    # normalize: only keep first 64 for safety (analyzer showed 64)
    header = bytearray(header[:HEADER_LEN])

    jpg = make_jpeg(img_path, quality=quality)
    jpg_len = len(jpg)

    # Patch length at offset 60 as u32 little-endian
    struct.pack_into("<I", header, LEN_FIELD_OFF, jpg_len)

    payload = bytes(header) + jpg + tail
    (PROJECT_ROOT / "assets" / "data" / "patched_payload.bin").write_bytes(payload)   # handy for reuse/looping

    print(f"JPEG {img_path}: {jpg_len} bytes (q={quality})")
    print(f"Header[60:64] -> {jpg_len} (u32le)")
    print(f"Sending {len(payload)} bytes total")

    dev, intf, ep_out = open_dev()
    try:
        send(ep_out, payload)
    finally:
        usb.util.release_interface(dev, intf.bInterfaceNumber)
        try:
            dev.attach_kernel_driver(intf.bInterfaceNumber)
        except Exception:
            pass
    print("Done. Expect ~3s watchdog revert unless you loop it.")

if __name__ == "__main__":
    main()
