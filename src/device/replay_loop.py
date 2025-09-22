#!/usr/bin/env python3
import time, usb.core, usb.util, sys
from pathlib import Path

VENDOR = 0x87ad
PRODUCT = 0x70db

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
INTERVAL_S = 2.9       # try 1 Hz first; we can tweak

def find_out_endpoint(dev):
    cfg = dev.get_active_configuration()
    for intf in cfg:
        try:
            if dev.is_kernel_driver_active(intf.bInterfaceNumber):
                dev.detach_kernel_driver(intf.bInterfaceNumber)
        except Exception:
            pass
        usb.util.claim_interface(dev, intf.bInterfaceNumber)
        for ep in intf:
            if usb.util.endpoint_direction(ep.bEndpointAddress) == usb.util.ENDPOINT_OUT:
                return intf, ep
    raise RuntimeError("No OUT endpoint found")

def main():
    data = (PROJECT_ROOT / "assets" / "data" / "patched_payload.bin").read_bytes()
    if not data:
        print("patched_payload.bin empty"); sys.exit(1)

    dev = usb.core.find(idVendor=VENDOR, idProduct=PRODUCT)
    if dev is None:
        print("Device 87ad:70db not found"); sys.exit(2)

    dev.set_configuration()
    intf, ep_out = find_out_endpoint(dev)
    mps = ep_out.wMaxPacketSize
    print(f"Using OUT 0x{ep_out.bEndpointAddress:02x}, maxpkt={mps}, payload={len(data)} bytes")

    try:
        t0 = time.time()
        count = 0
        while True:
            off = 0
            while off < len(data):
                n = ep_out.write(data[off:off+4096], timeout=3000)
                off += n

            # If exact multiple of max packet size, consider sending a ZLP
            if len(data) % mps == 0:
                # Some devices need a ZLP; try both ways. Start without it.
                pass

            count += 1
            # simple status each second
            print(f"[{count}] sent {len(data)} bytes; elapsed {time.time()-t0:.1f}s")
            time.sleep(INTERVAL_S)
    finally:
        usb.util.release_interface(dev, intf.bInterfaceNumber)
        try:
            dev.attach_kernel_driver(intf.bInterfaceNumber)
        except Exception:
            pass

if __name__ == "__main__":
    main()
