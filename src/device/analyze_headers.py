#!/usr/bin/env python3
import sys, struct, zlib, itertools, binascii
from pathlib import Path

def split_frame_bytes(b):
    soi = b.find(b"\xff\xd8")
    eoi = b.rfind(b"\xff\xd9")
    if soi < 0 or eoi < 0:
        raise ValueError("SOI or EOI not found")
    return b[:soi], b[soi:eoi+2], b[eoi+2:]

def u16le(b, off): return struct.unpack_from("<H", b, off)[0]
def u16be(b, off): return struct.unpack_from(">H", b, off)[0]
def u32le(b, off): return struct.unpack_from("<I", b, off)[0]
def u32be(b, off): return struct.unpack_from(">I", b, off)[0]

def crc16_ccitt_false(data, init=0xFFFF, poly=0x1021):
    crc = init
    for x in data:
        crc ^= (x << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc

def crc16_ibm(data, init=0x0000, poly=0x8005):
    crc = init
    for x in data:
        crc ^= x
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001  # reflected 0x8005
            else:
                crc >>= 1
    return crc & 0xFFFF

def scan_candidates(headers, jpg_lens, payload_lens):
    """
    Look for offsets in the header that match jpg length or payload length
    as 16 or 32 bit little or big endian fields across all samples.
    Also try to spot a monotonic frame counter.
    """
    H = headers
    # ensure all headers have same length for a clean diff
    hlen = min(len(h) for h in H)
    H = [h[:hlen] for h in H]

    print(f"Common header length considered: {hlen} bytes for all samples")

    matches = []
    for off in range(0, hlen - 1):
        vals16le = [u16le(h, off) for h in H]
        vals16be = [u16be(h, off) for h in H]
        vals32le = [u32le(h, off) for h in H] if off + 4 <= hlen else None
        vals32be = [u32be(h, off) for h in H] if off + 4 <= hlen else None

        for name, vals in [
            ("u16le", vals16le),
            ("u16be", vals16be),
            ("u32le", vals32le),
            ("u32be", vals32be),
        ]:
            if vals is None: 
                continue
            if vals == jpg_lens:
                matches.append((off, name, "equals_jpg_len"))
            if vals == payload_lens:
                matches.append((off, name, "equals_payload_len"))

    # crude counter guess: fields that strictly increase by one
    counters = []
    for off in range(0, hlen - 1):
        for bits, read in [("u16le", u16le), ("u16be", u16be)]:
            seq = [read(h, off) for h in H]
            inc1 = all((seq[i+1] - seq[i]) % 0x10000 == 1 for i in range(len(seq)-1))
            if inc1:
                counters.append((off, bits, seq))
    for off in range(0, hlen - 3):
        for bits, read in [("u32le", u32le), ("u32be", u32be)]:
            seq = [read(h, off) for h in H]
            inc1 = all((seq[i+1] - seq[i]) % 0x100000000 == 1 for i in range(len(seq)-1))
            if inc1:
                counters.append((off, bits, seq))

    print("\nCandidate length fields that match exactly across all samples:")
    for off, kind, what in matches:
        print(f"  offset {off} as {kind} -> {what}")

    if counters:
        print("\nPossible frame counters that increment by one:")
        for off, kind, seq in counters:
            print(f"  offset {off} as {kind} values {seq}")

    # checksum hunt over the JPEG only, and over header plus JPEG
    print("\nChecksum hunt (common ones) trying to match any 16 or 32 bit word in header:")
    def check_and_print(name, values, endian="le"):
        # try to find these values inside each header at the same offset
        # requires same header length and same offsets
        offs = []
        for i, h in enumerate(H):
            v = values[i]
            if endian == "le":
                needle16 = struct.pack("<H", v & 0xFFFF)
                needle32 = struct.pack("<I", v & 0xFFFFFFFF)
            else:
                needle16 = struct.pack(">H", v & 0xFFFF)
                needle32 = struct.pack(">I", v & 0xFFFFFFFF)
            # collect offsets where we find exact bytes
            pos16 = h.find(needle16)
            pos32 = h.find(needle32)
            offs.append((pos16, pos32))
        # require a consistent offset across all samples
        for idx in [0, 1]:
            pos = [o[idx] for o in offs]
            if all(p >= 0 and p == pos[0] for p in pos):
                size = 16 if idx == 0 else 32
                print(f"  {name} {size} bit {endian} appears at consistent offset {pos[0]}")

    jpg_crc32  = [zlib.crc32(j) & 0xFFFFFFFF for j in jpgs]
    jpg_ccitt  = [crc16_ccitt_false(j) for j in jpgs]
    jpg_ibm    = [crc16_ibm(j) for j in jpgs]
    all_crc32  = [zlib.crc32(H[i] + jpgs[i]) & 0xFFFFFFFF for i in range(len(H))]
    all_ccitt  = [crc16_ccitt_false(H[i] + jpgs[i]) for i in range(len(H))]
    all_ibm    = [crc16_ibm(H[i] + jpgs[i]) for i in range(len(H))]

    check_and_print("jpg_crc32", jpg_crc32, "le")
    check_and_print("jpg_crc32", jpg_crc32, "be")
    check_and_print("jpg_crc16_ccitt", jpg_ccitt, "le")
    check_and_print("jpg_crc16_ccitt", jpg_ccitt, "be")
    check_and_print("jpg_crc16_ibm", jpg_ibm, "le")
    check_and_print("jpg_crc16_ibm", jpg_ibm, "be")
    check_and_print("all_crc32", all_crc32, "le")
    check_and_print("all_crc32", all_crc32, "be")
    check_and_print("all_crc16_ccitt", all_ccitt, "le")
    check_and_print("all_crc16_ccitt", all_ccitt, "be")
    check_and_print("all_crc16_ibm", all_ibm, "le")
    check_and_print("all_crc16_ibm", all_ibm, "be")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: analyze_headers.py frame_*.bin")
        sys.exit(1)

    headers = []
    jpgs = []
    tails = []
    jpg_lens = []
    payload_lens = []

    for p in sys.argv[1:]:
        b = Path(p).read_bytes()
        h, j, t = split_frame_bytes(b)
        headers.append(h)
        jpgs.append(j)
        tails.append(t)
        jpg_lens.append(len(j))
        payload_lens.append(len(b))

    print(f"Loaded {len(headers)} frames")
    print("JPEG lengths:", jpg_lens)
    print("Payload lengths:", payload_lens)

    scan_candidates(headers, jpg_lens, payload_lens)
