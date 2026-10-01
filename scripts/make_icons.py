"""Render the GRAVITY app icons (PNG) from the favicon design — stdlib only.

Ring (platinum) above a falling red bar on near-black; supersampled 4x for
smooth edges. Writes docs/assets/img/icon-{180,192,512}.png and a maskable
512 with extra safe-zone padding.
"""
import struct
import zlib
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "img"
BG = (7, 8, 10)
FG = (238, 241, 244)
RED = (255, 46, 77)


def render(size: int, pad: float = 0.0) -> bytes:
    ss = 4
    n = size * ss
    scale = n * (1 - 2 * pad) / 32.0
    off = n * pad
    cx, cy, r, sw = 16 * scale + off, 11 * scale + off, 5.5 * scale, 2 * scale
    bx0, bx1 = 15 * scale + off, 17 * scale + off
    by0, by1 = 17.5 * scale + off, 26.5 * scale + off
    rows = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            acc = [0, 0, 0]
            for sy in range(ss):
                for sx in range(ss):
                    px, py = x * ss + sx + 0.5, y * ss + sy + 0.5
                    d = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
                    if abs(d - r) <= sw / 2:
                        c = FG
                    elif bx0 <= px <= bx1 and by0 <= py <= by1:
                        c = RED
                    else:
                        c = BG
                    acc[0] += c[0]; acc[1] += c[1]; acc[2] += c[2]
            k = ss * ss
            row += bytes((acc[0] // k, acc[1] // k, acc[2] // k))
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for s in (180, 192, 512):
        (OUT / f"icon-{s}.png").write_bytes(render(s, pad=0.08))
    (OUT / "icon-maskable-512.png").write_bytes(render(512, pad=0.2))
    print("icons written")
