from __future__ import annotations

from pathlib import Path

from PIL import Image


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src_png = root / "assets" / "favicon.png"
    out_dir = root / "src" / "fabric_mvp" / "static"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src_png.exists():
        raise SystemExit(f"Missing source image: {src_png}")

    img = Image.open(src_png).convert("RGBA")

    # Create ICO with multiple sizes for better compatibility.
    ico_path = out_dir / "favicon.ico"
    img.save(ico_path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])

    # Also keep a PNG variant for browsers that prefer it.
    png_path = out_dir / "favicon.png"
    img.save(png_path, format="PNG", optimize=True)

    print(f"Wrote: {ico_path}")
    print(f"Wrote: {png_path}")


if __name__ == "__main__":
    main()

