#!/usr/bin/env python3
"""
crop_figures.py - Tight bounding-box cropping and aspect-ratio harmonization for Manim figures.

Trims excess white background margins and adds a professional padding so that figures
fill LaTeX subfigure frames cleanly without excessive whitespace or optical scaling mismatches.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence
import numpy as np
from PIL import Image


def get_content_bbox(
    img: Image.Image,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    threshold: int = 5,
) -> tuple[int, int, int, int]:
    """
    Find the bounding box (left, top, right, bottom) containing non-background pixels.
    """
    arr = np.array(img.convert("RGB"))
    bg = np.array(bg_color, dtype=np.int16).reshape(1, 1, 3)
    diff = np.abs(arr.astype(np.int16) - bg)
    mask = (diff > threshold).any(axis=2)

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return (0, 0, img.width, img.height)

    return (int(cols[0]), int(rows[0]), int(cols[-1] + 1), int(rows[-1] + 1))


def crop_image_tight(
    img: Image.Image,
    padding: int = 30,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    threshold: int = 5,
) -> Image.Image:
    """
    Crop an image tightly around its content and add a uniform padding around all 4 sides.
    """
    left, top, right, bottom = get_content_bbox(img, bg_color=bg_color, threshold=threshold)
    content_w = right - left
    content_h = bottom - top

    target_w = content_w + 2 * padding
    target_h = content_h + 2 * padding

    content = img.crop((left, top, right, bottom))
    canvas = Image.new("RGB", (target_w, target_h), bg_color)
    canvas.paste(content, (padding, padding))
    return canvas


def crop_and_harmonize(
    images: Sequence[Image.Image],
    padding: int = 30,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    threshold: int = 5,
) -> list[Image.Image]:
    """
    Crop a batch of images to their visual content and harmonize canvas dimensions
    so that all images share the same width and height while preserving a minimum padding.

    This ensures that when included in LaTeX subfigures with \\includegraphics[width=\\linewidth],
    both subfigures have identical heights, optical scales, and baseline alignment.
    """
    bboxes = [get_content_bbox(im, bg_color=bg_color, threshold=threshold) for im in images]

    max_content_w = max(r - l for l, t, r, b in bboxes)
    max_content_h = max(b - t for l, t, r, b in bboxes)

    target_w = max_content_w + 2 * padding
    target_h = max_content_h + 2 * padding

    harmonized = []
    for im, (l, t, r, b) in zip(images, bboxes):
        content = im.crop((l, t, r, b))
        cw = r - l
        ch = b - t
        pad_x = (target_w - cw) // 2
        pad_y = (target_h - ch) // 2

        canvas = Image.new("RGB", (target_w, target_h), bg_color)
        canvas.paste(content, (pad_x, pad_y))
        harmonized.append(canvas)

    return harmonized


def process_figures(
    input_paths: list[Path],
    output_paths: list[Path] | None = None,
    padding: int = 30,
    harmonize: bool = True,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    threshold: int = 5,
) -> list[tuple[Path, tuple[int, int], Path, tuple[int, int]]]:
    """
    Load, crop, and save figures. Returns a summary list of (in_path, in_size, out_path, out_size).
    """
    images = [Image.open(p) for p in input_paths]

    if harmonize and len(images) > 1:
        processed = crop_and_harmonize(
            images, padding=padding, bg_color=bg_color, threshold=threshold
        )
    else:
        processed = [
            crop_image_tight(im, padding=padding, bg_color=bg_color, threshold=threshold)
            for im in images
        ]

    results = []
    for in_path, in_img, out_img, out_path in zip(input_paths, images, processed, output_paths):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(out_path, format="PNG", optimize=True)
        results.append((in_path, in_img.size, out_path, out_img.size))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop excessive white margins from Manim figures and harmonize proportions for LaTeX."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input image files. If omitted, default flow_viz figures will be processed.",
    )
    parser.add_argument(
        "--padding",
        "-p",
        type=int,
        default=30,
        help="Padding in pixels around visual content bounding box (default: 30).",
    )
    parser.add_argument(
        "--no-harmonize",
        action="store_true",
        help="Disable aspect ratio / dimension harmonization across images (perform independent tight crops).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory for cropped images (default: same directory as inputs).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_cropped",
        help="Suffix added to filename if output name not explicitly given (default: _cropped).",
    )

    args = parser.parse_args()

    default_dir = Path(__file__).resolve().parent.parent / "outputs" / "images" / "flow_viz"

    if not args.inputs:
        default_euc = default_dir / "EuclideanFlow_ManimCE_v0.21.0.png"
        default_cyl = default_dir / "CylindricalFlow_ManimCE_v0.21.0.png"

        # Fallback check if unversioned exists
        if not default_euc.exists():
            candidates = list(default_dir.glob("EuclideanFlow*.png"))
            if candidates:
                default_euc = candidates[0]
        if not default_cyl.exists():
            candidates = list(default_dir.glob("CylindricalFlow*.png"))
            if candidates:
                default_cyl = candidates[0]

        input_paths = [default_euc, default_cyl]
        output_paths = [
            (args.output_dir or default_dir) / "EuclideanFlow_cropped.png",
            (args.output_dir or default_dir) / "CylindricalFlow_cropped.png",
        ]
    else:
        input_paths = []
        for p in args.inputs:
            if p.is_dir():
                input_paths.extend(sorted(p.glob("*.png")))
            elif p.is_file():
                input_paths.append(p)
            else:
                print(f"Warning: {p} does not exist. Skipping.")

        output_paths = []
        for p in input_paths:
            out_dir = args.output_dir or p.parent
            # Clean Manim CE version suffix if present for cleaner names
            stem = p.stem.split("_ManimCE_")[0]
            out_name = f"{stem}{args.suffix}.png"
            output_paths.append(out_dir / out_name)

    if not input_paths:
        print("No valid input images found.")
        return

    harmonize = not args.no_harmonize
    print(
        f"Processing {len(input_paths)} figure(s) [padding={args.padding}px, harmonize={harmonize}]..."
    )

    results = process_figures(
        input_paths=input_paths,
        output_paths=output_paths,
        padding=args.padding,
        harmonize=harmonize,
    )

    print("\nCropping Results:")
    print("-" * 80)
    for in_p, in_sz, out_p, out_sz in results:
        print(f"File:   {in_p.name} -> {out_p.name}")
        print(f"Before: {in_sz[0]}x{in_sz[1]} px")
        print(f"After:  {out_sz[0]}x{out_sz[1]} px (Aspect Ratio: {out_sz[0] / out_sz[1]:.3f})")
        print(f"Path:   {out_p.resolve()}")
        print("-" * 80)


if __name__ == "__main__":
    main()
