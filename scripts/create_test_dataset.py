#!/usr/bin/env python3
"""Create a small test dataset by extracting a subset of TMA cores from the full development data.

Designed to run on a MacBook Air (16GB RAM) - reads large files in a streaming
fashion so the full dataset never needs to be in memory at once.

Usage:
    python scripts/create_test_dataset.py
    python scripts/create_test_dataset.py --cores A-4,A-5 --padding 50
"""

import argparse
import csv
import gc
import decimal
import json

# Ensure print output is unbuffered for real-time progress
import functools
print = functools.partial(print, flush=True)
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import ijson
import numpy as np
import tifffile


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that converts Decimal objects to float."""

    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return float(o)
        return super().default(o)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract a subset of TMA cores into a small test dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("development-data"),
        help="Directory containing the full development data files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test-data"),
        help="Directory where the test dataset will be written.",
    )
    parser.add_argument(
        "--cores",
        type=str,
        default="A-4,A-5,A-6,A-7",
        help="Comma-separated list of core names to include (e.g. A-4,A-5,A-6,A-7).",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=100,
        help="Padding (pixels) around the combined core bounding box.",
    )
    parser.add_argument(
        "--skip-image",
        action="store_true",
        help="Skip qptiff cropping if test_image.tiff already exists.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Cores geojson helpers
# ---------------------------------------------------------------------------
def load_cores_geojson(path: Path) -> dict:
    print(f"[INFO] Loading cores geojson from: {path}")
    with open(path) as f:
        data = json.load(f)
    print(f"[INFO] Found {len(data['features'])} cores")
    return data


def compute_combined_bbox(
    cores_data: dict,
    selected_cores: list[str],
    padding: int,
    img_height: int,
    img_width: int,
) -> tuple[int, int, int, int]:
    """Compute the union bounding box of selected cores with padding.

    Returns (x_min, y_min, x_max, y_max) clamped to image bounds.
    """
    all_xs = []
    all_ys = []
    found = set()

    for feat in cores_data["features"]:
        name = feat["properties"]["name"]
        if name not in selected_cores:
            continue
        found.add(name)
        if feat["properties"].get("isMissing", False):
            print(f"[WARNING] Core {name} is marked as missing")
        coords = feat["geometry"]["coordinates"][0]
        for x, y in coords:
            all_xs.append(x)
            all_ys.append(y)

    missing = set(selected_cores) - found
    if missing:
        print(f"[ERROR] Cores not found in geojson: {missing}")
        sys.exit(1)

    x_min = max(0, int(min(all_xs)) - padding)
    y_min = max(0, int(min(all_ys)) - padding)
    x_max = min(img_width, int(max(all_xs) + 0.5) + padding)
    y_max = min(img_height, int(max(all_ys) + 0.5) + padding)

    print(
        f"[INFO] Combined bounding box: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}] "
        f"({x_max - x_min} x {y_max - y_min} pixels)"
    )
    return x_min, y_min, x_max, y_max


# ---------------------------------------------------------------------------
# QPTIFF channel info
# ---------------------------------------------------------------------------
def get_channel_info(qptiff_path: Path) -> tuple[list[str], list[int], int, int]:
    """Read channel names and full-resolution page indices from a qptiff.

    Returns (channel_names, page_indices, img_height, img_width).
    """
    print(f"[INFO] Reading channel info from: {qptiff_path}")
    channel_names = []
    page_indices = []
    img_height = 0
    img_width = 0

    with tifffile.TiffFile(qptiff_path) as tif:
        for idx, page in enumerate(tif.pages):
            if not page.description:
                continue
            try:
                root = ET.fromstring(page.description)
            except ET.ParseError:
                continue
            image_type = root.find("ImageType")
            if image_type is None or image_type.text != "FullResolution":
                continue
            biomarker = root.find("Biomarker")
            if biomarker is not None and biomarker.text:
                channel_names.append(biomarker.text)
                page_indices.append(idx)
                if img_height == 0:
                    img_height, img_width = page.shape[:2]

    print(
        f"[INFO] Found {len(channel_names)} channels, "
        f"image size: {img_height} x {img_width}"
    )
    return channel_names, page_indices, img_height, img_width


# ---------------------------------------------------------------------------
# Crop QPTIFF (page by page)
# ---------------------------------------------------------------------------
def crop_qptiff(
    input_path: Path,
    output_path: Path,
    channel_names_path: Path,
    bbox: tuple[int, int, int, int],
    channel_names: list[str],
    page_indices: list[int],
):
    """Crop each full-resolution channel to the bounding box and write a new TIFF.

    Reads one page at a time to limit memory usage.
    """
    x_min, y_min, x_max, y_max = bbox
    n_channels = len(page_indices)

    print(f"[INFO] Cropping {n_channels} channels from qptiff...")
    print(f"[INFO] Output: {output_path}")

    crop_h = y_max - y_min
    crop_w = x_max - x_min

    with tifffile.TiffFile(input_path) as tif:
        with tifffile.TiffWriter(output_path, bigtiff=True) as writer:
            for i, page_idx in enumerate(page_indices):
                t0 = time.time()
                page = tif.pages[page_idx]
                full_channel = page.asarray()
                cropped = full_channel[y_min:y_max, x_min:x_max].copy()
                del full_channel
                gc.collect()
                # Write as part of a shaped (C, H, W) series so imread
                # returns a 3D array
                metadata = {"axes": "CYX"} if i == 0 else {}
                writer.write(
                    cropped,
                    contiguous=True,
                    metadata=metadata,
                )
                elapsed = time.time() - t0
                print(
                    f"[INFO]   Channel {i + 1}/{n_channels} "
                    f"({channel_names[i]}): {elapsed:.1f}s"
                )
                del cropped

    # Save channel names
    safe_names = [
        name.replace(" ", "_").replace("/", "-") for name in channel_names
    ]
    names_data = {
        "channel_names": channel_names,
        "channel_names_safe": safe_names,
    }
    with open(channel_names_path, "w") as f:
        json.dump(names_data, f, indent=2)
    print(f"[INFO] Saved channel names to: {channel_names_path}")


# ---------------------------------------------------------------------------
# Filter cores geojson
# ---------------------------------------------------------------------------
def offset_coords(coords_list: list, x_offset: int, y_offset: int) -> list:
    """Subtract (x_offset, y_offset) from coordinate structures.

    Handles both flat point lists [[x,y], ...] and nested ring lists
    [[[x,y], ...], ...] by checking whether the first element is a point
    (number) or another list (ring).
    """
    if not coords_list:
        return coords_list
    # Check if this is a list of points or a list of rings
    first = coords_list[0]
    if isinstance(first, list) and first and isinstance(first[0], list):
        # Nested rings: recurse
        return [offset_coords(ring, x_offset, y_offset) for ring in coords_list]
    # Flat list of points
    return [[pt[0] - x_offset, pt[1] - y_offset] + pt[2:] for pt in coords_list]


def filter_cores_geojson(
    cores_data: dict,
    selected_cores: set[str],
    crop_origin: tuple[int, int],
    output_path: Path,
):
    """Filter cores geojson to selected cores and offset coordinates."""
    x_off, y_off = crop_origin
    filtered_features = []

    for feat in cores_data["features"]:
        if feat["properties"]["name"] not in selected_cores:
            continue
        feat = json.loads(json.dumps(feat))  # deep copy
        for ring_idx in range(len(feat["geometry"]["coordinates"])):
            feat["geometry"]["coordinates"][ring_idx] = offset_coords(
                feat["geometry"]["coordinates"][ring_idx], x_off, y_off
            )
        filtered_features.append(feat)

    output = {"type": "FeatureCollection", "features": filtered_features}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(
        f"[INFO] Wrote {len(filtered_features)} cores to: {output_path}"
    )


# ---------------------------------------------------------------------------
# Filter cell segmentation geojson (streaming)
# ---------------------------------------------------------------------------
def filter_cell_geojson(
    input_path: Path,
    output_path: Path,
    bbox: tuple[int, int, int, int],
    crop_origin: tuple[int, int],
):
    """Stream through cell geojson, keeping cells whose centroid is in the bbox.

    Writes matching features incrementally to avoid memory buildup.
    """
    x_min, y_min, x_max, y_max = bbox
    x_off, y_off = crop_origin
    total = 0
    kept = 0

    print(f"[INFO] Streaming cell geojson from: {input_path}")
    print(f"[INFO] Output: {output_path}")

    with open(input_path, "rb") as fin, open(output_path, "w") as fout:
        fout.write('{"type": "FeatureCollection", "features": [\n')
        first = True

        for feature in ijson.items(fin, "features.item"):
            total += 1
            if total % 100000 == 0:
                print(
                    f"[INFO]   Processed {total:,} cells, kept {kept:,}..."
                )

            # Compute centroid from cell geometry (first ring of first polygon)
            coords = feature["geometry"]["coordinates"][0]
            # Handle extra nesting (e.g. MultiPolygon or nested rings)
            while coords and isinstance(coords[0], list) and isinstance(coords[0][0], list):
                coords = coords[0]
            cx = sum(float(pt[0]) for pt in coords) / len(coords)
            cy = sum(float(pt[1]) for pt in coords) / len(coords)

            if not (x_min <= cx <= x_max and y_min <= cy <= y_max):
                continue

            # Offset geometry coordinates
            feature["geometry"]["coordinates"] = offset_coords(
                feature["geometry"]["coordinates"], x_off, y_off
            )

            # Offset nucleus geometry if present
            if "nucleusGeometry" in feature:
                feature["nucleusGeometry"]["coordinates"] = offset_coords(
                    feature["nucleusGeometry"]["coordinates"], x_off, y_off
                )

            if not first:
                fout.write(",\n")
            json.dump(feature, fout, cls=DecimalEncoder)
            first = False
            kept += 1

        fout.write("\n]}\n")

    print(f"[INFO] Cell geojson: kept {kept:,} / {total:,} cells")


# ---------------------------------------------------------------------------
# Filter CSV (streaming)
# ---------------------------------------------------------------------------
def filter_csv(
    input_path: Path,
    output_path: Path,
    selected_cores: set[str],
):
    """Stream through CSV, keeping rows where TMA Core is in selected_cores."""
    total = 0
    kept = 0

    print(f"[INFO] Streaming CSV from: {input_path}")
    print(f"[INFO] Output: {output_path}")

    with open(input_path, newline="") as fin, open(output_path, "w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        header = next(reader)
        writer.writerow(header)

        # Find TMA Core column index
        try:
            core_col = header.index("TMA Core")
        except ValueError:
            print("[ERROR] 'TMA Core' column not found in CSV header")
            sys.exit(1)

        for row in reader:
            total += 1
            if total % 100000 == 0:
                print(f"[INFO]   Processed {total:,} rows, kept {kept:,}...")

            if row[core_col] in selected_cores:
                writer.writerow(row)
                kept += 1

    print(f"[INFO] CSV: kept {kept:,} / {total:,} rows")


# ---------------------------------------------------------------------------
# Discover input files
# ---------------------------------------------------------------------------
def find_input_files(input_dir: Path) -> dict[str, Path]:
    """Find the expected input files in the input directory."""
    files = {}
    for f in input_dir.iterdir():
        name = f.name
        if name.endswith(".er.qptiff"):
            files["qptiff"] = f
        elif name.endswith("_cores.geojson"):
            files["cores_geojson"] = f
        elif name.endswith(".Instanseg.geojson"):
            files["cell_geojson"] = f
        elif name.endswith(".csv"):
            files["csv"] = f

    required = ["qptiff", "cores_geojson", "cell_geojson", "csv"]
    missing = [k for k in required if k not in files]
    if missing:
        print(f"[ERROR] Missing input files: {missing}")
        print(f"[ERROR] Found: {list(files.keys())}")
        sys.exit(1)

    for key, path in files.items():
        print(f"[INFO] Input {key}: {path.name}")
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    selected_cores = [c.strip() for c in args.cores.split(",")]
    selected_cores_set = set(selected_cores)

    print(f"[INFO] Selected cores: {selected_cores}")
    print(f"[INFO] Input directory: {args.input_dir}")
    print(f"[INFO] Output directory: {args.output_dir}")

    # Discover input files
    input_files = find_input_files(args.input_dir)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get channel info from qptiff (metadata only)
    channel_names, page_indices, img_h, img_w = get_channel_info(
        input_files["qptiff"]
    )

    # Step 2: Load cores geojson and compute bbox
    cores_data = load_cores_geojson(input_files["cores_geojson"])
    bbox = compute_combined_bbox(
        cores_data, selected_cores, args.padding, img_h, img_w
    )
    x_min, y_min, x_max, y_max = bbox
    crop_origin = (x_min, y_min)

    # Step 3: Crop qptiff
    image_output = args.output_dir / "test_image.tiff"
    if args.skip_image and image_output.exists():
        print(f"[INFO] Skipping qptiff crop — {image_output} already exists")
    else:
        t0 = time.time()
        crop_qptiff(
            input_path=input_files["qptiff"],
            output_path=image_output,
            channel_names_path=args.output_dir / "channel_names.json",
            bbox=bbox,
            channel_names=channel_names,
            page_indices=page_indices,
        )
        print(f"[INFO] QPTIFF crop completed in {time.time() - t0:.1f}s")

    # Step 4: Filter cores geojson
    filter_cores_geojson(
        cores_data=cores_data,
        selected_cores=selected_cores_set,
        crop_origin=crop_origin,
        output_path=args.output_dir / "test_cores.geojson",
    )

    # Step 5: Filter cell geojson (streaming)
    t0 = time.time()
    filter_cell_geojson(
        input_path=input_files["cell_geojson"],
        output_path=args.output_dir / "test_cells.geojson",
        bbox=bbox,
        crop_origin=crop_origin,
    )
    print(f"[INFO] Cell geojson filtering completed in {time.time() - t0:.1f}s")

    # Step 6: Filter CSV (streaming)
    t0 = time.time()
    filter_csv(
        input_path=input_files["csv"],
        output_path=args.output_dir / "test_measurements.csv",
        selected_cores=selected_cores_set,
    )
    print(f"[INFO] CSV filtering completed in {time.time() - t0:.1f}s")

    # Summary
    print("\n[INFO] === Summary ===")
    for f in sorted(args.output_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        if size_mb > 1024:
            print(f"[INFO]   {f.name}: {size_mb / 1024:.2f} GB")
        else:
            print(f"[INFO]   {f.name}: {size_mb:.1f} MB")
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
