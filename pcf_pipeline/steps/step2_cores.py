"""Step 2: Split TMA image into per-core images using core geojson boundaries."""

import json
from pathlib import Path

import numpy as np
import tifffile

from pcf_pipeline import log
from pcf_pipeline.config import PipelineConfig
from pcf_pipeline.utils import get_channel_info, get_channel_names_safe


def _load_cores(geojson_path: Path, skip_missing: bool) -> list[dict]:
    """Load core features from GeoJSON, optionally skipping missing cores."""
    with open(geojson_path) as f:
        data = json.load(f)

    cores = []
    n_skipped = 0
    for feature in data["features"]:
        props = feature["properties"]
        if skip_missing and props.get("isMissing", False):
            n_skipped += 1
            continue
        coords = np.array(feature["geometry"]["coordinates"][0])
        cores.append(
            {
                "id": feature["id"],
                "name": props.get("name", str(feature["id"])),
                "coords": coords,
            }
        )

    if n_skipped:
        log.info(f"Skipped {n_skipped} missing cores")
    return cores


def _compute_bbox(
    coords: np.ndarray,
    padding: int,
    square: bool,
    img_height: int,
    img_width: int,
) -> tuple[int, int, int, int]:
    """Return (min_x, min_y, max_x, max_y) bounding box for a core polygon."""
    min_x = int(np.min(coords[:, 0]))
    max_x = int(np.max(coords[:, 0]))
    min_y = int(np.min(coords[:, 1]))
    max_y = int(np.max(coords[:, 1]))

    if square:
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        size = max(max_x - min_x, max_y - min_y) + 2 * padding
        half = size / 2
        min_x = int(center_x - half)
        max_x = int(center_x + half)
        min_y = int(center_y - half)
        max_y = int(center_y + half)
    else:
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding

    return (
        max(0, min_x),
        max(0, min_y),
        min(img_width, max_x),
        min(img_height, max_y),
    )


def _get_image_info(
    config: PipelineConfig,
) -> tuple[list[str], list[int] | None, int, int]:
    """Return (channel_names, page_indices_or_None, height, width).

    For qptiff files, page_indices points to full-resolution pages.
    For plain TIFFs, page_indices is None and the whole array is read directly.
    """
    names, indices, h, w = get_channel_info(config.inputs.image)
    if names:
        return names, indices, h, w

    # Plain TIFF fallback — infer shape from series
    with tifffile.TiffFile(config.inputs.image) as tif:
        shape = tif.series[0].shape  # (C, H, W)
    n_channels, h, w = shape

    if config.inputs.channel_names_json and config.inputs.channel_names_json.exists():
        with open(config.inputs.channel_names_json) as f:
            raw = json.load(f)
        # Support both a flat list and a dict with a "channel_names" key
        names = raw["channel_names"] if isinstance(raw, dict) else raw
        log.info(f"Loaded {len(names)} channel names from {config.inputs.channel_names_json.name}")
    else:
        names = [f"channel_{i:03d}" for i in range(n_channels)]
        log.warning("No channel names available, using defaults (channel_000, ...)")

    return names, None, h, w


def _iter_channels(
    image_path: Path, page_indices: list[int] | None, n_channels: int
):
    """Yield (ch_idx, 2-D array) one channel at a time."""
    with tifffile.TiffFile(image_path) as tif:
        if page_indices is not None:
            for ch_idx, page_idx in enumerate(page_indices):
                yield ch_idx, tif.pages[page_idx].asarray()
        else:
            arr = tif.series[0].asarray()
            for ch_idx in range(n_channels):
                yield ch_idx, arr[ch_idx]


def _crop_mask(mask_path: Path | None, min_x: int, min_y: int, max_x: int, max_y: int):
    """Load a 2-D mask TIFF and return the cropped region, or None if unavailable."""
    if mask_path is None:
        return None
    if not mask_path.exists():
        log.warning(f"Mask not found: {mask_path}")
        return None
    mask = tifffile.imread(mask_path)
    return mask[min_y:max_y, min_x:max_x]


def run_step2(config: PipelineConfig) -> None:
    """Split the TMA image into per-core cropped images with cell/nucleus masks."""
    t0 = log.log_step_start("Step 2: Core splitting")

    # Output directories
    images_dir = config.sample_images_dir / "images"
    cell_masks_dir = config.sample_images_dir / "cell_masks"
    nuclei_masks_dir = config.sample_images_dir / "nuclei_masks"
    for d in (images_dir, cell_masks_dir, nuclei_masks_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Load cores and channel info
    cores = _load_cores(config.inputs.cores_geojson, config.cores.skip_missing)
    log.info(f"Found {len(cores)} cores to process")

    channel_names, page_indices, img_height, img_width = _get_image_info(config)
    safe_names = get_channel_names_safe(channel_names)
    n_channels = len(channel_names)
    log.info(f"Image: {img_height} x {img_width} px, {n_channels} channels")

    # Compute bounding boxes
    bboxes = {}
    for core in cores:
        bboxes[core["name"]] = _compute_bbox(
            core["coords"],
            config.cores.padding,
            config.cores.square,
            img_height,
            img_width,
        )

    # Filter out already-completed cores
    cores_todo = [
        c
        for c in cores
        if not (
            config.pipeline.skip_completed
            and (images_dir / f"core_{c['name']}.tiff").exists()
        )
    ]
    n_skipped = len(cores) - len(cores_todo)
    if n_skipped:
        log.info(f"Skipping {n_skipped} already-completed cores")
    if not cores_todo:
        log.info("All cores already processed")
        log.log_step_end("Step 2: Core splitting", t0)
        return

    # Determine dtype from the source image
    with tifffile.TiffFile(config.inputs.image) as tif:
        dtype = (
            tif.pages[page_indices[0]].dtype
            if page_indices is not None
            else tif.series[0].dtype
        )

    # Allocate per-core output arrays
    core_arrays: dict[str, np.ndarray] = {}
    for core in cores_todo:
        name = core["name"]
        min_x, min_y, max_x, max_y = bboxes[name]
        core_arrays[name] = np.zeros(
            (n_channels, max_y - min_y, max_x - min_x), dtype=dtype
        )

    # Read image one channel at a time, crop all cores in each pass
    log.info(f"Reading image and cropping {len(cores_todo)} cores ({n_channels} channels)...")
    for ch_idx, channel_data in _iter_channels(config.inputs.image, page_indices, n_channels):
        log.log_progress(safe_names[ch_idx], ch_idx + 1, n_channels)
        for core in cores_todo:
            name = core["name"]
            min_x, min_y, max_x, max_y = bboxes[name]
            core_arrays[name][ch_idx] = channel_data[min_y:max_y, min_x:max_x]

    # Write core images, masks, and metadata
    metadata_records = []
    for idx, core in enumerate(cores_todo):
        name = core["name"]
        log.log_progress(name, idx + 1, len(cores_todo))
        min_x, min_y, max_x, max_y = bboxes[name]

        # Core image
        img_path = images_dir / f"core_{name}.tiff"
        tifffile.imwrite(
            img_path,
            core_arrays[name],
            photometric="minisblack",
            compression=config.cores.compression,
        )

        # Cell mask
        cell_mask = _crop_mask(config.masks.cell_mask_path, min_x, min_y, max_x, max_y)
        if cell_mask is not None:
            tifffile.imwrite(
                cell_masks_dir / f"core_{name}_cell_mask.tiff",
                cell_mask,
                compression=config.cores.compression,
            )

        # Nuclei mask
        nuc_mask = _crop_mask(config.masks.nucleus_mask_path, min_x, min_y, max_x, max_y)
        if nuc_mask is not None:
            tifffile.imwrite(
                nuclei_masks_dir / f"core_{name}_nuclei_mask.tiff",
                nuc_mask,
                compression=config.cores.compression,
            )

        metadata_records.append(
            {
                "core_name": name,
                "core_id": core["id"],
                "min_x": min_x,
                "min_y": min_y,
                "max_x": max_x,
                "max_y": max_y,
                "width": max_x - min_x,
                "height": max_y - min_y,
            }
        )

    # Save metadata JSON for downstream steps
    meta_path = config.sample_images_dir / "core_metadata.json"
    existing = []
    if meta_path.exists():
        with open(meta_path) as f:
            existing = json.load(f).get("cores", [])
    all_records = {r["core_name"]: r for r in existing}
    all_records.update({r["core_name"]: r for r in metadata_records})

    with open(meta_path, "w") as f:
        json.dump(
            {
                "channel_names": safe_names,
                "original_channel_names": channel_names,
                "cores": list(all_records.values()),
            },
            f,
            indent=2,
        )
    log.info(f"Saved core metadata: {meta_path}")

    log.log_step_end("Step 2: Core splitting", t0)
