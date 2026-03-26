"""Step 1: Create cell and nucleus segmentation masks via geojson2masks."""

import subprocess
import sys
from pathlib import Path

from pcf_pipeline import log
from pcf_pipeline.config import PipelineConfig
from pcf_pipeline.utils import get_channel_info


def _run_geojson2masks(
    cells_geojson: Path,
    output_dir: Path,
    width: int,
    height: int,
) -> tuple[Path, Path, Path]:
    """Call geojson2masks as a subprocess to generate cell/nucleus mask TIFFs.

    Returns (cell_mask_path, nucleus_mask_path, id_mapping_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = cells_geojson.stem  # e.g. "SomeDataset.Instanseg"

    cmd = [
        "geojson2masks",
        str(cells_geojson),
        "--width", str(width),
        "--height", str(height),
        "--output-dir", str(output_dir),
    ]
    log.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"geojson2masks failed (exit {result.returncode}):\n{result.stderr}")
        sys.exit(result.returncode)

    if result.stdout:
        for line in result.stdout.strip().splitlines():
            log.info(f"[geojson2masks] {line}")

    cell_mask = output_dir / f"{stem}_cell_mask.tif"
    nuc_mask = output_dir / f"{stem}_nucleus_mask.tif"
    id_mapping = output_dir / f"{stem}_id_mapping.csv"

    for name, path in [("cell mask", cell_mask), ("nucleus mask", nuc_mask), ("id mapping", id_mapping)]:
        if not path.exists():
            log.error(f"geojson2masks did not produce expected {name}: {path}")
            sys.exit(1)

    return cell_mask, nuc_mask, id_mapping


def run_step1(config: PipelineConfig) -> tuple[Path, Path, Path | None]:
    """Create segmentation masks from the cell geojson.

    If pre-generated mask paths are provided in config.masks, validates they
    exist and returns them. Otherwise calls geojson2masks as a subprocess,
    reading image dimensions automatically from the input image.

    Returns (cell_mask_path, nucleus_mask_path, id_mapping_path).
    """
    t0 = log.log_step_start("Step 1: Create segmentation masks")

    # Use pre-generated masks if provided
    if config.masks.cell_mask_path and config.masks.nucleus_mask_path:
        cell_mask = config.masks.cell_mask_path
        nuc_mask = config.masks.nucleus_mask_path
        id_mapping = config.masks.id_mapping_path

        for name, path in [("cell_mask", cell_mask), ("nucleus_mask", nuc_mask)]:
            if not path.exists():
                log.error(f"Pre-generated mask not found: masks.{name}_path = {path}")
                raise FileNotFoundError(path)

        log.info(f"Using pre-generated masks: {cell_mask.name}, {nuc_mask.name}")
        log.log_step_end("Step 1: Create segmentation masks", t0)
        return cell_mask, nuc_mask, id_mapping

    # Auto-generate masks via geojson2masks
    log.info("No pre-generated masks provided — running geojson2masks")

    # Get image dimensions from the source image
    _, _, img_height, img_width = get_channel_info(config.inputs.image)
    if img_height == 0 or img_width == 0:
        # Plain TIFF fallback: read shape directly
        import tifffile
        with tifffile.TiffFile(config.inputs.image) as tif:
            shape = tif.series[0].shape  # (C, H, W) or (H, W)
        img_height, img_width = shape[-2], shape[-1]

    log.info(f"Image dimensions: {img_height} x {img_width} px")

    masks_dir = config.masks_dir
    cell_mask, nuc_mask, id_mapping = _run_geojson2masks(
        cells_geojson=config.inputs.cells_geojson,
        output_dir=masks_dir,
        width=img_width,
        height=img_height,
    )

    log.info(f"Cell mask:    {cell_mask}")
    log.info(f"Nucleus mask: {nuc_mask}")
    log.info(f"ID mapping:   {id_mapping}")

    log.log_step_end("Step 1: Create segmentation masks", t0)
    return cell_mask, nuc_mask, id_mapping
