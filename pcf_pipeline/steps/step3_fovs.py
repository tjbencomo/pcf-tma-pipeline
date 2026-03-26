"""Step 3: Create FOV tiles for each core."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops

from pcf_pipeline import log
from pcf_pipeline.config import PipelineConfig


# ---------------------------------------------------------------------------
# FOV grid helpers
# ---------------------------------------------------------------------------


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Tile start positions covering the full extent of *length*."""
    if length <= tile_size:
        return [0]
    starts = [s for s in range(0, length, stride) if s + tile_size <= length]
    edge = length - tile_size
    if edge not in starts:
        starts.append(edge)
    return sorted(set(starts))


def _compute_fov_grid(
    height: int, width: int, tile_h: int, tile_w: int, stride_h: int, stride_w: int
) -> list[dict]:
    """Return list of FOV dicts with keys fov_id, y0, y1, x0, x1."""
    fovs = []
    idx = 0
    for y0 in _tile_starts(height, tile_h, stride_h):
        for x0 in _tile_starts(width, tile_w, stride_w):
            fovs.append(
                {
                    "fov_id": f"fov_{idx:03d}",
                    "fov_index": idx,
                    "y0": y0,
                    "y1": y0 + tile_h,
                    "x0": x0,
                    "x1": x0 + tile_w,
                }
            )
            idx += 1
    return fovs


def _compute_cell_geometry(cell_mask: np.ndarray) -> dict[int, dict]:
    """Return per-label dict with centroid, bbox from regionprops."""
    props = regionprops(cell_mask)
    result = {}
    for p in props:
        ymin, xmin, ymax, xmax = p.bbox
        cy, cx = p.centroid
        result[p.label] = {
            "label": p.label,
            "centroid_y": float(cy),
            "centroid_x": float(cx),
            "ymin": int(ymin),
            "xmin": int(xmin),
            "ymax": int(ymax),
            "xmax": int(xmax),
        }
    return result


def _assign_cells_to_fovs(cell_props: dict, fovs: list[dict]) -> pd.DataFrame:
    """Assign each cell to the FOV where its bbox is fully contained with maximum margin."""
    records = []
    for label, info in cell_props.items():
        ymin, xmin, ymax, xmax = info["ymin"], info["xmin"], info["ymax"], info["xmax"]
        cy, cx = info["centroid_y"], info["centroid_x"]

        best = None
        for f in fovs:
            y0, y1, x0, x1 = f["y0"], f["y1"], f["x0"], f["x1"]
            if ymin >= y0 and ymax <= y1 and xmin >= x0 and xmax <= x1:
                margin = min(cy - y0, (y1 - 1) - cy, cx - x0, (x1 - 1) - cx)
                if best is None or margin > best[0]:
                    best = (margin, f["fov_id"], f["fov_index"])

        if best is not None:
            assigned_fov_id = best[1]
            assigned_fov_index = best[2]
            fully_contained = True
        else:
            assigned_fov_id = None
            assigned_fov_index = None
            fully_contained = False

        records.append(
            {
                "label": label,
                "centroid_y": cy,
                "centroid_x": cx,
                "ymin": ymin,
                "xmin": xmin,
                "ymax": ymax,
                "xmax": xmax,
                "assigned_fov_id": assigned_fov_id,
                "assigned_fov_index": assigned_fov_index,
                "fully_contained": fully_contained,
            }
        )

    columns = [
        "label", "centroid_y", "centroid_x", "ymin", "xmin", "ymax", "xmax",
        "assigned_fov_id", "assigned_fov_index", "fully_contained",
    ]
    df = pd.DataFrame.from_records(records, columns=columns) if records else pd.DataFrame(columns=columns)
    n_contained = int(df["fully_contained"].sum()) if len(df) else 0
    log.info(f"Cells fully contained in a FOV: {n_contained} / {len(df)}")
    return df


def _fallback_fovs(
    unassigned: pd.DataFrame,
    tile_h: int,
    tile_w: int,
    img_h: int,
    img_w: int,
    start_idx: int,
) -> list[dict]:
    """Create one centred FOV per unassigned cell (deduped by position)."""
    seen: set[tuple[int, int]] = set()
    fovs = []
    skipped = []
    idx = start_idx

    for _, row in unassigned.iterrows():
        if (row["ymax"] - row["ymin"]) > tile_h or (row["xmax"] - row["xmin"]) > tile_w:
            skipped.append(int(row["label"]))
            continue
        cy = (row["ymin"] + row["ymax"]) / 2.0
        cx = (row["xmin"] + row["xmax"]) / 2.0
        y0 = int(round(cy - tile_h / 2.0))
        x0 = int(round(cx - tile_w / 2.0))
        y0 = max(0, min(y0, img_h - tile_h))
        x0 = max(0, min(x0, img_w - tile_w))
        if (y0, x0) in seen:
            continue
        seen.add((y0, x0))
        fovs.append(
            {
                "fov_id": f"fov_{idx:03d}",
                "fov_index": idx,
                "y0": y0,
                "y1": y0 + tile_h,
                "x0": x0,
                "x1": x0 + tile_w,
            }
        )
        idx += 1

    if skipped:
        log.warning(
            f"{len(skipped)} cell(s) have bounding boxes larger than the tile size "
            f"({tile_h}x{tile_w}) and cannot be fully contained. "
            f"Consider increasing fovs.tile_size. Labels: {skipped}"
        )
    return fovs


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _write_segmentation_fovs(
    cell_mask: np.ndarray,
    nuc_mask: np.ndarray,
    fovs: list[dict],
    seg_dir: Path,
) -> None:
    seg_dir.mkdir(parents=True, exist_ok=True)
    H, W = cell_mask.shape
    for f in fovs:
        y0, y1, x0, x1 = f["y0"], f["y1"], f["x0"], f["x1"]
        if y1 > H or x1 > W:
            continue
        fov_id = f["fov_id"]
        tifffile.imwrite(seg_dir / f"{fov_id}_whole_cell.tiff", cell_mask[y0:y1, x0:x1])
        tifffile.imwrite(seg_dir / f"{fov_id}_nuclear.tiff", nuc_mask[y0:y1, x0:x1])


def _write_image_fovs(
    img: np.ndarray,
    fovs: list[dict],
    images_dir: Path,
    channel_names: list[str],
    remove_offset: int | None,
) -> None:
    C, H, W = img.shape
    images_dir.mkdir(parents=True, exist_ok=True)
    for f in fovs:
        y0, y1, x0, x1 = f["y0"], f["y1"], f["x0"], f["x1"]
        if y1 > H or x1 > W:
            continue
        fov_dir = images_dir / f["fov_id"]
        fov_dir.mkdir(parents=True, exist_ok=True)
        for c in range(C):
            tile = img[c, y0:y1, x0:x1]
            if remove_offset is not None:
                tile = np.where(tile < remove_offset, 0, tile - np.uint16(remove_offset))
            tifffile.imwrite(fov_dir / f"{channel_names[c]}.tif", tile)


# ---------------------------------------------------------------------------
# Per-core FOV pipeline
# ---------------------------------------------------------------------------


def _process_core(
    core_name: str,
    img: np.ndarray,
    cell_mask: np.ndarray,
    nuc_mask: np.ndarray,
    core_output_dir: Path,
    channel_names: list[str],
    tile_h: int,
    tile_w: int,
    stride_h: int,
    stride_w: int,
    remove_offset: int | None,
) -> None:
    """Run the full FOV tiling pipeline for one core."""
    C, H, W = img.shape
    if H < tile_h or W < tile_w:
        log.warning(
            f"Core {core_name} ({H}x{W}) is smaller than tile size "
            f"({tile_h}x{tile_w}) — skipping"
        )
        return

    # Cell geometry
    log.info(f"Computing cell geometry for {core_name}")
    cell_props = _compute_cell_geometry(cell_mask)
    log.info(f"Found {len(cell_props)} cells in {core_name}")

    # FOV grid
    fovs = _compute_fov_grid(H, W, tile_h, tile_w, stride_h, stride_w)
    log.info(f"FOV grid for {core_name}: {len(fovs)} tiles ({tile_h}x{tile_w}, stride {stride_h}x{stride_w})")

    # Assign cells
    cell_to_fov = _assign_cells_to_fovs(cell_props, fovs)

    # Fallback FOVs for unassigned cells
    unassigned = cell_to_fov[~cell_to_fov["fully_contained"]]
    if len(unassigned) > 0:
        log.info(f"{len(unassigned)} unassigned cells — generating fallback FOVs")
        extra = _fallback_fovs(unassigned, tile_h, tile_w, H, W, len(fovs))
        fovs.extend(extra)
        log.info(f"Added {len(extra)} fallback FOVs (total: {len(fovs)})")
        cell_to_fov = _assign_cells_to_fovs(cell_props, fovs)
        still_unassigned = cell_to_fov[~cell_to_fov["fully_contained"]].shape[0]
        if still_unassigned:
            log.warning(
                f"{still_unassigned} cell(s) in {core_name} still unassigned — "
                "their bounding boxes exceed the tile size"
            )

    # Save cell-to-FOV map
    core_output_dir.mkdir(parents=True, exist_ok=True)
    cell_to_fov.to_csv(core_output_dir / "cell_to_fov.csv", index=False)

    # Write segmentation tiles
    _write_segmentation_fovs(
        cell_mask, nuc_mask, fovs, core_output_dir / "segmentation"
    )

    # Write image tiles
    _write_image_fovs(
        img, fovs, core_output_dir / "images", channel_names, remove_offset
    )


# ---------------------------------------------------------------------------
# Step entry point
# ---------------------------------------------------------------------------


def run_step3(config: PipelineConfig) -> None:
    """Generate FOV grid, tile images and segmentation masks, assign cells to FOVs."""
    t0 = log.log_step_start("Step 3: FOV creation")

    # Load core metadata written by step 2
    meta_path = config.sample_images_dir / "core_metadata.json"
    if not meta_path.exists():
        log.error(f"core_metadata.json not found: {meta_path} — run Step 2 first")
        raise FileNotFoundError(meta_path)

    with open(meta_path) as f:
        meta = json.load(f)

    channel_names = meta["channel_names"]
    cores = meta["cores"]
    log.info(f"Processing {len(cores)} cores with {len(channel_names)} channels")

    tile_h = tile_w = config.fovs.tile_size
    stride_h = stride_w = config.fovs.tile_size - config.fovs.overlap

    images_dir = config.sample_images_dir / "images"
    cell_masks_dir = config.sample_images_dir / "cell_masks"
    nuclei_masks_dir = config.sample_images_dir / "nuclei_masks"

    for idx, core_record in enumerate(cores):
        name = core_record["core_name"]
        log.log_progress(name, idx + 1, len(cores))

        core_output_dir = config.fovs_dir / name
        cell_to_fov_path = core_output_dir / "cell_to_fov.csv"

        if config.pipeline.skip_completed and cell_to_fov_path.exists():
            log.info(f"Skipping {name} (cell_to_fov.csv exists)")
            continue

        # Load core image
        img_path = images_dir / f"core_{name}.tiff"
        if not img_path.exists():
            log.error(f"Core image not found: {img_path} — run Step 2 first")
            raise FileNotFoundError(img_path)
        img = tifffile.imread(img_path)
        if img.ndim == 2:
            img = img[np.newaxis]  # treat single-channel as (1, H, W)

        H, W = img.shape[1], img.shape[2]

        # Load or synthesise masks
        cell_mask_path = cell_masks_dir / f"core_{name}_cell_mask.tiff"
        nuc_mask_path = nuclei_masks_dir / f"core_{name}_nuclei_mask.tiff"

        if cell_mask_path.exists():
            cell_mask = tifffile.imread(cell_mask_path)
        else:
            log.warning(f"Cell mask not found for {name} — using empty mask")
            cell_mask = np.zeros((H, W), dtype=np.uint32)

        if nuc_mask_path.exists():
            nuc_mask = tifffile.imread(nuc_mask_path)
        else:
            log.warning(f"Nuclei mask not found for {name} — using empty mask")
            nuc_mask = np.zeros((H, W), dtype=np.uint32)

        _process_core(
            core_name=name,
            img=img,
            cell_mask=cell_mask,
            nuc_mask=nuc_mask,
            core_output_dir=core_output_dir,
            channel_names=channel_names,
            tile_h=tile_h,
            tile_w=tile_w,
            stride_h=stride_h,
            stride_w=stride_w,
            remove_offset=config.fovs.remove_offset,
        )

    log.log_step_end("Step 3: FOV creation", t0)
