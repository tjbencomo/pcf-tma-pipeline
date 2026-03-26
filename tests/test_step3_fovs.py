"""Tests for Step 3: FOV creation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from pcf_pipeline.config import (
    CoresConfig,
    FovsConfig,
    InputsConfig,
    MasksConfig,
    PipelineConfig,
    PipelineRunConfig,
)
from pcf_pipeline.steps.step2_cores import run_step2
from pcf_pipeline.steps.step3_fovs import (
    _assign_cells_to_fovs,
    _compute_cell_geometry,
    _compute_fov_grid,
    run_step3,
)


TEST_DATA_DIR = Path(__file__).parent.parent / "test-data"


@pytest.fixture
def pipeline_config(tmp_path):
    return PipelineConfig(
        dataset_id="test_dataset",
        inputs=InputsConfig(
            image=TEST_DATA_DIR / "test_image.tiff",
            cores_geojson=TEST_DATA_DIR / "test_cores.geojson",
            cells_geojson=TEST_DATA_DIR / "test_cells.geojson",
            measurements_csv=TEST_DATA_DIR / "test_measurements.csv",
            channel_names_json=TEST_DATA_DIR / "channel_names.json",
        ),
        output_dir=tmp_path,
        masks=MasksConfig(),
        cores=CoresConfig(padding=50, square=True, skip_missing=True),
        fovs=FovsConfig(tile_size=512, overlap=64),  # smaller tiles for faster tests
        pipeline=PipelineRunConfig(skip_completed=False),
    )


@pytest.fixture
def config_after_step2(pipeline_config):
    """Run step 2 first so step 3 has inputs."""
    run_step2(pipeline_config)
    return pipeline_config


# ---------------------------------------------------------------------------
# Unit tests for grid helpers
# ---------------------------------------------------------------------------


def test_fov_grid_covers_all_pixels():
    """Every pixel in the image falls within at least one FOV."""
    H, W = 1200, 900
    tile = 512
    stride = 384
    fovs = _compute_fov_grid(H, W, tile, tile, stride, stride)

    covered = np.zeros((H, W), dtype=bool)
    for f in fovs:
        covered[f["y0"] : f["y1"], f["x0"] : f["x1"]] = True

    assert covered.all(), "Some pixels are not covered by any FOV"


def test_fov_grid_small_image():
    """Image smaller than tile — produces exactly one FOV starting at (0,0)."""
    fovs = _compute_fov_grid(300, 400, 512, 512, 384, 384)
    assert len(fovs) == 1
    assert fovs[0]["y0"] == 0 and fovs[0]["x0"] == 0


def test_assign_cells_fully_contained():
    """Cells whose bbox fits inside a FOV are assigned to it."""
    cell_mask = np.zeros((512, 512), dtype=np.uint32)
    cell_mask[100:120, 100:120] = 1  # cell 1
    cell_mask[300:330, 300:340] = 2  # cell 2
    cell_props = _compute_cell_geometry(cell_mask)

    fovs = _compute_fov_grid(512, 512, 512, 512, 512, 512)
    df = _assign_cells_to_fovs(cell_props, fovs)

    assert df["fully_contained"].all(), "All cells should be contained in the single FOV"
    assert (df["assigned_fov_id"] == "fov_000").all()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_run_step3_creates_outputs(config_after_step2):
    run_step3(config_after_step2)

    fovs_dir = config_after_step2.fovs_dir
    core_dirs = list(fovs_dir.iterdir())
    assert len(core_dirs) >= 1, "No core directories created in fovs/"

    for core_dir in core_dirs:
        assert (core_dir / "cell_to_fov.csv").exists(), \
            f"cell_to_fov.csv missing in {core_dir}"
        assert (core_dir / "segmentation").exists(), \
            f"segmentation/ dir missing in {core_dir}"
        assert (core_dir / "images").exists(), \
            f"images/ dir missing in {core_dir}"


def test_run_step3_cell_to_fov_columns(config_after_step2):
    """cell_to_fov.csv has the expected columns."""
    run_step3(config_after_step2)

    for core_dir in config_after_step2.fovs_dir.iterdir():
        df = pd.read_csv(core_dir / "cell_to_fov.csv")
        for col in ("label", "assigned_fov_id", "fully_contained"):
            assert col in df.columns, f"Missing column '{col}' in {core_dir.name}"


def test_run_step3_segmentation_tiles_exist(config_after_step2):
    """Each FOV referenced in cell_to_fov.csv has whole_cell and nuclear tiffs."""
    run_step3(config_after_step2)

    for core_dir in config_after_step2.fovs_dir.iterdir():
        df = pd.read_csv(core_dir / "cell_to_fov.csv")
        fov_ids = df["assigned_fov_id"].dropna().unique()
        seg_dir = core_dir / "segmentation"
        for fov_id in fov_ids:
            assert (seg_dir / f"{fov_id}_whole_cell.tiff").exists()
            assert (seg_dir / f"{fov_id}_nuclear.tiff").exists()


def test_run_step3_image_tiles_shape(config_after_step2):
    """Channel tiles are 2-D and have the expected tile dimensions."""
    run_step3(config_after_step2)

    tile_size = config_after_step2.fovs.tile_size
    for core_dir in config_after_step2.fovs_dir.iterdir():
        images_dir = core_dir / "images"
        fov_dirs = list(images_dir.iterdir())
        assert len(fov_dirs) >= 1
        # Check first FOV's first channel tile
        first_fov = sorted(fov_dirs)[0]
        tif_files = list(first_fov.glob("*.tif"))
        assert len(tif_files) >= 1
        arr = tifffile.imread(tif_files[0])
        assert arr.ndim == 2
        assert arr.shape[0] <= tile_size
        assert arr.shape[1] <= tile_size


def test_run_step3_skip_completed(config_after_step2):
    """Re-running with skip_completed=True does not overwrite existing outputs."""
    run_step3(config_after_step2)

    mtimes = {
        p: p.stat().st_mtime
        for p in config_after_step2.fovs_dir.rglob("cell_to_fov.csv")
    }

    config_after_step2.pipeline.skip_completed = True
    run_step3(config_after_step2)

    for p, mtime in mtimes.items():
        assert p.stat().st_mtime == mtime, f"{p} was overwritten despite skip_completed"


def test_run_step3_missing_metadata_raises(pipeline_config):
    """step3 raises FileNotFoundError when step 2 hasn't been run."""
    with pytest.raises(FileNotFoundError):
        run_step3(pipeline_config)
