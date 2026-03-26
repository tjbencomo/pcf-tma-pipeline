"""Tests for Step 2: TMA core splitting."""

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from pcf_pipeline.config import (
    CoresConfig,
    InputsConfig,
    MasksConfig,
    PipelineConfig,
    PipelineRunConfig,
)
from pcf_pipeline.steps.step2_cores import run_step2


TEST_DATA_DIR = Path(__file__).parent.parent / "test-data"


@pytest.fixture
def base_config(tmp_path):
    """Minimal PipelineConfig pointing at test data, outputting to tmp_path."""
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
        cores=CoresConfig(padding=50, square=True, skip_missing=True, compression="zlib"),
        pipeline=PipelineRunConfig(skip_completed=False),
    )


def test_run_step2_creates_expected_files(base_config):
    """Core images and metadata JSON are written for each non-missing core."""
    run_step2(base_config)

    images_dir = base_config.sample_images_dir / "images"
    meta_path = base_config.sample_images_dir / "core_metadata.json"

    assert meta_path.exists(), "core_metadata.json not written"

    with open(meta_path) as f:
        meta = json.load(f)

    core_names = {r["core_name"] for r in meta["cores"]}
    assert len(core_names) >= 1, "No cores in metadata"

    for name in core_names:
        img_path = images_dir / f"core_{name}.tiff"
        assert img_path.exists(), f"Missing core image: {img_path}"


def test_run_step2_image_shape(base_config):
    """Each saved core image has shape (C, H, W) matching channel count."""
    run_step2(base_config)

    with open(base_config.inputs.channel_names_json) as f:
        raw = json.load(f)
    channel_names = raw["channel_names"] if isinstance(raw, dict) else raw
    n_channels = len(channel_names)

    images_dir = base_config.sample_images_dir / "images"
    for img_path in images_dir.glob("core_*.tiff"):
        arr = tifffile.imread(img_path)
        assert arr.ndim == 3, f"Expected 3-D array, got shape {arr.shape}"
        assert arr.shape[0] == n_channels, (
            f"{img_path.name}: expected {n_channels} channels, got {arr.shape[0]}"
        )


def test_run_step2_metadata_bboxes(base_config):
    """Bounding boxes in metadata are consistent with saved image dimensions."""
    run_step2(base_config)

    with open(base_config.sample_images_dir / "core_metadata.json") as f:
        meta = json.load(f)

    for record in meta["cores"]:
        assert record["width"] == record["max_x"] - record["min_x"]
        assert record["height"] == record["max_y"] - record["min_y"]
        assert record["width"] > 0
        assert record["height"] > 0


def test_run_step2_skip_completed(base_config, tmp_path):
    """Second run with skip_completed=True skips cores whose output already exists."""
    run_step2(base_config)

    # Record modification times
    images_dir = base_config.sample_images_dir / "images"
    mtimes_before = {p: p.stat().st_mtime for p in images_dir.glob("core_*.tiff")}

    # Run again with skip_completed
    base_config.pipeline.skip_completed = True
    run_step2(base_config)

    mtimes_after = {p: p.stat().st_mtime for p in images_dir.glob("core_*.tiff")}
    assert mtimes_before == mtimes_after, "Files were re-written despite skip_completed=True"


def test_run_step2_no_channel_names_json(base_config):
    """Falls back to default channel names when channel_names_json is not provided."""
    base_config.inputs.channel_names_json = None
    run_step2(base_config)

    with open(base_config.sample_images_dir / "core_metadata.json") as f:
        meta = json.load(f)

    assert meta["channel_names"][0].startswith("channel_"), (
        "Expected fallback channel names like 'channel_000'"
    )
