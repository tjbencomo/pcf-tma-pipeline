"""Tests for Step 4: Nimbus inference.

These tests run actual Nimbus inference on the test dataset (GPU required).
They are marked slow and can be skipped with: pytest -m "not slow"
"""

from pathlib import Path

import pandas as pd
import pytest

from pcf_pipeline.config import (
    CoresConfig,
    FovsConfig,
    InputsConfig,
    MasksConfig,
    NimbusConfig,
    PipelineConfig,
    PipelineRunConfig,
)
from pcf_pipeline.steps.step2_cores import run_step2
from pcf_pipeline.steps.step3_fovs import run_step3
from pcf_pipeline.steps.step4_nimbus import run_step4

TEST_DATA_DIR = Path(__file__).parent.parent / "test-data"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def config_after_step3(tmp_path_factory):
    """Run steps 2 and 3 once for the whole module, then return config."""
    tmp_path = tmp_path_factory.mktemp("nimbus")
    config = PipelineConfig(
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
        fovs=FovsConfig(tile_size=1024, overlap=128),
        nimbus=NimbusConfig(batch_size=4, test_time_aug=False),
        pipeline=PipelineRunConfig(skip_completed=False),
    )
    run_step2(config)
    run_step3(config)
    return config


def test_run_step4_creates_parquets(config_after_step3):
    """A parquet file is written for each core."""
    run_step4(config_after_step3)

    core_dirs = sorted(config_after_step3.fovs_dir.iterdir())
    assert len(core_dirs) >= 1

    for core_dir in core_dirs:
        parquet = core_dir / "nimbus_output" / f"{core_dir.name}_nimbus_cell_predicted_probs.parquet"
        assert parquet.exists(), f"Missing parquet for core {core_dir.name}"


def test_run_step4_parquet_columns(config_after_step3):
    """Parquet files contain label, fov, and at least one probability column."""
    run_step4(config_after_step3)

    for core_dir in sorted(config_after_step3.fovs_dir.iterdir()):
        parquet = core_dir / "nimbus_output" / f"{core_dir.name}_nimbus_cell_predicted_probs.parquet"
        df = pd.read_parquet(parquet)
        assert "label" in df.columns
        assert "assigned_fov_id" in df.columns
        # At least one marker probability column should exist
        prob_cols = [c for c in df.columns if c not in (
            "label", "fov", "assigned_fov_id", "fully_contained",
            "centroid_y", "centroid_x", "ymin", "xmin", "ymax", "xmax",
            "assigned_fov_index",
        )]
        assert len(prob_cols) >= 1, f"No probability columns found in {core_dir.name}"


def test_run_step4_skip_completed(config_after_step3):
    """Re-running with skip_completed=True does not overwrite existing parquets."""
    run_step4(config_after_step3)

    mtimes = {
        p: p.stat().st_mtime
        for core_dir in config_after_step3.fovs_dir.iterdir()
        for p in (core_dir / "nimbus_output").glob("*.parquet")
    }

    config_after_step3.pipeline.skip_completed = True
    run_step4(config_after_step3)

    for p, mtime in mtimes.items():
        assert p.stat().st_mtime == mtime, f"{p.name} was overwritten despite skip_completed"


def test_run_step4_missing_fovs_raises(tmp_path):
    """step4 raises FileNotFoundError when step 3 hasn't been run."""
    config = PipelineConfig(
        dataset_id="test_dataset",
        inputs=InputsConfig(
            image=TEST_DATA_DIR / "test_image.tiff",
            cores_geojson=TEST_DATA_DIR / "test_cores.geojson",
            cells_geojson=TEST_DATA_DIR / "test_cells.geojson",
            measurements_csv=TEST_DATA_DIR / "test_measurements.csv",
        ),
        output_dir=tmp_path,
        pipeline=PipelineRunConfig(skip_completed=False),
    )
    with pytest.raises(FileNotFoundError):
        run_step4(config)
