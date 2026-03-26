"""Tests for Step 1: segmentation mask creation."""

from pathlib import Path
from unittest.mock import patch

import pytest

from pcf_pipeline.config import (
    InputsConfig,
    MasksConfig,
    PipelineConfig,
    PipelineRunConfig,
)
from pcf_pipeline.steps.step1_masks import run_step1

TEST_DATA_DIR = Path(__file__).parent.parent / "test-data"


@pytest.fixture
def base_config(tmp_path):
    return PipelineConfig(
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


def test_run_step1_pregenerated_masks(base_config, tmp_path):
    """Pre-generated masks are returned directly without calling geojson2masks."""
    cell_mask = tmp_path / "cell.tif"
    nuc_mask = tmp_path / "nuc.tif"
    id_map = tmp_path / "id_mapping.csv"
    cell_mask.write_bytes(b"")
    nuc_mask.write_bytes(b"")
    id_map.write_bytes(b"")

    base_config.masks = MasksConfig(
        cell_mask_path=cell_mask,
        nucleus_mask_path=nuc_mask,
        id_mapping_path=id_map,
    )

    with patch("pcf_pipeline.steps.step1_masks._run_geojson2masks") as mock_g2m:
        result = run_step1(base_config)

    mock_g2m.assert_not_called()
    assert result == (cell_mask, nuc_mask, id_map)


def test_run_step1_pregenerated_missing_raises(base_config, tmp_path):
    """Missing pre-generated mask raises FileNotFoundError."""
    base_config.masks = MasksConfig(
        cell_mask_path=tmp_path / "nonexistent_cell.tif",
        nucleus_mask_path=tmp_path / "nonexistent_nuc.tif",
    )
    with pytest.raises(FileNotFoundError):
        run_step1(base_config)


def test_run_step1_calls_geojson2masks_with_dimensions(base_config, tmp_path):
    """Auto-generation passes correct image dimensions to geojson2masks."""
    base_config.masks = MasksConfig()  # no pre-generated paths

    fake_cell = tmp_path / "fake_cell.tif"
    fake_nuc = tmp_path / "fake_nuc.tif"
    fake_map = tmp_path / "fake_id_mapping.csv"

    with patch("pcf_pipeline.steps.step1_masks._run_geojson2masks") as mock_g2m:
        mock_g2m.return_value = (fake_cell, fake_nuc, fake_map)
        run_step1(base_config)

    mock_g2m.assert_called_once()
    _, kwargs = mock_g2m.call_args
    # Dimensions must be non-zero and match the test image (3167 x 12999)
    assert kwargs["width"] == 12999
    assert kwargs["height"] == 3167


def test_run_step1_geojson2masks_output_paths(base_config, tmp_path):
    """Auto-generation returns the paths produced by geojson2masks."""
    base_config.masks = MasksConfig()

    fake_cell = tmp_path / "out_cell.tif"
    fake_nuc = tmp_path / "out_nuc.tif"
    fake_map = tmp_path / "out_id_mapping.csv"

    with patch("pcf_pipeline.steps.step1_masks._run_geojson2masks") as mock_g2m:
        mock_g2m.return_value = (fake_cell, fake_nuc, fake_map)
        cell, nuc, mapping = run_step1(base_config)

    assert cell == fake_cell
    assert nuc == fake_nuc
    assert mapping == fake_map
