"""Tests for config loading."""

from pathlib import Path

import pytest
import yaml

from pcf_pipeline.config import load_config


@pytest.fixture
def minimal_config(tmp_path, test_data_dir):
    """Create a minimal valid config file pointing to test data."""
    config = {
        "dataset_id": "test_dataset",
        "inputs": {
            "image": str(test_data_dir / "test_image.tiff"),
            "cores_geojson": str(test_data_dir / "test_cores.geojson"),
            "cells_geojson": str(test_data_dir / "test_cells.geojson"),
            "measurements_csv": str(test_data_dir / "test_measurements.csv"),
        },
        "output_dir": str(tmp_path / "output"),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


def test_load_config_minimal(minimal_config):
    """Loading a minimal config should produce valid PipelineConfig with defaults."""
    cfg = load_config(minimal_config)
    assert cfg.dataset_id == "test_dataset"
    assert cfg.cores.padding == 100
    assert cfg.fovs.tile_size == 1024
    assert cfg.pipeline.steps == [1, 2, 3, 4, 5]


def test_load_config_missing_field(tmp_path):
    """Missing required field should cause sys.exit."""
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.dump({"dataset_id": "x"}))
    with pytest.raises(SystemExit):
        load_config(config_path)
