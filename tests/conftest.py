"""Shared test fixtures pointing to test-data/."""

from pathlib import Path

import pytest


TEST_DATA_DIR = Path(__file__).parent.parent / "test-data"


@pytest.fixture
def test_data_dir():
    """Path to the test-data/ directory."""
    return TEST_DATA_DIR


@pytest.fixture
def test_image_path(test_data_dir):
    return test_data_dir / "test_image.tiff"


@pytest.fixture
def test_cores_geojson(test_data_dir):
    return test_data_dir / "test_cores.geojson"


@pytest.fixture
def test_cells_geojson(test_data_dir):
    return test_data_dir / "test_cells.geojson"


@pytest.fixture
def test_measurements_csv(test_data_dir):
    return test_data_dir / "test_measurements.csv"


@pytest.fixture
def channel_names_json(test_data_dir):
    return test_data_dir / "channel_names.json"
