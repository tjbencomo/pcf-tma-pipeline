"""Tests for Step 5: Merge into AnnData."""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from pcf_pipeline.config import (
    InputsConfig,
    MasksConfig,
    MergeConfig,
    PipelineConfig,
    PipelineRunConfig,
)
from pcf_pipeline.steps.step5_merge import (
    _build_anndata,
    _load_qupath,
    run_step5,
)

TEST_DATA_DIR = Path(__file__).parent.parent / "test-data"

# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

CHANNELS_ORIG = ["DAPI", "CD3e", "CD8", "Pan-Cytokeratin"]
CHANNELS_SAFE = ["DAPI", "CD3e", "CD8", "Pan-Cytokeratin"]
MARKERS_NIMBUS = ["CD3e", "CD8", "Pan-Cytokeratin"]  # DAPI excluded
CORE = "A-4"
N_CELLS = 10


def _make_id_mapping(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mask_id": range(1, n + 1),
            "geojson_id": [f"uuid-{i:04d}" for i in range(1, n + 1)],
        }
    )


def _make_qupath_csv(n: int, channels_orig: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = {
        "Object ID": [f"uuid-{i:04d}" for i in range(1, n + 1)],
        "TMA Core": [CORE] * n,
        "Centroid X µm": rng.uniform(0, 1000, n),
        "Centroid Y µm": rng.uniform(0, 1000, n),
        "Cell: Area µm^2": rng.uniform(50, 200, n),
        "Nucleus: Area µm^2": rng.uniform(20, 100, n),
        "Nucleus: Circularity": rng.uniform(0.5, 1.0, n),
    }
    for ch in channels_orig:
        rows[f"Cell: {ch}: Mean"] = rng.uniform(0, 5000, n).astype(np.float32)
    return pd.DataFrame(rows)


def _make_nimbus_parquet(n: int, markers: list[str], core: str) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = {
        "label": list(range(1, n + 1)),
        "assigned_fov_id": ["fov_000"] * n,
        "fov": ["fov_000"] * n,
        "fully_contained": [True] * n,
        "centroid_y": rng.uniform(0, 1000, n),
        "centroid_x": rng.uniform(0, 1000, n),
    }
    for m in markers:
        rows[m] = rng.uniform(0, 1, n).astype(np.float32)
    return pd.DataFrame(rows)


def _make_core_metadata(channels_safe: list[str], core: str) -> dict:
    return {
        "channel_names": channels_safe,
        "cores": [
            {
                "core_name": core,
                "core_id": "feat-1",
                "min_x": 0,
                "min_y": 0,
                "max_x": 500,
                "max_y": 500,
                "width": 500,
                "height": 500,
            }
        ],
    }


@pytest.fixture
def synthetic_config(tmp_path):
    """Config backed by fully synthetic inputs."""
    # Write id_mapping
    id_map = _make_id_mapping(N_CELLS)
    id_map_path = tmp_path / "id_mapping.csv"
    id_map.to_csv(id_map_path, index=False)

    # Write QuPath CSV
    qupath_df = _make_qupath_csv(N_CELLS, CHANNELS_ORIG)
    csv_path = tmp_path / "measurements.csv"
    qupath_df.to_csv(csv_path, index=False)

    # Write channel_names.json
    ch_json = tmp_path / "channel_names.json"
    ch_json.write_text(
        json.dumps({"channel_names": CHANNELS_ORIG, "channel_names_safe": CHANNELS_SAFE})
    )

    # Write Nimbus parquet
    nimbus_df = _make_nimbus_parquet(N_CELLS, MARKERS_NIMBUS, CORE)
    dataset_dir = tmp_path / "out" / "test_ds"
    nimbus_out = dataset_dir / "fovs" / CORE / "nimbus_output"
    nimbus_out.mkdir(parents=True)
    nimbus_df.to_parquet(nimbus_out / f"{CORE}_nimbus_cell_predicted_probs.parquet")

    # Write core_metadata.json
    sample_images_dir = dataset_dir / "sample-images"
    sample_images_dir.mkdir(parents=True)
    meta = _make_core_metadata(CHANNELS_SAFE, CORE)
    (sample_images_dir / "core_metadata.json").write_text(json.dumps(meta))

    # Dummy input files that must exist for config validation
    image_path = tmp_path / "image.tiff"
    image_path.write_bytes(b"")
    cores_geojson = tmp_path / "cores.geojson"
    cores_geojson.write_text("{}")
    cells_geojson = tmp_path / "cells.geojson"
    cells_geojson.write_text("{}")

    return PipelineConfig(
        dataset_id="test_ds",
        inputs=InputsConfig(
            image=image_path,
            cores_geojson=cores_geojson,
            cells_geojson=cells_geojson,
            measurements_csv=csv_path,
            channel_names_json=ch_json,
        ),
        output_dir=tmp_path / "out",
        masks=MasksConfig(id_mapping_path=id_map_path),
        merge=MergeConfig(positivity_threshold=0.5),
        pipeline=PipelineRunConfig(skip_completed=False),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_qupath_returns_correct_channels():
    """_load_qupath extracts Cell Mean columns and maps to safe names."""
    df = _make_qupath_csv(5, CHANNELS_ORIG)
    csv_path = Path("/tmp/_test_qupath.csv")
    df.to_csv(csv_path, index=False)

    result, channels, extra_cols = _load_qupath(csv_path, CHANNELS_ORIG, CHANNELS_SAFE)
    assert set(channels) == set(CHANNELS_SAFE)
    assert "core" in result.columns
    for ch in CHANNELS_SAFE:
        assert ch in result.columns
    # Extra morphological columns should be captured
    assert len(extra_cols) >= 2
    assert "nucleus_area_m_2" in extra_cols or any("nucleus" in c for c in extra_cols)


def test_build_anndata_shape():
    """_build_anndata produces correct obs × var shape."""
    rng = np.random.default_rng(0)
    n = 8
    merged = pd.DataFrame(
        {
            "object_id": [f"uuid-{i}" for i in range(n)],
            "core": [CORE] * n,
            "label": range(1, n + 1),
            "centroid_x_um": rng.uniform(0, 100, n),
            "centroid_y_um": rng.uniform(0, 100, n),
            "cell_area_um2": rng.uniform(50, 200, n),
            "CD3e": rng.uniform(0, 1, n),
            "CD8": rng.uniform(0, 1, n),
            "DAPI": rng.uniform(0, 5000, n),
        }
    )
    adata = _build_anndata(
        merged=merged,
        intensity_channels=["DAPI", "CD3e", "CD8"],
        nimbus_channels=["CD3e", "CD8"],
        all_channels=["DAPI", "CD3e", "CD8"],
        threshold=0.5,
        dataset_id="test_ds",
        extra_obs_cols=[],
    )
    assert adata.n_obs == n
    assert adata.n_vars == 3
    assert "raw_intensity" in adata.layers
    assert "nimbus_probabilities" in adata.layers


def test_build_anndata_positivity_columns():
    """Binary positivity columns are created at the correct threshold."""
    rng = np.random.default_rng(42)
    n = 20
    probs = rng.uniform(0, 1, n).astype(np.float32)
    # _build_anndata expects Nimbus probs under "_prob_<ch>" (post-merge rename)
    merged = pd.DataFrame(
        {
            "object_id": [f"uuid-{i}" for i in range(n)],
            "core": [CORE] * n,
            "label": range(1, n + 1),
            "centroid_x_um": np.zeros(n),
            "centroid_y_um": np.zeros(n),
            "cell_area_um2": np.ones(n),
            "_prob_CD3e": probs,
        }
    )
    adata = _build_anndata(
        merged=merged,
        intensity_channels=[],
        nimbus_channels=["CD3e"],
        all_channels=["CD3e"],
        threshold=0.5,
        dataset_id="test_ds",
        extra_obs_cols=[],
    )
    expected = probs >= 0.5
    np.testing.assert_array_equal(adata.obs["CD3e_positive"].values, expected)


def test_build_anndata_nimbus_nan_for_excluded():
    """Channels not in nimbus_channels have NaN in the nimbus_probabilities layer."""
    n = 5
    merged = pd.DataFrame(
        {
            "object_id": [f"uuid-{i}" for i in range(n)],
            "core": [CORE] * n,
            "label": range(1, n + 1),
            "centroid_x_um": np.zeros(n),
            "centroid_y_um": np.zeros(n),
            "cell_area_um2": np.ones(n),
            "DAPI": np.full(n, 1000.0),  # intensity column (no prefix)
            "CD3e": np.full(n, 800.0),  # intensity column (no prefix)
            "_prob_CD3e": np.full(n, 0.8),  # Nimbus prob column (prefixed)
        }
    )
    adata = _build_anndata(
        merged=merged,
        intensity_channels=["DAPI", "CD3e"],
        nimbus_channels=["CD3e"],  # DAPI excluded from Nimbus
        all_channels=["DAPI", "CD3e"],
        threshold=0.5,
        dataset_id="test_ds",
        extra_obs_cols=[],
    )
    dapi_idx = list(adata.var.index).index("DAPI")
    cd3e_idx = list(adata.var.index).index("CD3e")
    assert np.all(np.isnan(adata.layers["nimbus_probabilities"][:, dapi_idx]))
    assert not np.any(np.isnan(adata.layers["nimbus_probabilities"][:, cd3e_idx]))


def test_run_step5_creates_h5ad(synthetic_config):
    """run_step5 writes an h5ad file and returns a valid AnnData."""
    adata = run_step5(synthetic_config)
    assert isinstance(adata, ad.AnnData)
    out_path = synthetic_config.results_dir / f"{synthetic_config.dataset_id}_combined.h5ad"
    assert out_path.exists()


def test_run_step5_anndata_structure(synthetic_config):
    """Output AnnData has expected layers and obs columns."""
    adata = run_step5(synthetic_config)
    assert "raw_intensity" in adata.layers
    assert "nimbus_probabilities" in adata.layers
    assert adata.n_obs == N_CELLS
    assert adata.n_vars == len(CHANNELS_SAFE)
    # Binary positivity columns for Nimbus markers
    for marker in MARKERS_NIMBUS:
        assert f"{marker}_positive" in adata.obs.columns, f"Missing positivity column for {marker}"
    # spatial obsm
    assert "spatial" in adata.obsm
    assert adata.obsm["spatial"].shape == (N_CELLS, 2)
    # dataset column
    assert "dataset" in adata.obs.columns
    assert (adata.obs["dataset"] == "test_ds").all()
    # extra QuPath metadata columns
    assert any("nucleus" in c for c in adata.obs.columns)


def test_run_step5_missing_id_mapping_raises(tmp_path):
    """run_step5 raises ValueError when id_mapping_path is not configured."""
    config = PipelineConfig(
        dataset_id="x",
        inputs=InputsConfig(
            image=tmp_path / "img.tiff",
            cores_geojson=tmp_path / "c.geojson",
            cells_geojson=tmp_path / "cells.geojson",
            measurements_csv=tmp_path / "m.csv",
        ),
        output_dir=tmp_path,
        masks=MasksConfig(id_mapping_path=None),
        pipeline=PipelineRunConfig(skip_completed=False),
    )
    with pytest.raises(ValueError, match="id_mapping_path"):
        run_step5(config)
