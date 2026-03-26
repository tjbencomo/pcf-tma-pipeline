"""End-to-end integration test: runs all 5 pipeline steps on the test dataset."""

from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from pcf_pipeline.config import (
    CoresConfig,
    FovsConfig,
    InputsConfig,
    MasksConfig,
    MergeConfig,
    NimbusConfig,
    PipelineConfig,
    PipelineRunConfig,
)
from pcf_pipeline.steps.step1_masks import run_step1
from pcf_pipeline.steps.step2_cores import run_step2
from pcf_pipeline.steps.step3_fovs import run_step3
from pcf_pipeline.steps.step4_nimbus import run_step4
from pcf_pipeline.steps.step5_merge import run_step5

TEST_DATA_DIR = Path(__file__).parent.parent / "test-data"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def full_pipeline_output(tmp_path_factory):
    """Run all 5 steps once and return (config, adata)."""
    tmp_path = tmp_path_factory.mktemp("integration")

    config = PipelineConfig(
        dataset_id="integration_test",
        inputs=InputsConfig(
            image=TEST_DATA_DIR / "test_image.tiff",
            cores_geojson=TEST_DATA_DIR / "test_cores.geojson",
            cells_geojson=TEST_DATA_DIR / "test_cells.geojson",
            measurements_csv=TEST_DATA_DIR / "test_measurements.csv",
            channel_names_json=TEST_DATA_DIR / "channel_names.json",
        ),
        output_dir=tmp_path,
        masks=MasksConfig(),  # auto-generate via geojson2masks
        cores=CoresConfig(padding=50, square=True, skip_missing=True),
        fovs=FovsConfig(tile_size=1024, overlap=128),
        nimbus=NimbusConfig(batch_size=4, test_time_aug=False),
        merge=MergeConfig(positivity_threshold=0.5),
        pipeline=PipelineRunConfig(skip_completed=True),
    )

    # Step 1: generate masks
    cell_mask, nuc_mask, id_mapping = run_step1(config)
    config.masks.cell_mask_path = cell_mask
    config.masks.nucleus_mask_path = nuc_mask
    config.masks.id_mapping_path = id_mapping

    # Steps 2-5
    run_step2(config)
    run_step3(config)
    run_step4(config)
    adata = run_step5(config)

    return config, adata


# ---------------------------------------------------------------------------
# Step 1 outputs
# ---------------------------------------------------------------------------


def test_step1_masks_exist(full_pipeline_output):
    config, _ = full_pipeline_output
    assert config.masks.cell_mask_path.exists()
    assert config.masks.nucleus_mask_path.exists()
    assert config.masks.id_mapping_path.exists()


def test_step1_id_mapping_has_rows(full_pipeline_output):
    import pandas as pd
    config, _ = full_pipeline_output
    df = pd.read_csv(config.masks.id_mapping_path)
    assert len(df) > 0
    assert "mask_id" in df.columns
    assert "geojson_id" in df.columns


# ---------------------------------------------------------------------------
# Step 2 outputs
# ---------------------------------------------------------------------------


def test_step2_core_images_exist(full_pipeline_output):
    config, _ = full_pipeline_output
    images_dir = config.sample_images_dir / "images"
    core_tiffs = list(images_dir.glob("core_*.tiff"))
    assert len(core_tiffs) >= 1


def test_step2_core_metadata_exists(full_pipeline_output):
    config, _ = full_pipeline_output
    assert (config.sample_images_dir / "core_metadata.json").exists()


# ---------------------------------------------------------------------------
# Step 3 outputs
# ---------------------------------------------------------------------------


def test_step3_cell_to_fov_exists(full_pipeline_output):
    config, _ = full_pipeline_output
    for core_dir in config.fovs_dir.iterdir():
        assert (core_dir / "cell_to_fov.csv").exists(), \
            f"cell_to_fov.csv missing for {core_dir.name}"


def test_step3_fov_tiles_exist(full_pipeline_output):
    config, _ = full_pipeline_output
    for core_dir in config.fovs_dir.iterdir():
        fov_dirs = list((core_dir / "images").iterdir())
        assert len(fov_dirs) >= 1, f"No FOV dirs for {core_dir.name}"


# ---------------------------------------------------------------------------
# Step 4 outputs
# ---------------------------------------------------------------------------


def test_step4_parquets_exist(full_pipeline_output):
    config, _ = full_pipeline_output
    for core_dir in config.fovs_dir.iterdir():
        parquet = core_dir / "nimbus_output" / f"{core_dir.name}_nimbus_cell_predicted_probs.parquet"
        assert parquet.exists(), f"Parquet missing for {core_dir.name}"


def test_step4_parquets_have_predictions(full_pipeline_output):
    import pandas as pd
    config, _ = full_pipeline_output
    for core_dir in config.fovs_dir.iterdir():
        parquet = core_dir / "nimbus_output" / f"{core_dir.name}_nimbus_cell_predicted_probs.parquet"
        df = pd.read_parquet(parquet)
        assert len(df) > 0, f"Empty parquet for {core_dir.name}"


# ---------------------------------------------------------------------------
# Step 5 outputs
# ---------------------------------------------------------------------------


def test_step5_h5ad_exists(full_pipeline_output):
    config, adata = full_pipeline_output
    out_path = config.results_dir / f"{config.dataset_id}_combined.h5ad"
    assert out_path.exists()


def test_step5_anndata_layers(full_pipeline_output):
    _, adata = full_pipeline_output
    assert isinstance(adata, ad.AnnData)
    assert "intensities" in adata.layers
    assert "nimbus_probabilities" in adata.layers


def test_step5_anndata_has_cells(full_pipeline_output):
    _, adata = full_pipeline_output
    assert adata.n_obs > 0
    assert adata.n_vars > 0


def test_step5_anndata_positivity_columns(full_pipeline_output):
    config, adata = full_pipeline_output
    # At least one positivity column should exist
    pos_cols = [c for c in adata.obs.columns if c.endswith("_positive")]
    assert len(pos_cols) > 0


def test_step5_intensities_not_all_nan(full_pipeline_output):
    _, adata = full_pipeline_output
    assert not np.all(np.isnan(adata.layers["intensities"]))


def test_step5_h5ad_reloadable(full_pipeline_output):
    config, _ = full_pipeline_output
    out_path = config.results_dir / f"{config.dataset_id}_combined.h5ad"
    reloaded = ad.read_h5ad(out_path)
    assert reloaded.n_obs > 0
