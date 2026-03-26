"""Step 4: Run Nimbus inference on each core's FOVs."""

import os
from pathlib import Path

import pandas as pd

from pcf_pipeline import log
from pcf_pipeline.config import PipelineConfig


def _get_include_channels(tiff_dir: Path, exclude_channels: list[str]) -> list[str]:
    """Return sorted list of channel names in a FOV dir, minus excluded channels."""
    fov_dirs = sorted(
        d for d in tiff_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not fov_dirs:
        raise FileNotFoundError(f"No FOV subdirectories found in {tiff_dir}")
    all_channels = sorted(
        f.stem for f in fov_dirs[0].glob("*.tif") if not f.name.startswith(".")
    )
    include = [ch for ch in all_channels if ch not in exclude_channels]
    log.info(
        f"Channels: {len(all_channels)} total, {len(include)} included "
        f"({len(all_channels) - len(include)} excluded)"
    )
    return include


def _get_fov_paths(tiff_dir: Path) -> list[str]:
    """Return sorted list of FOV directory paths as strings."""
    return sorted(
        str(d)
        for d in tiff_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def _run_nimbus_for_core(
    core_id: str,
    core_fov_dir: Path,
    config: PipelineConfig,
) -> None:
    """Run Nimbus inference for a single core and save the parquet result."""
    # Deferred import so the rest of the pipeline works without nimbus installed
    from nimbus_inference.nimbus import Nimbus, prep_naming_convention
    from nimbus_inference.utils import MultiplexDataset

    tiff_dir = core_fov_dir / "images"
    segmentation_dir = core_fov_dir / "segmentation"
    nimbus_output_dir = core_fov_dir / "nimbus_output"
    nimbus_output_dir.mkdir(parents=True, exist_ok=True)

    include_channels = _get_include_channels(tiff_dir, config.nimbus.exclude_channels)
    fov_paths = _get_fov_paths(tiff_dir)
    log.info(f"Found {len(fov_paths)} FOVs for core {core_id}")

    segmentation_naming_convention = prep_naming_convention(segmentation_dir)

    # Validate naming convention against first FOV
    first_seg = segmentation_naming_convention(fov_paths[0])
    if not os.path.exists(first_seg):
        log.warning(
            f"Segmentation file not found for first FOV: {first_seg} — "
            "check that Step 3 ran successfully"
        )

    dataset = MultiplexDataset(
        fov_paths=fov_paths,
        suffix="tif",
        include_channels=include_channels,
        segmentation_naming_convention=segmentation_naming_convention,
        output_dir=nimbus_output_dir,
    )

    nimbus = Nimbus(
        dataset=dataset,
        save_predictions=True,
        batch_size=config.nimbus.batch_size,
        test_time_aug=config.nimbus.test_time_aug,
        input_shape=config.nimbus.input_shape,
        device=config.nimbus.device,
        output_dir=nimbus_output_dir,
    )
    nimbus.check_inputs()

    log.info(f"Computing normalization dict for {core_id}...")
    dataset.prepare_normalization_dict(
        quantile=config.nimbus.normalization_quantile,
        n_subset=config.nimbus.normalization_n_subset,
        clip_values=tuple(config.nimbus.normalization_clip),
        multiprocessing=True,
        overwrite=True,
    )

    log.info(f"Running Nimbus inference for {core_id}...")
    cell_table = nimbus.predict_fovs()

    # Merge with cell-to-FOV map
    cells2fov = pd.read_csv(core_fov_dir / "cell_to_fov.csv")
    cell_df = pd.merge(
        cells2fov,
        cell_table,
        left_on=["label", "assigned_fov_id"],
        right_on=["label", "fov"],
        how="inner",
    )

    out_path = nimbus_output_dir / f"{core_id}_nimbus_cell_predicted_probs.parquet"
    cell_df.to_parquet(out_path)
    log.info(f"Saved {len(cell_df)} cell predictions: {out_path}")


def run_step4(config: PipelineConfig) -> None:
    """Run Nimbus deep-learning inference to predict marker positivity per cell."""
    t0 = log.log_step_start("Step 4: Nimbus inference")

    fovs_dir = config.fovs_dir
    if not fovs_dir.exists():
        log.error(f"FOVs directory not found: {fovs_dir} — run Step 3 first")
        raise FileNotFoundError(fovs_dir)

    core_dirs = sorted(
        d for d in fovs_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not core_dirs:
        log.error(f"No core directories found in {fovs_dir}")
        raise FileNotFoundError(f"No cores in {fovs_dir}")

    log.info(f"Found {len(core_dirs)} cores to process")
    log.info(f"Excluding channels: {config.nimbus.exclude_channels}")

    for idx, core_dir in enumerate(core_dirs):
        core_id = core_dir.name
        log.log_progress(core_id, idx + 1, len(core_dirs))

        out_parquet = (
            core_dir / "nimbus_output" / f"{core_id}_nimbus_cell_predicted_probs.parquet"
        )
        if config.pipeline.skip_completed and out_parquet.exists():
            log.info(f"Skipping {core_id} (parquet exists)")
            continue

        _run_nimbus_for_core(core_id, core_dir, config)

    log.log_step_end("Step 4: Nimbus inference", t0)
