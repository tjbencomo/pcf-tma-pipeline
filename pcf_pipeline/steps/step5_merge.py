"""Step 5: Merge Nimbus predictions with QuPath measurements into AnnData."""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from pcf_pipeline import log
from pcf_pipeline.config import PipelineConfig


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_id_mapping(id_mapping_path: Path) -> pd.DataFrame:
    """Load mask_id → geojson_id mapping CSV produced by geojson2masks."""
    df = pd.read_csv(id_mapping_path, dtype={"mask_id": int, "geojson_id": str})
    log.info(f"Loaded id_mapping: {len(df)} entries from {id_mapping_path.name}")
    return df


def _load_qupath(
    csv_path: Path, channel_names: list[str], safe_names: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Load QuPath CSV and extract per-cell intensity measurements.

    Returns
    -------
    qupath_df : DataFrame indexed by Object ID with columns:
        core, centroid_x_um, centroid_y_um, cell_area_um2,
        and one column per safe channel name (Cell mean intensities).
    intensity_channels : list of safe channel names that had a matching
        Cell Mean column in the CSV.
    """
    df = pd.read_csv(csv_path, dtype={"Object ID": str})
    log.info(f"Loaded QuPath CSV: {len(df)} rows, {len(df.columns)} columns")

    # Build original → safe name mapping
    orig_to_safe = dict(zip(channel_names, safe_names))

    # Extract intensity columns: "Cell: <original_name>: Mean"
    intensity_cols = {}
    for orig, safe in orig_to_safe.items():
        col = f"Cell: {orig}: Mean"
        if col in df.columns:
            intensity_cols[col] = safe

    if not intensity_cols:
        raise ValueError(
            "No 'Cell: <marker>: Mean' columns found in QuPath CSV — "
            "check that channel names match the CSV headers"
        )
    log.info(f"Found {len(intensity_cols)} intensity columns in QuPath CSV")

    # Build output dataframe
    out = pd.DataFrame(index=df["Object ID"])
    out.index.name = "object_id"
    out["core"] = df["TMA Core"].values
    out["centroid_x_um"] = df["Centroid X µm"].values
    out["centroid_y_um"] = df["Centroid Y µm"].values
    out["cell_area_um2"] = df["Cell: Area µm^2"].values if "Cell: Area µm^2" in df.columns else np.nan

    for orig_col, safe_name in intensity_cols.items():
        out[safe_name] = df[orig_col].values

    intensity_channels = list(intensity_cols.values())
    return out, intensity_channels


def _load_nimbus_parquets(fovs_dir: Path, core_names: list[str]) -> pd.DataFrame:
    """Concatenate Nimbus parquet files across all cores.

    Returns a DataFrame with columns: label, core, <marker_prob>...
    """
    dfs = []
    for core in core_names:
        parquet = fovs_dir / core / "nimbus_output" / f"{core}_nimbus_cell_predicted_probs.parquet"
        if not parquet.exists():
            log.warning(f"Nimbus parquet not found for core {core}: {parquet}")
            continue
        df = pd.read_parquet(parquet)
        df["core"] = core
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No Nimbus parquet files found in {fovs_dir} — run Step 4 first"
        )

    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"Loaded Nimbus parquets: {len(combined)} rows across {len(dfs)} cores")
    return combined


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------


def _merge_all(
    nimbus_df: pd.DataFrame,
    id_mapping: pd.DataFrame,
    qupath_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join Nimbus → id_mapping → QuPath into one per-cell DataFrame."""
    # Nimbus label → geojson_id
    merged = nimbus_df.merge(
        id_mapping.rename(columns={"mask_id": "label", "geojson_id": "object_id"}),
        on="label",
        how="inner",
    )
    log.info(f"After Nimbus↔id_mapping join: {len(merged)} rows")

    # Rename Nimbus probability columns to avoid collisions with QuPath intensity columns
    meta_cols = {
        "label", "centroid_y", "centroid_x", "ymin", "xmin", "ymax", "xmax",
        "assigned_fov_id", "assigned_fov_index", "fully_contained", "fov", "core",
        "object_id",
    }
    prob_cols = [c for c in merged.columns if c not in meta_cols]
    merged = merged.rename(columns={c: f"_prob_{c}" for c in prob_cols})

    # geojson_id → QuPath row
    merged = merged.merge(
        qupath_df.reset_index(),  # brings object_id back as a column
        on="object_id",
        how="inner",
    )
    log.info(f"After Nimbus↔QuPath join: {len(merged)} rows")
    return merged


# ---------------------------------------------------------------------------
# AnnData builder
# ---------------------------------------------------------------------------


def _build_anndata(
    merged: pd.DataFrame,
    intensity_channels: list[str],
    nimbus_channels: list[str],
    all_channels: list[str],
    threshold: float,
) -> ad.AnnData:
    """Construct AnnData from the merged per-cell DataFrame.

    Parameters
    ----------
    merged : per-cell DataFrame after all joins
    intensity_channels : safe channel names present in the QuPath CSV
    nimbus_channels : safe channel names present in Nimbus parquets
    all_channels : union (ordered) of all safe channel names for var index
    threshold : positivity threshold applied to Nimbus probabilities
    """
    n_cells = len(merged)
    n_vars = len(all_channels)
    var_index = pd.Index(all_channels, name="channel")

    # Intensities layer (QuPath Cell Mean) — NaN for channels not in CSV
    intensity_matrix = np.full((n_cells, n_vars), np.nan, dtype=np.float32)
    for i, ch in enumerate(all_channels):
        if ch in intensity_channels and ch in merged.columns:
            intensity_matrix[:, i] = merged[ch].values.astype(np.float32)

    # Nimbus probabilities layer — stored under "_prob_<ch>" after merge rename
    # NaN for excluded channels
    nimbus_matrix = np.full((n_cells, n_vars), np.nan, dtype=np.float32)
    for i, ch in enumerate(all_channels):
        prob_col = f"_prob_{ch}"
        if ch in nimbus_channels and prob_col in merged.columns:
            nimbus_matrix[:, i] = merged[prob_col].values.astype(np.float32)

    # obs: cell metadata
    obs_cols = ["object_id", "core", "label", "centroid_x_um", "centroid_y_um", "cell_area_um2"]
    obs = merged[[c for c in obs_cols if c in merged.columns]].copy()
    obs = obs.set_index("object_id")
    obs.index.name = "cell_id"

    # Binary positivity columns for Nimbus channels
    for ch in nimbus_channels:
        prob_col = f"_prob_{ch}"
        if prob_col in merged.columns:
            obs[f"{ch}_positive"] = (merged[prob_col].values >= threshold).astype(bool)

    adata = ad.AnnData(
        X=intensity_matrix,
        obs=obs,
        var=pd.DataFrame(index=var_index),
        layers={
            "intensities": intensity_matrix,
            "nimbus_probabilities": nimbus_matrix,
        },
    )
    return adata


# ---------------------------------------------------------------------------
# Step entry point
# ---------------------------------------------------------------------------


def run_step5(config: PipelineConfig) -> ad.AnnData:
    """Combine fluorescence intensities, Nimbus probabilities, and metadata into AnnData."""
    t0 = log.log_step_start("Step 5: Merge into AnnData")

    # Require id_mapping
    if not config.masks.id_mapping_path:
        log.error(
            "masks.id_mapping_path is required for Step 5 — "
            "provide the id_mapping CSV produced by geojson2masks"
        )
        raise ValueError("masks.id_mapping_path not set in config")
    if not config.masks.id_mapping_path.exists():
        log.error(f"id_mapping file not found: {config.masks.id_mapping_path}")
        raise FileNotFoundError(config.masks.id_mapping_path)

    # Load channel names from core metadata
    meta_path = config.sample_images_dir / "core_metadata.json"
    if not meta_path.exists():
        log.error(f"core_metadata.json not found: {meta_path} — run Step 2 first")
        raise FileNotFoundError(meta_path)

    with open(meta_path) as f:
        meta = json.load(f)

    safe_names: list[str] = meta["channel_names"]  # already safe names from step 2
    core_records: list[dict] = meta["cores"]
    core_names = [r["core_name"] for r in core_records]

    # Recover original channel names from channel_names_json if available
    if config.inputs.channel_names_json and config.inputs.channel_names_json.exists():
        with open(config.inputs.channel_names_json) as f:
            raw = json.load(f)
        original_names = raw["channel_names"] if isinstance(raw, dict) else raw
    else:
        # Fall back: reverse the safe transformation (best-effort)
        original_names = [s.replace("_", " ").replace("-", "/") for s in safe_names]
        log.warning("channel_names_json not provided — using approximate original names for QuPath join")

    log.info(f"Processing {len(core_names)} cores, {len(safe_names)} channels")

    # Load data
    id_mapping = _load_id_mapping(config.masks.id_mapping_path)
    qupath_df, intensity_channels = _load_qupath(
        config.inputs.measurements_csv, original_names, safe_names
    )
    nimbus_df = _load_nimbus_parquets(config.fovs_dir, core_names)

    # Identify Nimbus probability columns (marker names only, before any renaming)
    _meta_cols = {
        "label", "centroid_y", "centroid_x", "ymin", "xmin", "ymax", "xmax",
        "assigned_fov_id", "assigned_fov_index", "fully_contained", "fov", "core",
    }
    nimbus_channels = [c for c in nimbus_df.columns if c not in _meta_cols]
    log.info(f"Nimbus probability columns: {len(nimbus_channels)}")

    # Merge
    merged = _merge_all(nimbus_df, id_mapping, qupath_df)
    if len(merged) == 0:
        log.warning("Merge produced 0 rows — check that id_mapping matches the masks used")

    # Build AnnData
    adata = _build_anndata(
        merged=merged,
        intensity_channels=intensity_channels,
        nimbus_channels=nimbus_channels,
        all_channels=safe_names,
        threshold=config.merge.positivity_threshold,
    )
    log.info(
        f"AnnData: {adata.n_obs} cells × {adata.n_vars} channels, "
        f"layers: {list(adata.layers.keys())}"
    )

    # Save
    config.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.results_dir / f"{config.dataset_id}_combined.h5ad"
    adata.write_h5ad(out_path)
    log.info(f"Saved: {out_path}")

    log.log_step_end("Step 5: Merge into AnnData", t0)
    return adata
