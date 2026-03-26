# PCF Pipeline

A modular data processing pipeline for **Phenocycler (PCF) multiplex immunofluorescence** data from tissue microarray (TMA) datasets. The pipeline runs [Nimbus](https://github.com/angelolab/Nimbus-Inference) deep-learning inference on PCF images and merges the resulting cell-level marker probability scores with QuPath single-cell fluorescence intensity measurements, producing an [AnnData](https://anndata.readthedocs.io) object ready for downstream analysis.

## Why use this pipeline?

Analysing PCF TMA data typically requires stitching together several bespoke steps — converting segmentation polygons to mask arrays, cropping per-core images, tiling for GPU inference, running Nimbus, and finally combining everything with the raw intensity measurements. This pipeline automates that entire workflow from raw `.qptiff` and QuPath exports through to a single `.h5ad` file, with:

- **Reproducibility** — all parameters live in a single YAML config file.
- **Resumability** — `skip_completed: true` lets you restart a job after a failure without re-running finished steps.
- **HPC-ready** — ships with a SLURM `sbatch` script and is designed for GPU nodes with large RAM.
- **Selective execution** — run the full pipeline or any subset of steps via the CLI.

---

## Computing Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 128 GB | 256 GB |
| GPU | Any CUDA GPU | NVIDIA GPU with ≥8 GB VRAM |
| CPU cores | 4 | 8 |
| Storage | Varies by dataset | ~2× raw image size for outputs |

Nimbus requires a CUDA-capable GPU. The pipeline is designed for SLURM HPC clusters but can also run on a workstation with sufficient resources.

---

## Installation

The pipeline is managed with [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html). On an HPC cluster with NVIDIA CUDA:

```bash
# Load CUDA (adjust for your cluster's module system)
ml CUDA

micromamba create -n pcf-pipeline python=3.10
micromamba activate pcf-pipeline
micromamba install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

pip install nimbus-inference
pip install tifffile numpy pandas anndata scikit-image shapely geojson \
            imagecodecs ijson ruff pyarrow pyyaml
pip install git+https://github.com/tjbencomo/geojson2masks.git

# Install the pipeline itself (from the repo root)
pip install -e .
```

Verify the GPU is accessible:

```bash
micromamba run -n pcf-pipeline python -c "import torch; print(torch.cuda.is_available())"
```

---

## Input Files

All four inputs are required and are specified in the config YAML.

| Parameter | File | Description |
|-----------|------|-------------|
| `inputs.image` | `*.er.qptiff` | PCF image — multi-channel TIFF exported from the scanner. Channel names are extracted automatically from qptiff metadata. |
| `inputs.cores_geojson` | `*_cores.geojson` | TMA core polygon segmentations exported from QuPath. |
| `inputs.cells_geojson` | `*.Instanseg.geojson` | Cell and nucleus segmentation polygons exported from QuPath (InstanSeg format). |
| `inputs.measurements_csv` | `*_all-cell-measurements*.csv` | Per-cell fluorescence intensity measurements exported from QuPath, including `Cell: <marker>: Mean` columns. |

An optional `inputs.channel_names_json` can provide channel names explicitly; if omitted they are parsed from the qptiff metadata.

---

## Pipeline Steps

### Step 1 — Segmentation Masks

Converts cell and nucleus segmentation polygons (GeoJSON) into integer-labelled TIFF mask arrays using [geojson2masks](https://github.com/tjbencomo/geojson2masks). Image dimensions are read automatically from the input image.

**Outputs** (written to `<output_dir>/<dataset_id>/masks/`):
- `<stem>_cell_mask.tif` — whole-cell label mask
- `<stem>_nucleus_mask.tif` — nuclear label mask
- `<stem>_id_mapping.csv` — maps integer mask labels to QuPath Object IDs (UUIDs)

**Skipping**: if `masks.cell_mask_path` and `masks.nucleus_mask_path` are set in the config, this step is skipped and the provided files are used directly.

---

### Step 2 — Core Splitting

Crops the full TMA image into individual per-core images using the core polygon boundaries. Also crops the cell and nucleus masks to each core. Core bounding boxes can be padded and optionally forced to be square.

**Outputs** (written to `<output_dir>/<dataset_id>/sample-images/`):
```
sample-images/
  images/
    core_A-1.tiff
    core_A-2.tiff
    ...
  cell_masks/
    core_A-1_cell_mask.tiff
    ...
  nuclei_masks/
    core_A-1_nuclei_mask.tiff
    ...
  core_metadata.json        # Bounding box offsets for each core
```

**Parameters** (`cores` section):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `padding` | `100` | Pixels of padding added around each core bounding box |
| `square` | `true` | Force bounding boxes to be square (centred on core) |
| `skip_missing` | `true` | Skip cores marked `isMissing` in the GeoJSON |
| `compression` | `"zlib"` | TIFF compression (`zlib`, `lzw`, or `null`) |

---

### Step 3 — FOV Creation

Tiles each per-core image into 1024×1024 Field-of-View (FOV) images with configurable overlap. Assigns every cell to exactly one FOV (the FOV in which its bounding box is fully contained with maximum centroid margin). Cells that don't fit any grid FOV receive a dedicated fallback FOV centred on their bounding box.

**Outputs** (written to `<output_dir>/<dataset_id>/fovs/<core_id>/`):
```
fovs/
  A-1/
    images/
      fov_000/
        DAPI.tif
        CD3e.tif
        ...
      fov_001/
        ...
    segmentation/
      fov_000_whole_cell.tiff
      fov_000_nuclear.tiff
      ...
    cell_to_fov.csv           # label, assigned_fov_id, centroid, bbox, ...
```

**Parameters** (`fovs` section):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tile_size` | `1024` | FOV height and width in pixels |
| `overlap` | `128` | Overlap between adjacent FOVs in pixels |
| `remove_offset` | `null` | Subtract a constant pixel offset from all values (or `null` to skip) |

---

### Step 4 — Nimbus Inference

Runs [Nimbus](https://github.com/angelolab/Nimbus-Inference) deep-learning inference on each core's FOV directory to predict per-cell marker positivity probabilities. Non-marker channels (e.g. DAPI, background channels) are excluded. Results are merged with the cell-to-FOV map and saved as a parquet file.

**Outputs** (written to `<output_dir>/<dataset_id>/fovs/<core_id>/nimbus_output/`):
- `<core_id>_nimbus_cell_predicted_probs.parquet` — one row per cell, one column per marker with probability scores in [0, 1]

**Parameters** (`nimbus` section):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `exclude_channels` | `["DAPI","BX010_750","T55","T60"]` | Channel names to exclude from Nimbus inference. All channels are still saved in core/FOV TIFFs and the final intensity layer. |
| `batch_size` | `4` | GPU batch size |
| `test_time_aug` | `true` | Enable test-time augmentation |
| `input_shape` | `[1024, 1024]` | Expected input tile size (should match `fovs.tile_size`) |
| `device` | `"auto"` | `"auto"`, `"cuda"`, `"cpu"`, or `"mps"` |
| `normalization_quantile` | `0.999` | Quantile for per-channel normalization |
| `normalization_n_subset` | `400` | Number of FOVs sampled for normalization |
| `normalization_clip` | `[0, 2]` | Clip range after normalization |

---

### Step 5 — Merge into AnnData

Combines all data sources into a single [AnnData](https://anndata.readthedocs.io) object:

- **`X`** — same as `layers['intensities']` (QuPath fluorescence intensities); provided as the default matrix for compatibility with standard AnnData workflows
- **`layers['intensities']`** — QuPath `Cell: <marker>: Mean` fluorescence intensities for all channels (cells × channels)
- **`layers['nimbus_probabilities']`** — Nimbus marker probability scores (cells × channels; `NaN` for channels excluded from inference)
- **`obs`** — per-cell metadata: `cell_id` (QuPath UUID), `core`, `mask_label`, `centroid_x_um`, `centroid_y_um`, `cell_area_um2`, plus binary `<marker>_positive` columns at the configured threshold
- **`var`** — channel names (index)

The merge uses the `id_mapping.csv` from Step 1 to link integer mask labels (used by Nimbus) to QuPath Object IDs (used in the measurements CSV).

**Output** (written to `<output_dir>/<dataset_id>/results/`):
- `<dataset_id>_combined.h5ad`

**Parameters** (`merge` section):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `positivity_threshold` | `0.5` | Nimbus probability threshold above which a cell is called positive for a marker |

---

## Configuration

Copy `config.example.yaml` and edit it for your dataset:

```bash
cp config.example.yaml my_dataset.yaml
```

```yaml
dataset_id: "MCC_51p_TMA_A1B1"

inputs:
  image: "path/to/image.er.qptiff"
  cores_geojson: "path/to/cores.geojson"
  cells_geojson: "path/to/cells.Instanseg.geojson"
  measurements_csv: "path/to/measurements.csv"
  channel_names_json: null   # optional

output_dir: "/path/to/output"

masks:
  cell_mask_path: null       # set to skip Step 1
  nucleus_mask_path: null
  id_mapping_path: null

cores:
  padding: 100
  square: true
  skip_missing: true

fovs:
  tile_size: 1024
  overlap: 128

nimbus:
  exclude_channels: ["DAPI", "BX010_750", "T55", "T60"]
  batch_size: 4
  test_time_aug: true
  device: "auto"

merge:
  positivity_threshold: 0.5

pipeline:
  skip_completed: true
  steps: [1, 2, 3, 4, 5]
```

All file paths in the config can be absolute or relative to the config file's directory.

---

## Running the Pipeline

### Full pipeline (recommended — via SLURM)

```bash
sbatch scripts/run_pipeline.sh /path/to/my_dataset.yaml
```

Logs are written to `logs/pcf-pipeline_<jobid>.out` and `.err`.

### Full pipeline (interactive)

```bash
micromamba run -n pcf-pipeline pcf-pipeline run --config my_dataset.yaml
```

### Individual steps

Run a single step by name — useful for re-running a failed step or testing:

```bash
micromamba run -n pcf-pipeline pcf-pipeline masks  --config my_dataset.yaml  # Step 1
micromamba run -n pcf-pipeline pcf-pipeline cores  --config my_dataset.yaml  # Step 2
micromamba run -n pcf-pipeline pcf-pipeline fovs   --config my_dataset.yaml  # Step 3
micromamba run -n pcf-pipeline pcf-pipeline nimbus --config my_dataset.yaml  # Step 4
micromamba run -n pcf-pipeline pcf-pipeline merge  --config my_dataset.yaml  # Step 5
```

### Running a subset of steps

Set `pipeline.steps` in the config to run only the steps you need:

```yaml
pipeline:
  steps: [4, 5]   # re-run Nimbus and merge only
```

---

## Output File Structure

```
<output_dir>/
  <dataset_id>/
    masks/
      <stem>_cell_mask.tif
      <stem>_nucleus_mask.tif
      <stem>_id_mapping.csv
    sample-images/
      core_metadata.json
      images/core_<id>.tiff
      cell_masks/core_<id>_cell_mask.tiff
      nuclei_masks/core_<id>_nuclei_mask.tiff
    fovs/
      <core_id>/
        images/
          fov_000/<channel>.tif
          fov_001/<channel>.tif
          ...
        segmentation/
          fov_000_whole_cell.tiff
          fov_000_nuclear.tiff
          ...
        nimbus_output/
          <core_id>_nimbus_cell_predicted_probs.parquet
        cell_to_fov.csv
    results/
      <dataset_id>_combined.h5ad
```

---

## Development

```bash
# Run fast tests (no GPU required)
micromamba run -n pcf-pipeline python -m pytest tests/ -m "not slow"

# Run all tests including Nimbus inference (GPU required, ~15 min)
micromamba run -n pcf-pipeline python -m pytest tests/

# Lint and format
micromamba run -n pcf-pipeline ruff check . && micromamba run -n pcf-pipeline ruff format --check .
micromamba run -n pcf-pipeline ruff check --fix . && micromamba run -n pcf-pipeline ruff format .
```
