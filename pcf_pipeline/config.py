"""YAML configuration loading and validation for the PCF pipeline."""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from pcf_pipeline import log


@dataclass
class InputsConfig:
    image: Path
    cores_geojson: Path
    cells_geojson: Path
    measurements_csv: Path
    channel_names_json: Path | None = None


@dataclass
class MasksConfig:
    cell_mask_path: Path | None = None
    nucleus_mask_path: Path | None = None
    id_mapping_path: Path | None = None


@dataclass
class CoresConfig:
    padding: int = 100
    square: bool = True
    skip_missing: bool = True
    compression: str = "zlib"


@dataclass
class FovsConfig:
    tile_size: int = 1024
    overlap: int = 128
    remove_offset: int | None = None


@dataclass
class NimbusConfig:
    exclude_channels: list[str] = field(
        default_factory=lambda: ["DAPI", "BX010_750", "T55", "T60"]
    )
    batch_size: int = 4
    test_time_aug: bool = True
    input_shape: list[int] = field(default_factory=lambda: [1024, 1024])
    device: str = "auto"
    normalization_quantile: float = 0.999
    normalization_n_subset: int = 400
    normalization_clip: list[float] = field(default_factory=lambda: [0.0, 2.0])


@dataclass
class MergeConfig:
    positivity_threshold: float = 0.5


@dataclass
class PipelineRunConfig:
    skip_completed: bool = True
    steps: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])


@dataclass
class PipelineConfig:
    dataset_id: str
    inputs: InputsConfig
    output_dir: Path
    masks: MasksConfig = field(default_factory=MasksConfig)
    cores: CoresConfig = field(default_factory=CoresConfig)
    fovs: FovsConfig = field(default_factory=FovsConfig)
    nimbus: NimbusConfig = field(default_factory=NimbusConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    pipeline: PipelineRunConfig = field(default_factory=PipelineRunConfig)

    @property
    def dataset_dir(self) -> Path:
        """Root output directory for this dataset."""
        return self.output_dir / self.dataset_id

    @property
    def sample_images_dir(self) -> Path:
        return self.dataset_dir / "sample-images"

    @property
    def fovs_dir(self) -> Path:
        return self.dataset_dir / "fovs"

    @property
    def results_dir(self) -> Path:
        return self.dataset_dir / "results"

    @property
    def masks_dir(self) -> Path:
        return self.dataset_dir / "masks"


def _resolve_path(base_dir: Path, value: str | None) -> Path | None:
    """Resolve a path relative to the config file's directory."""
    if value is None:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return base_dir / p


def _build_inputs(raw: dict, base_dir: Path) -> InputsConfig:
    required = ["image", "cores_geojson", "cells_geojson", "measurements_csv"]
    for key in required:
        if key not in raw:
            log.error(f"Missing required input: inputs.{key}")
            sys.exit(1)
    return InputsConfig(
        image=_resolve_path(base_dir, raw["image"]),
        cores_geojson=_resolve_path(base_dir, raw["cores_geojson"]),
        cells_geojson=_resolve_path(base_dir, raw["cells_geojson"]),
        measurements_csv=_resolve_path(base_dir, raw["measurements_csv"]),
        channel_names_json=_resolve_path(base_dir, raw.get("channel_names_json")),
    )


def _build_masks(raw: dict | None, base_dir: Path) -> MasksConfig:
    if raw is None:
        return MasksConfig()
    return MasksConfig(
        cell_mask_path=_resolve_path(base_dir, raw.get("cell_mask_path")),
        nucleus_mask_path=_resolve_path(base_dir, raw.get("nucleus_mask_path")),
        id_mapping_path=_resolve_path(base_dir, raw.get("id_mapping_path")),
    )


def load_config(config_path: Path) -> PipelineConfig:
    """Load and validate a pipeline YAML config file."""
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        log.error(f"Config file not found: {config_path}")
        sys.exit(1)

    log.info(f"Loading config from: {config_path}")
    base_dir = config_path.parent

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        log.error("Config file must be a YAML mapping")
        sys.exit(1)

    # Required top-level fields
    for key in ["dataset_id", "inputs", "output_dir"]:
        if key not in raw:
            log.error(f"Missing required config field: {key}")
            sys.exit(1)

    inputs = _build_inputs(raw["inputs"], base_dir)
    masks = _build_masks(raw.get("masks"), base_dir)

    # Validate input files exist
    for name, path in [
        ("image", inputs.image),
        ("cores_geojson", inputs.cores_geojson),
        ("cells_geojson", inputs.cells_geojson),
        ("measurements_csv", inputs.measurements_csv),
    ]:
        if not path.exists():
            log.error(f"Input file not found: inputs.{name} = {path}")
            sys.exit(1)

    raw_cores = raw.get("cores", {}) or {}
    raw_fovs = raw.get("fovs", {}) or {}
    raw_nimbus = raw.get("nimbus", {}) or {}
    raw_merge = raw.get("merge", {}) or {}
    raw_pipeline = raw.get("pipeline", {}) or {}

    return PipelineConfig(
        dataset_id=raw["dataset_id"],
        inputs=inputs,
        output_dir=_resolve_path(base_dir, raw["output_dir"]),
        masks=masks,
        cores=CoresConfig(
            padding=raw_cores.get("padding", 100),
            square=raw_cores.get("square", True),
            skip_missing=raw_cores.get("skip_missing", True),
            compression=raw_cores.get("compression", "zlib"),
        ),
        fovs=FovsConfig(
            tile_size=raw_fovs.get("tile_size", 1024),
            overlap=raw_fovs.get("overlap", 128),
            remove_offset=raw_fovs.get("remove_offset"),
        ),
        nimbus=NimbusConfig(
            exclude_channels=raw_nimbus.get(
                "exclude_channels", ["DAPI", "BX010_750", "T55", "T60"]
            ),
            batch_size=raw_nimbus.get("batch_size", 4),
            test_time_aug=raw_nimbus.get("test_time_aug", True),
            input_shape=raw_nimbus.get("input_shape", [1024, 1024]),
            device=raw_nimbus.get("device", "auto"),
            normalization_quantile=raw_nimbus.get("normalization_quantile", 0.999),
            normalization_n_subset=raw_nimbus.get("normalization_n_subset", 400),
            normalization_clip=raw_nimbus.get("normalization_clip", [0.0, 2.0]),
        ),
        merge=MergeConfig(
            positivity_threshold=raw_merge.get("positivity_threshold", 0.5),
        ),
        pipeline=PipelineRunConfig(
            skip_completed=raw_pipeline.get("skip_completed", True),
            steps=raw_pipeline.get("steps", [1, 2, 3, 4, 5]),
        ),
    )
