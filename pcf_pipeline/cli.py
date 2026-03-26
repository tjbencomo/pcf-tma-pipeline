"""Command-line interface for the PCF pipeline."""

import argparse
import sys
from pathlib import Path

from pcf_pipeline import __version__, log
from pcf_pipeline.config import PipelineConfig, load_config


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the pipeline YAML config file.",
    )


def cmd_run(args: argparse.Namespace) -> None:
    """Run all pipeline steps end-to-end."""
    config = load_config(args.config)
    steps = config.pipeline.steps

    step_funcs = {
        1: ("Step 1: Create segmentation masks", _run_step1),
        2: ("Step 2: Split TMA into per-core images", _run_step2),
        3: ("Step 3: Create FOVs per core", _run_step3),
        4: ("Step 4: Run Nimbus inference", _run_step4),
        5: ("Step 5: Merge into AnnData", _run_step5),
    }

    t0 = log.log_step_start("pipeline")
    for step_num in steps:
        if step_num not in step_funcs:
            log.warning(f"Unknown step number: {step_num}, skipping")
            continue
        name, func = step_funcs[step_num]
        func(config)
    log.log_step_end("pipeline", t0)


def _run_step1(config: PipelineConfig) -> None:
    from pcf_pipeline.steps.step1_masks import run_step1

    cell_mask, nuc_mask, id_mapping = run_step1(config)
    # If masks were auto-generated, update config so downstream steps can find them
    if not config.masks.cell_mask_path:
        config.masks.cell_mask_path = cell_mask
    if not config.masks.nucleus_mask_path:
        config.masks.nucleus_mask_path = nuc_mask
    if not config.masks.id_mapping_path:
        config.masks.id_mapping_path = id_mapping


def _run_step2(config: PipelineConfig) -> None:
    from pcf_pipeline.steps.step2_cores import run_step2

    run_step2(config)


def _run_step3(config: PipelineConfig) -> None:
    from pcf_pipeline.steps.step3_fovs import run_step3

    run_step3(config)


def _run_step4(config: PipelineConfig) -> None:
    from pcf_pipeline.steps.step4_nimbus import run_step4

    run_step4(config)


def _run_step5(config: PipelineConfig) -> None:
    from pcf_pipeline.steps.step5_merge import run_step5

    run_step5(config)


def cmd_masks(args: argparse.Namespace) -> None:
    """Run step 1: create segmentation masks."""
    config = load_config(args.config)
    _run_step1(config)


def cmd_cores(args: argparse.Namespace) -> None:
    """Run step 2: split TMA into per-core images."""
    config = load_config(args.config)
    _run_step2(config)


def cmd_fovs(args: argparse.Namespace) -> None:
    """Run step 3: create FOVs per core."""
    config = load_config(args.config)
    _run_step3(config)


def cmd_nimbus(args: argparse.Namespace) -> None:
    """Run step 4: Nimbus inference."""
    config = load_config(args.config)
    _run_step4(config)


def cmd_merge(args: argparse.Namespace) -> None:
    """Run step 5: merge into AnnData."""
    config = load_config(args.config)
    _run_step5(config)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pcf-pipeline",
        description="Phenocycler TMA data processing pipeline with Nimbus inference.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Pipeline commands")

    # run — all steps
    p_run = subparsers.add_parser("run", help="Run the full pipeline end-to-end.")
    _add_config_arg(p_run)
    p_run.set_defaults(func=cmd_run)

    # Individual step subcommands
    for name, help_text, func in [
        ("masks", "Step 1: Create segmentation masks from cell geojson.", cmd_masks),
        ("cores", "Step 2: Split TMA image into per-core images.", cmd_cores),
        ("fovs", "Step 3: Create FOVs for each core.", cmd_fovs),
        ("nimbus", "Step 4: Run Nimbus inference on each core.", cmd_nimbus),
        ("merge", "Step 5: Merge results into AnnData.", cmd_merge),
    ]:
        p = subparsers.add_parser(name, help=help_text)
        _add_config_arg(p)
        p.set_defaults(func=func)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)
