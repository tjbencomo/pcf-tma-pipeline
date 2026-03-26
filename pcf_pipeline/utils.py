"""Shared utility functions for the PCF pipeline."""

import xml.etree.ElementTree as ET
from pathlib import Path

import tifffile

from pcf_pipeline import log


def get_channel_info(image_path: Path) -> tuple[list[str], list[int], int, int]:
    """Read channel names and full-resolution page indices from a qptiff.

    Parses the XML metadata in each TIFF page to find FullResolution pages
    and extract Biomarker names.

    Returns
    -------
    channel_names : list[str]
        Original channel names (may contain spaces/special chars).
    page_indices : list[int]
        Indices of full-resolution pages in the TIFF.
    img_height : int
        Height of the full-resolution image in pixels.
    img_width : int
        Width of the full-resolution image in pixels.
    """
    log.info(f"Reading channel info from: {image_path}")
    channel_names = []
    page_indices = []
    img_height = 0
    img_width = 0

    with tifffile.TiffFile(image_path) as tif:
        for idx, page in enumerate(tif.pages):
            if not page.description:
                continue
            try:
                root = ET.fromstring(page.description)
            except ET.ParseError:
                continue
            image_type = root.find("ImageType")
            if image_type is None or image_type.text != "FullResolution":
                continue
            biomarker = root.find("Biomarker")
            if biomarker is not None and biomarker.text:
                channel_names.append(biomarker.text)
                page_indices.append(idx)
                if img_height == 0:
                    img_height, img_width = page.shape[:2]

    log.info(f"Found {len(channel_names)} channels, image size: {img_height} x {img_width}")
    return channel_names, page_indices, img_height, img_width


def get_channel_names_safe(channel_names: list[str]) -> list[str]:
    """Convert channel names to filesystem-safe versions.

    Replaces spaces with underscores and slashes with hyphens.
    """
    return [name.replace(" ", "_").replace("/", "-") for name in channel_names]


def check_output_exists(path: Path, skip_completed: bool) -> bool:
    """Check if an output file exists and should be skipped.

    Returns True if the file exists and skip_completed is enabled.
    """
    if skip_completed and path.exists():
        log.info(f"Skipping (output exists): {path}")
        return True
    return False
