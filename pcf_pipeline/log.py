"""Timestamped logging utilities for the PCF pipeline."""

import time
from datetime import datetime


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(level: str, msg: str) -> None:
    """Print a timestamped log message with the given level prefix."""
    print(f"[{level}] {_timestamp()} -- {msg}", flush=True)


def info(msg: str) -> None:
    log("INFO", msg)


def warning(msg: str) -> None:
    log("WARNING", msg)


def error(msg: str) -> None:
    log("ERROR", msg)


def log_step_start(step_name: str) -> float:
    """Log the start of a pipeline step. Returns the start time."""
    info(f"Starting {step_name}")
    return time.time()


def log_step_end(step_name: str, start_time: float) -> None:
    """Log the completion of a pipeline step with elapsed time."""
    elapsed = time.time() - start_time
    if elapsed >= 60:
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        info(f"Completed {step_name} in {minutes}m {seconds:.1f}s")
    else:
        info(f"Completed {step_name} in {elapsed:.1f}s")


def log_progress(item: str, index: int, total: int) -> None:
    """Log iteration progress for long-running loops."""
    info(f"Processing {item} ({index} / {total})")
