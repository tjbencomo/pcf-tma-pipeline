#!/bin/bash
# SLURM job submission script for the PCF pipeline.
#
# Usage:
#   sbatch scripts/run_pipeline.sh /path/to/config.yaml
#
# The config.yaml controls all pipeline parameters and paths.
# See config.example.yaml for a template.

#SBATCH --job-name=pcf-pipeline
#SBATCH --output=logs/pcf-pipeline_%j.out
#SBATCH --error=logs/pcf-pipeline_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gpus=1

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch scripts/run_pipeline.sh <config.yaml>"
    exit 1
fi

CONFIG="$1"

if [[ ! -f "$CONFIG" ]]; then
    echo "[ERROR] Config file not found: $CONFIG"
    exit 1
fi

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

ml CUDA

# Ensure log directory exists
mkdir -p logs

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') -- Starting PCF pipeline"
echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') -- Config: $CONFIG"
echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') -- Job ID: $SLURM_JOB_ID"
echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') -- Node: $SLURMD_NODENAME"
echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') -- CPUs: $SLURM_CPUS_PER_TASK"

micromamba run -n pcf-pipeline pcf-pipeline run --config "$CONFIG"

echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') -- Pipeline complete"
