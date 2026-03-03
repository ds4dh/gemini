#!/bin/bash

#SBATCH --partition=shared-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=build/logs/%x_%j.out
#SBATCH --error=build/logs/%x_%j.err
#SBATCH --job-name=build_vllm_container

# Apptainer file paths
WORK_DIR=$(pwd)
DEF_FILE_PATH="${WORK_DIR}/research-env.def"
SIF_FILE_PATH="${WORK_DIR}/research-env.sif"
BUILD_TMP_DIR="${WORK_DIR}/build/tmp"
BUILD_CACHE_DIR="${WORK_DIR}/build/cache"

# Local environment extraction to host after build
LOCAL_ENV_EXTRACTION="true"
HOST_VENV_TARGET="${WORK_DIR}/.venv"

# Remove existing .sif file if it exists
if [ -f "$SIF_FILE_PATH" ]; then
    rm "$SIF_FILE_PATH"
    echo "Removed existing $SIF_FILE_PATH"
fi

# Create directories
mkdir -p "$BUILD_TMP_DIR"
mkdir -p "$BUILD_CACHE_DIR"
mkdir -p "${WORK_DIR}/build/logs"

# Export the variables so Apptainer sees them
export APPTAINER_TMPDIR="$BUILD_TMP_DIR"
export APPTAINER_CACHEDIR="$BUILD_CACHE_DIR"

# Build the apptainer image
echo "Building Apptainer image..."
echo "Using Temp Dir: $APPTAINER_TMPDIR"
echo "Using Def File: $DEF_FILE_PATH"

# Note: VLLM compilation is heavy, so 16 CPUs and 64G RAM are requested above
apptainer build --fakeroot "$SIF_FILE_PATH" "$DEF_FILE_PATH"

# Verify if the build was successful
if [ $? -eq 0 ] && [ -f "$SIF_FILE_PATH" ]; then
    echo "SUCCESS: Apptainer image built at $SIF_FILE_PATH."
else
    echo "ERROR: Failed to build Apptainer image."
    exit 1
fi

# Setup local venv extraction
if [ "$LOCAL_ENV_EXTRACTION" = "true" ]; then
    echo "Starting extraction of /opt/venv to host..."
    PARENT_DIR=$(dirname "$HOST_VENV_TARGET")
    TARGET_NAME=$(basename "$HOST_VENV_TARGET")
    mkdir -p "$(dirname "$HOST_VENV_TARGET")"
    if [ -d "$HOST_VENV_TARGET" ]; then
        echo "removing old venv at $HOST_VENV_TARGET"
        rm -rf "$HOST_VENV_TARGET"
    fi

    # Attempt to copy the venv from the apptainer image to host
    apptainer exec \
        --bind "$PARENT_DIR:/mnt_out" \
        "$SIF_FILE_PATH" \
        cp -r /opt/venv "/mnt_out/$TARGET_NAME"
    if [ $? -eq 0 ]; then
        echo "Apptainer virtual environment extracted to: $HOST_VENV_TARGET"
    else
        echo "Failed to extract apptainer virual environment."
        exit 1
    fi
else
    echo "Skipping local venv extraction."
fi