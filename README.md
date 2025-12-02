# Structured data extraction from clinical text

This project uses generative Large Language Models (LLMs) with `vllm` to extract medical concepts from clinical text.

## Setup and installation

You can set up the environment locally for development or build a container for reproducible execution on an HPC cluster.

### Local development setup

For local development, we use `uv` for fast dependency management, reading configuration directly from `pyproject.toml`.

1.  **Install `uv`**

    If you don't have `uv` installed, you can install it with:
    ```bash
    # On macOS and Linux
    curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
    ```

2.  **Create a virtual environment and install dependencies**

    From the root of the project, run the following commands to create the environment and install the project in "editable" mode (changes to code reflect immediately):
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -e .
    ```
    *Note: This installs dependencies defined in `pyproject.toml`.*

### HPC setup using Apptainer

For running experiments on an HPC cluster, an Apptainer (formerly Singularity) container is used to ensure a consistent and reproducible environment.

1.  **Prerequisites**

    -   Access to an HPC cluster with Apptainer/Singularity installed.
    -   A SLURM workload manager.

2.  **Build the Container image**

    The `research-env.def` file defines the container environment. It sets up a virtual environment and installs the project and dependencies via `pyproject.toml`.

    To build the image, submit the build script to the SLURM scheduler:

    ```bash
    sbatch research-env.sbatch
    ```

    This script will:
    -   Build the Apptainer image (e.g., `research-env.sif`) from `research-env.def`.
    -   Store build logs in `build/logs/`.

## Usage

Once the setup is complete, you can run your experiments.

### Running experiments locally

Make sure your virtual environment is activated:
```bash
source .venv/bin/activate
./experiments/experiment_1_desktop.sh  # check parameters in this file first
```

### Running experiments on HPC

To run the experiment from an HPC, use the following script:
```bash
./experiments/experiment_1.sh  # after checking the parameters in this file
```
