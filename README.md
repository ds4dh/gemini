# GEMINI: Clinical Text Variable Extraction Pipeline

`gemini` is a flexible, high-performance data extraction pipeline built to extract structured clinical variables from free-form medical text (such as discharge letters, consultation notes, and imaging reports) to construct databases for digital twin models.

---

## Features

- **Modular Configuration (`configs/`)**: Manage model execution parameters (`configs/run_cfg.yaml`) and target extraction schemas & prompts (`configs/extraction_cfgs/*.yaml`) cleanly from separate files.
- **Dynamic Schema Engine**: Automatically generates Pydantic validation schemas at runtime based on field specifications (e.g., Modified Rankin Scale score `mRS`, `smoking_status`, `aneurysm_size_mm`, or custom clinical parameters).
- **Multi-Backend Inference**:
  - `vllm` / `vllm-serve-async`: High-throughput GPU inference engine for desktop and HPC environments.
  - `llama-cpp`: GGUF format model execution via llama.cpp.
  - `mock`: Instant offline execution mode for verifying pipeline logic and prompt structure.
- **Cross-Platform Compatibility**: Fully supported on Windows desktop environments and Linux HPC clusters with Apptainer/Singularity container support.

---

## Installation & Environment Setup

### 1. Local Development Setup (`uv`)

Dependency management and environment isolation are handled via `uv`.

#### Install `uv`
- **Windows (PowerShell)**:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **Linux / macOS**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

#### Create Virtual Environment & Install Dependencies
Run the following commands from the repository root:
```bash
# Create virtual environment
uv venv

# Activate environment:
# PowerShell (Windows): .venv\Scripts\Activate.ps1
# Command Prompt (cmd): .venv\Scripts\activate.bat
# Linux / macOS:        source .venv/bin/activate

# Install dependencies in editable mode
uv pip install -e .
```

*Note for Windows vLLM installation*:
To run `vllm` natively on Windows, download the matching `.whl` binary wheel from [SystemPanic/vllm-windows Releases](https://github.com/SystemPanic/vllm-windows/releases) and install it using:
```powershell
uv pip install path/to/downloaded/vllm-0.26.0+cu132-cp312-cp312-win_amd64.whl --extra-index-url https://download.pytorch.org/whl/cu130
```

---

### 2. HPC Cluster Setup (Apptainer / Singularity)

For execution on HPC environments (e.g., Baobab / Bamboo with Slurm):

Submit the container build script to Slurm to generate `research-env.sif`:
```bash
sbatch research-env.sbatch
```

---

## Usage Guide

### 1. Pipeline Execution (`scripts/run_pipeline.py`)

Run the pipeline using the configuration files in `configs/`:

#### Offline Test Run (Mock Backend)
```bash
python scripts/run_pipeline.py --backend mock
```

#### High-Throughput GPU Inference (vLLM Backend)
```bash
# Uses default run_cfg.yaml and extraction config path specified inside run_cfg.yaml (under data section)
python scripts/run_pipeline.py

# Or specify custom run and extraction configs via CLI
python scripts/run_pipeline.py --run-config configs/run_cfg.yaml --extraction-config configs/extraction_cfgs/clinical_variables.yaml
```

---

### 2. Target Variable Configuration

Default run configurations are located in `configs/run_cfg.yaml`. The path to the default extraction configuration is specified under the `data` section in `run_cfg.yaml`:

```yaml
data:
  extraction_config_path: "configs/extraction_cfgs/mrs_score.yaml"
```

Target clinical fields are defined under `schema.fields` in extraction YAML files located in `configs/extraction_cfgs/` (such as `mrs_score.yaml` or `clinical_variables.yaml`):

```yaml
schema:
  name: "MultiVariableClinicalExtractionSchema"
  fields:
    mRS:
      type: "int"
      description: "Modified Rankin Scale score (0 to 6, or null if unmentioned)"
      ge: 0
      le: 6
      default: null

    smoking_status:
      type: "enum"
      description: "Patient smoking status"
      enum_values: ["Smoker", "Non-smoker", "Former-smoker", "Unknown"]
      default: "Unknown"

    aneurysm_size_mm:
      type: "float"
      description: "Maximal intracranial aneurysm diameter in mm, or null if none"
      default: null
```

---

### 3. Dynamic CLI Overrides

Override configuration parameters via command-line arguments without modifying YAML files:

```bash
python scripts/run_pipeline.py \
    --run-config configs/run_cfg.yaml \
    --extraction-config configs/extraction_cfgs/clinical_variables.yaml \
    --model "cyankiwi/Qwen3.6-27B-AWQ-INT4" \
    --backend "vllm-serve-async" \
    --gpu-memory-utilization 0.90 \
    --input-path "data/synthetic_clinical_notes.csv" \
    --output-dir "./results"
```

---

### 4. Running Multi-Model Experiment Batches

- **Local Batch Runs**:
  ```bash
  ./experiments/experiment_curated.sh
  ```
- **HPC Slurm Batch Runs**:
  ```bash
  ./experiments/experiment_1.sh
  ```

---

## Synthetic Test Data Generation

Generate a synthetic clinical dataset (containing sample clinical notes and ground truth labels) for offline testing:

```bash
python scripts/generate_synthetic_data.py
```
Outputs `data/synthetic_clinical_notes.csv`.

---

## Repository Structure

```
gemini/
├── configs/
│   ├── run_cfg.yaml                   # Model, dataset, backend, and output paths
│   └── extraction_cfgs/
│       ├── mrs_score.yaml             # Single-variable mRS score extraction schema & prompt
│       └── clinical_variables.yaml    # Multi-variable (mRS, smoking, aneurysm) schema & prompt
├── pyproject.toml                     # Project dependencies and packaging metadata
├── data/                              # Input datasets and synthetic clinical notes
│   └── synthetic_clinical_notes.csv
├── results/                           # Output extracted CSV databases
├── src/
│   ├── data/                          # Schema engine, prompting, data loading
│   ├── models/                        # LLM loaders and inference backends
│   └── utils/                         # Environment and configuration utilities
├── scripts/
│   ├── run_pipeline.py                # Primary CLI entrypoint script
│   ├── generate_synthetic_data.py     # Synthetic dataset generator
│   ├── create_dataset.py              # Dataset formatting and encryption script
│   ├── plot_figures.py                # Benchmark figures plotting script
│   └── run_benchmark.sh               # Slurm submission runner script
└── experiments/                       # Experiment execution scripts
```
