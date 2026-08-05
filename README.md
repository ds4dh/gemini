# GEMINI: Clinical Text Variable Extraction Pipeline

`gemini` is a flexible, high-performance data extraction pipeline built to extract structured clinical variables from free-form medical text (such as discharge letters, consultation notes, and imaging reports) to construct databases for digital twin models.

---

## Features

- **Single-File Configuration (`config.yaml`)**: Manage target extraction variables, prompt templates, model selection, inference backends, and output paths from a single configuration file.
- **Dynamic Schema Engine**: Automatically generates Pydantic validation schemas at runtime based on field specifications (e.g., Modified Rankin Scale score `mRS`, `smoking_status`, `aneurysm_size_mm`, or custom clinical parameters).
- **Multi-Backend Inference**:
  - `transformers`: Local PyTorch inference using HuggingFace models.
  - `vllm` / `vllm-serve-async`: High-throughput GPU inference engine for desktop and HPC environments.
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
Standard local execution on Windows uses the `transformers` or `mock` backends. To run `vllm` natively on Windows, download the matching `.whl` binary wheel from [SystemPanic/vllm-windows Releases](https://github.com/SystemPanic/vllm-windows/releases) and install it using:
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

### 1. Pipeline Execution (`run_pipeline.py`)

Run the pipeline using `config.yaml`:

#### Offline Test Run (Mock Backend)
```bash
python run_pipeline.py --config config.yaml --backend mock
```

#### Local Desktop Inference (HuggingFace Transformers Backend)
```bash
python run_pipeline.py --config config.yaml --backend transformers --model Qwen/Qwen2.5-0.5B-Instruct
```

#### High-Throughput GPU Inference (vLLM Backend)
```bash
python run_pipeline.py --config config.yaml --backend vllm
```

---

### 2. Target Variable Configuration

Target clinical fields are defined under `schema.fields` in `config.yaml`:

```yaml
schema:
  name: "ClinicalVariablesExtractionSchema"
  fields:
    mRS:
      type: "int"
      description: "Modified Rankin Scale score (0 to 6, or -1 if unmentioned)"
      ge: -1
      le: 6
      default: -1

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

Override configuration parameters via command-line arguments without modifying `config.yaml`:

```bash
python run_pipeline.py \
    --config config.yaml \
    --model "unsloth/Qwen3-8B-GGUF" \
    --quant-scheme "Q6_K_XL" \
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
├── config.yaml                # Primary configuration file
├── run_pipeline.py            # Primary CLI entrypoint script
├── pyproject.toml             # Project dependencies and packaging metadata
├── data/                      # Input datasets and synthetic clinical notes
│   └── synthetic_clinical_notes.csv
├── results/                   # Output extracted CSV databases
│   └── extracted_clinical_database.csv
├── src/
│   ├── data/                  # Schema engine, prompting, data loading
│   ├── models/                # LLM loaders and inference backends
│   └── utils/                 # Environment and configuration utilities
├── scripts/                   # Utility scripts (synthetic data, benchmarking)
└── experiments/               # Experiment execution scripts
```
