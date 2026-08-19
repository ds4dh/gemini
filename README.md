# GEMINI: Clinical Text Variable Extraction Pipeline

`gemini` is a flexible, modular, high-performance data extraction pipeline engineered to extract structured clinical variables from free-form medical text (such as discharge summaries, consultation notes, and imaging reports) using Large Language Models (LLMs) to build databases for digital twin models.

---

## Key Features

- **Modular Configuration Architecture (`configs/`)**:
  - Separate run execution parameters (`configs/run_cfg.yaml`) from extraction schemas & prompt templates (`configs/extraction_cfgs/*.yaml`).
  - Supports unified config loading or dynamic CLI overrides for run/extraction parameters.
  - Automatically snapshots configuration files (`run_cfg.yaml`, `extraction_cfg.yaml`) into timestamped output folders (`results/run_YYYYMMDD_HHMMSS/`) for exact experimental reproducibility.

- **Integrated & Hard Reasoning Control**:
  - Fine-grained reasoning configuration (`model.reasoning`): `enabled` (`auto` | `true` | `false`), `effort` (`auto` | `off` | `low` | `medium` | `high` | `xhigh`), and `preserve_thinking` toggle for multi-turn conversations.
  - **Hard Thinking Token Budget**: Enforces custom hard caps on thought generation (`hard_thinking_token_budget`), automatically injecting thought-ending tokens (`</think>`) when the budget limit is reached.
  - **Dynamic Prompt Context**: Adapts system prompt instructions based on reasoning capabilities (`context_data_nothinking`).

- **XGrammar Constrained Decoding & Guided JSON Generation**:
  - Integrated stateful custom vLLM logits processor (`ThinkingJSONAdapterProcessor`) using **XGrammar**.
  - Seamlessly orchestrates generation phases:
    1. **`THINKING` Phase**: Freeform or hard-capped reasoning generation.
    2. **`FINAL_JSON` Phase**: Enforces strict JSON Schema compliance at the logit level via next-token bitmasking once reasoning completes.

- **Multi-Stage Resilient Parsing & Regex Fallback**:
  - Multi-stage extraction engine (`extract_structured_output`):
    1. Post-think block parsing (extracts JSON target following `</think>`, `</thought>`, or `</reasoning>`).
    2. Code fence extraction (` ```json ... ``` `).
    3. Balanced stack-based JSON object/array matching.
    4. Truncated JSON auto-repair.
    5. Type-aware field-by-field regex extraction fallback.

- **Expanded Multi-Backend Inference**:
  - **`vllm-serve-async`**: High-throughput asynchronous vLLM OpenAI API client supporting multi-request concurrency (`max_concurrent_requests`) and server sequence handling (`max_concurrent_inferences` / `--max-num-seqs`).
  - **`vllm` / `vllm-serve`**: Synchronous direct Python or server-based GPU inference.
  - **`llama-cpp`**: GGUF model execution with automated multi-shard Hugging Face downloading (`download_gguf_by_quant`) and automatic `gguf-split` merging.
  - **`mock`**: Instant offline execution mode for verifying pipeline logic, schema compilation, and prompt formatting without GPU resources.

- **Modern Quantization Formats**:
  - Native support for **NVFP4** (NVIDIA FP4 quantization, e.g., `unsloth/Qwen3.8-27B-NVFP4`), **AWQ** (`awq_marlin`), **FP8**, **GPTQ**, and **GGUF** models.

- **Stateful Resuming & Incremental Checkpointing**:
  - Incremental chunk saving (`save_chunk_size`) writes progress periodically to `_chunks.csv`.
  - **Run Continuation**: Enables `resume_previous_run: true` to detect interrupted runs, filtering out already processed patient/record IDs to resume seamlessly without duplicate processing.

- **Cross-Platform & HPC Support**:
  - Dynamic Windows CUDA DLL path configuration (`CUDA_PATH`, `CUDA_LIB_PATH`) and DLL search directory registration.
  - Automated network interface detection for PyTorch Distributed / vLLM on Linux HPC nodes (Gloo/NCCL socket interface binding).

---

## Installation & Environment Setup

### 1. Local Development Setup (`uv`)

Dependency management and virtual environment isolation are handled via `uv`.

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

*Note for Windows vLLM Native Installation*:
To run `vllm` natively on Windows, download the matching `.whl` binary wheel from [SystemPanic/vllm-windows Releases](https://github.com/SystemPanic/vllm-windows/releases) and install it using:
```powershell
uv pip install path/to/downloaded/vllm-0.26.0+cu132-cp312-cp312-win_amd64.whl --extra-index-url https://download.pytorch.org/whl/cu130
```

---

### 2. HPC Cluster Setup (Apptainer / Singularity)

For execution on HPC environments (e.g., Slurm cluster nodes):

Submit the container build script to Slurm to generate `research-env.sif`:
```bash
sbatch research-env.sbatch
```

---

## Pipeline Architecture & Configuration

The configuration framework is divided into **Run Execution Configuration** and **Extraction Configuration**.

### 1. Run Execution Configuration (`configs/run_cfg.yaml`)
Controls hardware, server options, model parameters, reasoning limits, and output destinations:

```yaml
data:
  extraction_config_path: "configs/extraction_cfgs/mrs_score.yaml"
  input_path: "data/synthetic_clinical_notes.csv"
  input_text_column: "input_text"
  id_column: "patient_id"
  max_samples: null
  remove_samples_without_label: false

output:
  result_dir: "./results"
  use_timestamp_subfolder: true
  detailed_filename: "detailed_clinical_database.csv"
  summary_filename: "summary_clinical_database.csv"
  report_filename: "extraction_report.json"
  save_chunk_size: 100
  resume_previous_run: true

model:
  model_path: "unsloth/Qwen3.8-27B-NVFP4"
  inference_backend: "vllm-serve-async" # vllm | vllm-serve | vllm-serve-async | llama-cpp | mock
  n_inference_repeats: 10
  max_concurrent_requests: 3
  max_concurrent_inferences: 30
  gpu_memory_utilization: 0.90
  enforce_eager: true

  # Integrated reasoning control
  reasoning:
    enabled: auto # true | false | auto
    effort: auto # auto | off | low | medium | high | xhigh
    preserve_thinking: false
    hard_thinking_token_budget: 768

  use_output_guide: true
  logits_processors:
    - "src.models.logits_processors:ThinkingJSONAdapterProcessor"

  # Sampling parameters
  temperature: 1.0
  top_p: 0.95
  max_new_tokens: 1024
  max_context_length: 5000
```

---

### 2. Target Extraction Configuration (`configs/extraction_cfgs/*.yaml`)
Defines the runtime Pydantic schema, prompt templates, and domain guidelines.

Example schema (`configs/extraction_cfgs/clinical_variables.yaml`):

```yaml
schema:
  name: "MultiVariableClinicalExtractionSchema"
  fields:
    mRS:
      type: "int"
      description: "Modified Rankin Scale score (0 to 6, or null)"
      ge: 0
      le: 6
      default: null

    smoking_status:
      type: "enum"
      description: "Patient smoking habit classification"
      enum_values: ["Smoker", "Non-smoker", "Former-smoker", "Unknown"]
      default: "Unknown"

    aneurysm_size_mm:
      type: "float"
      description: "Maximal intracranial aneurysm diameter in mm, or null"
      default: null

prompt:
  system_template: |
    {task_description}
    {domain_knowledge}
    {output_specifications}

  user_template: |
    Voici le texte d'entrée:
    DEBUT DU TEXTE:
    {input_text}
    FIN DU TEXTE
```

---

## Usage Guide

### 1. Running the Pipeline (`scripts/run_pipeline.py`)

Run the pipeline using configuration files or command-line arguments:

#### Offline Mock Run (Logic & Schema Validation)
```bash
python scripts/run_pipeline.py --backend mock
```

#### Standard Execution (Default YAML Configs)
```bash
python scripts/run_pipeline.py
```

#### Custom Config & Model Execution
```bash
python scripts/run_pipeline.py \
    --run-config configs/run_cfg.yaml \
    --extraction-config configs/extraction_cfgs/clinical_variables.yaml \
    --model "unsloth/Qwen3.8-27B-NVFP4" \
    --backend "vllm-serve-async" \
    --input-path "data/synthetic_clinical_notes.csv" \
    --output-dir "./results"
```

---

### 2. Dynamic CLI Arguments Reference

| Parameter | Short | Description |
| :--- | :--- | :--- |
| `--run-config` | `-rc` | Path to run configuration YAML file (default: `configs/run_cfg.yaml`) |
| `--extraction-config` | `-ec` | Path to extraction task YAML file |
| `--config` | `-c` | Path to single unified configuration file |
| `--input-path` | `-i` | Override path to input dataset CSV/Excel |
| `--curated-data-path` | `-cd` | Override path to non-encrypted dataset CSV |
| `--backend` | `-b` | Inference backend (`vllm`, `vllm-serve-async`, `vllm-serve`, `llama-cpp`, `mock`) |
| `--model` | `-m` | Hugging Face model repository ID or local path |
| `--quant-scheme` | `-qs` | GGUF quantization scheme override (e.g., `Q6_K_XL`, `Q8_0`) |
| `--gpu-memory-utilization` | `-gpu` | GPU memory utilization fraction (`0.50` - `0.95`) |
| `--reasoning-parser` | `-rp` | Reasoning parser specification (e.g., `qwen3`, `deepseek_r1`) |
| `--output-dir` | `-o` | Output directory path for results |
| `--save-chunk-size` | | Number of records processed before writing incremental CSV checkpoints |

---

### 3. Resuming Interrupted Runs

To resume an incomplete extraction run, set `resume_previous_run: true` in `configs/run_cfg.yaml`.
The pipeline automatically scans the output directory for recent `run_*` folders containing `_chunks.csv` files, identifies previously extracted patient/record IDs, and processes only the remaining unprocessed entries.

---

### 4. Running Experiment Batches

- **Local Execution Script**:
  ```bash
  ./experiments/experiment_curated.sh
  ```
- **HPC Slurm Execution**:
  ```bash
  sbatch scripts/run_benchmark.sh
  ```

---

## Outputs & Reporting Structure

Execution outputs are saved inside timestamped subfolders under the designated output directory (`results/run_YYYYMMDD_HHMMSS/`):

```
results/run_20260819_120000/
├── run_cfg.yaml                     # Snapshot of run execution config
├── extraction_cfg.yaml              # Snapshot of target schema & prompt config
├── detailed_clinical_database.csv   # Comprehensive row-by-row extraction database
├── summary_clinical_database.csv    # Consolidated consensus values across repeats
├── extraction_report.json           # Computational metrics, token counts, and accuracy stats
└── report.md                        # Formatted Markdown report summary
```

- **Detailed Database (`detailed_clinical_database.csv`)**: Contains individual output iterations, raw text generation, reasoning thinking trace, parsed JSON fields, and validation status per repeat.
- **Summary Database (`summary_clinical_database.csv`)**: Contains aggregated consensus values computed across inference repeats (`n_inference_repeats`).

---

## Synthetic Dataset Generation

Generate synthetic clinical notes with ground-truth labels for offline development and validation:

```bash
python scripts/generate_synthetic_data.py
```
Outputs `data/synthetic_clinical_notes.csv`.

---

## Repository Structure

```
gemini/
├── configs/
│   ├── run_cfg.yaml                   # Model execution, backend, hardware, and reasoning settings
│   └── extraction_cfgs/
│       ├── mrs_score.yaml             # Single-variable mRS score extraction schema & prompt
│       └── clinical_variables.yaml    # Multi-variable (mRS, smoking, aneurysm) schema & prompt
├── pyproject.toml                     # Project packaging and dependency specs
├── data/                              # Datasets and synthetic test records
│   └── synthetic_clinical_notes.csv
├── results/                           # Timestamped experiment output directories
├── src/
│   ├── data/                          # Data loading, dynamic schema generation, prompting, parsing
│   ├── models/                        # LLM loader, logits processors, reasoning engine, evaluation
│   └── utils/                         # Config normalization, GPU environment setup, plot utilities
├── scripts/
│   ├── run_pipeline.py                # Main CLI execution entrypoint
│   ├── generate_synthetic_data.py     # Synthetic clinical text generator
│   ├── create_dataset.py              # Dataset formatting and encryption utility
│   ├── plot_figures.py                # Benchmark visualization script
│   └── run_benchmark.sh               # Slurm runner script
└── experiments/                       # Shell scripts for batch model experiments
```
