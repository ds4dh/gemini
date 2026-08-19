# GEMINI: Clinical text variable extraction pipeline

`gemini` is a flexible, modular, high-performance pipeline for extracting structured clinical variables from free-form medical text (such as discharge summaries, consultation notes, and imaging reports) using Large Language Models (LLMs) to build databases for digital twin models.

---

## Key features

- **Modular configuration architecture (`configs/`)**: Separates run execution settings (`configs/run_cfg.yaml`) from extraction schemas and prompts (`configs/extraction_cfgs/*.yaml`). Configuration snapshots are automatically saved to output directories for exact reproducibility.
- **Integrated and hard reasoning control**: Configurable reasoning (`enabled`, `effort`, `preserve_thinking`) with an optional `hard_thinking_token_budget` that forces a `</think>` token when token limits are met.
- **XGrammar constrained decoding and guided JSON generation**: Custom vLLM logits processor (`ThinkingJSONAdapterProcessor`) enforcing two-phase generation: freeform/hard-capped reasoning (`THINKING`) followed by strict logit-level JSON Schema mask (`FINAL_JSON`).
- **Multi-stage resilient parsing and regex fallback**: Automatic parsing order from post-think blocks (`</think>`), markdown fences, stacked JSON structures, repaired truncated JSONs, down to field-by-field regex fallback.
- **Expanded multi-backend inference**: Supports `vllm-serve-async` (high-throughput async client with request/sequence concurrency), `vllm`, `vllm-serve`, `llama-cpp` (GGUF with auto-downloading and shard merging), and `mock` (offline testing).
- **Modern quantization formats**: Built-in support for NVFP4 (NVIDIA FP4), AWQ (`awq_marlin`), FP8, GPTQ, and GGUF models.
- **Stateful resuming and incremental checkpointing**: Writes progress periodically (`save_chunk_size`) to `_chunks.csv` and supports `resume_previous_run: true` to skip already extracted records.
- **Cross-platform and HPC support**: Automated CUDA DLL path setup on Windows and network socket interface detection (Gloo/NCCL) on Slurm/Apptainer HPC nodes.

---

## Installation and environment setup

### 1. Local development setup (`uv`)

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

#### Create virtual environment and install dependencies
```bash
# Create and activate virtual environment
uv venv

# PowerShell (Windows): .venv\Scripts\Activate.ps1
# Linux / macOS:        source .venv/bin/activate

# Install dependencies in editable mode
uv pip install -e .
```

*Note for native Windows vLLM installation*:
Download the matching `.whl` wheel from [SystemPanic/vllm-windows Releases](https://github.com/SystemPanic/vllm-windows/releases) and install using:
```powershell
uv pip install path/to/vllm-0.26.0+cu132-cp312-cp312-win_amd64.whl --extra-index-url https://download.pytorch.org/whl/cu130
```

---

### 2. HPC cluster setup (Apptainer / Singularity)

To build the container image on Slurm HPC clusters:
```bash
sbatch research-env.sbatch
```

---

## Pipeline architecture and configuration

Configuration is decoupled into **run execution configuration** and **target extraction configuration**.

### 1. Run execution configuration (`configs/run_cfg.yaml`)
Controls backend options, model parameters, reasoning settings, and output options:

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
  reasoning_parser: auto # auto | qwen3 | deepseek_r1 | granite | hunyuan | null
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

### 2. Target extraction configuration (`configs/extraction_cfgs/*.yaml`)
Defines the Pydantic schema, prompt templates, and extraction guidelines.

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

## Usage guide

### 1. Running the pipeline (`scripts/run_pipeline.py`)

#### Offline mock run (logic and schema validation)
```bash
python scripts/run_pipeline.py --backend mock
```

#### Standard execution (default YAML configs)
```bash
python scripts/run_pipeline.py
```

#### Custom config and model execution
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

### 2. Dynamic CLI arguments reference

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

### 3. Resuming interrupted runs

Set `resume_previous_run: true` in `configs/run_cfg.yaml`. The pipeline scans the output folder for existing `_chunks.csv` files, tracks completed IDs, and resumes processing remaining records.

---

### 4. Running experiment batches

- **Local execution script**: `./experiments/experiment_curated.sh`
- **HPC Slurm execution**: `sbatch scripts/run_benchmark.sh`

---

## Outputs and reporting structure

Results are saved inside timestamped subfolders under the output directory (`results/run_YYYYMMDD_HHMMSS/`):

```
results/run_20260819_120000/
├── run_cfg.yaml                     # Snapshot of run execution config
├── extraction_cfg.yaml              # Snapshot of target schema and prompt config
├── detailed_clinical_database.csv   # Row-by-row extraction database with raw outputs and traces
├── summary_clinical_database.csv    # Consolidated consensus values across repeats
├── extraction_report.json           # Computational metrics, token counts, and accuracy statistics
└── report.md                        # Formatted Markdown report summary
```

---

## Synthetic dataset generation

Generate synthetic clinical notes with ground-truth labels for development and testing:

```bash
python scripts/generate_synthetic_data.py
```
Outputs `data/synthetic_clinical_notes.csv`.

---

## Repository structure

```
gemini/
├── configs/
│   ├── run_cfg.yaml                   # Model execution, backend, hardware, and reasoning settings
│   └── extraction_cfgs/
│       ├── mrs_score.yaml             # Single-variable mRS score extraction schema and prompt
│       └── clinical_variables.yaml    # Multi-variable (mRS, smoking, aneurysm) schema and prompt
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
