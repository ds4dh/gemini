import os
import gc
import shutil
import signal
import argparse
import datetime
import subprocess
from functools import partial
from typing import Any

import torch
import pandas as pd
from datasets import Dataset

from src.data.data_loading import load_data_formatted_for_benchmarking
from src.data.prompting import build_messages
from src.models.llm_evaluation import generate_extraction_summary_and_reports
from src.models.llm_inference import process_samples
from src.models.llm_loader import load_model
from src.utils.run_utils import load_config_files, set_distributed_environment


def main() -> None:
    """
    Main entrypoint for the clinical variable extraction pipeline
    """
    set_distributed_environment()

    # Load configuration from YAML files or CLI options
    args = parse_cli_args()
    cfg = load_config_files(args)

    # Apply CLI argument overrides
    if args.input_path:
        cfg["input_path"] = args.input_path
    if args.backend:
        cfg["inference_backend"] = args.backend
    if args.model:
        cfg["model_path"] = args.model
    if args.quant_scheme:
        cfg["quant_scheme"] = args.quant_scheme
    if args.gpu_memory_utilization:
        cfg["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.reasoning_parser:
        cfg["reasoning_parser"] = args.reasoning_parser
    if args.output_dir:
        cfg["result_dir"] = args.output_dir
    if args.save_chunk_size:
        cfg["save_chunk_size"] = args.save_chunk_size

    active_rc = cfg.get("_active_run_config_path", args.run_config)
    active_ec = cfg.get("_active_extraction_config_path", args.extraction_config or "configs/extraction_cfgs/mrs_score.yaml")

    print("================================================================")
    print(" GEMINI CLINICAL VARIABLE EXTRACTION PIPELINE")
    print("================================================================")
    print(f"Run Config:        {active_rc}")
    print(f"Extraction Config: {active_ec}")
    print(f"Model:             {cfg['model_path']}")
    print(f"Inference Backend: {cfg['inference_backend']}")
    print(f"Input Data Path:   {cfg.get('input_path', 'default')}")
    print(f"Result Directory:  {cfg['result_dir']}")
    print("================================================================\n")

    # Load dataset
    data_loading_kwargs = cfg.get("data_loading_arguments", {})
    if "input_path" in cfg:
        data_loading_kwargs["input_path"] = cfg["input_path"]
    if "input_text_column" in cfg:
        data_loading_kwargs["input_text_column"] = cfg["input_text_column"]

    dataset = load_data_formatted_for_benchmarking(args, **data_loading_kwargs)
    print(f"Loaded {len(dataset)} clinical text records for extraction.\n")

    # Load LLM model and execute pipeline
    model, server_process = None, None
    try:
        model, server_process = load_model(**cfg)
        run_model(model=model, dataset=dataset, args=args, cfg=cfg)

    except Exception as e:
        print(f"\nAn error occurred during pipeline execution: {e}")
        raise

    finally:
        cleanup_vllm_server(server_process)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def parse_cli_args() -> argparse.Namespace:
    """
    Parses command-line arguments for configuration overrides.
    """
    parser = argparse.ArgumentParser(
        description="Unified Clinical Variable Extraction Pipeline using LLMs."
    )
    parser.add_argument(
        "--run-config", "-rc",
        type=str,
        default="configs/run_cfg.yaml",
        help="Path to run configuration YAML file (default: configs/run_cfg.yaml)",
    )
    parser.add_argument(
        "--extraction-config", "-ec",
        type=str,
        default=None,
        help="Path to extraction configuration YAML file (overrides extraction_config_path in run_cfg.yaml)",
    )
    parser.add_argument(
        "--input-path", "-i",
        type=str,
        default=None,
        help="Override input dataset CSV/Excel/Parquet path",
    )
    parser.add_argument(
        "--backend", "-b",
        type=str,
        default=None,
        help="Override inference backend (vllm, vllm-serve-async, llama-cpp, mock)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Override model path or HuggingFace ID",
    )
    parser.add_argument(
        "--quant-scheme", "-qs",
        type=str,
        default=None,
        help="Override GGUF/quantization scheme (e.g., Q6_K_XL, Q8_0)",
    )
    parser.add_argument(
        "--gpu-memory-utilization", "-gpu",
        type=float,
        default=None,
        help="Override GPU memory utilization fraction (0.50 - 0.95)",
    )
    parser.add_argument(
        "--reasoning-parser", "-rp",
        type=str,
        default=None,
        help="Override reasoning parser (e.g., qwen3, deepseek_r1)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Override output directory for results CSV",
    )
    parser.add_argument(
        "--save-chunk-size",
        type=int,
        default=None,
        help="Override incremental CSV save chunk size",
    )

    return parser.parse_args()


def run_model(
    model: Any,
    dataset: Dataset,
    args: argparse.Namespace,
    cfg: dict[str, Any],
) -> None:
    """
    Constructs prompts, executes model inference incrementally, and saves detailed/summary outputs.
    Supports resuming previous runs if resume_previous_run is enabled.
    """
    print("Building prompt messages for LLM...")
    dataset = dataset.map(
        function=partial(build_messages, cfg=cfg),
        desc="Constructing chat prompts",
    )

    base_output_dir = cfg.get("result_dir", "./results")
    use_timestamp = cfg.get("use_timestamp_subfolder", True)
    resume_run = cfg.get("resume_previous_run", False)
    detailed_filename = cfg.get("output", {}).get("detailed_filename", "detailed_clinical_database.csv")
    if not detailed_filename.endswith(".csv"):
        detailed_filename = "detailed_clinical_database.csv"

    # Determine run output directory
    run_dir = None
    if use_timestamp:
        # Check if resuming and an existing run subfolder with progress exists
        if resume_run and os.path.exists(base_output_dir):
            existing_runs = sorted(
                [d for d in os.listdir(base_output_dir) if d.startswith("run_") and os.path.isdir(os.path.join(base_output_dir, d))],
                reverse=True,
            )
            for er in existing_runs:
                candidate_dir = os.path.join(base_output_dir, er)
                chunks_file = os.path.join(candidate_dir, detailed_filename.replace(".csv", "_chunks.csv"))
                detailed_file = os.path.join(candidate_dir, detailed_filename)
                if os.path.exists(chunks_file) or os.path.exists(detailed_file):
                    run_dir = candidate_dir
                    print(f"Resuming existing run folder: {run_dir}")
                    break

        if run_dir is None:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(base_output_dir, f"run_{timestamp_str}")
    else:
        run_dir = base_output_dir

    os.makedirs(run_dir, exist_ok=True)

    # Save configuration snapshots for reproducibility
    try:
        rc_dest = os.path.join(run_dir, "run_cfg.yaml")
        ec_dest = os.path.join(run_dir, "extraction_cfg.yaml")
        rc_src = cfg.get("_active_run_config_path", getattr(args, "run_config", "configs/run_cfg.yaml"))
        ec_src = cfg.get("_active_extraction_config_path", getattr(args, "extraction_config", "configs/extraction_cfgs/mrs_score.yaml"))

        if rc_src and os.path.exists(rc_src):
            shutil.copyfile(rc_src, rc_dest)
        if ec_src and os.path.exists(ec_src):
            shutil.copyfile(ec_src, ec_dest)

        print(f"Saved config snapshots at: {run_dir}")
    except Exception as e:
        print(f"Notice: Could not write config snapshot ({e})")

    detailed_csv_path = os.path.join(run_dir, detailed_filename)
    chunks_csv_path = detailed_csv_path.replace(".csv", "_chunks.csv")

    # Determine unique ID column to track already processed records
    id_col = cfg.get("id_column") or "patient_id"
    if id_col not in dataset.column_names:
        id_col = "input_text"

    # Handle resume logic vs fresh run
    processed_ids: set[Any] = set()
    if resume_run:
        target_resume_file = chunks_csv_path if os.path.exists(chunks_csv_path) else (detailed_csv_path if os.path.exists(detailed_csv_path) else None)
        if target_resume_file:
            try:
                df_existing = pd.read_csv(target_resume_file)
                if id_col in df_existing.columns:
                    processed_ids = set(df_existing[id_col].dropna())
                elif "input_text" in df_existing.columns:
                    processed_ids = set(df_existing["input_text"].dropna())
                print(f"RESUME ACTIVE: Found {len(processed_ids)} already processed records in {target_resume_file}.")
            except Exception as e:
                print(f"Warning: Could not read existing progress file for resume ({e}). Starting fresh.")
    else:
        # If not resuming, remove any leftover chunk file
        if os.path.exists(chunks_csv_path):
            os.remove(chunks_csv_path)

    # Filter out already processed records
    if processed_ids:
        initial_len = len(dataset)
        dataset = dataset.filter(
            lambda row: row.get(id_col) not in processed_ids and row.get("input_text") not in processed_ids
        )
        print(f"Filtered dataset: {initial_len} total records -> {len(dataset)} remaining records to process.")

    num_samples = len(dataset)

    if num_samples == 0:
        print("\nAll records have already been processed! No new inference required.")
        if os.path.exists(chunks_csv_path):
            df_results = pd.read_csv(chunks_csv_path)
            df_results.to_csv(detailed_csv_path, index=False)
        elif os.path.exists(detailed_csv_path):
            df_results = pd.read_csv(detailed_csv_path)
        else:
            raise RuntimeError("No records to process and no existing results CSV found.")
    else:
        # Run inference incrementally in chunks
        infer_cfg: dict[str, Any] = {
            "inference_backend": cfg["inference_backend"],
            "n_inference_repeats": cfg["n_inference_repeats"],
            "reasoning_config": cfg["model"]["reasoning"],
            "model_path": cfg["model_path"],
            "use_output_guide": cfg["use_output_guide"],
            "output_schema_name": cfg.get("output_schema_name"),
            "max_new_tokens": cfg["max_new_tokens"],
            "temperature": cfg["temperature"],
            "top_p": cfg["top_p"],
            "top_k": cfg["top_k"],
            "min_p": cfg["min_p"],
            "presence_penalty": cfg["presence_penalty"],
            "repetition_penalty": cfg["repetition_penalty"],
            "max_concurrent_requests": cfg["max_concurrent_requests"],
            "max_context_length": cfg.get("max_context_length"),
        }
        if cfg.get("schema_config") is not None:
            infer_cfg["schema_config"] = cfg["schema_config"]
        elif cfg.get("schema") is not None:
            infer_cfg["schema"] = cfg["schema"]

        chunk_size = cfg.get("save_chunk_size", 100)
        print(f"\nRunning extraction pipeline in incremental chunks of {chunk_size} ({cfg['inference_backend']} backend)...")
        for i in range(0, num_samples, chunk_size):
            chunk_end = min(i + chunk_size, num_samples)
            print(f"Processing batch chunk [{i + 1} to {chunk_end}] of {num_samples} remaining samples...")

            chunk_dataset = dataset.select(range(i, chunk_end))
            chunk_with_outputs = process_samples(model=model, dataset=chunk_dataset, **infer_cfg)
            
            df_chunk: pd.DataFrame = chunk_with_outputs.to_pandas()
            header_needed = not os.path.exists(chunks_csv_path)
            df_chunk.to_csv(chunks_csv_path, mode="a", index=False, header=header_needed)

            print(f"  Saved chunk [{i + 1}..{chunk_end}] to {chunks_csv_path}")

        # Combine all processed records into final detailed CSV
        df_results = pd.read_csv(chunks_csv_path)
        df_results.to_csv(detailed_csv_path, index=False)

    # Generate summary metrics, report JSON, and report MD
    summary_df, report_json = generate_extraction_summary_and_reports(df_results, cfg, run_dir)

    print("\n================================================================")
    print(" EXTRACTION & EVALUATION COMPLETED SUCCESSFULLY!")
    print(f" Total Records in Database: {len(df_results)}")
    print(f" Output Directory:         {os.path.abspath(run_dir)}")
    print(f" Detailed CSV:            {detailed_filename}")
    print(f" Summary CSV:             {cfg.get('output', {}).get('summary_filename', 'summary_clinical_database.csv')}")
    print(" Report Files:            extraction_report.json & report.md")
    print("================================================================")

    print("\nSummary Consensus Database Preview:")
    print(summary_df.head())


def cleanup_vllm_server(server_process: subprocess.Popen | None) -> None:
    """
    Gracefully terminates the background vLLM server using SIGINT signal.
    """
    if server_process is None or server_process.poll() is not None:
        return

    print("\nTerminating background server process gracefully...")

    try:
        server_process.send_signal(signal.SIGINT)
        server_process.wait(timeout=15)
        print("vLLM server process stopped successfully.")
    except subprocess.TimeoutExpired:
        print("vLLM server took too long to stop. Terminating process...")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()


if __name__ == "__main__":
    main()
