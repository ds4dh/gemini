import argparse
import datetime
import gc
import os
import shutil
import sys
from functools import partial

import pandas as pd
import torch

from src.data.data_loading import load_data_formatted_for_benchmarking
from src.data.prompting import build_messages
from src.models.llm_evaluation import generate_extraction_summary_and_reports
from src.models.llm_inference import process_samples
from src.models.llm_loader import load_model
from src.utils.run_utils import load_config_files, set_distributed_environment


def main():
    """
    Main entrypoint for the Gemini Clinical Variable Extraction Pipeline.
    """
    set_distributed_environment()

    parser = argparse.ArgumentParser(
        description="Unified Clinical Variable Extraction Pipeline using LLMs."
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to unified configuration YAML file (default: config.yaml)"
    )
    parser.add_argument(
        "--input-path", "-i",
        type=str,
        default=None,
        help="Override input dataset CSV/Excel path"
    )
    parser.add_argument(
        "--curated-data-path", "-cd",
        type=str,
        default=None,
        help="Path to non-encrypted curated dataset CSV"
    )
    parser.add_argument(
        "--encrypted-data-path", "-ed",
        type=str,
        default=None,
        help="Path to encrypted dataset CSV"
    )
    parser.add_argument(
        "--backend", "-b",
        type=str,
        default=None,
        help="Override inference backend (transformers, mock, vllm, vllm-serve-async, llama-cpp)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Override model path or HuggingFace ID"
    )
    parser.add_argument(
        "--quant-scheme", "-qs",
        type=str,
        default=None,
        help="Override GGUF/quantization scheme (e.g., Q6_K_XL, Q8_0)"
    )
    parser.add_argument(
        "--gpu-memory-utilization", "-gpu",
        type=float,
        default=None,
        help="Override GPU memory utilization fraction (0.50 - 0.95)"
    )
    parser.add_argument(
        "--reasoning-parser", "-rp",
        type=str,
        default=None,
        help="Override reasoning parser (e.g., qwen3, deepseek_r1)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Override output directory for results CSV"
    )
    parser.add_argument(
        "--save-chunk-size",
        type=int,
        default=None,
        help="Override incremental CSV save chunk size"
    )

    args = parser.parse_args()

    # 1. Load configuration from single unified YAML file
    cfg = load_config_files(args)

    # CLI Overrides
    if args.input_path:
        cfg["input_path"] = args.input_path
    if args.curated_data_path:
        cfg["input_path"] = args.curated_data_path
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

    print("================================================================")
    print(" GEMINI CLINICAL VARIABLE EXTRACTION PIPELINE")
    print("================================================================")
    print(f"Config path:       {args.config}")
    print(f"Model:             {cfg['model_path']}")
    print(f"Inference Backend: {cfg['inference_backend']}")
    print(f"Input Data Path:   {cfg.get('input_path', 'default')}")
    print(f"Result Directory:  {cfg['result_dir']}")
    print("================================================================\n")

    # 2. Load dataset
    data_loading_kwargs = cfg.get("data_loading_arguments", {})
    if "input_path" in cfg:
        data_loading_kwargs["input_path"] = cfg["input_path"]
    if "input_text_column" in cfg:
        data_loading_kwargs["input_text_column"] = cfg["input_text_column"]

    dataset = load_data_formatted_for_benchmarking(args, **data_loading_kwargs)
    print(f"Loaded {len(dataset)} clinical text records for extraction.\n")

    # 3. Load LLM model
    model, server_process = None, None
    try:
        model, server_process = load_model(**cfg)

        # 4. Build prompt messages using prompt templates and context data
        print("Building prompt messages for LLM...")
        dataset = dataset.map(
            function=partial(build_messages, cfg=cfg),
            desc="Constructing chat prompts",
        )

        # 5. Execute inference and structured variable extraction
        print(f"\nRunning extraction pipeline ({cfg['inference_backend']} backend)...")
        infer_cfg = {k: v for k, v in cfg.items() if k not in ("model", "dataset")}
        dataset_with_outputs = process_samples(model=model, dataset=dataset, **infer_cfg)

        # 6. Save results to CSV
        # 6. Save detailed and summary results to timestamped subfolder
        base_output_dir = cfg.get("result_dir", "./results")
        if cfg.get("use_timestamp_subfolder", True):
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(base_output_dir, f"run_{timestamp_str}")
        else:
            run_dir = base_output_dir

        os.makedirs(run_dir, exist_ok=True)

        # Save config file snapshot for complete experiment reproducibility
        run_config_path = os.path.join(run_dir, "config.yaml")
        try:
            if os.path.exists(args.config):
                shutil.copyfile(args.config, run_config_path)
            else:
                with open(run_config_path, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False)
            print(f"Saved config snapshot at: {run_config_path}")
        except Exception as e:
            print(f"Notice: Could not write config snapshot ({e})")

        detailed_filename = cfg.get("output", {}).get("detailed_filename", "detailed_clinical_database.csv")
        if not detailed_filename.endswith(".csv"):
            detailed_filename = "detailed_clinical_database.csv"
        detailed_csv_path = os.path.join(run_dir, detailed_filename)

        df_results: pd.DataFrame = dataset_with_outputs.to_pandas()
        df_results.to_csv(detailed_csv_path, index=False)

        # Generate consensus summary database & evaluation reports (JSON & MD)
        summary_df, report_json = generate_extraction_summary_and_reports(df_results, cfg, run_dir)

        print("\n================================================================")
        print(" EXTRACTION & EVALUATION COMPLETED SUCCESSFULLY!")
        print(f" Total Processed: {len(df_results)} records")
        print(f" Output Directory:{os.path.abspath(run_dir)}")
        print(f" Detailed CSV:    {detailed_filename}")
        print(f" Summary CSV:     {cfg.get('output', {}).get('summary_filename', 'summary_clinical_database.csv')}")
        print(f" Report Files:    extraction_report.json & report.md")
        print("================================================================")

        # Display preview of consensus extracted data
        print("\nSummary Consensus Database Preview:")
        print(summary_df.head())

    except Exception as e:
        print(f"\nAn error occurred during pipeline execution: {e}")
        raise

    finally:
        # Cleanup server and memory
        if server_process is not None and server_process.poll() is None:
            print("Terminating background server process...")
            server_process.terminate()
            server_process.wait()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
