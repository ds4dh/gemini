import os
import gc
import pandas as pd
import argparse
import subprocess
from typing import Any
from functools import partial

# Set distributed environment setup before importing torch, vLLM, or transformer
from src.utils.run_utils import set_distributed_environment
set_distributed_environment()

import torch
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp
from transformers import AutoModelForCausalLM
from datasets import Dataset

from src.models.llm_evaluation import (
    get_gpu_memory_usage_by_pid,
    compute_and_save_metrics,
    print_gpu_info,
)
from src.models.llm_inference import process_samples
from src.models.llm_loader import load_model
from src.data.prompting import build_messages
from src.data.data_loading import load_data_formatted_for_benchmarking
from src.utils.plot_utils import plot_metrics
from src.utils.run_utils import (
    load_config_files,
    add_model_arguments,
    add_data_arguments,
    extract_quant_method,
    clean_model_cache,
)


def main(args: argparse.Namespace):
    """
    Run benchmarks on different generative large language models in separate
    processes to avoid GPU memory leak or accumulation
    """
    # Force a fresh run for each new process
    torch_mp.set_start_method("spawn", force=True)

    # Load configuration from yaml files
    cfg = load_config_files(args)

    # Simply re-plot previous results if specified
    if args.plot_only:
        save_benchmark_results(cfg=cfg)

    # Generate json and png from an existing csv file
    elif getattr(args, "load_csv_results", None):
        print(f"Loading previous results from {args.load_csv_results}...")
        dataset = Dataset.from_pandas(pd.read_csv(args.load_csv_results))
        # Pass dummy time/memory metrics since we skipped inference
        benchmark_results = {"dataset": dataset, "time": 0.0, "memory": 0.0}
        save_benchmark_results(cfg=cfg, benchmark_results=benchmark_results)

    # Load data and run the benchmark
    else:
        data_loading_args = cfg["data_loading_arguments"]
        dataset = load_data_formatted_for_benchmarking(args, **data_loading_args)
        run_kwargs = {"cfg": cfg, "dataset": dataset, "debug": args.debug}
        record_one_benchmark(**run_kwargs)

def record_one_benchmark(
    cfg: dict,
    dataset: Dataset,
    debug: bool = False,
) -> None:
    """ Run the benchmark for a single model and record metrics
    """
    # Try to benchmark the model
    model, server_process = None, None
    try:

        # Model loading (and server start if needed)
        print(f"Benchmarking {cfg['model_path']} with {cfg['inference_backend']} backend")
        print_gpu_info()
        model, server_process = load_model(**cfg, debug=debug)
        
        # Build messages using the model configuration
        dataset = dataset.map(
            function=partial(build_messages, cfg=cfg),
            desc="Building messages for prompting",
        )

        # Pre-calculate the output path for incremental saving
        output_subdir = cfg["inference_backend"]
        if cfg.get("use_output_guide"): output_subdir = f"{output_subdir}_guided"
        output_dir = os.path.join(cfg["result_dir"], output_subdir)
        os.makedirs(output_dir, exist_ok=True)
        
        quant_method = extract_quant_method(cfg["model_path"])
        if quant_method is not None and quant_method == "gguf":
            model_result_path = f"{cfg['model_path']}-{cfg['quant_scheme']}.csv"
        else:
            model_result_path = f"{cfg['model_path']}-no_quant_scheme.csv"

        # Build output path and prevent appending rows to an older aborted run
        output_path = os.path.join(output_dir, model_result_path)
        output_path_chunks = output_path.replace(".csv", "_chunks.csv")
        os.makedirs(os.path.split(output_path_chunks)[0], exist_ok=True)
        if os.path.exists(output_path_chunks):
            os.remove(output_path_chunks)

        # Record results in chunks to prevent data loss on crashes
        all_times, all_mems = [], []
        num_samples = len(dataset)
        chunk_size = cfg.get("save_chunk_size", 100)
        for i in range(0, num_samples, chunk_size):
            chunk_end = min(i + chunk_size, num_samples)
            print(f"\nProcessing chunk {i} to {chunk_end} of {num_samples}...")
            
            # Select the chunk and run inference
            chunk = dataset.select(range(i, chunk_end))
            chunk_results = benchmark_one_model(
                cfg=cfg,
                model=model,
                dataset=chunk,
            )
            
            # Append chunk to CSV on the fly
            df_chunk: Dataset = chunk_results["dataset"].to_pandas()
            df_chunk.to_csv(
                output_path_chunks, mode="a", index=False,
                header=not os.path.exists(output_path_chunks),
            )
            
            # Store metrics (weighting time by chunk size to get correct final average)
            all_times.append(chunk_results["time"] * len(chunk))
            all_mems.append(chunk_results["memory"])

        # Reconstruct the results dictionary with aggregated metrics
        full_dataset = Dataset.from_pandas(pd.read_csv(output_path_chunks))
        benchmark_results = {
            "dataset": full_dataset,
            "time": sum(all_times) / num_samples if num_samples > 0 else 0.0,
            "memory": max(all_mems) if all_mems else 0.0,
        }

        # Compute and save benchmark results (this will also safely overwrite the CSV with final formatting)
        save_benchmark_results(cfg=cfg, benchmark_results=benchmark_results)
        print("Benchmarked %s" % cfg["model_path"])

    # Error handling logic
    except Exception as e:
        print(f"An error occurred during benchmarking: {e}")
        raise 

    # Cleanup logic
    finally:
        print("Cleaning up resources...")

        # Terminate server process if it was started
        if server_process is not None and server_process.poll() is None:
            print("Terminating vLLM server...")
            server_process.terminate()  # send a polite SIGTERM signal
            
            # Wait for a graceful shutdown
            try:
                server_process.wait(timeout=30)
                print("Server terminated gracefully.")
            
            # Force killing the process with SIGKILL
            except subprocess.TimeoutExpired:
                print("Server did not terminate within 30 seconds, killing it.")
                server_process.kill()
                server_process.wait()  # wait for the OS to clean up the killed process
                print("Server killed.")

        # Clean GPU memory and distributed processes if any
        if "model" in locals() and cfg.get('inference_backend') != 'vllm-serve':
            del model
        if torch_dist.is_initialized(): torch_dist.destroy_process_group()
        torch.cuda.empty_cache()
        gc.collect()
        print_gpu_info()
        print("Cleaned memory")

        # Clean model cache
        if cfg["delete_model_cache_after_run"]:
            clean_model_cache(cfg["model_path"], cfg["quant_scheme"])


def benchmark_one_model(
    cfg: dict[str, Any],
    model: AutoModelForCausalLM,
    dataset: Dataset,
) -> dict[str, Dataset|float]:
    """
    Prompts a generative LLM with medical questions and computes metrics
    about computation time and GPU memory usage
    --> https://triton-lang.org/main/python-api/generated/triton.testing.do_bench

    """
    # Initialize events, clear cache and reset peak memory stats
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Time execution time for processing samples
    start_event.record()
    dataset_with_outputs = process_samples(model, dataset, **cfg)
    end_event.record()

    # Wait for all kernels on all GPUs to finish BEFORE measuring time.
    torch.cuda.synchronize()
    print("Devices synchronized.")

    # Record the time and memory usage
    time = start_event.elapsed_time(end_event) / 1000  # in seconds
    time = time / len(dataset)  # time per sample
    # time = time / cfg["n_inference_repeats"]  # time per inference (?)
    memory = get_gpu_memory_usage_by_pid()  # in GB

    # Return benchmark results for metric computation and plotting
    print("Model successfully benchmarked.")
    return {"dataset": dataset_with_outputs, "time": time, "memory": memory}


def save_benchmark_results(
    cfg: dict,
    benchmark_results: dict | None = None,
) -> None:
    """
    Saves benchmark results to a CSV file and plot metrics
    """
    # Build unique output path given configuration
    output_subdir = cfg["inference_backend"]
    if cfg["use_output_guide"]: output_subdir = f"{output_subdir}_guided"
    output_dir = os.path.join(cfg["result_dir"], output_subdir)
    quant_method = extract_quant_method(cfg["model_path"])
    if quant_method is not None and quant_method == "gguf":
        model_result_path = f"{cfg['model_path']}-{cfg['quant_scheme']}.csv"
    else:
        model_result_path = f"{cfg['model_path']}-no_quant_scheme.csv"
    output_path = os.path.join(output_dir, model_result_path)

    # If provided, save model results to a csv file
    if benchmark_results is not None:
        compute_and_save_metrics(
            benchmark_results=benchmark_results,
            model_path=cfg["model_path"],
            output_path=output_path,
        )

    # Metrics are plotted by loading saved benchmark results
    metric_path = output_path.replace(".csv", ".json")
    plot_metrics(metric_path=metric_path)


def set_torch_cuda_arch_list() -> None:
    """
    Sets the TORCH_CUDA_ARCH_LIST environment variable to the CUDA compute
    capability of the current GPU(s). This prevents PyTorch from compiling
    kernels for all possible architectures, reducing compilation time.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping TORCH_CUDA_ARCH_LIST setup.")
        return

    # Get the current GPU's compute capability
    major, minor = torch.cuda.get_device_capability(device=None)
    cuda_arch_value = f"{major}.{minor}"  # required format is 'X.Y', not 'sm_XY'

    # Check if the environment variable is already set correctly
    if os.environ.get('TORCH_CUDA_ARCH_LIST') != cuda_arch_value:
        print(f"Setting TORCH_CUDA_ARCH_LIST to '{cuda_arch_value}' for efficient compilation.")
        os.environ['TORCH_CUDA_ARCH_LIST'] = cuda_arch_value
    else:
        print(f"TORCH_CUDA_ARCH_LIST is already set to '{cuda_arch_value}'.")


if __name__ == "__main__":
    """
    Entry point for running the benchmark script
    """
    # Provide environment variables to help the inference backend
    set_torch_cuda_arch_list()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Benchmark an LLM in inference mode for data extraction."
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Run the script in debug mode with a smaller dataset and fewer repetitions."
    )
    parser.add_argument(
        "--plot-only", "-po", action="store_true",
        help="Only plot the results from a previouly run benchmark."
    )
    parser.add_argument(
        "-l", "--load-csv-results", type=str, default=None,
        help="Path to an existing CSV file to load results from, skipping inference."
    )
    add_model_arguments(parser)
    add_data_arguments(parser)
    args = parser.parse_args()

    # Start the main script
    main(args)
