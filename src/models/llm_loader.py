import os
import re
import socket
import subprocess
import sys
import time
from warnings import warn

import httpx
import psutil
import torch
from typing import Any
from huggingface_hub import HfApi, hf_hub_download, list_repo_files, snapshot_download
from transformers import AutoTokenizer

try:
    from vllm import LLM
except ImportError:
    LLM = None

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:
    OpenAI, AsyncOpenAI = None, None

from src.utils.run_utils import extract_quant_method


def _detect_hf_config_quant_method(model_path: str) -> str | None:
    """
    Extracts quantization method directly from HF model config.json if available.
    """
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        q_cfg = getattr(config, "quantization_config", None)
        if isinstance(q_cfg, dict):
            return q_cfg.get("quant_method")
        elif hasattr(q_cfg, "quant_method"):
            return getattr(q_cfg, "quant_method")
    except Exception:
        pass
    return None


def _load_model_vllm(
    model_path: str,
    quant_scheme: str|None = None,
    quant_method: str|None = None,
    max_context_length: int|None = None,
    num_gpus_to_use: int|None = None,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = True,
    *args, **kwargs
):
    """
    Load a model using the vLLM backend
    """
    if LLM is None:
        raise ImportError(
            "vLLM package is not installed in this Python environment.\n"
            "Note: Standard vLLM cannot be installed via default `pip install` on native Windows.\n\n"
            "Options on Windows:\n"
            "  1. Switch to `inference_backend: llama-cpp` in config.yaml for GGUF models.\n"
            "  2. Install a prebuilt Windows vLLM wheel (see README.md).\n"
            "  3. Run inside WSL2 (Windows Subsystem for Linux) or on a Linux GPU server."
        )

    if "VLLM_DP_MASTER_PORT" not in os.environ:
        from src.utils.run_utils import set_distributed_environment
        set_distributed_environment()

    # Initialize vLLM model arguments
    model_args = {
        "trust_remote_code": True,
        "max_model_len": max_context_length,
        "tensor_parallel_size": num_gpus_to_use,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
    }

    # Check if quantization method is already defined in model's config.json
    config_quant = _detect_hf_config_quant_method(model_path)

    model_args.update({"model": model_path})

    if config_quant:
        print(f"Detected quantization method '{config_quant}' in model config.json. Letting vLLM auto-detect.")
        model_args["quantization"] = None
    elif quant_method == "bnb":
        raise ValueError(f"vLLM does not support format {quant_method}")
    elif quant_method == "gguf":
        raise ValueError(
            "vLLM does not support GGUF quantization format.\n"
            "Supported vLLM quantizations are: ['awq', 'gptq', 'fp8', 'bitsandbytes', 'torchao', etc.].\n\n"
            "To use GGUF models, please change config.yaml to:\n"
            "  inference_backend: \"llama-cpp\"\n"
            "And install llama-cpp-python with: `uv pip install llama-cpp-python`"
        )
    else:
        model_args["quantization"] = quant_method
        if quant_method == "awq":
            model_args.update({"dtype": "float16", "quantization": "awq_marlin"})

    try:
        return LLM(**model_args)
    except Exception as e:
        err_msg = str(e)
        if "Quantization method specified" in err_msg or "ValidationError" in err_msg or "quantization" in err_msg.lower():
            print(f"Notice: Explicit quantization setting failed ({e}). Retrying with vLLM auto-detection (quantization=None)...")
            model_args["quantization"] = None
            if "dtype" in model_args:
                del model_args["dtype"]
            return LLM(**model_args)
        raise e


def _load_model_vllm_server(
    model_path: str,
    quant_method: str | None = None,
    quant_scheme: str | None = None,
    reasoning_parser: str | None = None,
    logits_processors: list[str] | None = None,
    max_context_length: int | None = None,
    max_concurrent_inferences: int | None = None,
    num_gpus_to_use: int = 1,
    gpu_memory_utilization: float = 0.90,
    enforce_eager: bool = True,
    max_swap_space_gb: int = 8,
    max_batched_tokens: int = 32768,
    host: str = "localhost",
    port: int | None = None,
    client_timeout: int | float = 43200,  # 7200
    async_mode: bool = False,
    *args, **kwargs,
) -> tuple[OpenAI | AsyncOpenAI, subprocess.Popen]:
    """
    Launches a vLLM OpenAI-compatible server as a background process,
    allowing its output to stream to the terminal, and returns a client
    and the server process handle.
    """
    tokenizer_name = get_tokenizer_name(model_path)
    
    # Special case for GGUF models: download locally and force eager mode
    if quant_method == "gguf":
        enforce_eager = True  # solves unnecessary torch compile crashes on GGUF
        model_path = download_gguf_by_quant(model_path, quant_scheme)

    config_quant = _detect_hf_config_quant_method(model_path)
    vllm_quant = None if config_quant else quant_method

    # Build the server command declaratively
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    if port is None: 
        port = find_free_port()

    params = {
        # Server configuration
        "--host": host,
        "--port": port,
        "--model": model_path,
        "--tokenizer": tokenizer_name,
        "--tensor-parallel-size": num_gpus_to_use,
        "--reasoning-parser": reasoning_parser,
        "--logits-processors": logits_processors,

        # Memory and batching
        "--gpu-memory-utilization": gpu_memory_utilization,
        "--max-num-seqs": max_concurrent_inferences,
        "--max-model-len": max_context_length,

        # Performance
        "--dtype": "auto",
        "--quantization": vllm_quant,
        "--enforce-eager": enforce_eager,
    }

    # Convert parameters to command-line arguments
    for key, value in params.items():
        if value is None or value is False:
            continue
        cmd.append(key)
        if isinstance(value, bool):
            continue  # for True booleans, only key is added (e.g. --enforce-eager)
        if isinstance(value, (list, tuple)):
            cmd.extend([str(v) for v in value])
        else:
            cmd.append(str(value))

    # Launch the server and wait for it to be ready
    base_url = f"http://{host}:{port}"
    server_process = subprocess.Popen(cmd)
    print(f"\nStarting vLLM server at {base_url}")
    try:
        wait_for_vllm_server_ready(server_process, base_url)

    # Ensure the process is terminated if the server fails to start
    except (RuntimeError, TimeoutError) as e:
        server_process.terminate()
        server_process.wait()
        raise e

    # Choose the appropriate client (sync or async)
    print("\nvLLM server is running. Client is being configured.")
    if async_mode:
        client = AsyncOpenAI(base_url=f"{base_url}/v1", api_key="vllm", timeout=client_timeout)
    else:
        client = OpenAI(base_url=f"{base_url}/v1", api_key="vllm", timeout=client_timeout)

    return client, server_process


def _load_model_llama_cpp(
    model_path: str,
    quant_scheme: str|None = None,
    quant_method: str|None = None,
    max_context_length: int|None = None,
    use_flash_attention: bool = False,
    *args, **kwargs
):
    """
    Load a model using the llama-cpp backend
    """
    if Llama is None:
        raise ImportError(
            "llama-cpp-python package is not installed in this environment.\n"
            "Install it via: `uv pip install llama-cpp-python`"
        )

    # Quantization method check
    if quant_method != "gguf":
        raise ValueError(f"Llama-cpp does not support format {quant_method}")

    return Llama.from_pretrained(
        repo_id=model_path,
        filename=f"*{quant_scheme}.gguf",
        n_gpu_layers=-1,  # 0 for not using GPU
        n_ctx=max_context_length,
        flash_attn=use_flash_attention,
        verbose=False,
    )


def load_model(
    model_path: str,
    inference_backend: str,
    quant_scheme: str|None = None,
    reasoning_parser: str|None = None,
    logits_processors: list[str]|None = None,
    max_context_length: int|None = None,
    max_concurrent_inferences: int|None = None,
    use_flash_attention: bool = False,
    num_gpus_to_use: int|None = None,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = True,
    *args, **kwargs,
) -> tuple[Any, subprocess.Popen | None]:
    """
    Create an LLM-based inference generator for solving a task
    """
    model_path = model_path.strip().strip("'").strip('"').rstrip("\\").rstrip("/")
    # Determine number of GPUs to use
    if num_gpus_to_use is None:
        num_gpus_to_use = torch.cuda.device_count()
        print(f"Selected all available GPUs by default ({num_gpus_to_use})")    
    max_num_gpus = torch.cuda.device_count()
    if num_gpus_to_use > max_num_gpus:
        print("Warning: selected number of GPUs larger than what is available")
        print(f"Will try to load the model on {max_num_gpus} GPUs")
        num_gpus_to_use = max_num_gpus

    # Define arguments for the model loaders
    quant_method = extract_quant_method(model_path)
    load_args = {
        "model_path": model_path,
        "quant_scheme": quant_scheme,
        "quant_method": quant_method,
        "reasoning_parser": reasoning_parser,
        "logits_processors": logits_processors,
        "max_concurrent_inferences": max_concurrent_inferences,
        "max_context_length": max_context_length,
        "num_gpus_to_use": num_gpus_to_use,
        "gpu_memory_utilization": gpu_memory_utilization,
        "use_flash_attention": use_flash_attention,
        "enforce_eager": enforce_eager,
    }

    # Load model and tokenizer
    server_process = None
    match inference_backend:
        case "vllm": model = _load_model_vllm(**load_args)
        case "vllm-serve": model, server_process = _load_model_vllm_server(**load_args)
        case "vllm-serve-async": model, server_process = _load_model_vllm_server(async_mode=True, **load_args)
        case "llama-cpp": model = _load_model_llama_cpp(**load_args)
        case "mock": model = _load_model_mock(**load_args)
        case _: raise ValueError(f"Unknown inference backend: {inference_backend}")

    return model, server_process




def _load_model_mock(*args, **kwargs):
    """Load mock model for fast offline testing."""
    print("Mock inference backend initialized.")
    return ("mock_model", None)



def get_tokenizer_name(
    model_id: str,
    chat_template_required: bool=True,
    default_tokenizer_name: str="Qwen/Qwen3-8B",
) -> str:
    """
    Identify base model from which any model was quantized, in order to load
    the correct tokenizer
    """
    # Look for base model in the "cardData" (where model tree info is stored)
    api = HfApi()
    model_info = api.model_info(model_id)
    card_data = model_info.card_data or {}
    tokenizer_name = card_data.get("base_model")
    if isinstance(tokenizer_name, list):  # sometimes a list?
        tokenizer_name = tokenizer_name[0]

    # Alternatively, inspect tags or siblings
    if not tokenizer_name:
        for tag in model_info.tags:
            if "base_model:" in tag:
                tokenizer_name = tag.split(":")[1]
                break

    # Check for chat template, may fall back on Llama-3.2-3B-Instruct (most common)
    if chat_template_required and tokenizer_name:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if tokenizer.chat_template is None:
                raise ValueError("chat_template missing")
        except Exception as e:
            warn(
                f"The tokenizer '{tokenizer_name}' does not support chat templating "
                f"(reason: {e}). Falling back to '{default_tokenizer_name}'."
            )
            return default_tokenizer_name

    return tokenizer_name


# def download_gguf_by_quant(model_id: str, quant_scheme: str) -> str:
#     """
#     Download the first matching GGUF file in a model repository
#     """
#     quant_scheme_lower = quant_scheme.lower()
#     files = list_repo_files(model_id)
#     for file in files:
#         file_lower = file.lower()
#         if file_lower.endswith(".gguf") and quant_scheme_lower in file_lower:
#             return hf_hub_download(repo_id=model_id, filename=file)
        
#     raise FileNotFoundError(f"No GGUF file found with quantization scheme '{quant_scheme}' in repo '{model_id}'")


def download_gguf_by_quant(model_id: str, quant_scheme: str) -> str:
    """
    Orchestrates the download of GGUF files. If split files are detected,
    downloads all parts and calls the merge function.
    """
    quant_scheme_lower = quant_scheme.lower()
    files = list_repo_files(model_id)
    
    # Identify the primary file
    target_file = None
    for file in files:
        file_lower = file.lower()
        if file_lower.endswith(".gguf") and quant_scheme_lower in file_lower:
            # Prioritize part 1 of a split set
            if "00001-of-" in file_lower:
                target_file = file
                break
            if target_file is None:
                target_file = file

    if target_file is None:
        raise FileNotFoundError(
            f"No GGUF file found with quantization scheme '{quant_scheme}' in repo '{model_id}'"
        )

    # Check for split files and return single path if existing
    split_match = re.search(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", target_file)
    if not split_match:
        print(f"Detected single GGUF file: {target_file}")
        return hf_hub_download(repo_id=model_id, filename=target_file)
    
    # Handle split download
    base_name = split_match.group(1)
    total_parts = int(split_match.group(3))
    print(f"Detected split GGUF model ({total_parts} parts). Downloading...")
    first_part_path = hf_hub_download(repo_id=model_id, filename=target_file)
    for i in range(2, total_parts + 1):
        shard_name = f"{base_name}-{i:05d}-of-{total_parts:05d}.gguf"
        print(f"Downloading part {i}/{total_parts}: {shard_name}")
        hf_hub_download(repo_id=model_id, filename=shard_name)

    # Merge all GGUF files into a single one, for vLLM
    return merge_gguf_shards(first_part_path)


def merge_gguf_shards(first_part_path: str) -> str:
    """
    Merges split GGUF files using the installed 'gguf-split' CLI tool.
    
    Args:
        first_part_path (str): Path to the first file (e.g., model-00001-of-00002.gguf)
        
    Returns:
        str: The path to the successfully merged file.
    """
    # Construct the output filename (removing the split suffix)
    base_dir = os.path.dirname(first_part_path)
    original_filename = os.path.basename(first_part_path)
    merged_filename = re.sub(r"-\d{5}-of-\d{5}", "", original_filename)
    
    # If regex didn't change anything (unexpected naming), append "-merged"
    if merged_filename == original_filename:
        merged_filename = f"{original_filename.replace('.gguf', '')}-merged.gguf"
        
    # If the merged file already exists, return it to save time
    merged_path = os.path.join(base_dir, merged_filename)
    if os.path.exists(merged_path):
        print(f"Found previously merged file: {merged_path}")
        return merged_path

    # Usage: gguf-split --merge <first-split-file-path> <output-file-path>
    print(f"Merging shards into: {merged_path}")
    try:
        cmd = ["gguf-split", "--merge", first_part_path, merged_path]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Merge successful! Single file built at {merged_path}")
        return merged_path
        
    except FileNotFoundError:
        raise RuntimeError("The 'gguf-split' tool was not found in the system PATH.\n")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gguf-split failed to merge files. Error: {e}")


def download_model(model_id):
    """
    Download a model from the HuggingFaceHub to the local cache
    """
    # The snapshot_download function downloads all files from a repo
    print(f"Downloading model {model_id} to the cache")
    try:
        cached_path = snapshot_download(
            repo_id=model_id,
            repo_type="model",
            local_files_only=False,
        )
        print(f"Model downloaded successfully and cached at: {cached_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


def find_free_port() -> int:
    """
    Finds and returns an available, unused port on the host machine.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Bind to port 0. The OS will assign a random available port.
        s.bind(("", 0))
        # Get the port number assigned by the OS.
        port = s.getsockname()[1]
        return port


def get_swap_space_gb(percentage=0.5, max=None):
    """
    Calculates the recommended vLLM swap space in GB based on a percentage
    of the total available CPU RAM.
    """
    total_ram_bytes = psutil.virtual_memory().total
    total_ram_gb = total_ram_bytes / (1024 ** 3) # Convert bytes to gigabytes
    
    # Calculate swap space and convert to an integer
    swap_gb = int(total_ram_gb * percentage)
    if max is not None and swap_gb > max:
        swap_gb = max
    
    return swap_gb


def wait_for_vllm_server_ready(
    server_process: subprocess.Popen,
    url: str,
    timeout: int = 1800,
):
    """
    Waits for the vLLM server to be ready and become healthy
    """
    print("\nWaiting for server to become available (server logs coming below)")
    start_time = time.time()

    # Check if the server process terminated unexpectedly
    while time.time() - start_time < timeout:
        if server_process.poll() is not None:
            raise RuntimeError(
                "\nvLLM server process terminated unexpectedly "
                f"with exit code {server_process.returncode}.\n"
                "Check the server logs above for the cause of the error."
            )

        # Attempt to connect to the server's health endpoint
        try:
            resp = httpx.get(f"{url}/health", timeout=1.0)
            if resp.status_code == 200:

                # Once healthy, double-check that the model is loaded and ready
                model_resp = httpx.get(f"{url}/v1/models", timeout=2.0)
                if model_resp.status_code == 200 and model_resp.json().get("data"):
                    time.sleep(5)  # extra wait to ensure readiness
                    return  # success, server is ready
        
        # Server is not yet available, wait and retry
        except httpx.RequestError:
            time.sleep(2)
            continue
        
        time.sleep(2)

    # If the loop completes, it means we've timed out
    server_process.terminate()
    server_process.wait()
    raise TimeoutError(
        f"\nvLLM server failed to start within {timeout} seconds.\n"
        "Check the server logs above for the cause of the error."
    )