from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from typing import Any
from warnings import warn

import httpx
import psutil
import torch
from huggingface_hub import HfApi, hf_hub_download, list_repo_files, snapshot_download
from transformers import AutoTokenizer

from vllm import LLM
from openai import AsyncOpenAI, OpenAI

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from src.utils.run_utils import extract_quant_method



THINKING_JSON_ADAPTER_PROCESSOR = (
    "src.models.logits_processors:ThinkingJSONAdapterProcessor"
)


def select_server_logits_processors(
    use_output_guide: bool,
) -> list[str]:
    """Select the only custom processor used for guided server requests."""
    if use_output_guide:
        return [THINKING_JSON_ADAPTER_PROCESSOR]
    return []


def _detect_hf_config_quant_method(model_path: str) -> str | None:
    """Read a Hugging Face model's declared quantization method, if present."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        quantization_config = getattr(config, "quantization_config", None)

        if isinstance(quantization_config, dict):
            return quantization_config.get("quant_method")

        if hasattr(quantization_config, "quant_method"):
            return quantization_config.quant_method
    except Exception:
        pass

    return None


def _prepare_gguf_model(
    model_path: str,
    quant_scheme: str | None,
) -> tuple[str, str]:
    """Download/merge one GGUF model file and resolve its base tokenizer."""
    if not quant_scheme:
        raise ValueError(
            "quant_scheme is required when model_path identifies a GGUF repository."
        )

    tokenizer_name = get_tokenizer_name(model_path)
    if not tokenizer_name:
        raise ValueError(
            f"Could not determine the base tokenizer for GGUF repository "
            f"{model_path!r}. Add base_model metadata to the Hugging Face card "
            "or provide a repository that exposes it."
        )

    local_gguf_path = download_gguf_by_quant(
        model_id=model_path,
        quant_scheme=quant_scheme,
    )
    return local_gguf_path, tokenizer_name


def _load_model_vllm(
    model_path: str,
    quant_scheme: str | None = None,
    quant_method: str | None = None,
    max_context_length: int | None = None,
    num_gpus_to_use: int | None = None,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = True,
    *args: Any,
    **kwargs: Any,
) -> LLM:
    """Load a model with direct Python vLLM, including local GGUF support."""
    if LLM is None:
        raise ImportError(
            "vLLM is not installed. On Windows, use a compatible vLLM wheel, "
            "WSL2, Linux, or the llama-cpp backend."
        )

    if "VLLM_DP_MASTER_PORT" not in os.environ:
        from src.utils.run_utils import set_distributed_environment

        set_distributed_environment()

    model_args: dict[str, Any] = {
        "model": model_path,
        "trust_remote_code": True,
        "max_model_len": max_context_length,
        "tensor_parallel_size": num_gpus_to_use,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
    }

    if quant_method == "gguf":
        local_gguf_path, tokenizer_name = _prepare_gguf_model(
            model_path=model_path,
            quant_scheme=quant_scheme,
        )
        model_args.update(
            {
                "model": local_gguf_path,
                "tokenizer": tokenizer_name,
                "quantization": None,
                "enforce_eager": True,
            }
        )
    else:
        config_quant_method = _detect_hf_config_quant_method(model_path)

        if config_quant_method:
            print(
                f"Detected quantization method {config_quant_method!r} in "
                "model config; letting vLLM auto-detect it."
            )
            model_args["quantization"] = None
        elif quant_method == "bnb":
            raise ValueError("vLLM does not support the legacy 'bnb' alias.")
        else:
            model_args["quantization"] = quant_method
            if quant_method == "awq":
                model_args.update(
                    {
                        "dtype": "float16",
                        "quantization": "awq_marlin",
                    }
                )

    try:
        return LLM(**model_args)
    except Exception as exc:
        error_message = str(exc).lower()
        if quant_method != "gguf" and (
            "quantization" in error_message or "validationerror" in error_message
        ):
            print(
                "Explicit quantization setting failed; retrying with vLLM "
                "auto-detection."
            )
            model_args["quantization"] = None
            model_args.pop("dtype", None)
            return LLM(**model_args)
        raise


def _load_model_vllm_server(
    model_path: str,
    quant_method: str | None = None,
    quant_scheme: str | None = None,
    reasoning_parser: str | None = None,
    logits_processors: list[str] | None = None,
    use_output_guide: bool = False,
    max_context_length: int | None = None,
    max_concurrent_inferences: int | None = None,
    num_gpus_to_use: int = 1,
    gpu_memory_utilization: float = 0.90,
    enforce_eager: bool = True,
    max_swap_space_gb: int = 8,
    max_batched_tokens: int = 32768,
    host: str = "localhost",
    port: int | None = None,
    client_timeout: int | float = 43200,
    async_mode: bool = False,
    *args: Any,
    **kwargs: Any,
) -> tuple[OpenAI | AsyncOpenAI, subprocess.Popen]:
    """Start vLLM's OpenAI server, supporting HF and local downloaded GGUF."""
    if OpenAI is None or AsyncOpenAI is None:
        raise ImportError("The openai package is required for vLLM server clients.")

    served_model_path = model_path
    tokenizer_name: str | None
    vllm_quant: str | None

    if quant_method == "gguf":
        # vLLM detects a local .gguf file; do not pass --quantization gguf.
        served_model_path, tokenizer_name = _prepare_gguf_model(
            model_path=model_path,
            quant_scheme=quant_scheme,
        )
        vllm_quant = None
        enforce_eager = True
    else:
        tokenizer_name = get_tokenizer_name(model_path)
        config_quant_method = _detect_hf_config_quant_method(model_path)
        vllm_quant = None if config_quant_method else quant_method

    if not tokenizer_name:
        raise ValueError(
            f"Could not determine a tokenizer for model {model_path!r}. "
            "Provide Hugging Face base_model metadata or update "
            "get_tokenizer_name()."
        )

    selected_processors = select_server_logits_processors(
        use_output_guide=use_output_guide,
    )

    if logits_processors is not None:
        requested_processors = set(logits_processors)
        expected_processors = set(selected_processors)
        allowed_processors = {THINKING_JSON_ADAPTER_PROCESSOR}
        unsupported_processors = requested_processors - allowed_processors

        if unsupported_processors:
            raise ValueError(
                "Unsupported custom logits processor(s): "
                f"{sorted(unsupported_processors)}. Only "
                f"{THINKING_JSON_ADAPTER_PROCESSOR!r} is supported."
            )

        if requested_processors != expected_processors:
            raise ValueError(
                "Configured logits_processors does not match required runtime "
                f"selection. Configured={sorted(requested_processors)}, "
                f"required={sorted(expected_processors)}."
            )

    print(
        "vLLM server custom-guidance configuration: "
        f"use_output_guide={use_output_guide}, "
        f"logits_processors={selected_processors or 'none'}"
    )

    if port is None:
        port = find_free_port()

    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]

    parameters: dict[str, Any] = {
        "--host": host,
        "--port": port,
        "--model": served_model_path,
        "--tokenizer": tokenizer_name,
        "--tensor-parallel-size": num_gpus_to_use,
        "--reasoning-parser": reasoning_parser,
        "--logits-processors": selected_processors or None,
        "--gpu-memory-utilization": gpu_memory_utilization,
        "--max-num-seqs": max_concurrent_inferences,
        "--max-num-batched-tokens": max_batched_tokens,
        "--max-model-len": max_context_length,
        "--dtype": "auto",
        "--quantization": vllm_quant,
        "--enforce-eager": enforce_eager,
    }

    for key, value in parameters.items():
        if value is None or value is False:
            continue

        command.append(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (list, tuple)):
            command.extend(str(item) for item in value)
        else:
            command.append(str(value))

    base_url = f"http://{host}:{port}"
    print("Launching vLLM command:")
    print(" ".join(command))

    server_process = subprocess.Popen(command)
    print(f"\nStarting vLLM server at {base_url}")

    try:
        wait_for_vllm_server_ready(server_process, base_url)
    except (RuntimeError, TimeoutError):
        server_process.terminate()
        server_process.wait()
        raise

    client_base_url = f"{base_url}/v1"
    if async_mode:
        client = AsyncOpenAI(
            base_url=client_base_url,
            api_key="vllm",
            timeout=client_timeout,
        )
    else:
        client = OpenAI(
            base_url=client_base_url,
            api_key="vllm",
            timeout=client_timeout,
        )

    return client, server_process


def _load_model_llama_cpp(
    model_path: str,
    quant_scheme: str | None = None,
    quant_method: str | None = None,
    max_context_length: int | None = None,
    use_flash_attention: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Llama:
    """Load a GGUF model through llama-cpp-python."""
    if Llama is None:
        raise ImportError(
            "llama-cpp-python is not installed. Install it with "
            "`uv pip install llama-cpp-python`."
        )

    if quant_method != "gguf":
        raise ValueError(
            f"llama-cpp requires a GGUF model, received quant_method={quant_method!r}."
        )
    if not quant_scheme:
        raise ValueError("quant_scheme is required for llama-cpp GGUF loading.")

    return Llama.from_pretrained(
        repo_id=model_path,
        filename=f"*{quant_scheme}.gguf",
        n_gpu_layers=-1,
        n_ctx=max_context_length,
        flash_attn=use_flash_attention,
        verbose=False,
    )


def load_model(
    model_path: str,
    inference_backend: str,
    quant_scheme: str | None = None,
    reasoning_parser: str | None = None,
    logits_processors: list[str] | None = None,
    use_output_guide: bool = False,
    max_context_length: int | None = None,
    max_concurrent_inferences: int | None = None,
    use_flash_attention: bool = False,
    num_gpus_to_use: int | None = None,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = True,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, subprocess.Popen | None]:
    """Load the configured model/backend without legacy reasoning-budget args."""
    model_path = model_path.strip().strip("'").strip('"').rstrip("\\").rstrip("/")

    available_gpus = torch.cuda.device_count()
    if num_gpus_to_use is None:
        num_gpus_to_use = available_gpus
        print(f"Selected all available GPUs by default ({num_gpus_to_use}).")

    if num_gpus_to_use < 1:
        raise RuntimeError(
            "No CUDA GPU selected. Use llama-cpp or mock, or configure a "
            "vLLM-capable CUDA environment."
        )

    if num_gpus_to_use > available_gpus:
        print("Warning: selected GPU count exceeds available GPUs.")
        num_gpus_to_use = available_gpus

    if inference_backend == "vllm" and logits_processors:
        print(
            "Note: direct vLLM uses native StructuredOutputsParams; configured "
            "server logits processors are ignored."
        )

    quant_method = extract_quant_method(model_path)
    load_args = {
        "model_path": model_path,
        "quant_scheme": quant_scheme,
        "quant_method": quant_method,
        "reasoning_parser": reasoning_parser,
        "logits_processors": logits_processors,
        "use_output_guide": use_output_guide,
        "max_concurrent_inferences": max_concurrent_inferences,
        "max_context_length": max_context_length,
        "num_gpus_to_use": num_gpus_to_use,
        "gpu_memory_utilization": gpu_memory_utilization,
        "use_flash_attention": use_flash_attention,
        "enforce_eager": enforce_eager,
    }

    server_process: subprocess.Popen | None = None

    match inference_backend:
        case "vllm":
            model = _load_model_vllm(**load_args)
        case "vllm-serve":
            model, server_process = _load_model_vllm_server(**load_args)
        case "vllm-serve-async":
            model, server_process = _load_model_vllm_server(
                async_mode=True,
                **load_args,
            )
        case "llama-cpp":
            model = _load_model_llama_cpp(**load_args)
        case "mock":
            model = _load_model_mock(**load_args)
        case _:
            raise ValueError(f"Unknown inference backend: {inference_backend}")

    return model, server_process


def _load_model_mock(*args: Any, **kwargs: Any) -> tuple[str, None]:
    """Load an offline mock backend."""
    print("Mock inference backend initialized.")
    return "mock_model", None


def get_tokenizer_name(
    model_id: str,
    chat_template_required: bool = True,
    default_tokenizer_name: str = "Qwen/Qwen3-8B",
) -> str | None:
    """Resolve the base tokenizer name declared by a quantized model repository."""
    api = HfApi()
    model_info = api.model_info(model_id)
    card_data = model_info.card_data or {}
    tokenizer_name = card_data.get("base_model")

    if isinstance(tokenizer_name, list):
        tokenizer_name = tokenizer_name[0] if tokenizer_name else None

    if not tokenizer_name:
        for tag in model_info.tags or []:
            if tag.startswith("base_model:"):
                tokenizer_name = tag.split(":", maxsplit=1)[1]
                break

    if not tokenizer_name:
        return None

    if chat_template_required:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True,
            )
            if tokenizer.chat_template is None:
                raise ValueError("chat_template missing")
        except Exception as exc:
            warn(
                f"Tokenizer {tokenizer_name!r} has no usable chat template "
                f"({exc}); using {default_tokenizer_name!r}."
            )
            return default_tokenizer_name

    return tokenizer_name


def download_gguf_by_quant(model_id: str, quant_scheme: str) -> str:
    """Download the requested GGUF quantization and merge split shards if needed."""
    quant_scheme_lower = quant_scheme.lower()
    target_file: str | None = None

    for file_name in list_repo_files(model_id):
        file_name_lower = file_name.lower()
        if not file_name_lower.endswith(".gguf"):
            continue
        if quant_scheme_lower not in file_name_lower:
            continue

        if "00001-of-" in file_name_lower:
            target_file = file_name
            break
        if target_file is None:
            target_file = file_name

    if target_file is None:
        raise FileNotFoundError(
            f"No GGUF file matching quant_scheme={quant_scheme!r} was found "
            f"in {model_id!r}."
        )

    split_match = re.search(
        r"^(.*)-(\d{5})-of-(\d{5})\.gguf$",
        target_file,
    )

    if split_match is None:
        print(f"Downloading GGUF file: {target_file}")
        return hf_hub_download(repo_id=model_id, filename=target_file)

    base_name = split_match.group(1)
    total_parts = int(split_match.group(3))
    print(f"Downloading split GGUF model with {total_parts} shards.")

    first_part_path = hf_hub_download(repo_id=model_id, filename=target_file)
    for shard_index in range(2, total_parts + 1):
        shard_name = (
            f"{base_name}-{shard_index:05d}-of-{total_parts:05d}.gguf"
        )
        print(f"Downloading shard {shard_index}/{total_parts}: {shard_name}")
        hf_hub_download(repo_id=model_id, filename=shard_name)

    return merge_gguf_shards(first_part_path)


def merge_gguf_shards(first_part_path: str) -> str:
    """Merge local GGUF shards with the gguf-split command-line utility."""
    base_dir = os.path.dirname(first_part_path)
    original_filename = os.path.basename(first_part_path)
    merged_filename = re.sub(
        r"-\d{5}-of-\d{5}",
        "",
        original_filename,
    )

    if merged_filename == original_filename:
        merged_filename = original_filename.replace(".gguf", "-merged.gguf")

    merged_path = os.path.join(base_dir, merged_filename)
    if os.path.exists(merged_path):
        print(f"Using existing merged GGUF file: {merged_path}")
        return merged_path

    command = ["gguf-split", "--merge", first_part_path, merged_path]
    print(f"Merging GGUF shards into: {merged_path}")

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The gguf-split command was not found in PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"gguf-split failed to merge GGUF shards: {exc.stderr}"
        ) from exc

    return merged_path


def download_model(model_id: str) -> None:
    """Download a Hugging Face model repository into the local cache."""
    print(f"Downloading model {model_id!r} to the Hugging Face cache.")
    snapshot_download(
        repo_id=model_id,
        repo_type="model",
        local_files_only=False,
    )


def find_free_port() -> int:
    """Ask the operating system for an available local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_handle:
        socket_handle.bind(("", 0))
        return socket_handle.getsockname()[1]


def get_swap_space_gb(
    percentage: float = 0.5,
    max_gb: int | None = None,
) -> int:
    """Return a RAM-based vLLM swap-space recommendation in GiB."""
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    swap_space_gb = int(total_ram_gb * percentage)

    if max_gb is not None:
        swap_space_gb = min(swap_space_gb, max_gb)

    return swap_space_gb


def wait_for_vllm_server_ready(
    server_process: subprocess.Popen,
    url: str,
    timeout: int = 1800,
) -> None:
    """Wait until the vLLM health and model-list endpoints are available."""
    print("\nWaiting for vLLM server readiness.")
    start_time = time.time()

    while time.time() - start_time < timeout:
        if server_process.poll() is not None:
            raise RuntimeError(
                "vLLM server terminated unexpectedly with exit code "
                f"{server_process.returncode}."
            )

        try:
            health_response = httpx.get(f"{url}/health", timeout=1.0)
            if health_response.status_code == 200:
                model_response = httpx.get(f"{url}/v1/models", timeout=2.0)
                if model_response.status_code == 200 and model_response.json().get("data"):
                    time.sleep(5)
                    return
        except httpx.RequestError:
            pass

        time.sleep(2)

    server_process.terminate()
    server_process.wait()
    raise TimeoutError(
        f"vLLM server did not become ready within {timeout} seconds."
    )
