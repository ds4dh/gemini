from argparse import ArgumentParser
import os
import re
import shutil
import socket
import sys

import psutil
import yaml
from huggingface_hub import scan_cache_dir


def add_model_arguments(parser: ArgumentParser) -> None:
    """ Parse and validate model arguments
    """
    model_group = parser.add_argument_group(
        title="Model configuration",
        description="Configuration options for model benchmarking and extraction",
    )

    model_group.add_argument(
        "-c", "--config", "--config-path",
        default="config.yaml",
        help="Path to the primary unified configuration file (default: config.yaml)"
    )

    model_group.add_argument(
        "-mc", "--model-config-path",
        default=None,
        help="Legacy model config path (optional)"
    )

    model_group.add_argument(
        "-pc", "--prompt-config-path",
        default=None,
        help="Legacy prompt config path (optional)"
    )


def add_data_arguments(parser: ArgumentParser) -> None:
    """
    Add arguments required for dataset access.
    """
    data_group = parser.add_argument_group(
        title="Data access and remote env configuration",
        description="Arguments for local/remote dataset access and encryption."
    )

    data_group.add_argument(
        "-dc", "--data-config-path",
        default=None,
        help="Legacy data config path (optional)"
    )

    data_group.add_argument(
        "--encrypted-data-path",
        "-ed",
        type=str,
        default="default_encrypted_data_path.encrypted.csv",
        help="Path to the local encrypted data file.",
    )

    data_group.add_argument(
        "--curated-data-path",
        "-cd",
        type=str,
        default="default_non_encrypted_data_path.csv",
        help="Path to the local non-encrypted data file.",
    )

    data_group.add_argument(
        "--key-name",
        "-kn",
        type=str,
        default="GEMINI",
        help="Name of the encryption key variable in the .env file on the remote server.",
    )

    data_group.add_argument(
        "--hostname",
        "-hn",
        type=str,
        default="host.name.is.required.com",
        help="Hostname or IP address of the remote server.",
    )

    data_group.add_argument(
        "--username",
        "-un",
        type=str,
        default="username_is_required",
        help="Username for the SSH connection.",
    )

    data_group.add_argument(
        "--remote-env-path",
        "-re",
        type=str,
        default="/home/username/project/required/env/path/.env",
        help="Path to the .env file on the remote server.",
    )

    data_group.add_argument(
        "--port",
        type=int,
        default=22,
        help="SSH port on the remote server (default: 22).",
    )


def _load_config_from_yaml(config_file_path: str) -> dict:
    """ Load configuration from a YAML file
    
    Args:
        config_file_path (str): Path to the YAML
    """
    try:
        with open(config_file_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_file_path}")
        exit(1)


def load_config_files(script_args) -> dict:
    """ Load configurations from split YAML files (configs/run_cfg.yaml & configs/extraction_cfg.yaml)
        or fallback single unified YAML file specified in script_args.
    """
    # Check if an explicit single unified config file path is provided via --config
    single_config_path = getattr(script_args, "config", None) or getattr(script_args, "config_path", None)

    if single_config_path and os.path.exists(single_config_path):
        print(f"Loading unified configuration from: {single_config_path}")
        raw_cfg = _load_config_from_yaml(single_config_path)
        return normalize_pipeline_config(raw_cfg)

    # Load split configurations: run_cfg.yaml and extraction_cfg.yaml
    run_config_path = getattr(script_args, "run_config", None) or getattr(script_args, "run_config_path", None) or "configs/run_cfg.yaml"
    extraction_config_path = getattr(script_args, "extraction_config", None) or getattr(script_args, "extraction_config_path", None) or "configs/extraction_cfg.yaml"

    run_cfg = _load_config_from_yaml(run_config_path) if os.path.exists(run_config_path) else {}
    extraction_cfg = _load_config_from_yaml(extraction_config_path) if os.path.exists(extraction_config_path) else {}

    if run_config_path and os.path.exists(run_config_path):
        print(f"Loading run configuration from: {run_config_path}")
    if extraction_config_path and os.path.exists(extraction_config_path):
        print(f"Loading extraction configuration from: {extraction_config_path}")

    merged_cfg = {**run_cfg, **extraction_cfg}
    return normalize_pipeline_config(merged_cfg)


def normalize_pipeline_config(raw_cfg: dict) -> dict:
    """
    Normalizes and flattens unified or legacy configuration dicts so all modules
    can access parameters reliably via top-level keys or nested sections.
    """
    cfg = raw_cfg.copy()

    # Extract nested sections if present
    model_section = cfg.get("model", {})
    data_section = cfg.get("data", {})
    prompt_section = cfg.get("prompt", {})
    output_section = cfg.get("output", {})
    schema_section = cfg.get("schema", {})

    # Flatten nested parameters into top-level dict for backwards compatibility
    for section in (model_section, data_section, prompt_section, output_section):
        if isinstance(section, dict):
            for k, v in section.items():
                if k not in cfg or cfg[k] is None:
                    cfg[k] = v

    # Normalize prompt template structures
    if "prompt_templates" not in cfg:
        cfg["prompt_templates"] = {
            "system_template": prompt_section.get("system_template") or cfg.get("system_template") or "{task_description}\n{domain_knowledge}\n{output_specifications}",
            "user_template": prompt_section.get("user_template") or cfg.get("user_template") or "Voici le texte d'entrée:\nDEBUT DU TEXTE:\n{input_text}\nFIN DU TEXTE",
        }

    if "context_data" not in cfg:
        cfg["context_data"] = prompt_section.get("context_data") or cfg.get("context_data") or {
            "task_description": "Tu es un expert médical.",
            "domain_knowledge": "",
            "output_specifications": "",
        }

    # Normalize schema
    if "schema" not in cfg and "output_schema_name" in cfg:
        cfg["schema"] = cfg["output_schema_name"]
    elif "schema" in cfg:
        cfg["schema_config"] = schema_section
        if isinstance(schema_section, dict) and "name" in schema_section:
            cfg["output_schema_name"] = schema_section["name"]

    # Normalize default data loading arguments
    if "data_loading_arguments" not in cfg:
        cfg["data_loading_arguments"] = {
            "use_curated_dataset": cfg.get("use_curated_dataset", False),
            "add_curated_dataset": cfg.get("add_curated_dataset", False),
            "remove_samples_without_label": cfg.get("remove_samples_without_label", False),
            "sample_small_dataset": cfg.get("sample_small_dataset", False),
            "min_samples_per_class": cfg.get("min_samples_per_class", 15),
        }

    # Defaults for essential fields
    cfg.setdefault("result_dir", "./results")
    cfg.setdefault("resume_previous_run", True)
    cfg.setdefault("save_chunk_size", 100)
    cfg.setdefault("inference_backend", "vllm")
    cfg.setdefault("model_path", "Qwen/Qwen2.5-0.5B-Instruct")
    cfg.setdefault("quant_scheme", None)
    cfg.setdefault("n_inference_repeats", 1)
    cfg.setdefault("enable_thinking", False)
    cfg.setdefault("max_thinking_tokens", None)
    cfg.setdefault("max_new_tokens", 512)
    cfg.setdefault("temperature", 0.1)
    cfg.setdefault("top_p", 0.9)
    cfg.setdefault("use_output_guide", False)
    cfg.setdefault("delete_model_cache_after_run", False)
    cfg.setdefault("logits_processors", [
        "src.models.logits_processors:ThinkingBudgetProcessor",
        "src.models.logits_processors:JSONParsingProcessor",
    ])

    return cfg



def extract_quant_method(
    model_id_or_path: str,
    quant_map: dict = {
        "awq": "awq",
        "gptq": "gptq",
        "gguf": "gguf",
        "fp8": "fp8",
        "eetq": "eetq",
        "aqlm": "aqlm",
        "hqq": "hqq",
    },  # schema: {name_in_model_id_or_path: name_in_like_vllm}
) -> str | None:
    """
    Extracts the quantization method from a model ID or path
    """
    # Standardize the input for case-insensitive matching.
    lower_model_id = model_id_or_path.lower()

    # Split the model ID by common delimiters to get potential keywords.
    # Delimiters include '/', '-', and '_'.
    parts = re.split(r'[/_-]', lower_model_id)

    # Iterate through the parts from right to left, as the quantization
    # method is almost always at the end of the name.
    for part in reversed(parts):
        if part in quant_map:
            return quant_map[part]

    # If no known quantization method is found, the model is likely in a 
    # native format like FP16 or BF16
    return None


def clean_model_cache(
    model_to_delete: str,
    quant_scheme: str | None = None,
):
    """
    Cleans specified cached revisions of a repository from the hugginface cache
    """
    # Find the cache repository to delete
    cache_info = scan_cache_dir()
    try:
        target_repo = next(r for r in cache_info.repos if r.repo_id == model_to_delete)
    except StopIteration:
        print(f"Model '{model_to_delete}' not found in cache.")
        return

    # Non-GGUF case (simply delete the model cache directory)
    if extract_quant_method(model_to_delete) != "gguf":
        print(f"Deleting entire model directory: {target_repo.repo_path}")
        shutil.rmtree(target_repo.repo_path)
        return

    # Identify GGUF-related cache files to delete using a generator
    files_to_delete = (
        file for revision in target_repo.revisions for file in revision.files
        if quant_scheme and quant_scheme in str(file.file_path)
    )

    # Iterate over the filtered files and delete them
    for file in files_to_delete:
        print(f"Deleting link: {file.file_path}")
        os.remove(file.file_path)
        if os.path.exists(file.blob_path):
            print(f"Deleting blob: {file.blob_path}")
            os.remove(file.blob_path)


def set_distributed_environment():
    """
    Automatically detects the primary network interface and sets environment
    variables for torch.distributed (Gloo/NCCL) to prevent socket binding errors
    on systems with multiple network interfaces. Also sets CUDA_LIB_PATH on Windows.
    """
    # On Windows, set CUDA_LIB_PATH and register DLL search directories for vLLM and flashinfer
    if sys.platform == "win32":
        import site, shutil
        for sp in site.getsitepackages():
            nv_path = os.path.join(sp, "nvidia")
            if os.path.exists(nv_path):
                for sub in os.listdir(nv_path):
                    if sub.startswith("cu"):
                        candidate = os.path.join(nv_path, sub)
                        cu_bin = os.path.join(candidate, "bin")
                        cu_x86 = os.path.join(cu_bin, "x86_64")
                        cu_lib = os.path.join(candidate, "lib")
                        torch_lib = os.path.join(sp, "torch", "lib")

                        # Ensure bin directory exists and contains cudart64_*.dll for flashinfer
                        os.makedirs(cu_bin, exist_ok=True)
                        for d in [cu_x86, torch_lib]:
                            if os.path.exists(d):
                                for f in os.listdir(d):
                                    if f.startswith("cudart64_") and f.endswith(".dll"):
                                        target_f = os.path.join(cu_bin, f)
                                        if not os.path.exists(target_f):
                                            try:
                                                shutil.copy2(os.path.join(d, f), target_f)
                                                print(f"Copied CUDA runtime DLL to: '{target_f}'")
                                            except Exception:
                                                pass

                        # Register Windows DLL directories
                        for dll_dir in [cu_bin, cu_x86, cu_lib, torch_lib]:
                            if os.path.exists(dll_dir):
                                try:
                                    os.add_dll_directory(dll_dir)
                                except Exception:
                                    pass
                                os.environ["PATH"] = dll_dir + ";" + os.environ.get("PATH", "")

                        # Add virtual environment Scripts directory to PATH for ninja execution
                        scripts_dir = os.path.dirname(sys.executable)
                        if scripts_dir and os.path.exists(scripts_dir):
                            os.environ["PATH"] = scripts_dir + ";" + os.environ.get("PATH", "")

                        os.environ["CUDA_PATH"] = candidate
                        os.environ["CUDA_LIB_PATH"] = cu_lib
                        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
                        print(f"Automatically set CUDA_PATH to: '{candidate}' and CUDA_LIB_PATH to: '{cu_lib}'")
                        break

        # On Windows, set MASTER_ADDR to loopback and unset socket ifnames so Gloo uses standard Windows loopback
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ.pop("GLOO_SOCKET_IFNAME", None)
        os.environ.pop("NCCL_SOCKET_IFNAME", None)

        # Assign random non-ephemeral ports to avoid socket TIME_WAIT state and default port 29550 collisions
        import random
        base_port = random.randint(30000, 44000)
        os.environ["VLLM_DP_MASTER_PORT"] = str(base_port)
        os.environ["MASTER_PORT"] = str(base_port)
        os.environ["VLLM_PORT"] = str(base_port + 20)
        return

    # Linux multi-interface detection
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            primary_ip = s.getsockname()[0]
    except Exception as e:
        print(f"Could not automatically determine the primary IP address: {e}")
        return

    # Find the interface name that corresponds to this IP address
    interface_name = None
    all_interfaces = psutil.net_if_addrs()
    for if_name, addrs in all_interfaces.items():
        for addr in addrs:
            if addr.family == socket.AF_INET and addr.address == primary_ip:
                interface_name = if_name
                break
        if interface_name:
            break

    # Set the environment variables for both Gloo (CPU) and NCCL (GPU) backends
    if interface_name:
        print(f"Automatically detected primary network interface: '{interface_name}' with IP: {primary_ip}")
        os.environ['GLOO_SOCKET_IFNAME'] = interface_name
        os.environ['NCCL_SOCKET_IFNAME'] = interface_name
        print(f"Successfully set GLOO_SOCKET_IFNAME and NCCL_SOCKET_IFNAME to '{interface_name}'")