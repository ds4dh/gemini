import os
import re
import yaml
import socket
import psutil
import shutil
from argparse import ArgumentParser
from huggingface_hub import scan_cache_dir


def add_model_arguments(parser: ArgumentParser) -> None:
    """ Parse and validate model arguments
    """
    model_group = parser.add_argument_group(
        title="Model configuration",
        description="Configuration options for model benchmarking",
    )

    model_group.add_argument(
        "-mc", "--model-config-path",
        default="configs/model_config.yaml",
        help="Path to the model configuration file"
    )

    model_group.add_argument(
        "-pc", "--prompt-config-path",
        default="configs/prompt_config.yaml",
        help="Path to the prompt configuration file"
    )


def add_data_arguments(parser: ArgumentParser) -> None:
    """
    Add arguments required for reading an encrypted file by fetching a remote key.
    """
    data_group = parser.add_argument_group(
        title="Data access and remote env configuration",
        description="Arguments for the local encrypted file and key identifier."
    )

    data_group.add_argument(
        "-dc", "--data-config-path",
        default="configs/data_config.yaml",
        help="Path to the data configuration file"
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
        default="default_non_encrypted_data_path.csv"
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
    """ Load configurations from YAML files specified in script_args
    """
    # Load configurations
    model_config = _load_config_from_yaml(script_args.model_config_path)
    data_config = _load_config_from_yaml(script_args.data_config_path)
    prompt_config = _load_config_from_yaml(script_args.prompt_config_path)

    # Combine all configurations into a single dictionary
    run_config = {**model_config, **data_config, **prompt_config}

    return run_config


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
    on systems with multiple network interfaces.
    """
    # Connects to a public DNS server to find the OS's preferred outbound IP
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            primary_ip = s.getsockname()[0]
    except Exception as e:
        print(f"Could not automatically determine the primary IP address: {e}")
        print("You may need to set GLOO_SOCKET_IFNAME and NCCL_SOCKET_IFNAME manually.")
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
    else:
        print("Failed to find a network interface matching the primary IP.")
        print("You may need to set GLOO_SOCKET_IFNAME and NCCL_SOCKET_IFNAME manually.")