from argparse import ArgumentParser
import os
import re
import yaml
import shutil
import psutil
import socket
import sys

from typing import Any
from copy import deepcopy
from huggingface_hub import scan_cache_dir

THINKING_JSON_ADAPTER_PROCESSOR = (
    "src.models.logits_processors:ThinkingJSONAdapterProcessor"
)
VALID_REASONING_EFFORTS = {"auto", "off", "low", "medium", "high", "xhigh"}



def add_model_arguments(parser: ArgumentParser) -> None:
    """ Parse and validate model arguments
    """
    model_group = parser.add_argument_group(
        title="Model configuration",
        description="Configuration options for model benchmarking and extraction",
    )

    model_group.add_argument(
        "-rc", "--run-config",
        default="configs/run_cfg.yaml",
        help="Path to the primary run configuration file (default: configs/run_cfg.yaml)"
    )

    model_group.add_argument(
        "-ec", "--extraction-config",
        default=None,
        help="Path to the specific extraction configuration file (optional override)"
    )

    model_group.add_argument(
        "-c", "--config",
        default=None,
        help="Path to unified single configuration file (optional override)"
    )


def add_data_arguments(parser: ArgumentParser) -> None:
    """
    Add arguments required for dataset access.
    """
    data_group = parser.add_argument_group(
        title="Data access configuration",
        description="Arguments for local/remote dataset access and encryption."
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
        help="Name of the encryption key variable in the .env file.",
    )


def _load_config_from_yaml(config_file_path: str) -> dict[str, Any]:
    """Load and validate one YAML configuration mapping."""
    try:
        with open(config_file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Configuration file not found: {config_file_path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML configuration: {config_file_path}"
        ) from exc

    if not isinstance(config, dict):
        raise TypeError(
            f"Configuration root must be a YAML mapping: {config_file_path}"
        )

    return config


def _deep_merge_config(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge config dictionaries; override takes precedence."""
    merged = deepcopy(base)

    for key, override_value in override.items():
        base_value = merged.get(key)

        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = _deep_merge_config(base_value, override_value)
        else:
            merged[key] = deepcopy(override_value)

    return merged


def load_config_files(script_args) -> dict[str, Any]:
    """
    Load either a unified YAML configuration or a run configuration plus
    an extraction configuration, then normalize the combined result.
    """
    single_config_path = (
        getattr(script_args, "config", None)
        or getattr(script_args, "config_path", None)
    )

    if single_config_path:
        print(f"Loading unified configuration from: {single_config_path}")
        return normalize_pipeline_config(
            _load_config_from_yaml(single_config_path)
        )

    run_config_path = (
        getattr(script_args, "run_config", None)
        or getattr(script_args, "run_config_path", None)
        or "configs/run_cfg.yaml"
    )

    print(f"Loading run configuration from: {run_config_path}")
    run_cfg = _load_config_from_yaml(run_config_path)

    cli_extraction_path = (
        getattr(script_args, "extraction_config", None)
        or getattr(script_args, "extraction_config_path", None)
    )

    extraction_config_path = (
        cli_extraction_path
        or run_cfg.get("data", {}).get("extraction_config_path")
        or run_cfg.get("extraction_config_path")
        or "configs/extraction_cfgs/mrs_score.yaml"
    )

    print(f"Loading extraction configuration from: {extraction_config_path}")
    extraction_cfg = _load_config_from_yaml(extraction_config_path)

    merged_cfg = _deep_merge_config(run_cfg, extraction_cfg)
    merged_cfg["_active_run_config_path"] = run_config_path
    merged_cfg["_active_extraction_config_path"] = extraction_config_path

    return normalize_pipeline_config(merged_cfg)


def normalize_pipeline_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Validate the nested configuration and expose non-reasoning model/data/output
    values at top level for existing modules.

    Reasoning remains canonical and nested at:
        cfg["model"]["reasoning"]
    """
    cfg = deepcopy(raw_cfg)

    model_section = cfg.get("model") or {}
    data_section = cfg.get("data") or {}
    prompt_section = cfg.get("prompt") or {}
    output_section = cfg.get("output") or {}

    for section_name, section in (
        ("model", model_section),
        ("data", data_section),
        ("prompt", prompt_section),
        ("output", output_section),
    ):
        if not isinstance(section, dict):
            raise TypeError(
                f"Configuration section '{section_name}' must be a mapping."
            )

    # Keep the nested sections intact.
    cfg["model"] = model_section
    cfg["data"] = data_section
    cfg["prompt"] = prompt_section
    cfg["output"] = output_section

    # Flatten non-reasoning values for existing modules that still expect them.
    # Do not flatten model.reasoning: it remains model.reasoning only.
    for key, value in model_section.items():
        if key != "reasoning":
            cfg.setdefault(key, value)

    for section in (data_section, prompt_section, output_section):
        for key, value in section.items():
            cfg.setdefault(key, value)

    # --------------------------------------------------------------------------
    # Prompt configuration
    # --------------------------------------------------------------------------
    cfg.setdefault(
        "prompt_templates",
        {
            "system_template": (
                prompt_section.get("system_template")
                or cfg.get("system_template")
                or (
                    "{task_description}\n"
                    "{domain_knowledge}\n"
                    "{output_specifications}"
                )
            ),
            "user_template": (
                prompt_section.get("user_template")
                or cfg.get("user_template")
                or (
                    "Voici le texte d'entrée:\n"
                    "DEBUT DU TEXTE:\n"
                    "{input_text}\n"
                    "FIN DU TEXTE"
                )
            ),
        },
    )

    cfg.setdefault(
        "context_data",
        prompt_section.get("context_data")
        or cfg.get("context_data")
        or {
            "task_description": "Tu es un expert médical.",
            "domain_knowledge": "",
            "output_specifications": "",
        },
    )

    # --------------------------------------------------------------------------
    # Schema configuration
    # --------------------------------------------------------------------------
    schema_value = cfg.get("schema")

    if isinstance(schema_value, dict):
        cfg["schema_config"] = schema_value

        if "name" in schema_value:
            cfg["output_schema_name"] = schema_value["name"]

    elif schema_value is None and "output_schema_name" in cfg:
        cfg["schema"] = cfg["output_schema_name"]

    # --------------------------------------------------------------------------
    # Data-loading configuration
    # --------------------------------------------------------------------------
    cfg.setdefault(
        "data_loading_arguments",
        {
            "use_curated_dataset": cfg.get("use_curated_dataset", False),
            "add_curated_dataset": cfg.get("add_curated_dataset", False),
            "remove_samples_without_label": cfg.get(
                "remove_samples_without_label",
                False,
            ),
            "max_samples": cfg.get("max_samples"),
        },
    )

    # --------------------------------------------------------------------------
    # General defaults
    # --------------------------------------------------------------------------
    cfg.setdefault("result_dir", "./results")
    cfg.setdefault("resume_previous_run", True)
    cfg.setdefault("save_chunk_size", 100)

    cfg.setdefault("inference_backend", "vllm")
    cfg.setdefault("model_path", "Qwen/Qwen2.5-0.5B-Instruct")
    cfg.setdefault("quant_scheme", None)

    raw_reasoning_parser = (
        model_section.get("reasoning_parser")
        if "reasoning_parser" in model_section
        else cfg.get("reasoning_parser", "auto")
    )
    cfg["reasoning_parser"] = resolve_reasoning_parser(
        cfg["model_path"],
        raw_reasoning_parser,
    )

    cfg.setdefault("n_inference_repeats", 1)
    cfg.setdefault("max_concurrent_requests", 1)
    cfg.setdefault("max_concurrent_inferences", 1)

    cfg.setdefault("max_new_tokens", 512)
    cfg.setdefault("max_context_length", None)

    cfg.setdefault("temperature", 0.1)
    cfg.setdefault("top_p", 0.9)
    cfg.setdefault("top_k", 0)
    cfg.setdefault("min_p", 0.0)
    cfg.setdefault("presence_penalty", 0.0)
    cfg.setdefault("repetition_penalty", 1.0)

    cfg.setdefault("use_output_guide", False)
    cfg.setdefault("delete_model_cache_after_run", False)
    cfg.setdefault("enforce_eager", False)

    # --------------------------------------------------------------------------
    # Canonical nested reasoning configuration
    # --------------------------------------------------------------------------
    reasoning = model_section.get("reasoning") or {}

    if not isinstance(reasoning, dict):
        raise TypeError("model.reasoning must be a mapping.")

    normalized_reasoning = {
        "enabled": reasoning.get("enabled", "auto"),
        "effort": reasoning.get("effort", "auto"),
        "preserve_thinking": reasoning.get("preserve_thinking", False),
        "hard_thinking_token_budget": reasoning.get(
            "hard_thinking_token_budget",
            None,
        ),
    }

    if normalized_reasoning["enabled"] not in (True, False, "auto"):
        raise ValueError(
            "model.reasoning.enabled must be true, false, or 'auto'."
        )

    if normalized_reasoning["effort"] not in VALID_REASONING_EFFORTS:
        raise ValueError(
            "model.reasoning.effort must be one of "
            f"{sorted(VALID_REASONING_EFFORTS)}."
        )

    if not isinstance(normalized_reasoning["preserve_thinking"], bool):
        raise TypeError(
            "model.reasoning.preserve_thinking must be boolean."
        )

    hard_budget = normalized_reasoning["hard_thinking_token_budget"]

    if hard_budget is not None and (
        not isinstance(hard_budget, int)
        or isinstance(hard_budget, bool)
        or hard_budget < 0
    ):
        raise ValueError(
            "model.reasoning.hard_thinking_token_budget must be a "
            "non-negative integer or null."
        )

    if hard_budget is not None and cfg["max_new_tokens"] <= hard_budget:
        raise ValueError(
            "max_new_tokens must exceed "
            "model.reasoning.hard_thinking_token_budget."
        )

    if (
        hard_budget is not None
        and cfg["inference_backend"] not in {
            "vllm-serve",
            "vllm-serve-async",
        }
    ):
        raise ValueError(
            "model.reasoning.hard_thinking_token_budget requires "
            "inference_backend='vllm-serve' or "
            "inference_backend='vllm-serve-async'."
        )

    cfg["model"]["reasoning"] = normalized_reasoning

    # --------------------------------------------------------------------------
    # Guided-output / custom processor validation
    # --------------------------------------------------------------------------
    if cfg["use_output_guide"]:
        configured_processors = cfg.get("logits_processors")

        if configured_processors is None:
            cfg["logits_processors"] = [
                THINKING_JSON_ADAPTER_PROCESSOR
            ]

        elif configured_processors != [THINKING_JSON_ADAPTER_PROCESSOR]:
            raise ValueError(
                "use_output_guide=True requires exactly "
                f"[{THINKING_JSON_ADAPTER_PROCESSOR!r}] in "
                "logits_processors."
            )

    else:
        cfg["logits_processors"] = []

    if hard_budget is not None and not cfg["use_output_guide"]:
        raise ValueError(
            "model.reasoning.hard_thinking_token_budget currently requires "
            "use_output_guide=True because ThinkingJSONAdapterProcessor owns "
            "the hard-cap logic."
        )

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


def resolve_reasoning_parser(
    model_id_or_path: str,
    reasoning_parser: str | None = "auto",
) -> str | None:
    """
    Resolves the reasoning parser to pass to vLLM's OpenAI server entrypoint.
    If reasoning_parser is 'auto', inspects the model ID/path and HuggingFace tags/card
    for known reasoning parser patterns (qwen3, deepseek_r1, granite, hunyuan).
    """
    if reasoning_parser is None or reasoning_parser is False or reasoning_parser == "":
        return None

    if isinstance(reasoning_parser, str) and reasoning_parser.lower() not in ("auto", ""):
        return reasoning_parser

    # Auto resolution based on model_id_or_path inspection
    lower_path = model_id_or_path.lower()

    if any(k in lower_path for k in ("qwen3", "qwen-3", "qwen_3")):
        return "qwen3"

    if any(k in lower_path for k in ("deepseek-r1", "deepseek_r1", "r1-distill", "r1_distill")):
        return "deepseek_r1"

    if "granite" in lower_path and any(k in lower_path for k in ("reasoning", "think")):
        return "granite"

    if "hunyuan" in lower_path and any(k in lower_path for k in ("reasoning", "think")):
        return "hunyuan"

    # Secondary check: Hugging Face hub metadata
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        model_info = api.model_info(model_id_or_path)

        card_data = model_info.card_data or {}
        base_model = card_data.get("base_model") or ""
        if isinstance(base_model, list):
            base_model = " ".join(base_model)

        tags = " ".join(model_info.tags or []).lower()
        search_target = f"{base_model} {tags}".lower()

        if any(k in search_target for k in ("qwen3", "qwen-3", "qwen_3")):
            return "qwen3"
        if any(k in search_target for k in ("deepseek-r1", "deepseek_r1", "r1-distill")):
            return "deepseek_r1"
        if "granite" in search_target and "reasoning" in search_target:
            return "granite"
        if "hunyuan" in search_target and "reasoning" in search_target:
            return "hunyuan"
    except Exception:
        pass

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
        
        # Add IPv4 force flags to prevent IPv6/hostname fallback errors
        os.environ['TP_SOCKET_IFNAME'] = interface_name
        os.environ['GLOO_FORCE_IPV4'] = '1'
        os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand if using standard Ethernet
        print(f"Successfully set GLOO_SOCKET_IFNAME and NCCL_SOCKET_IFNAME to '{interface_name}'")