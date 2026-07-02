import os
import re
import subprocess
import numpy as np
import torch
import json
from typing import Any
from collections import Counter
from datasets import Dataset
from huggingface_hub import HfApi
from requests.exceptions import ReadTimeout
from src.utils.plot_utils import POOLED_MODES


def pool_model_predictions(
    preds_and_labels: list[Dataset],
    pred_pool_mode: str,
    num_models: int | None = None,
) -> tuple[np.ndarray]:
    """
    Pool model predictions on the same set of samples given a pooling method
    """
    if num_models is not None and 0 < num_models <= len(preds_and_labels):
        preds_and_labels = preds_and_labels[:num_models]

    if pred_pool_mode == "single":
        if not preds_and_labels: return np.array([]), np.array([])
        y_true_pooled = preds_and_labels[0]["label"]
        y_pred_pooled = preds_and_labels[0]["mRS"]

    elif pred_pool_mode == "concatenation":
        y_true_pooled = sum([[v for v in col["label"]] for col in preds_and_labels], [])
        y_pred_pooled = sum([[v for v in col["mRS"]] for col in preds_and_labels], [])

    elif pred_pool_mode == "majority":
        if not preds_and_labels: return np.array([]), np.array([])
        y_true_pooled = preds_and_labels[0]["label"]
        y_pred_pooled = []
        num_samples = preds_and_labels[0].num_rows
        preds_by_model = [dataset["mRS"] for dataset in preds_and_labels]

        for i in range(num_samples):
            votes = [preds[i] for preds in preds_by_model]
            y_pred_pooled.append(Counter(votes).most_common(1)[0][0])

    else:
        raise ValueError("Invalid pooling mode (single, concatenation, majority)")

    return np.array(y_true_pooled), np.array(y_pred_pooled)


def extract_preds_and_labels(dataset: Dataset) -> list[Dataset]:
    """
    Extract a set of model predictions and output labels for different runs
    to a list of datasets with labels and predictions
    """
    num_models = sum(1 for f in dataset.features if f.startswith("mRS_"))
    dataset = dataset.map(lambda x: {"label": -1 if x["label"] is None else int(x["label"])})
    preds_and_labels = [{"label": dataset["label"]} for _ in range(num_models)]

    for feature in dataset.features:
        if feature.startswith("mRS_"):
            original_feature_name, index = feature.split("_")
            preds_and_labels[int(index)][original_feature_name] = dataset[feature]

    return [Dataset.from_dict(d) for d in preds_and_labels]


def convert_to_json_serializable(obj: Any) -> Any:
    """
    Recursively converts arrays and tensors in a dictionary to lists.
    """
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    elif isinstance(obj, tuple):
        return list(obj)
    return obj


def _record_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
    loc: tuple,
) -> dict:
    """
    Records performance metrics (Error Rate and Distance) for a given set of labels and predictions.
    """
    # Filter out samples without label (i.e., with y_true != -1)
    mask = y_true != -1
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Compute metrics
    error_rate = np.mean(y_true != y_pred, keepdims=True)
    distance = np.mean(np.abs(y_true - y_pred), keepdims=True)

    return {
        f"Error Rate\n({label})": {
            "values": error_rate,
            "unit": "%", "max_y": 1.0, "loc": (loc[0], 1), "color": "tab:red",
        },
        f"Distance\n({label})": {
            "values": distance,
            "unit": "mRS", "max_y": 5.0, "loc": (loc[0], 2), "color": "tab:orange",
        },
    }


def _record_metrics_for_pooling_modes(
    preds_and_labels: list[Dataset],
    pooling_modes: list[dict],
) -> tuple[dict, dict]:
    """
    Computes metrics for different pooling modes, dynamically handling any number of models.
    """
    y_true_dict = {}
    y_pred_dict = {}
    metric_dict = {}

    for i, mode in enumerate(pooling_modes, start=1):
        mode_name = mode["name"]
        y_true, y_pred = pool_model_predictions(
            preds_and_labels,
            mode["pred_pool_mode"],
            mode.get("num_models"),
        )
        y_true_dict[mode_name] = y_true
        y_pred_dict[mode_name] = y_pred
        
        # Add a conditional check to avoid errors if y_true is empty
        if y_true.size > 0:
            metrics = _record_metrics(y_true, y_pred, mode["label"], loc=(i, 1))
            metric_dict.update(metrics)

    return y_true_dict, y_pred_dict, metric_dict


def compute_and_save_metrics(
    benchmark_results: dict,
    model_path: str,
    output_path: str,
) -> dict:
    """
    Compute and save metrics for a set of model predictions
    """
    # Save inputs and raw outputs to a CSV file
    dataset: Dataset = benchmark_results.pop("dataset")
    os.makedirs(os.path.split(output_path)[0], exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"Saved raw results at {output_path}")

    # Extract predictions and labels
    preds_and_labels = extract_preds_and_labels(dataset)
    num_models = len(preds_and_labels)

    # Filter pooling modes based on the number of models
    pooling_modes = []
    for mode in POOLED_MODES:
        if mode.get("num_models", num_models) <= num_models:
            # For the 'all' mode, update the label with the actual number of models
            if mode["name"] == "all":
                mode = mode.copy() # Avoid modifying the global list
                mode["label"] = f"all {num_models} models"
            pooling_modes.append(mode)

    y_true_pooled, y_pred_pooled, pooled_metrics = _record_metrics_for_pooling_modes(
        preds_and_labels,
        pooling_modes,
    )

    # Basic model characteristics
    num_params = get_model_number_of_parameters(model_path)
    bits_per_param = get_model_bits_per_parameter(model_path, output_path)
    total_bits = num_params * bits_per_param
    params_weight = total_bits / (8 * 1024**3)

    # Build the final metric dictionary
    metric_dict = {
        "y_true": y_true_pooled,
        "y_pred": y_pred_pooled,
        **pooled_metrics,
        "Number of params": {
            "values": np.array([num_params / 10**9]),
            "unit": "Billions", "max_y": 75, "loc": (0, 0), "color": "tab:gray",
        },
        "Bits/param": {
            "values": np.array([bits_per_param]),
            "unit": "Bits", "max_y": 16, "loc": (0, 1), "color": "tab:brown",
        },
        "VRAM param usage": {
            "values": np.array([params_weight]),
            "unit": "GB", "max_y": 60.0, "loc": (0, 2), "color": "tab:blue",
        },
        "Time/sample": {
            "values": benchmark_results["time"],
            "unit": "s", "max_y": 100.0, "loc": (0, 3), "color": "tab:green",
        },
    }

    # Save plotted data to a handy json file
    metric_dict = convert_to_json_serializable(metric_dict)
    json_path = output_path.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump(metric_dict, f, indent=4)
        print(f"Saved metrics summary at {json_path}")

    return metric_dict


def print_gpu_info():
    """
    Print information about available GPU(s)
    """
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("GPU Information:\n")
        print(result.stdout)
    except FileNotFoundError:
        print("nvidia-smi not found. Ensure NVIDIA drivers are installed and accessible.")


def get_model_number_of_parameters(model_id: str) -> int:
    """
    Get the number of parameters in a huggingface model, using:
    - First, Hugging Face API metadata
    - Then, GGUF file metadata
    - Then, Safetensors metadata
    - Finally: fallback on regex pattern match using model_id
    """
    api = HfApi()
    model_info = None

    # Try fetching metadata
    try:
        model_info = api.model_info(model_id, timeout=30)
    except ReadTimeout:
        print(f"Timeout error for getting model info of {model_id}")

    # GGUF method
    try:
        return model_info.gguf["total"]
    except Exception:
        pass

    # Safetensors method
    try:
        return model_info.safetensors.total
    except Exception:
        pass

    # Regex fallback: look for things like "0.6B", "7B", "13B", "70B"
    match = re.search(r"(\d+(?:\.\d+)?)[ ]?B", model_id, re.IGNORECASE)
    if match:
        num = float(match.group(1))
        return int(num * 1e9)  # Convert B -> parameters

    match = re.search(r"(\d+(?:\.\d+)?)[ ]?M", model_id, re.IGNORECASE)
    if match:
        num = float(match.group(1))
        return int(num * 1e6)  # Convert M -> parameters

    print(f"Could not infer number of parameters for {model_id}, defaulting to 1")
    return 1


def get_model_bits_per_parameter(
    model_path: str,
    output_path: str,
) -> int:
    """
    Get the number of parameters using heuristics from my own file-naming system
    """
    # Identify quantization scheme
    scheme = output_path.split(model_path)[-1].strip("-").split(".")[0]
    scheme_check = output_path.split("-")[-1].split(".")[0]
    if scheme != scheme_check:
        print(f"Warning: quantization scheme might be wrong for {model_path}: {scheme}")

    # Model quantization without quantization scheme (non-GGUF)
    if "gptq" in model_path.lower(): return 4
    if "awq" in model_path.lower(): return 4
    if "fp4" in model_path.lower(): return 4
    if "fp8" in model_path.lower(): return 8

    # Try to identify the number of bits with my own heuristics
    match = re.search(r"^(?:I?Q)(\d+)", scheme)
    if match: return int(match.group(1))
    if scheme.lower() in ["f16", "fp16"]: return 16
    if scheme.lower() == "q8_0": return 8

    return 0


def get_gpu_memory_usage_by_pid():
    """
    Compute the amount of GPU memory used at the moment by the current PID
    
    Returns:
        float: GPU memory used in all available GPUs, in GB
    """
    # Collect the current process ID and the output of nvidia-smi
    pid = str(os.getpid())
    cmd = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,nounits,noheader"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    
    # Extract the amount of GPU memory used by the current PID
    MiB_used_by_pid = 0
    for line in result.stdout.strip().split('\n'):
        if not line.strip(): continue
        proc_pid, mem = [x.strip() for x in line.split(',')]
        if proc_pid == pid: MiB_used_by_pid += int(mem)
    
    # Convert to GB and return
    GB_used_by_pid = MiB_used_by_pid * 1024 ** 2 / 1000 ** 3
    return GB_used_by_pid


def save_consensus_predictions(
    dataset: Dataset,
    output_path: str,
    keep_columns: list[str] = ["text", "label", "case_id"],
) -> None:
    """
    Creates a CSV containing the 1st prediction, and majority votes for the
    first 3, 5, and 10 models for each sample.
    
    Args:
        dataset: The HuggingFace dataset containing raw model outputs.
        output_path: The base path where the main results were saved.
        keep_columns: List of original dataset columns to keep in the new CSV.
    """
    # Extract predictions per run using your existing helper
    preds_and_labels = extract_preds_and_labels(dataset)
    num_models = len(preds_and_labels)
    num_samples = len(dataset)

    # Extract all votes for sample across all runs
    new_rows = []
    for i in range(num_samples):
        votes = [run[i]["mRS"] for run in preds_and_labels]
        row = {col: dataset[i][col] for col in keep_columns if col in dataset.column_names}

        # Calculate consensi
        row["pred_1"] = votes[0] if num_models >= 1 else None
        if num_models >= 3:
            # .most_common(1) returns [(value, count)], we take [0][0] for the value
            row["pred_maj_3"] = Counter(votes[:3]).most_common(1)[0][0]
        else:
            row["pred_maj_3"] = None
        if num_models >= 5:
            row["pred_maj_5"] = Counter(votes[:5]).most_common(1)[0][0]
        else:
            row["pred_maj_5"] = None
        if num_models >= 10:
            row["pred_maj_10"] = Counter(votes[:10]).most_common(1)[0][0]
        else:
            row["pred_maj_10"] = None

        # Record data
        new_rows.append(row)

    # Save updated results
    consensus_dataset = Dataset.from_list(new_rows)
    dir_name, base_name = os.path.split(output_path)
    new_filename = base_name.replace(".csv", "_data_with_predictions.csv")
    if new_filename == base_name:  # handle case where .csv wasn't in the string
        new_filename = f"{base_name}_data_with_predictions.csv"        
    final_path = os.path.join(dir_name, new_filename)
    consensus_dataset.to_csv(final_path, index=False)
    print(f"Saved consensus predictions at {final_path}")
