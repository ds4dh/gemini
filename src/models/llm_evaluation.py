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
import pandas as pd
import datetime
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


def generate_extraction_summary_and_reports(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    run_dir: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Generates consensus summary DataFrame, saves summary_clinical_database.csv,
    and computes metrics against ground truth (if available) or missingness/completion statistics.
    Outputs extraction_report.json and report.md in the run_dir.
    """
    schema_fields = cfg.get("schema", {}).get("fields", {})
    target_field_names = list(schema_fields.keys()) if schema_fields else ["mRS"]

    patient_col = cfg.get("column_mapping", {}).get("patient_id", "patient_id")
    if patient_col not in df.columns:
        patient_col = "patient_id" if "patient_id" in df.columns else df.columns[0]

    summary_df = pd.DataFrame()
    summary_df[patient_col] = df[patient_col]

    field_evaluations = {}

    for field_name in target_field_names:
        # Find all repeat columns matching <field_name>_000, <field_name>_001, etc.
        repeat_cols = [c for c in df.columns if c == field_name or c.startswith(f"{field_name}_")]
        if not repeat_cols:
            continue

        field_def = schema_fields.get(field_name, {})
        field_type = field_def.get("type", "str").lower() if isinstance(field_def, dict) else "str"

        consensus_values = []
        for idx, row in df.iterrows():
            row_votes = [row[c] for c in repeat_cols if pd.notna(row[c])]
            if not row_votes:
                consensus_values.append(None)
            elif field_type in ("int", "integer", "float", "number"):
                # Numeric field: try median / mode
                num_votes = []
                for v in row_votes:
                    try:
                        num_votes.append(float(v))
                    except (ValueError, TypeError):
                        pass
                if num_votes:
                    val = float(np.median(num_votes))
                    consensus_values.append(int(val) if field_type in ("int", "integer") else round(val, 2))
                else:
                    consensus_values.append(None)
            else:
                # String / Enum field: majority vote
                str_votes = [str(v) for v in row_votes]
                most_common = Counter(str_votes).most_common(1)[0][0]
                consensus_values.append(most_common)

        summary_df[field_name] = consensus_values

        # Check for ground truth column
        gt_col_candidates = [
            f"ground_truth_{field_name}",
            f"{field_name}_gt",
            "ground_truth",
            "label"
        ]
        gt_col = next((c for c in gt_col_candidates if c in df.columns), None)

        valid_predictions = [v for v in consensus_values if v is not None]
        total_samples = len(df)
        completed_count = len(valid_predictions)
        completion_rate = completed_count / total_samples if total_samples > 0 else 0.0

        eval_data = {
            "field_name": field_name,
            "field_type": field_type,
            "total_samples": total_samples,
            "completed_samples": completed_count,
            "completion_rate": round(completion_rate * 100, 2),
            "ground_truth_available": gt_col is not None,
        }

        if gt_col is not None:
            gt_values = df[gt_col].tolist()
            summary_df[f"ground_truth_{field_name}"] = gt_values

            # Compute evaluation metrics against ground truth
            correct_count = 0
            evaluated_count = 0
            abs_errors = []

            for pred, gt in zip(consensus_values, gt_values):
                if pd.isna(gt) or gt is None or str(gt).strip() in ("", "nan", "None", "-1"):
                    continue
                evaluated_count += 1
                if str(pred).strip().lower() == str(gt).strip().lower():
                    correct_count += 1
                if field_type in ("int", "integer", "float", "number"):
                    try:
                        abs_errors.append(abs(float(pred) - float(gt)))
                    except (ValueError, TypeError):
                        pass

            accuracy = (correct_count / evaluated_count * 100) if evaluated_count > 0 else 0.0
            eval_data.update({
                "evaluated_samples_with_gt": evaluated_count,
                "accuracy": round(accuracy, 2),
                "mae": round(float(np.mean(abs_errors)), 3) if abs_errors else None,
            })
        else:
            eval_data.update({
                "evaluated_samples_with_gt": 0,
                "accuracy": None,
                "mae": None,
                "note": "Ground truth not provided (set to NaN)"
            })

        field_evaluations[field_name] = eval_data

    # Save summary CSV
    summary_filename = cfg.get("output", {}).get("summary_filename", "summary_clinical_database.csv")
    summary_path = os.path.join(run_dir, summary_filename)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary database at: {summary_path}")

    # Build report JSON
    report_json = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_path": cfg.get("model_path"),
        "inference_backend": cfg.get("inference_backend"),
        "quant_scheme": cfg.get("quant_scheme"),
        "total_records_processed": len(df),
        "fields_evaluation": field_evaluations,
    }

    report_filename = cfg.get("output", {}).get("report_filename", "extraction_report.json")
    report_json_path = os.path.join(run_dir, report_filename)
    with open(report_json_path, "w") as f:
        json.dump(report_json, f, indent=4)

    # Build human-readable Markdown report
    md_lines = [
        f"# Clinical Variable Extraction Report",
        f"",
        f"- **Timestamp**: `{report_json['timestamp']}`",
        f"- **Model**: `{report_json['model_path']}`",
        f"- **Backend**: `{report_json['inference_backend']}`",
        f"- **Total Records**: {report_json['total_records_processed']}",
        f"",
        f"## Variable Evaluation Summary",
        f"",
        f"| Variable | Type | Completion Rate | Evaluated (GT) | Accuracy / Match % | MAE |",
        f"| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for fname, feval in field_evaluations.items():
        acc_str = f"{feval['accuracy']}%" if feval['accuracy'] is not None else "N/A (No GT)"
        mae_str = f"{feval['mae']}" if feval['mae'] is not None else "N/A"
        gt_count = feval['evaluated_samples_with_gt']
        md_lines.append(
            f"| `{fname}` | `{feval['field_type']}` | {feval['completion_rate']}% | {gt_count} | {acc_str} | {mae_str} |"
        )

    report_md_path = os.path.join(run_dir, "report.md")
    with open(report_md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"Saved extraction report at: {report_json_path} and {report_md_path}")
    return summary_df, report_json
