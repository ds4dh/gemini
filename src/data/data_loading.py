import os
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset
from src.data.encryption import read_pandas_from_encrypted_file


def load_data_formatted_for_benchmarking(
    cfg_args: argparse.Namespace = None,
    use_curated_dataset: bool = False,
    add_curated_dataset: bool = False,
    remove_samples_without_label: bool = False,
    max_samples: int | None = None,
    input_path: str = None,
    input_text_column: str = "input_text",
    *args, **kwargs,
) -> Dataset:
    """
    Load and preprocess data for extraction or benchmarking.
    """
    df_data = None

    # Check for direct input_path override in kwargs or config
    direct_path = input_path or kwargs.get("input_path") or getattr(cfg_args, "input_path", None)

    if direct_path and os.path.exists(direct_path):
        print(f"Loading dataset directly from: {direct_path}")
        if direct_path.endswith(".csv"):
            df_data = pd.read_csv(direct_path)
        elif direct_path.endswith((".xlsx", ".xls")):
            df_data = pd.read_excel(direct_path)
        elif direct_path.endswith(".parquet"):
            df_data = pd.read_parquet(direct_path)
        else:
            df_data = pd.read_csv(direct_path)

    elif use_curated_dataset:
        if add_curated_dataset:
            raise ValueError(
                "Cannot use both 'use_curated_dataset' and 'add_curated_dataset' "
                "flags at the same time. Please choose one."
            )
        curated_path = getattr(cfg_args, "curated_data_path", "data/curated_dataset.csv")
        print(f"Loading curated dataset from: {curated_path}")
        df_data = pd.read_csv(curated_path)

    elif getattr(cfg_args, "encrypted_data_path", None) and os.path.exists(getattr(cfg_args, "encrypted_data_path", "")):
        print("Loading encrypted dataset...")
        df_data = read_pandas_from_encrypted_file(
            encrypted_file_path=cfg_args.encrypted_data_path,
            encryption_key_var_name=getattr(cfg_args, "key_name", "GEMINI"),
            hostname=getattr(cfg_args, "hostname", ""),
            username=getattr(cfg_args, "username", ""),
            remote_env_path=getattr(cfg_args, "remote_env_path", ""),
            port=getattr(cfg_args, "port", 22),
        )

    else:
        # Fallback search for synthetic or local default data
        default_paths = [
            "data/synthetic_clinical_notes.csv",
            "data/data_2024/processed/dataset.csv",
        ]
        for path in default_paths:
            if os.path.exists(path):
                print(f"Loading fallback dataset from: {path}")
                df_data = pd.read_csv(path)
                break

    if df_data is None:
        raise FileNotFoundError(
            "Could not locate dataset. Please specify a valid 'input_path' in configuration or CLI arguments."
        )

    # Rename custom text column to standard 'input_text' if needed
    if input_text_column and input_text_column in df_data.columns and input_text_column != "input_text":
        df_data = df_data.rename(columns={input_text_column: "input_text"})

    # Ensure mandatory input_text column exists
    if "input_text" not in df_data.columns:
        # If there's a text column with a similar name, use it
        possible_text_cols = [c for c in df_data.columns if "text" in c.lower() or "letter" in c.lower() or "note" in c.lower()]
        if possible_text_cols:
            df_data = df_data.rename(columns={possible_text_cols[0]: "input_text"})
        else:
            raise KeyError(f"Missing mandatory input text column in dataset. Available columns: {list(df_data.columns)}")

    gt_columns = [col for col in df_data.columns if col.endswith("_ground_truth")]
    if remove_samples_without_label:
        if gt_columns:
            original_count = len(df_data)  # keep a row if at least one target feature has ground truth
            df_data = df_data.dropna(subset=gt_columns, how="all").reset_index(drop=True)
            print("Removed rows without any ground-truth target: " f"{original_count} -> {len(df_data)}.")
        else:
            print("Warning: remove_samples_without_label=True, but no '*_ground_truth' columns were found. No rows removed.")

    # Optional subsampling for debug
    if max_samples is not None:
        df_data = sample_small_balanced_dataset(df_data=df_data, max_samples=max_samples, gt_columns=gt_columns)

    return Dataset.from_pandas(df_data)


def sample_small_balanced_dataset(
    df_data: pd.DataFrame,
    max_samples: int = 200,
    gt_columns: list[str] | None = None,
    random_state: int = 1234,
) -> pd.DataFrame:
    """
    Build a small, reproducible debug subset distributed across combined ground-truth profiles.
    A profile is the tuple of all values in the available '<feature_name>_ground_truth' columns for one row.
    This works with:
    - multiple target features;
    - categorical, ordinal, binary, and continuous target values;
    - rows with partial or fully missing ground truth;
    - datasets without any ground-truth columns.
    The returned DataFrame contains at most max_samples rows.
    """
    if max_samples < 1:
        raise ValueError(f"max_samples must be at least 1; received {max_samples}.")
    if df_data.empty:
        print("Dataset is empty; no debug subsampling performed.")
        return df_data.copy()

    target_size = min(max_samples, len(df_data))
    if gt_columns is None:
        gt_columns = [column for column in df_data.columns if column.endswith("_ground_truth")]
    gt_columns = [column for column in gt_columns if column in df_data.columns]

    # No labels: retain a random subset rather than the first N rows.
    if not gt_columns:
        print(f"No '*_ground_truth' columns found. Randomly sampling {target_size}/{len(df_data)} rows.")
        return df_data.sample(n=target_size, random_state=random_state).reset_index(drop=True)

    # Convert NaN, pd.NA, and None to a common hashable sentinel.
    print(f"Building distributed debug subset using ground-truth columns: {gt_columns}")
    missing_sentinel = "<MISSING>"
    label_frame = df_data[gt_columns].astype(object).where(df_data[gt_columns].notna(), missing_sentinel)

    # A tuple creates a multi-feature profile without assuming labels are categorical or discrete.
    profile_series = label_frame.apply(lambda row: tuple(row.tolist()), axis=1)
    working_df = df_data.copy()
    working_df["_debug_label_profile"] = profile_series
    profile_groups = [
        group.sample(frac=1.0, random_state=random_state + group_index).index.tolist()
        for group_index, (_, group) in enumerate(
            working_df.groupby("_debug_label_profile", sort=False, dropna=False)
        )
    ]

    # Shuffle profile order, independently from row order inside profiles.
    rng = np.random.default_rng(random_state)
    rng.shuffle(profile_groups)

    # Round-robin sampling ensures broad profile coverage before repeatedly drawing from common profiles.
    selected_indices: list[int] = []
    while profile_groups and len(selected_indices) < target_size:
        remaining_groups: list[list[int]] = []

        for group_indices in profile_groups:
            if len(selected_indices) >= target_size:
                break

            selected_indices.append(group_indices.pop())
            if group_indices:
                remaining_groups.append(group_indices)

        profile_groups = remaining_groups

    result = df_data.loc[selected_indices].sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    print(f"Selected distributed debug subset: {len(result)}/{len(df_data)} rows; {len(profile_groups)} profiles still had unselected rows.")

    return result


def estimate_token_distribution(
    dataset: Dataset,
    text_column: str = "input_text",
    tokenizer_name_or_path: str | None = None,
) -> dict:
    """
    Computes exact or heuristic token counts across dataset entries and prints distribution metrics.
    """
    texts = [str(t) for t in dataset[text_column] if t is not None]
    if not texts:
        print("No valid text records found to compute token counts.")
        return {}

    token_counts = []
    mode = "heuristic"

    # Attempt exact tokenization if a model/tokenizer identifier is provided
    if tokenizer_name_or_path:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
            print(f"Tokenizing using exact tokenizer: '{tokenizer_name_or_path}'...")
            # Fast batch encoding without special token overhead
            encodings = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
            token_counts = [len(ids) for ids in encodings]
            mode = "exact"
        except Exception as err:
            print(f"Failed to load tokenizer '{tokenizer_name_or_path}' ({err}). Falling back to heuristic estimation.")

    # Heuristic fallback: Clinical texts average ~3.8-4.2 characters or ~1.3 tokens per whitespace word
    if not token_counts:
        print("Using heuristic estimation (~1 token ≈ 4 characters / 0.75 words)...")
        token_counts = [
            max(1, int(np.ceil(max(len(t) / 4.0, len(t.split()) / 0.75))))
            for t in texts
        ]

    counts_arr = np.array(token_counts)
    stats = {
        "count": len(counts_arr),
        "mean": float(np.mean(counts_arr)),
        "std": float(np.std(counts_arr)),
        "min": int(np.min(counts_arr)),
        "p25": int(np.percentile(counts_arr, 25)),
        "median (p50)": int(np.median(counts_arr)),
        "p75": int(np.percentile(counts_arr, 75)),
        "p95": int(np.percentile(counts_arr, 95)),
        "p99": int(np.percentile(counts_arr, 99)),
        "max": int(np.max(counts_arr)),
    }

    print("\n" + "=" * 55)
    print(f" TOKEN LENGTH DISTRIBUTION ANALYSIS ({mode.upper()}) ")
    print("=" * 55)
    print(f"Records Evaluated : {stats['count']}")
    print(f"Mean ± Std        : {stats['mean']:.1f} ± {stats['std']:.1f}")
    print(f"Min / Max         : {stats['min']} / {stats['max']}")
    print(f"Median (p50)      : {stats['median (p50)']}")
    print(f"p75 / p95 / p99   : {stats['p75']} / {stats['p95']} / {stats['p99']}")
    print("=" * 55)

    # Context length recommendation
    recommended_len = int(np.ceil(stats["p99"] + 1024))  # account for generation/thinking budget
    print(f"Suggested vLLM --max-model-len: {min(recommended_len, 8192)} (based on p99 + 1024 budget)\n")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test data loading and compute input token distribution.")
    parser.add_argument("--input-path", type=str, default="data/data_2024/processed/dataset.csv", help="Path to input dataset file.")
    parser.add_argument("--text-column", type=str, default="input_text", help="Name of input text column.")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3.8-27B", help="Hugging Face tokenizer name or local path.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max sample count.")
    args = parser.parse_args()

    # Load dataset
    ds = load_data_formatted_for_benchmarking(
        input_path=args.input_path,
        input_text_column=args.text_column,
        max_samples=args.max_samples,
    )

    # Estimate distribution
    estimate_token_distribution(
        dataset=ds,
        text_column="input_text",
        tokenizer_name_or_path=args.tokenizer,
    )