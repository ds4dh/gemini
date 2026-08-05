import os
import argparse
import pandas as pd
from datasets import Dataset
from src.data.encryption import read_pandas_from_encrypted_file


def load_data_formatted_for_benchmarking(
    cfg_args: argparse.Namespace = None,
    use_curated_dataset: bool = False,
    add_curated_dataset: bool = False,
    remove_samples_without_label: bool = False,
    sample_small_dataset: bool = False,
    min_samples_per_class: int = 200,
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

    # Optional label filtering (only if label column exists)
    if "label" in df_data.columns and remove_samples_without_label:
        print("Filtering out samples without labels.")
        df_data = df_data.dropna(subset=["label"])

    # Optional small balanced dataset sampling (only if label column exists)
    if sample_small_dataset and "label" in df_data.columns:
        df_data = sample_small_balanced_dataset(df_data, min_samples_per_class)
    elif sample_small_dataset and len(df_data) > min_samples_per_class:
        print(f"Sampling small dataset of {min_samples_per_class} rows.")
        df_data = df_data.head(min_samples_per_class)

    return Dataset.from_pandas(df_data)


def sample_small_balanced_dataset(
    df_data: pd.DataFrame,
    min_samples_per_class: int = 200,
) -> pd.DataFrame:
    """
    Select a small portion of the data that has more or less balanced classes.
    """
    print("Sampling a small, balanced dataset.")
    sampled_chunks = []
    for _, group in df_data.groupby("label"):
        chunk = group.sample(n=min(len(group), min_samples_per_class))
        sampled_chunks.append(chunk)

    df_data = pd.concat(sampled_chunks, ignore_index=True)
    df_data = df_data.sample(frac=1)
    df_data = df_data.reset_index(drop=True)
    
    return df_data