import argparse
import pandas as pd
from datasets import Dataset
from src.data.encryption import read_pandas_from_encrypted_file


def load_data_formatted_for_benchmarking(
    cfg_args: argparse.Namespace,
    use_curated_dataset: bool = False,
    add_curated_dataset: bool = False,
    remove_samples_without_label: bool = False,
    sample_small_dataset: bool = False,
    min_samples_per_class: int = 200,
    *args, **kwargs,
) -> Dataset:
    """
    Load and preprocess data for benchmarking
    """
    # Load dataset file (small, curated dataset)
    if use_curated_dataset:
        if add_curated_dataset:
            raise ValueError(
                "Cannot use both 'use_curated_dataset' and 'add_curated_dataset' "
                "flags at the same time. Please choose one."
            )

        print("Loading curated dataset...")
        df_data = pd.read_csv(cfg_args.curated_data_path)

    # Load dataset file (large, non-curated dataset + encrypted)
    else:
        print("Loading encrypted dataset...")
        df_data = read_pandas_from_encrypted_file(
            encrypted_file_path=cfg_args.encrypted_data_path,
            encryption_key_var_name=cfg_args.key_name,
            hostname=cfg_args.hostname,
            username=cfg_args.username,
            remote_env_path=cfg_args.remote_env_path,
            port=cfg_args.port,
        )

    # Add small, curated dataset if required
    if add_curated_dataset:
        print("Adding curated dataset...")
        df_curated = pd.read_csv(cfg_args.curated_data_path)
        df_data = pd.concat([df_data, df_curated], ignore_index=True)

    # Check for the presence of benchmarking fields
    if "input_text" not in df_data.columns or "label" not in df_data.columns:
        raise KeyError("Missing expected columns: 'input_text', 'label'")

    # Replace label values and / or filter out samples without labels if specified
    if remove_samples_without_label:
        print("Filtering out samples without labels.")
        df_data = df_data.dropna(subset=["label"])

    # Benchmark the chosen model
    if sample_small_dataset:
        df_data = sample_small_balanced_dataset(df_data, min_samples_per_class)

    return Dataset.from_pandas(df_data)


# def sample_small_balanced_dataset(
#     df_data: pd.DataFrame,
#     min_samples_per_class: int = 200,
# ) -> pd.DataFrame:
#     """
#     Select a small portion of the data that has more or less balanced classes
#     """
#     print("Sampling a small, balanced dataset.")
#     df_data = df_data.groupby("label", group_keys=False)
#     df_data = df_data.apply(
#         lambda x: x.sample(n=min(len(x), min_samples_per_class)), 
#         include_groups=True,
#     )
#     df_data = df_data.sample(frac=1)
#     df_data = df_data.reset_index(drop=True)

#     return df_data


def sample_small_balanced_dataset(
    df_data: pd.DataFrame,
    min_samples_per_class: int = 200,
) -> pd.DataFrame:
    """
    Select a small portion of the data that has more or less balanced classes.
    """
    print("Sampling a small, balanced dataset.")
    
    # Iterate explicitly over the groups
    sampled_chunks = []
    for _, group in df_data.groupby("label"):
        # Sample from the group (which preserves all columns, including 'label')
        chunk = group.sample(n=min(len(group), min_samples_per_class))
        sampled_chunks.append(chunk)

    # Concatenate all chunks back together
    df_data = pd.concat(sampled_chunks, ignore_index=True)
    df_data = df_data.sample(frac=1)
    df_data = df_data.reset_index(drop=True)
    
    return df_data