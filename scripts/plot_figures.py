import os
import json
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import statsmodels.formula.api as smf


INPUT_DIR = "results/vllm-serve-async_guided"
OUTPUT_DIR = os.path.join(INPUT_DIR, "pooled")
OUTPUT_NAME = "pooled_results"
MAIN_X_VARIABLE = "vram"  # "vram", "nparams", "nbits"
TARGET_VARIABLES = ["error", "distance"]
CASE_MAPPING = {
    "single model": "single",
    # "maj-pooling-3": "maj_3",
    "maj-pooling-5": "maj_5", 
    "maj-pooling-10": "maj_10",
    "all 10 models": "all",
}
CASES = list(CASE_MAPPING.keys())
X_CONFIGS = {
    "vram": {"key": "VRAM param usage", "unit": "GB", "lim": [0.1, 80.0], "log": True},
    "nparams": {"key": "Number of params", "unit": "Billion", "lim": None, "log": True},
    "nbits": {"key": "Bits/param", "unit": "[]", "lim": None, "log": False},
}
Y_CONFIGS = {
    "error": {"key": "Error Rate", "unit": "%", "lim": [0.0, 1.1], "tick": 0.2, "log": False},
    "distance": {"key": "Distance", "unit": "mRS unit", "lim": [0.1, 3.1], "tick": 1.0, "log": False},
}
GROUP_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]


def _extract_metric_value(
    data_source: dict,
    key: str,
    default_val: float=None,
) -> float:
    """ Helper to extract a single metric value
    """
    item_data = data_source.get(key)
    if item_data and isinstance(item_data.get("values"), list) and item_data["values"]:
        try:
            return float(item_data["values"][0])  # Get the first element
        except (ValueError, TypeError, IndexError) as e:
            print(f"Warning: Could not parse value for key {key} from {item_data.get('values')}: {e}. Using default: {default_val}.")

    return default_val


def generate_pooled_metric_plots(
    result_path_group: dict[str, list],
    output_name: str,
    output_dir: str,
    target_variable: str,
) -> None:
    """ Generate a single figure with subplots for Error Rate vs VRAM,
        each model group plotted with a different color within each subplot
    """
    # Plot each case data to a subplot
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    num_cols = 2
    num_rows = math.ceil(len(CASES) / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 4 * num_rows), squeeze=False)
    axes_flat = axes.flatten()
    csv_data = []
    
    for i, case_name in enumerate(CASES):
        raw_key = CASE_MAPPING.get(case_name, "single")  # get internal key
        
        for group_idx, (group_label, result_paths_in_group) in enumerate(result_path_group.items()):
            group_data_points = []
            group_color = GROUP_COLORS[group_idx % len(GROUP_COLORS)]
            for result_path in result_paths_in_group:

                # Load data
                try:
                    with open(result_path, "r") as f:
                        data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Warning: Could not load or parse JSON file {result_path}: {e}. Skipping.")
                    continue

                extracted_data = {"model": group_label}

                # Extract X values
                for x_id in X_CONFIGS:
                    query_key = X_CONFIGS[x_id]["key"]
                    extracted_data[x_id] = _extract_metric_value(data, query_key)

                # Extract Y values and compute confidence intervals
                for y_id in Y_CONFIGS:
                    # Get the mean value
                    query_key = f"{Y_CONFIGS[y_id]['key']}\n({case_name})"
                    y_id_cased = f"{y_id} - {case_name}"
                    extracted_data[y_id_cased] = _extract_metric_value(data, query_key)

                    # Bootstrap confidence intervals using raw data
                    try:
                        raw_true = data.get("y_true", {}).get(raw_key, [])
                        raw_pred = data.get("y_pred", {}).get(raw_key, [])
                        low_err, up_err = calculate_bootstrap_ci(raw_true, raw_pred, y_id)
                        extracted_data[f"{y_id_cased}_err_low"] = low_err
                        extracted_data[f"{y_id_cased}_err_high"] = up_err
                    except Exception as e:
                        print(f"Warning: CI calc failed for {result_path}: {e}")
                        extracted_data[f"{y_id_cased}_err_low"] = 0.0
                        extracted_data[f"{y_id_cased}_err_high"] = 0.0

                # Temporary fix for fp8
                if "fp8" in result_path.lower():
                    extracted_data["nbits"] = extracted_data["nbits"] * 2
                    extracted_data["vram"] = extracted_data["vram"] * 2
                
                group_data_points.append(extracted_data)
            
            # Remove points where the X or Y data is None (missing in JSON)
            plotted_y_key = f"{target_variable} - {case_name}"
            valid_points = []
            for dp in group_data_points:
                x_val = dp.get(MAIN_X_VARIABLE)
                y_val = dp.get(plotted_y_key)
                if x_val is not None and y_val is not None:
                    valid_points.append(dp)
            group_data_points = valid_points
            if not group_data_points: continue

            # Scatter plot for the current group
            x_values = [dp[MAIN_X_VARIABLE] for dp in group_data_points]
            plotted_y_id_cased = f"{target_variable} - {case_name}"
            y_values = [dp[plotted_y_id_cased] for dp in group_data_points]
            sizes = [200 * dp["nbits"] / 16 for dp in group_data_points]

            # Plot the scatter points
            axes_flat[i].scatter(
                x_values, y_values, color=group_color, label=group_label,
                marker="o", alpha=0.9, s=sizes, edgecolors='white', linewidth=0.5, zorder=1,
            )

            # Plot the error bars
            rgb = mcolors.to_rgb(group_color)
            darker_color = tuple(c * 0.5 for c in rgb)
            y_err_low = [dp[f"{plotted_y_id_cased}_err_low"] for dp in group_data_points]
            y_err_high = [dp[f"{plotted_y_id_cased}_err_high"] for dp in group_data_points]
            axes_flat[i].errorbar(
                x_values, y_values, yerr=[y_err_low, y_err_high],
                elinewidth=0.75, markeredgewidth=0.75, fmt='none',
                ecolor=darker_color, alpha=0.7, capsize=3, zorder=2,
            )

            # Record data for pooled json/csv
            csv_data.extend(group_data_points)

        # Configure subplot
        x_label = f"{X_CONFIGS[MAIN_X_VARIABLE]['key']} [{X_CONFIGS[MAIN_X_VARIABLE]['unit']}]"
        y_label = f"{Y_CONFIGS[target_variable]['key']} [{Y_CONFIGS[target_variable]['unit']}]"
        if X_CONFIGS[MAIN_X_VARIABLE]['log']:
            axes_flat[i].set_xscale('log')
        if Y_CONFIGS[target_variable]['log']:
            axes_flat[i].set_yscale('log')
            axes_flat[i].yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
            axes_flat[i].yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs="auto", numticks=10))
        else:
            tick_dist = Y_CONFIGS[target_variable]["tick"]
            if tick_dist is not None:
                axes_flat[i].yaxis.set_major_locator(ticker.MultipleLocator(tick_dist))
                axes_flat[i].yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        axes_flat[i].set_xlabel(x_label, fontsize=12)
        axes_flat[i].set_ylabel(y_label, fontsize=12)
        if X_CONFIGS[MAIN_X_VARIABLE]["lim"] is not None:
            axes_flat[i].set_xlim(X_CONFIGS[MAIN_X_VARIABLE]["lim"])
        if Y_CONFIGS[target_variable]["lim"] is not None:
            axes_flat[i].set_ylim(Y_CONFIGS[target_variable]["lim"])
        axes_flat[i].tick_params(axis="y", labelsize=10)
        axes_flat[i].tick_params(axis="x", labelsize=10)
        axes_flat[i].grid(True, linestyle="--", alpha=0.6)
        axes_flat[i].set_title(f"Prediction with {case_name}", fontsize=14, pad=10)
        axes_flat[i].legend(loc="upper right", fontsize=10, fancybox=True, ncol=2)

    # Save the pooled results figure
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_full_path = os.path.join(output_dir, f"{output_name}_{target_variable}.png")
    plt.savefig(plot_full_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"Combined plot saved: {plot_full_path}")

    # Save the pooled data to a csv file using pandas
    csv_df = pd.DataFrame(csv_data)
    csv_df = csv_df.groupby(["model", "vram", "nparams", "nbits"]).first().reset_index()
    csv_full_path = os.path.join(output_dir, f"{output_name}_{target_variable}.csv")
    csv_df.to_csv(csv_full_path, index=False)
    print(f"Pooled data saved: {csv_full_path}")

    return plot_full_path, csv_full_path


# def fit_error_model_lme(
#     df_input: pd.DataFrame,
#     dependent_variable: str="error - all 10 models",
#     fixed_effect_0: str="nparams",
#     fixed_effect_1: str="nbits",
#     include_interaction: bool=True,
#     random_intercept_group: str="model",
# ):
#     """
#     Fits a linear mixed-effects model to the provided dataframe
#     """
#     # Load data
#     df = df_input.copy()
#     df.rename(columns={dependent_variable: "target"}, inplace=True)

#     # Correctly define the model formula
#     if include_interaction:
#         # The '*' operator includes main effects AND the interaction
#         model_formula = f"target ~ {fixed_effect_0} * {fixed_effect_1}"
#     else:
#         # The '+' operator includes main effects ONLY
#         model_formula = f"target ~ {fixed_effect_0} + {fixed_effect_1}"

#     # Create and fit the mixed-effects model
#     md = smf.mixedlm(model_formula, df, groups=df[random_intercept_group])
#     return md.fit()

def fit_error_model_lme(
    df_input: pd.DataFrame,
    dependent_variable: str = "error - all 10 models",
    fixed_effect_0: str = "nparams",
    fixed_effect_1: str = "nbits",
    include_interaction: bool = True,
    random_intercept_group: str = "model",
):
    """
    Fits a linear mixed-effects model to the provided dataframe
    """
    # Load data
    df = df_input.copy()

    # Log-transform nparams to account for LLM scaling laws
    df["log_nparams"] = np.log10(df[fixed_effect_0])
    fe_0_term = "log_nparams"  # update the fixed effect name to use the log version
    
    # Keep nbits linear since we take 2, 3, 4, 5, 6, 8 and we have no further hypothesis
    fe_1_term = fixed_effect_1
    
    # Keep the target variable linear because no further hypothesis
    df["target"] = df[dependent_variable]
    
    # Correctly define the model formula
    if include_interaction:
        model_formula = f"target ~ {fe_0_term} * {fe_1_term}"  # "*" combines "+" and "x"
    else:
        model_formula = f"target ~ {fe_0_term} + {fe_1_term}"
    print(f"Fitting LME with formula: {model_formula}")

    # Create and fit the mixed-effects model
    md = smf.mixedlm(model_formula, df, groups=df[random_intercept_group])
    return md.fit()


def calculate_bootstrap_ci(
    y_true: list | np.ndarray, 
    y_pred: list | np.ndarray, 
    metric_type: str, 
    n_bootstraps: int = 1000,
) -> tuple[float, float]:
    """
    Calculate 95% CI (lower_diff, upper_diff) relative to the mean using bootstrapping.
    """
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    
    # Filter out invalid labels (-1)
    mask = y_t != -1
    y_t, y_p = y_t[mask], y_p[mask]
    
    if len(y_t) == 0:
        return 0.0, 0.0
    
    # Define metric function
    if metric_type == "error":
        metric_func = lambda t, p: np.mean(t != p)
    elif metric_type == "distance":
        metric_func = lambda t, p: np.mean(np.abs(t - p))
    else:
        return 0.0, 0.0
    
    # Original score
    original_score = metric_func(y_t, y_p)
    
    # Bootstrap
    boot_scores = []
    rng = np.random.default_rng(seed=1234)
    indices = np.arange(len(y_t))
    for _ in range(n_bootstraps):
        resample_idx = rng.choice(indices, size=len(indices), replace=True)
        score = metric_func(y_t[resample_idx], y_p[resample_idx])
        boot_scores.append(score)
    
    # Get 2.5th and 97.5th percentiles
    lower = np.percentile(boot_scores, 2.5)
    upper = np.percentile(boot_scores, 97.5)
    
    # Return distances from the mean (for matplotlib yerr)
    return original_score - lower, upper - original_score


if __name__ == "__main__":

    # Define paths to all plotted models
    result_path_group = {

        "Qwen3-0.6B": [
            "unsloth/Qwen3-0.6B-GGUF-Q2_K_XL.json",
            "unsloth/Qwen3-0.6B-GGUF-Q3_K_XL.json",
            "unsloth/Qwen3-0.6B-GGUF-Q4_K_XL.json",
            "unsloth/Qwen3-0.6B-GGUF-Q5_K_XL.json",
            "unsloth/Qwen3-0.6B-GGUF-Q6_K_XL.json",
            "unsloth/Qwen3-0.6B-GGUF-Q8_0.json",
            # "Qwen/Qwen3-0.6B-FP8-no_quant_scheme.json",
        ],

        "Qwen3-1.7B": [
            "unsloth/Qwen3-1.7B-GGUF-Q2_K_XL.json",
            "unsloth/Qwen3-1.7B-GGUF-Q3_K_XL.json",
            "unsloth/Qwen3-1.7B-GGUF-Q4_K_XL.json",
            "unsloth/Qwen3-1.7B-GGUF-Q5_K_XL.json",
            "unsloth/Qwen3-1.7B-GGUF-Q6_K_XL.json",
            "unsloth/Qwen3-1.7B-GGUF-Q8_0.json",
            # "Qwen/Qwen3-1.7B-FP8-no_quant_scheme.json",
        ],

        "Qwen3-4B": [
            "unsloth/Qwen3-4B-GGUF-Q2_K_XL.json",
            "unsloth/Qwen3-4B-GGUF-Q3_K_XL.json",
            "unsloth/Qwen3-4B-GGUF-Q4_K_XL.json",
            "unsloth/Qwen3-4B-GGUF-Q5_K_XL.json",
            "unsloth/Qwen3-4B-GGUF-Q6_K_XL.json",
            "unsloth/Qwen3-4B-GGUF-Q8_0.json",
            # "Qwen/Qwen3-4B-AWQ-no_quant_scheme.json",
            # "Qwen/Qwen3-4B-FP8-no_quant_scheme.json",
        ],

        "Qwen3-8B": [
            "unsloth/Qwen3-8B-GGUF-Q2_K_XL.json",
            "unsloth/Qwen3-8B-GGUF-Q3_K_XL.json",
            "unsloth/Qwen3-8B-GGUF-Q4_K_XL.json",
            "unsloth/Qwen3-8B-GGUF-Q5_K_XL.json",
            "unsloth/Qwen3-8B-GGUF-Q6_K_XL.json",
            "unsloth/Qwen3-8B-GGUF-Q8_0.json",
            # "Qwen/Qwen3-8B-AWQ-no_quant_scheme.json",
            # "Qwen/Qwen3-8B-FP8-no_quant_scheme.json",
        ],

        "Qwen3-14B": [
            "unsloth/Qwen3-14B-GGUF-Q2_K_XL.json",
            "unsloth/Qwen3-14B-GGUF-Q3_K_XL.json",
            "unsloth/Qwen3-14B-GGUF-Q4_K_XL.json",
            "unsloth/Qwen3-14B-GGUF-Q5_K_XL.json",
            "unsloth/Qwen3-14B-GGUF-Q6_K_XL.json",
            "unsloth/Qwen3-14B-GGUF-Q8_0.json",
            # "Qwen/Qwen3-14B-AWQ-no_quant_scheme.json",
            # "Qwen/Qwen3-14B-FP8-no_quant_scheme.json",
        ],

        "Qwen3-32B": [
            "unsloth/Qwen3-32B-GGUF-Q2_K_XL.json",
            "unsloth/Qwen3-32B-GGUF-Q3_K_XL.json",
            "unsloth/Qwen3-32B-GGUF-Q4_K_XL.json",
            "unsloth/Qwen3-32B-GGUF-Q5_K_XL.json",
            "unsloth/Qwen3-32B-GGUF-Q6_K_XL.json",
            "unsloth/Qwen3-32B-GGUF-Q8_0.json",
            # "Qwen/Qwen3-32B-AWQ-no_quant_scheme.json",
            # "Qwen/Qwen3-32B-FP8-no_quant_scheme.json",
        ],

        "DS-R1-Distill-Qwen3-32B": [
            "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF-Q2_K_L.json",
            "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF-Q3_K_M.json",
            "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF-Q4_K_M.json",
            "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF-Q5_K_M.json",
            "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF-Q6_K.json",
            "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF-Q8_0.json",
        ],
        
        
        "DS-R1-Distill-Llama-70B": [
            "unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF-Q2_K_XL.json",
            "unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF-Q3_K_XL.json",
            "unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF-Q4_K_XL.json",
            "unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF-Q5_K_XL.json",
            "unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF-Q6_K_XL.json",
            "unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF-Q8_0.json",
        ],

    }

    # Prepend input directory to all result paths
    result_path_group = {
        group: [os.path.join(INPUT_DIR, path) for path in paths]
        for group, paths in result_path_group.items()
    }

    # Do a full analysis for all required target variables
    for target_variable in TARGET_VARIABLES:
        
        # Pool results and plot them
        output_png_path, output_csv_path = generate_pooled_metric_plots(
            result_path_group,
            output_name=OUTPUT_NAME,
            output_dir=OUTPUT_DIR,
            target_variable=target_variable,
        )

        # Identify statistical patterns using linear mixed-effects models
        lme_results = fit_error_model_lme(
            df_input=pd.read_csv(output_csv_path),
            dependent_variable=f"{target_variable} - all 10 models",
        )
        print(lme_results.summary())