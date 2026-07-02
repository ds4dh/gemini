import os
import json
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import seaborn as sns
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
    item_data = data_source.get(key)
    if item_data and isinstance(item_data.get("values"), list) and item_data["values"]:
        try:
            return float(item_data["values"][0])
        except (ValueError, TypeError, IndexError) as e:
            print(f"Warning: Could not parse value for key {key}: {e}.")
    return default_val


def generate_pooled_metric_plots(
    result_path_group: dict[str, list],
    output_name: str,
    output_dir: str,
    target_variable: str,
) -> None:
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    num_cols = 2
    num_rows = math.ceil(len(CASES) / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 4 * num_rows), squeeze=False)
    axes_flat = axes.flatten()
    csv_data = []
    
    for i, case_name in enumerate(CASES):
        raw_key = CASE_MAPPING.get(case_name, "single")  
        
        for group_idx, (group_label, result_paths_in_group) in enumerate(result_path_group.items()):
            group_data_points = []
            group_color = GROUP_COLORS[group_idx % len(GROUP_COLORS)]
            for result_path in result_paths_in_group:

                try:
                    with open(result_path, "r") as f:
                        data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    continue

                extracted_data = {"model": group_label}

                for x_id in X_CONFIGS:
                    query_key = X_CONFIGS[x_id]["key"]
                    extracted_data[x_id] = _extract_metric_value(data, query_key)

                for y_id in Y_CONFIGS:
                    query_key = f"{Y_CONFIGS[y_id]['key']}\n({case_name})"
                    y_id_cased = f"{y_id} - {case_name}"
                    extracted_data[y_id_cased] = _extract_metric_value(data, query_key)

                    try:
                        raw_true = data.get("y_true", {}).get(raw_key, [])
                        raw_pred = data.get("y_pred", {}).get(raw_key, [])
                        low_err, up_err = calculate_bootstrap_ci(raw_true, raw_pred, y_id)
                        extracted_data[f"{y_id_cased}_err_low"] = low_err
                        extracted_data[f"{y_id_cased}_err_high"] = up_err
                    except Exception:
                        extracted_data[f"{y_id_cased}_err_low"] = 0.0
                        extracted_data[f"{y_id_cased}_err_high"] = 0.0

                if "fp8" in result_path.lower():
                    extracted_data["nbits"] = extracted_data["nbits"] * 2
                    extracted_data["vram"] = extracted_data["vram"] * 2
                
                group_data_points.append(extracted_data)
            
            plotted_y_key = f"{target_variable} - {case_name}"
            valid_points = []
            for dp in group_data_points:
                x_val = dp.get(MAIN_X_VARIABLE)
                y_val = dp.get(plotted_y_key)
                if x_val is not None and y_val is not None:
                    valid_points.append(dp)
            group_data_points = valid_points
            if not group_data_points: continue

            x_values = [dp[MAIN_X_VARIABLE] for dp in group_data_points]
            plotted_y_id_cased = f"{target_variable} - {case_name}"
            y_values = [dp[plotted_y_id_cased] for dp in group_data_points]
            sizes = [200 * dp["nbits"] / 16 for dp in group_data_points]

            axes_flat[i].scatter(
                x_values, y_values, color=group_color, label=group_label,
                marker="o", alpha=0.9, s=sizes, edgecolors='white', linewidth=0.5, zorder=1,
            )

            rgb = mcolors.to_rgb(group_color)
            darker_color = tuple(c * 0.5 for c in rgb)
            y_err_low = [dp[f"{plotted_y_id_cased}_err_low"] for dp in group_data_points]
            y_err_high = [dp[f"{plotted_y_id_cased}_err_high"] for dp in group_data_points]
            axes_flat[i].errorbar(
                x_values, y_values, yerr=[y_err_low, y_err_high],
                elinewidth=0.75, markeredgewidth=0.75, fmt='none',
                ecolor=darker_color, alpha=0.7, capsize=3, zorder=2,
            )

            csv_data.extend(group_data_points)

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
        axes_flat[i].set_xlabel(x_label, fontsize=14)
        axes_flat[i].set_ylabel(y_label, fontsize=14)
        if X_CONFIGS[MAIN_X_VARIABLE]["lim"] is not None:
            axes_flat[i].set_xlim(X_CONFIGS[MAIN_X_VARIABLE]["lim"])
        if Y_CONFIGS[target_variable]["lim"] is not None:
            axes_flat[i].set_ylim(Y_CONFIGS[target_variable]["lim"])
        axes_flat[i].tick_params(axis="both", labelsize=12)
        axes_flat[i].grid(True, linestyle="--", alpha=0.6)
        axes_flat[i].set_title(f"Prediction with {case_name}", fontsize=16, pad=10)
        axes_flat[i].legend(loc="upper right", fontsize=10, fancybox=True, ncol=2)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_full_path = os.path.join(output_dir, f"{output_name}_{target_variable}.png")
    plt.savefig(plot_full_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"Combined plot saved: {plot_full_path}")

    csv_df = pd.DataFrame(csv_data)
    csv_df = csv_df.groupby(["model", "vram", "nparams", "nbits"]).first().reset_index()
    csv_full_path = os.path.join(output_dir, f"{output_name}_{target_variable}.csv")
    csv_df.to_csv(csv_full_path, index=False)
    print(f"Pooled data saved: {csv_full_path}")

    return plot_full_path, csv_full_path


def fit_error_model_lme(
    df_input: pd.DataFrame,
    dependent_variable: str = "error - all 10 models",
    fixed_effect_0: str = "nparams",
    fixed_effect_1: str = "nbits",
    include_interaction: bool = True,
    random_intercept_group: str = "model",
):
    df = df_input.copy()
    df["log_nparams"] = np.log10(df[fixed_effect_0])
    fe_0_term = "log_nparams"  
    fe_1_term = fixed_effect_1
    df["target"] = df[dependent_variable]
    
    if include_interaction:
        model_formula = f"target ~ {fe_0_term} * {fe_1_term}"
    else:
        model_formula = f"target ~ {fe_0_term} + {fe_1_term}"
    print(f"\nFitting LME with formula: {model_formula}")

    md = smf.mixedlm(model_formula, df, groups=df[random_intercept_group])
    return md.fit()


def calculate_bootstrap_ci(
    y_true: list | np.ndarray, 
    y_pred: list | np.ndarray, 
    metric_type: str, 
    n_bootstraps: int = 1000,
) -> tuple[float, float]:
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    
    mask = y_t != -1
    y_t, y_p = y_t[mask], y_p[mask]
    
    if len(y_t) == 0:
        return 0.0, 0.0
    
    if metric_type == "error":
        metric_func = lambda t, p: np.mean(t != p)
    elif metric_type == "distance":
        metric_func = lambda t, p: np.mean(np.abs(t - p))
    else:
        return 0.0, 0.0
    
    original_score = metric_func(y_t, y_p)
    boot_scores = []
    rng = np.random.default_rng(seed=1234)
    indices = np.arange(len(y_t))
    for _ in range(n_bootstraps):
        resample_idx = rng.choice(indices, size=len(indices), replace=True)
        score = metric_func(y_t[resample_idx], y_p[resample_idx])
        boot_scores.append(score)
    
    lower = np.percentile(boot_scores, 2.5)
    upper = np.percentile(boot_scores, 97.5)
    return original_score - lower, upper - original_score


def add_model_annotations(ax, df, col_y):
    """Helper to add bespoke, spread-out arrows for models to avoid overlapping borders and each other."""
    model_stats = df.groupby('model').agg({'nparams': 'first', col_y: 'mean'}).reset_index()
    
    # Hand-tuned (x_offset, y_offset) coordinates to keep labels fully inside the plot 
    # and safely separated from one another.
    custom_offsets = {
        "Qwen3-0.6B": (65, -20),
        "Qwen3-1.7B": (-50, -40),
        "Qwen3-4B": (-60, -30),
        "Qwen3-8B": (-40, -50),
        "Qwen3-14B": (-20, 50),
        "Qwen3-32B": (-60, -40),
        "DS-R1-Distill-Qwen3-32B": (-30, 100),
        "DS-R1-Distill-Llama-70B": (-30, 40),
    }

    for idx, row in model_stats.iterrows():
        model_key = row['model']
        # Fallback to an alternating pattern if an unexpected model is added
        x_offset, y_offset = custom_offsets.get(model_key, (50 if idx % 2 == 0 else -50, 60 if idx % 2 == 0 else -60))
        
        short_model_name = model_key.replace("DS-R1-Distill-", "DS-")
        
        ax.annotate(
            short_model_name,
            xy=(row['nparams'], row[col_y]),
            xytext=(x_offset, y_offset),
            textcoords='offset points',
            ha='center',
            fontsize=12,
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.9),
            arrowprops=dict(arrowstyle="-|>", mutation_scale=20, connectionstyle="arc3,rad=0.25", color="black", lw=1.5, alpha=0.8)
        )


def generate_aggregated_main_effects_plot(
    csv_error_path: str, 
    csv_distance_path: str, 
    output_dir: str,
    lme_models: dict
):
    df_err = pd.read_csv(csv_error_path)
    df_dist = pd.read_csv(csv_distance_path)
    
    df_err['nparams'] = df_err['nparams'].round(1)
    df_dist['nparams'] = df_dist['nparams'].round(1)

    target_case = "all 10 models"
    col_err = f"error - {target_case}"
    col_dist = f"distance - {target_case}"

    if col_err not in df_err.columns or col_dist not in df_dist.columns:
        print("Warning: Target columns for 'all 10 models' missing. Cannot generate main effects plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    sns.set_theme(style="whitegrid")

    # Fixed explicit ticks for the right column log scale
    explicit_x_ticks = [0.5, 1, 5, 10, 50]

    # --- ROW 1: ERROR RATE ---
    coef_err_nbits = lme_models['error'].params['nbits']
    label_err_nbits = f"LME coeff.: {coef_err_nbits:.3f}"
    
    sns.regplot(
        data=df_err, x='nbits', y=col_err, ax=axes[0, 0], 
        color='tab:blue', x_estimator=np.mean, x_ci=68,
        scatter_kws={'s': 70, 'edgecolor': 'white'}, 
        line_kws={'color': 'black', 'alpha': 0.8, 'linewidth': 2, 'label': label_err_nbits}
    )
    axes[0, 0].set_title("Main effect of quantization on error rate", fontsize=16)
    axes[0, 0].set_ylabel("Error rate [%]", fontsize=14)
    axes[0, 0].set_xlabel("Bits/param", fontsize=14)
    axes[0, 0].set_xticks(sorted(df_err['nbits'].dropna().unique()))
    axes[0, 0].tick_params(axis='both', labelsize=12)
    axes[0, 0].legend(loc="best", fontsize=10)

    coef_err_nparams = lme_models['error'].params['log_nparams']
    label_err_nparams = f"LME coeff.: {coef_err_nparams:.3f}"

    sns.regplot(
        data=df_err, x='nparams', y=col_err, ax=axes[0, 1], 
        color='tab:orange', logx=True, x_estimator=np.mean, x_ci=68,
        scatter_kws={'s': 70, 'edgecolor': 'white'}, 
        line_kws={'color': 'black', 'alpha': 0.8, 'linewidth': 2, 'label': label_err_nparams}
    )
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xticks(explicit_x_ticks)
    axes[0, 1].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    axes[0, 1].set_title("Main effect of model size on error rate", fontsize=16)
    axes[0, 1].set_ylabel("Error rate [%]", fontsize=14)
    axes[0, 1].set_xlabel("Number of parameters (billion) - Log scale", fontsize=14)
    axes[0, 1].tick_params(axis='both', labelsize=12)
    axes[0, 1].legend(loc="best", fontsize=10)
    add_model_annotations(axes[0, 1], df_err, col_err)

    # --- ROW 2: DISTANCE ---
    coef_dist_nbits = lme_models['distance'].params['nbits']
    label_dist_nbits = f"LME coeff.: {coef_dist_nbits:.3f}"

    sns.regplot(
        data=df_dist, x='nbits', y=col_dist, ax=axes[1, 0], 
        color='tab:green', x_estimator=np.mean, x_ci=68,
        scatter_kws={'s': 70, 'edgecolor': 'white'}, 
        line_kws={'color': 'black', 'alpha': 0.8, 'linewidth': 2, 'label': label_dist_nbits}
    )
    axes[1, 0].set_title("Main effect of quantization on distance", fontsize=16)
    axes[1, 0].set_ylabel("Distance [mRS unit]", fontsize=14)
    axes[1, 0].set_xlabel("Bits/param", fontsize=14)
    axes[1, 0].set_xticks(sorted(df_dist['nbits'].dropna().unique()))
    axes[1, 0].tick_params(axis='both', labelsize=12)
    axes[1, 0].legend(loc="best", fontsize=10)

    coef_dist_nparams = lme_models['distance'].params['log_nparams']
    label_dist_nparams = f"LME coeff.: {coef_dist_nparams:.3f}"

    sns.regplot(
        data=df_dist, x='nparams', y=col_dist, ax=axes[1, 1], 
        color='tab:red', logx=True, x_estimator=np.mean, x_ci=68,
        scatter_kws={'s': 70, 'edgecolor': 'white'}, 
        line_kws={'color': 'black', 'alpha': 0.8, 'linewidth': 2, 'label': label_dist_nparams}
    )
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_xticks(explicit_x_ticks)
    axes[1, 1].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    axes[1, 1].set_title("Main effect of model size on distance", fontsize=16)
    axes[1, 1].set_ylabel("Distance [mRS unit]", fontsize=14)
    axes[1, 1].set_xlabel("Number of parameters (billion) - Log scale", fontsize=14)
    axes[1, 1].tick_params(axis='both', labelsize=12)
    axes[1, 1].legend(loc="best", fontsize=10)
    add_model_annotations(axes[1, 1], df_dist, col_dist)

    fig.tight_layout(pad=3.0)
    out_path = os.path.join(output_dir, f"{OUTPUT_NAME}_main_effects_aggregated.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"Aggregated main effects plot saved: {out_path}")


if __name__ == "__main__":

    result_path_group = {
        "Qwen3-0.6B": [
            "unsloth/Qwen3-0.6B-GGUF-Q2_K_XL.json",
            "unsloth/Qwen3-0.6B-GGUF-Q3_K_XL.json",
            "unsloth/Qwen3-0.6B-GGUF-Q4_K_XL.json",
            "unsloth/Qwen3-0.6B-GGUF-Q5_K_XL.json",
            "unsloth/Qwen3-0.6B-GGUF-Q6_K_XL.json",
            "unsloth/Qwen3-0.6B-GGUF-Q8_0.json",
        ],
        "Qwen3-1.7B": [
            "unsloth/Qwen3-1.7B-GGUF-Q2_K_XL.json",
            "unsloth/Qwen3-1.7B-GGUF-Q3_K_XL.json",
            "unsloth/Qwen3-1.7B-GGUF-Q4_K_XL.json",
            "unsloth/Qwen3-1.7B-GGUF-Q5_K_XL.json",
            "unsloth/Qwen3-1.7B-GGUF-Q6_K_XL.json",
            "unsloth/Qwen3-1.7B-GGUF-Q8_0.json",
        ],
        "Qwen3-4B": [
            "unsloth/Qwen3-4B-GGUF-Q2_K_XL.json",
            "unsloth/Qwen3-4B-GGUF-Q3_K_XL.json",
            "unsloth/Qwen3-4B-GGUF-Q4_K_XL.json",
            "unsloth/Qwen3-4B-GGUF-Q5_K_XL.json",
            "unsloth/Qwen3-4B-GGUF-Q6_K_XL.json",
            "unsloth/Qwen3-4B-GGUF-Q8_0.json",
        ],
        "Qwen3-8B": [
            "unsloth/Qwen3-8B-GGUF-Q2_K_XL.json",
            "unsloth/Qwen3-8B-GGUF-Q3_K_XL.json",
            "unsloth/Qwen3-8B-GGUF-Q4_K_XL.json",
            "unsloth/Qwen3-8B-GGUF-Q5_K_XL.json",
            "unsloth/Qwen3-8B-GGUF-Q6_K_XL.json",
            "unsloth/Qwen3-8B-GGUF-Q8_0.json",
        ],
        "Qwen3-14B": [
            "unsloth/Qwen3-14B-GGUF-Q2_K_XL.json",
            "unsloth/Qwen3-14B-GGUF-Q3_K_XL.json",
            "unsloth/Qwen3-14B-GGUF-Q4_K_XL.json",
            "unsloth/Qwen3-14B-GGUF-Q5_K_XL.json",
            "unsloth/Qwen3-14B-GGUF-Q6_K_XL.json",
            "unsloth/Qwen3-14B-GGUF-Q8_0.json",
        ],
        "Qwen3-32B": [
            "unsloth/Qwen3-32B-GGUF-Q2_K_XL.json",
            "unsloth/Qwen3-32B-GGUF-Q3_K_XL.json",
            "unsloth/Qwen3-32B-GGUF-Q4_K_XL.json",
            "unsloth/Qwen3-32B-GGUF-Q5_K_XL.json",
            "unsloth/Qwen3-32B-GGUF-Q6_K_XL.json",
            "unsloth/Qwen3-32B-GGUF-Q8_0.json",
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

    result_path_group = {
        group: [os.path.join(INPUT_DIR, path) for path in paths]
        for group, paths in result_path_group.items()
    }

    generated_csvs = {}
    lme_models = {}

    for target_variable in TARGET_VARIABLES:
        
        output_png_path, output_csv_path = generate_pooled_metric_plots(
            result_path_group,
            output_name=OUTPUT_NAME,
            output_dir=OUTPUT_DIR,
            target_variable=target_variable,
        )
        generated_csvs[target_variable] = output_csv_path

        lme_results = fit_error_model_lme(
            df_input=pd.read_csv(output_csv_path),
            dependent_variable=f"{target_variable} - all 10 models",
        )
        print(lme_results.summary())
        lme_models[target_variable] = lme_results

    if "error" in generated_csvs and "distance" in generated_csvs:
        generate_aggregated_main_effects_plot(
            csv_error_path=generated_csvs["error"],
            csv_distance_path=generated_csvs["distance"],
            output_dir=OUTPUT_DIR,
            lme_models=lme_models
        )