import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections.abc import Iterable
from sklearn.metrics import confusion_matrix

POOLED_MODES = [
    {"name": "all", "label": "all models", "pred_pool_mode": "concatenation"},
    {"name": "single", "label": "single model", "pred_pool_mode": "single"},
    {"name": "maj_3", "label": "maj-pooling-3", "pred_pool_mode": "majority", "num_models": 3},
    {"name": "maj_5", "label": "maj-pooling-5", "pred_pool_mode": "majority", "num_models": 5},
    {"name": "maj_10", "label": "maj-pooling-10", "pred_pool_mode": "majority", "num_models": 10},
]


def plot_metrics(metric_path: str) -> None:
    """
    Plot metrics in a common plot for different prediction voting strategies
    """
    # Load data to plot to determine the number of rows needed
    with open(metric_path, "r") as f:
        metric_dict: dict = json.load(f)
    y_true = metric_dict.pop("y_true", {})
    y_pred = metric_dict.pop("y_pred", {})

    # Determine which rows will be plotted and build the confusion matrix config
    rows_to_plot = [0]
    cms_to_plot = []
    for mode_idx, mode in enumerate(POOLED_MODES):
        if mode["name"] in y_true and mode["name"] in y_pred:
            # Using mode_idx + 1 as a placeholder for the row index (hacky)
            rows_to_plot.append(mode_idx + 1)
            cms_to_plot.append({
                "key": mode["name"],
                "title": mode["label"],
            })

    # Create figure and gridspec dynamically
    num_rows = len(rows_to_plot)
    figsize_height = 2 + num_rows * 3  # rough estimate, tune this as needed
    fig = plt.figure(figsize=(7, figsize_height))
    gs = fig.add_gridspec(
        nrows=num_rows, ncols=4,
        width_ratios=[1, 1, 1, 1],
        height_ratios=[1] * num_rows,  # uniform height ratios are simpler here
    )
    ax = []
    for row_idx in range(num_rows):
        if rows_to_plot[row_idx] == 0:
            ax.append((
                fig.add_subplot(gs[row_idx, 0]), fig.add_subplot(gs[row_idx, 1]),
                fig.add_subplot(gs[row_idx, 2]), fig.add_subplot(gs[row_idx, 3]),
            ))
        else:
            ax.append((
                fig.add_subplot(gs[row_idx, 0:2]),
                fig.add_subplot(gs[row_idx, 2]), fig.add_subplot(gs[row_idx, 3]),
            ))

    # Plot confusion matrices
    row_map = {original: new for new, original in enumerate(sorted(rows_to_plot))}
    for cm_info in cms_to_plot:
        key = cm_info["key"]
        title = cm_info["title"]
        mode_idx = [i for i, mode in enumerate(POOLED_MODES) if mode["name"] == key][0]
        new_row_idx = row_map[mode_idx + 1]
        plot_cm(ax=ax[new_row_idx][0], y_true=y_true[key], y_pred=y_pred[key], title_flag=title)

    # Plot each metric in a separate subplot
    for metric_name, plot_dict in metric_dict.items():
        plot_values = plot_dict["values"]
        mean = np.mean(plot_values)
        bar_kwargs = {"capsize": 5, "alpha": 0.75, "color": plot_dict["color"]}
        if isinstance(plot_values, Iterable) and len(plot_values) > 1:
            bar_kwargs["yerr"] = np.std(plot_values, ddof=1) / np.sqrt(len(plot_values))
        else:
            bar_kwargs["yerr"] = None

        # Use the mapping to get the correct new row index
        i, j = plot_dict["loc"]
        if i in row_map:
            new_i = row_map[i]
            if new_i < len(ax) and j < len(ax[new_i]):
                current_ax = ax[new_i][j]
                current_ax.bar(0, mean, **bar_kwargs)
                current_ax.set_xticks([])
                current_ax.set_ylim([0.0, plot_dict["max_y"]])
                current_ax.set_ylabel(f"[{plot_dict['unit']}]")
                current_ax.set_title(metric_name.split("\n")[0])

    # Adjust layout and save plot
    plot_path = metric_path.replace(".json", ".png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved processed result plot at {plot_path}")


def plot_cm(
    ax: plt.Axes,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    title_flag: str,
    possible_labels: list[int] = [-1, 0, 1, 2, 3, 4, 5, 6],  # mRS in this case
    add_group_patches: bool = True,
) -> None:
    """
    Plot a confusion matrix given model prediction and true label values.
    Optionally adds red squares around specified groups of cells.
    """
    # Plot pooled confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=possible_labels)
    ax.imshow(cm, cmap="Blues", interpolation="none")
    ax.set_xticks(range(len(possible_labels)))
    ax.set_yticks(range(len(possible_labels)))
    ax.set_xticklabels(possible_labels)
    ax.set_yticklabels(possible_labels)
    ax.set_xlabel("Predicted mRS")
    ax.set_ylabel("True mRS")
    ax.set_title(f"Confusion Matrix ({title_flag})")
    
    # Polish confusion matrix
    max_cm_value = np.max(cm)
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            value = cm[i, j]
            if value > 0:
                color = "white" if value > max_cm_value / 2 else "black"
                ax.text(j, i, str(value), ha="center", va="center", color=color)

    # If required, add squares to highlight label groups
    if add_group_patches:
        # Group 1: no impairement (label = from 0 to 1)
        ax.add_patch(patches.Rectangle(
            xy=(0.5, 0.5), width=2, height=2, linewidth=2, zorder=10,
            edgecolor='tab:red', facecolor='none',
        ))

        # Group 2: impairement (label = from 2 to 5)
        ax.add_patch(patches.Rectangle(
            xy=(2.5, 2.5), width=4, height=4, linewidth=2, zorder=10,
            edgecolor='tab:red', facecolor='none',
        ))

        # Group 3: dead (label = 6)
        ax.add_patch(patches.Rectangle(
            xy=(6.5, 6.5), width=1, height=1, linewidth=2, zorder=10,
            edgecolor='tab:red', facecolor='none',
        ))


def _normalize_cat(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "Unknown"
    s = str(val).strip()
    if s.lower() in ("", "nan", "none", "null", "-1"):
        return "Unknown"
    return s


def _try_float(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if s.lower() in ("", "nan", "none", "null", "-1"):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def is_field_continuous(field_name: str, field_type: str, y_true: list) -> bool:
    return field_type in ("float", "double", "number") or (
        field_type in ("int", "integer") and field_name != "mRS" and any(_try_float(v) is not None and _try_float(v) > 10 for v in y_true)
    )


def plot_categorical_cm(ax: plt.Axes, field_name: str, y_true: list, y_pred: list):
    norm_gt = [_normalize_cat(g) for g in y_true]
    norm_pred = [_normalize_cat(p) for p in y_pred]

    labels = list(dict.fromkeys(norm_gt + norm_pred))
    def label_sort_key(x):
        try:
            return (0, float(x))
        except ValueError:
            return (2 if x.lower() in ("unknown", "missing", "none") else 1, x)

    labels = sorted(labels, key=label_sort_key)
    str_labels = [str(lbl) for lbl in labels]

    cm = confusion_matrix(norm_gt, norm_pred, labels=labels)

    ax.set_facecolor("#FFFFFF")
    im = ax.imshow(cm, cmap="YlGnBu", interpolation="none")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(str_labels, rotation=30 if len(str_labels) > 3 else 0, ha="right" if len(str_labels) > 3 else "center", fontsize=8.5, color="#1E293B")
    ax.set_yticklabels(str_labels, fontsize=8.5, color="#1E293B")
    ax.set_xlabel("Predicted", fontsize=9.5, fontweight="bold", color="#0F172A", labelpad=6)
    ax.set_ylabel("Ground Truth", fontsize=9.5, fontweight="bold", color="#0F172A", labelpad=6)
    ax.set_title(f"Confusion Matrix: {field_name}", fontsize=11, fontweight="bold", color="#0F172A", pad=10)

    # Grid lines between cells for card/tile look
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="#F1F5F9", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    max_val = np.max(cm) if cm.size > 0 else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            if val > 0:
                color = "white" if val > max_val * 0.55 else "#0F172A"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold", fontsize=9.5)


def plot_continuous_scatter_with_presence_inset(
    ax: plt.Axes,
    field_name: str,
    y_true: list,
    y_pred: list,
):
    clean_gt = []
    clean_pred = []
    tp, fn, fp, tn = 0, 0, 0, 0

    for g, p in zip(y_true, y_pred):
        g_f = _try_float(g)
        p_f = _try_float(p)

        if g_f is not None and p_f is not None:
            clean_gt.append(g_f)
            clean_pred.append(p_f)
            tp += 1
        elif g_f is not None and p_f is None:
            fn += 1
        elif g_f is None and p_f is not None:
            fp += 1
        elif g_f is None and p_f is None:
            tn += 1

    c_gt = np.array(clean_gt)
    c_pred = np.array(clean_pred)
    ax.set_facecolor("#FFFFFF")

    if len(c_gt) > 0:
        ax.scatter(c_gt, c_pred, color="#2563EB", alpha=0.88, edgecolors="#1E3A8A", s=55, label="Predictions", zorder=3)

        min_val = min(c_gt.min(), c_pred.min())
        max_val = max(c_gt.max(), c_pred.max())
        if min_val == max_val:
            min_val -= 1.0
            max_val += 1.0
        ax.plot([min_val, max_val], [min_val, max_val], color="#EF4444", linestyle="--", linewidth=2, alpha=0.85, label="Perfect (y=x)", zorder=2)

        mae = float(np.mean(np.abs(c_gt - c_pred)))
        r_str = ""
        if len(c_gt) > 1 and np.std(c_gt) > 0 and np.std(c_pred) > 0:
            r_val = float(np.corrcoef(c_gt, c_pred)[0, 1])
            r_str = f" | r = {r_val:.3f}"

        annot = f"N = {len(c_gt)} | MAE = {mae:.2f}{r_str}"
        ax.set_xlabel("Ground Truth", fontsize=9.5, fontweight="bold", color="#0F172A", labelpad=6)
        ax.set_ylabel("Predicted", fontsize=9.5, fontweight="bold", color="#0F172A", labelpad=6)
        ax.set_title(f"Correlation: {field_name}\n({annot})", fontsize=10.5, fontweight="bold", color="#0F172A", pad=8)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9, facecolor="#F8FAFC", edgecolor="#E2E8F0")
        ax.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1")
    else:
        ax.text(
            0.5, 0.5, "No paired numeric values", transform=ax.transAxes, ha="center", va="center",
            fontsize=9.5, color="#64748B", bbox=dict(boxstyle="round,pad=0.4", facecolor="#F1F5F9", edgecolor="#CBD5E1")
        )
        ax.set_title(f"Correlation: {field_name}", fontsize=10.5, fontweight="bold", color="#0F172A", pad=8)

    # Mini 2x2 Presence CM Heatmap Card Inset inside top-left corner
    ax_inset = ax.inset_axes([0.05, 0.48, 0.40, 0.40])
    ax_inset.set_facecolor("#FFFFFF")
    presence_cm = np.array([[tp, fn], [fp, tn]])

    im = ax_inset.imshow(presence_cm, cmap="Purples", interpolation="none")

    ax_inset.set_xticks([0, 1])
    ax_inset.set_yticks([0, 1])
    ax_inset.set_xticklabels(["Pres", "Abs"], fontsize=7, color="#1E293B")
    ax_inset.set_yticklabels(["Pres", "Abs"], fontsize=7, color="#1E293B")
    ax_inset.set_xlabel("Pred", fontsize=7.5, labelpad=2, fontweight="bold", color="#0F172A")
    ax_inset.set_ylabel("GT", fontsize=7.5, labelpad=2, fontweight="bold", color="#0F172A")
    ax_inset.set_title("Presence CM", fontsize=8, pad=3, fontweight="bold", color="#0F172A")

    # Inset cell grid lines
    ax_inset.set_xticks([-0.5, 0.5, 1.5], minor=True)
    ax_inset.set_yticks([-0.5, 0.5, 1.5], minor=True)
    ax_inset.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=1.2)
    ax_inset.tick_params(which="minor", bottom=False, left=False)

    cell_tags = [["TP", "FN"], ["FP", "TN"]]
    max_v = np.max(presence_cm) if presence_cm.size > 0 else 1

    for i in range(2):
        for j in range(2):
            cnt = presence_cm[i, j]
            tag = cell_tags[i][j]
            color = "white" if cnt > max_v * 0.55 else "#0F172A"
            ax_inset.text(j, i, f"{tag}:{cnt}", ha="center", va="center", color=color, fontweight="bold", fontsize=7.5)


def generate_combined_evaluation_dashboard(
    fields_eval_data: dict[str, dict],
    output_path: str,
) -> str | None:
    """
    Generates a single consolidated evaluation dashboard figure (evaluation_dashboard.png)
    containing subplots for all extracted variables (1 subplot per variable).
    """
    if not fields_eval_data:
        return None

    import math
    all_fields = list(fields_eval_data.items())
    num_fields = len(all_fields)
    if num_fields == 0:
        return None

    # Dynamic balanced layout sizing
    if num_fields <= 3:
        ncols = num_fields
        nrows = 1
    elif num_fields == 4:
        ncols = 2
        nrows = 2
    else:
        ncols = 3
        nrows = math.ceil(num_fields / 3)

    fig_width = 4.8 * ncols
    fig_height = 4.4 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("#F8FAFC")

    if num_fields == 1:
        axes_flat = [axes]
    elif hasattr(axes, "flatten"):
        axes_flat = list(axes.flatten())
    else:
        axes_flat = list(axes)

    for idx, (fname, fdata) in enumerate(all_fields):
        ax = axes_flat[idx]
        if is_field_continuous(fname, fdata["field_type"], fdata["y_true"]):
            plot_continuous_scatter_with_presence_inset(ax, fname, fdata["y_true"], fdata["y_pred"])
        else:
            plot_categorical_cm(ax, fname, fdata["y_true"], fdata["y_pred"])

    # Hide unused extra subplots if total features is not a multiple of ncols
    for unused_idx in range(num_fields, len(axes_flat)):
        axes_flat[unused_idx].set_visible(False)

    fig.suptitle("Clinical Variable Extraction - Evaluation Dashboard", fontsize=15, fontweight="bold", color="#0F172A", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return output_path

