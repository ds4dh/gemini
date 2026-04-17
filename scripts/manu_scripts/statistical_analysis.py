import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from sklearn.metrics import (
    cohen_kappa_score, mean_absolute_error, accuracy_score,
    confusion_matrix, classification_report, f1_score,
)

# Configuration
CSV_FILE = './data/paper/patient_data_with_results.csv'
CVV_FILE_DETAILED = './data/paper/consensus_details.csv'
MODEL_DIR = './llm-project/results_paper/vllm-serve-async_guided/unsloth'
BASE_OUTPUT_DIR = './stat_results'

# Original column names
RAW_MODEL_COL = 'Qwen3-32B-GGUF-Q4_K_XL-maj_10'
RAW_COMPETITORS = ['Etudiants', 'Internes']
RAW_EXPERT = 'Chefs_de_Clinique'

# Output directory (model specific)
OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/{RAW_MODEL_COL}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Display names
NAME_MAPPING = {
    RAW_MODEL_COL: 'Model',
    'Etudiants': 'Students',
    'Internes': 'Residents',
    RAW_EXPERT: 'Experts',
}

# Define groups
GROUPS = [NAME_MAPPING[c] for c in RAW_COMPETITORS] + [NAME_MAPPING[RAW_MODEL_COL]]
EXPERT_COL = NAME_MAPPING[RAW_EXPERT]
LABELS = [0, 1, 2, 3, 4, 5, 6]

# Broad Clinical Categories
CLINICAL_MAP = {
    0: '(0-1)', 1: '(0-1)',
    2: '(2-5)', 3: '(2-5)', 4: '(2-5)', 5: '(2-5)',
    6: '(6)'
}
CLINICAL_LABELS = ['(0-1)', '(2-5)', '(6)']

# Increase global font size for better readability
plt.rcParams.update({'font.size': 12}) 
TITLE_SIZE = 14
LABEL_SIZE = 12


def load_and_prep_data(filename):
    """Loads data and renames columns."""
    try:
        df = pd.read_csv(filename)
        print(f"Loaded data from {filename}")
    except FileNotFoundError:
        print(f"ERROR: '{filename}' not found.")
        return None

    df = df.rename(columns=NAME_MAPPING)
    return df


def calculate_metrics(df):
    """Calculates metrics and 95% CIs using Bootstrapping and Pingouin."""
    results = []
    for group in GROUPS:
        y_t, y_p = df[EXPERT_COL], df[group]
        
        # Calculate base metrics
        qwk = cohen_kappa_score(y_t, y_p, weights='quadratic')
        mae = mean_absolute_error(y_t, y_p)
        acc_ex = exact_acc(y_t, y_p)
        acc_t1 = tol_1_acc(y_t, y_p)
        
        # Calculate 95% CIs
        qwk_ci = compute_bootstrap_ci(y_t, y_p, cohen_kappa_score, weights='quadratic')
        mae_ci = compute_bootstrap_ci(y_t, y_p, mean_absolute_error)
        acc_ex_ci = compute_bootstrap_ci(y_t, y_p, exact_acc)
        acc_t1_ci = compute_bootstrap_ci(y_t, y_p, tol_1_acc)
        
        # ICC (Pingouin calculates the CI automatically)
        subset = df[[EXPERT_COL, group]].reset_index().melt(id_vars='index', var_name='Rater', value_name='Score')
        icc = pg.intraclass_corr(data=subset, targets='index', raters='Rater', ratings='Score').set_index('Type')
        icc_val = icc.loc['ICC3', 'ICC']
        icc_ci = icc.loc['ICC3', 'CI95%']
        
        results.append({
            'Group': group,
            'QWK': qwk, # Kept raw for run_statistical_tests
            'MAE': mae, # Kept raw for run_statistical_tests
            'QWK (95% CI)': f"{qwk:.3f} [{qwk_ci[0]:.3f}-{qwk_ci[1]:.3f}]",
            'MAE (95% CI)': f"{mae:.3f} [{mae_ci[0]:.3f}-{mae_ci[1]:.3f}]",
            'ICC (95% CI)': f"{icc_val:.3f} [{icc_ci[0]:.3f}-{icc_ci[1]:.3f}]",
            'Acc Exact (95% CI)': f"{acc_ex:.3f} [{acc_ex_ci[0]:.3f}-{acc_ex_ci[1]:.3f}]",
            'Acc ±1 (95% CI)': f"{acc_t1:.3f} [{acc_t1_ci[0]:.3f}-{acc_t1_ci[1]:.3f}]",
        })
    
    return pd.DataFrame(results).set_index('Group')


def compute_bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, **kwargs):
    """Calculates 95% Confidence Intervals using bootstrapping."""
    np.random.seed(42) # For reproducibility
    y_t, y_p = np.array(y_true), np.array(y_pred)
    scores = []
    
    for _ in range(n_boot):
        # Sample with replacement
        idx = np.random.choice(len(y_t), len(y_t), replace=True)
        try:
            scores.append(metric_fn(y_t[idx], y_p[idx], **kwargs))
        except Exception:
            pass  # Skips edge cases (e.g., QWK needs >1 class in the sample)
            
    if not scores:
        return np.nan, np.nan
    return np.percentile(scores, [2.5, 97.5])


# Helper metric functions for accuracy bootstrapping
def exact_acc(y_t, y_p):
    return (np.abs(y_t - y_p) == 0).mean()
def tol_1_acc(y_t, y_p):
    return (np.abs(y_t - y_p) <= 1).mean()


# Helper functions for f1-score bootstrapping
def f1_mac(y_t, y_p):
    return f1_score(y_t, y_p, average='macro', zero_division=0)
def f1_wei(y_t, y_p):
    return f1_score(y_t, y_p, average='weighted', zero_division=0)


def run_statistical_tests(df, metrics_df):
    """
    Performs Wilcoxon Signed-Rank tests using Pingouin.
    Returns P-value, Effect Size (RBC), and Bayes Factor (BF10).
    """
    tests = []
    model_name = NAME_MAPPING[RAW_MODEL_COL]
    competitor_names = [NAME_MAPPING[c] for c in RAW_COMPETITORS]
    student_name = NAME_MAPPING['Etudiants']
    intern_name = NAME_MAPPING['Internes']
    
    def _run_single_test(group1_col, group2_col, label1, label2):
        """Helper function running statistical tests"""
        # Calculate absolute errors (magnitude of failure)
        err1 = np.abs(df[group1_col] - df[EXPERT_COL])
        err2 = np.abs(df[group2_col] - df[EXPERT_COL])
        
        # Run Wilcoxon via pingouin (returns: W-val, p-val, RBC, CLES, BF10)
        stats = pg.wilcoxon(err1, err2, alternative='two-sided')
        
        # Extract Metrics
        p_val = stats['p-val'].values[0]
        rbc = stats['RBC'].values[0]  # effect size (-1 to 1)
        
        # Calculate Jeffreys-Zellner-Siow Bayes factor (one-sample test against 0)
        ttest_res = pg.ttest(err1, err2, paired=True)
        bf10 = pd.to_numeric(ttest_res['BF10'], errors='coerce').values[0]
        
        # Determine who won, based on MAE
        mae1 = metrics_df.loc[label1, 'MAE']
        mae2 = metrics_df.loc[label2, 'MAE']
        if mae1 < mae2:
            direction = f"{label1} better"
        elif mae2 < mae1:
            direction = f"{label2} better"
        else:
            direction = "TIE"

        # Significance (frequentist approach)
        if p_val < 0.05:
            sig_label = "**Significant**"
        else:
            sig_label = "Not sig"

        # Evidence interpretation (bayesian approach)
        if bf10 > 3:
            evidence = "Favors difference"
        elif bf10 < 0.333:
            evidence = "Favors equivalence"
        else:
            evidence = "Inconclusive"

        return {
            'Comparison': f'{label1} vs {label2}', 
            'p-val': p_val, 
            'Significance': sig_label,
            'RBC (effect size)': rbc,
            'BF10': bf10,
            'Bayes interpretation': evidence,
        }

    # Model vs competitors
    for group in competitor_names:
        res = _run_single_test(model_name, group, 'Model', group)
        tests.append(res)

    # Students vs interns
    if student_name in df.columns and intern_name in df.columns:
        res = _run_single_test(student_name, intern_name, 'Students', 'Residents')
        tests.append(res)
    
    # Save stat report to a csv file
    stat_report = pd.DataFrame(tests)
    stat_path = os.path.join(OUTPUT_DIR, 'statistical_tests.csv')
    stat_report.to_csv(stat_path, index=False)
    
    return stat_report, stat_path


def plot_comprehensive_analysis(df):
    """Generates the 3x3 mega-figure."""
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])
    
    # --- ROW 1: Confusion matrices ---
    all_cms = [confusion_matrix(df[EXPERT_COL], df[group], labels=LABELS) for group in GROUPS]
    global_max = max(cm.max() for cm in all_cms)
    for i, group in enumerate(GROUPS):
        ax = fig.add_subplot(gs[0, i])
        cm = all_cms[i]
        mask = (cm == 0)  # makes cells with 0 value transparent (white background)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='YlGnBu', cbar=False, ax=ax, mask=mask, linecolor='k',
            linewidths=1, vmin=0.0, vmax=global_max, xticklabels=LABELS, yticklabels=LABELS,
        )

        # Re-enable the axes spines and match the heatmap's line properties
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('k')
            spine.set_linewidth(1)
                    
        ax.set_title(f'Confusion matrix: {group}', fontsize=TITLE_SIZE, fontweight='bold')
        ax.set_ylabel('True label (expert)', fontsize=LABEL_SIZE)
        ax.set_xlabel('Predicted label', fontsize=LABEL_SIZE)

    # --- ROW 2: Bland-Altman plots ---
    for i, group in enumerate(GROUPS):
        ax = fig.add_subplot(gs[1, i])
        mean = (df[EXPERT_COL] + df[group]) / 2
        diff = df[EXPERT_COL] - df[group]
        jitter = np.random.normal(0, 0.1, len(df))
        bias, std_diff = np.mean(diff), np.std(diff)

        ax.scatter(mean + jitter, diff + jitter, alpha=0.4, s=25)
        ax.axhline(bias, color='red', linestyle='--', label=f'Bias: {bias:.2f}')
        ax.axhline(0, color='black', linewidth=1)
        ax.axhline(bias + 1.96 * std_diff, color='gray', linestyle=':', label='+1.96 SD')
        ax.axhline(bias - 1.96 * std_diff, color='gray', linestyle=':', label='-1.96 SD')
        ax.set_ylim(-4.5, 4.5)
        ax.set_title(f'Bland-Altman: {group}', fontsize=TITLE_SIZE, fontweight='bold')
        ax.set_xlabel('Mean of (expert + pred)', fontsize=LABEL_SIZE)
        ax.set_ylabel('Diff (expert - pred)', fontsize=LABEL_SIZE)
        ax.legend(loc='upper right')

    # --- ROW 3: Advanced analysis ---
    
    # Error distribution histogram
    ax_hist = fig.add_subplot(gs[2, 0])
    bins = np.arange(-6.5, 7.5, 1)
    
    for group in GROUPS:
        errors = df[group] - df[EXPERT_COL]
        sns.histplot(errors, bins=bins, element="step", fill=False, 
                     label=f'{group}', ax=ax_hist, stat="density", common_norm=False, linewidth=2)
                     
    ax_hist.set_title('Error distribution (pred - expert)', fontsize=TITLE_SIZE, fontweight='bold')
    ax_hist.set_xlabel('Error (negative = underestimation)', fontsize=LABEL_SIZE)
    ax_hist.legend()

    # MAE per ground truth class
    ax_bar = fig.add_subplot(gs[2, 1])
    mae_data = []
    for label in LABELS:
        subset = df[df[EXPERT_COL] == label]
        if len(subset) > 0:
            for group in GROUPS:
                mae = mean_absolute_error(subset[EXPERT_COL], subset[group])
                mae_data.append({'True Label': label, 'Group': group, 'MAE': mae})
    
    mae_df = pd.DataFrame(mae_data)
    sns.barplot(data=mae_df, x='True Label', y='MAE', hue='Group', ax=ax_bar, palette='viridis')
    ax_bar.set_title('MAE per expert label', fontsize=TITLE_SIZE, fontweight='bold')
    ax_bar.set_xlabel('True label (expert)', fontsize=LABEL_SIZE)
    ax_bar.set_ylabel('Mean absolute error', fontsize=LABEL_SIZE)
    ax_bar.legend(loc='upper left')

    # Error correlation heatmap
    ax_corr = fig.add_subplot(gs[2, 2])
    error_df = pd.DataFrame()
    for group in GROUPS:
        error_df[group] = df[group] - df[EXPERT_COL]
    
    corr = error_df.corr(method='spearman')
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax_corr, fmt=".2f")
    ax_corr.set_title('Error correlation', fontsize=TITLE_SIZE, fontweight='bold')
    ax_corr.tick_params(axis='x', rotation=0)
    ax_corr.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'detailed_ordinal_analysis.png')
    plt.savefig(plot_path, dpi=300)
    return plot_path


def analyze_broad_classes(df):
    """Analyzes performance on broad clinical categories."""
    # Create a copy to avoid modifying the original
    df_broad = df.copy()
    cols_to_map = [EXPERT_COL] + GROUPS
    
    # DROP MISSING VALUES strictly for this analysis
    # This fixes the "float vs str" error by removing NaNs
    before_drop = len(df_broad)
    df_broad = df_broad.dropna(subset=cols_to_map)
    after_drop = len(df_broad)
    if before_drop != after_drop:
        print(f"Warning: Dropped {before_drop - after_drop} rows containing NaN values.")

    # Ensure inputs are integers so the dictionary mapping works
    for col in cols_to_map:
        df_broad[col] = df_broad[col].astype(int)

    # Map the values
    for col in cols_to_map:
        df_broad[col] = df_broad[col].map(CLINICAL_MAP)

    # Calculate metrics
    metrics = []
    for group in GROUPS:
        y_t, y_p = df_broad[EXPERT_COL], df_broad[group]

        # Calculate metrics
        acc = accuracy_score(y_t, y_p)
        f1_m = f1_mac(y_t, y_p)
        f1_w = f1_wei(y_t, y_p)

        # Calculate CIs
        acc_ci = compute_bootstrap_ci(y_t, y_p, accuracy_score)
        f1_m_ci = compute_bootstrap_ci(y_t, y_p, f1_mac)
        f1_w_ci = compute_bootstrap_ci(y_t, y_p, f1_wei)

        metrics.append({
            'Group': group,
            'Accuracy': f"{acc:.3f} [{acc_ci[0]:.3f}-{acc_ci[1]:.3f}]",
            'F1-macro': f"{f1_m:.3f} [{f1_m_ci[0]:.3f}-{f1_m_ci[1]:.3f}]",
            'F1-weighted': f"{f1_w:.3f} [{f1_w_ci[0]:.3f}-{f1_w_ci[1]:.3f}]"
        })
    
    print(pd.DataFrame(metrics).set_index('Group'))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for i, group in enumerate(GROUPS):
        cm = confusion_matrix(df_broad[EXPERT_COL], df_broad[group], labels=CLINICAL_LABELS)
        
        # Mask zeros
        mask = (cm == 0)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Oranges', cbar=False, 
            ax=axes[i], mask=mask, linecolor='k', linewidths=1,
            xticklabels=CLINICAL_LABELS, yticklabels=CLINICAL_LABELS,
        )

        # Re-enable the axes spines and match the heatmap's line properties
        for spine in axes[i].spines.values():
            spine.set_visible(True)
            spine.set_color('k')
            spine.set_linewidth(1)
        
        axes[i].set_title(f'{group}', fontsize=TITLE_SIZE)
        axes[i].set_ylabel('Expert label', fontsize=LABEL_SIZE)
        axes[i].set_xlabel('Predicted label', fontsize=LABEL_SIZE)
        axes[i].tick_params(axis='x')

    plot_path = os.path.join(OUTPUT_DIR, 'broad_clinical_analysis.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    return plot_path


def analyze_consensus_agreement():
    """Analyzes and prints the agreement between individual raters before consensus."""
    try:
        df_cons = pd.read_csv(CVV_FILE_DETAILED)
    except FileNotFoundError:
        print(f"ERROR: '{CVV_FILE_DETAILED}' not found.")
        return

    # Define the pairs of raters for each category based on the CSV
    rater_pairs = {
        'Students': ('Etudiants1', 'Etudiants2'),
        'Interns': ('Internes1', 'Internes2'),
        'Experts (CDC)': ('Chefs_de_Clinique1', 'Chefs_de_Clinique2')
    }
    
    # Calculate for human raters
    results = []
    for category, (rater1, rater2) in rater_pairs.items():
        subset = df_cons[[rater1, rater2]].dropna()
        if subset.empty: continue
            
        y_1, y_2 = subset[rater1], subset[rater2]
        
        qwk = cohen_kappa_score(y_1, y_2, weights='quadratic')
        qwk_ci = compute_bootstrap_ci(y_1, y_2, cohen_kappa_score, weights='quadratic')
        
        mae = mean_absolute_error(y_1, y_2)
        mae_ci = compute_bootstrap_ci(y_1, y_2, mean_absolute_error)
        
        acc = exact_acc(y_1, y_2)
        acc_ci = compute_bootstrap_ci(y_1, y_2, exact_acc)
        
        results.append({
            'Category': category,
            'QWK (inter-rater)': f"{qwk:.3f} [{qwk_ci[0]:.3f}-{qwk_ci[1]:.3f}]",
            'MAE': f"{mae:.3f} [{mae_ci[0]:.3f}-{mae_ci[1]:.3f}]",
            'Acc (exact)': f"{acc:.3f} [{acc_ci[0]:.3f}-{acc_ci[1]:.3f}]"
        })
        
    # Calculate for the model
    model_str = RAW_MODEL_COL.rsplit('-maj', 1)[0] if '-maj' in RAW_MODEL_COL else RAW_MODEL_COL.rsplit('-single', 1)[0]
    model_metrics = get_model_internal_agreement(MODEL_DIR, model_str)
    if model_metrics:
        results.append(model_metrics)

    # Build and print the table
    results_df = pd.DataFrame(results).set_index('Category')
    print(results_df.round(3))
    
    out_path = os.path.join(OUTPUT_DIR, 'inter_rater_agreement.csv')
    results_df.to_csv(out_path)
    print(f"Inter-rater agreement table saved to {out_path}")


def get_model_internal_agreement(model_dir, model_str, generations=["mRS_000", "mRS_009"]):
    """Calculates agreement between two generations of a model."""
    # Assuming raw results are saved as a CSV matching the model string
    raw_csv_path = os.path.join(model_dir, f"{model_str}.csv")
    try:
        df_model = pd.read_csv(raw_csv_path)
    except FileNotFoundError:
        print(f"Warning: Raw model file '{raw_csv_path}' not found. Skipping internal agreement.")
        return None

    # Check if required columns exist in the raw file
    if any([gen not in df_model.columns for gen in generations]):
        print(f"Warning: {generations} not all found in {raw_csv_path}. Skipping.")
        return None

    # Drop rows where either generation failed/is missing
    subset = df_model[generations].dropna()
    if subset.empty:
        return None
    
    y_1, y_2 = subset[generations[0]], subset[generations[1]]
    
    qwk = cohen_kappa_score(y_1, y_2, weights='quadratic')
    qwk_ci = compute_bootstrap_ci(y_1, y_2, cohen_kappa_score, weights='quadratic')
    
    mae = mean_absolute_error(y_1, y_2)
    mae_ci = compute_bootstrap_ci(y_1, y_2, mean_absolute_error)
    
    acc = exact_acc(y_1, y_2)
    acc_ci = compute_bootstrap_ci(y_1, y_2, exact_acc)
    
    return {
        'Category': 'Model (generations)',
        'QWK (inter-rater)': f"{qwk:.3f} [{qwk_ci[0]:.3f}-{qwk_ci[1]:.3f}]",
        'MAE': f"{mae:.3f} [{mae_ci[0]:.3f}-{mae_ci[1]:.3f}]",
        'Acc (exact)': f"{acc:.3f} [{acc_ci[0]:.3f}-{acc_ci[1]:.3f}]"
    }


if __name__ == "__main__":
    print(f"\n--- USING MODEL {RAW_MODEL_COL} ---")
    df = load_and_prep_data(CSV_FILE)
    
    if df is not None:
        print("\n--- PERFORMANCE METRICS ---")
        metrics_report = calculate_metrics(df)
        print(metrics_report.round(3))
        
        print("\n--- STATISTICAL TESTS (Wilcoxon) ---")
        stat_report, stat_path = run_statistical_tests(df, metrics_report)
        print(stat_report)
        print(f"\nStatistical tests saved to {stat_path}")

        print("\n--- COMPREHENSIVE ANALYSIS ---")
        path1 = plot_comprehensive_analysis(df)
        print(f"\nDetailed analysis plots saved to {path1}")

        print("\n--- BROAD CLINICAL CLASS ANALYSIS ---")
        path2 = analyze_broad_classes(df)
        print(f"\nBroad analysis plots saved to {path2}")

        print("\n--- CONSENSUS AGREEMENT DETAILS ---")
        analyze_consensus_agreement()
        
        print("\nDone.")