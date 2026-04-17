import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import norm, binom

def continuous_to_ordinal(continuous_scores, thresholds):
    """
    Maps continuous severity scores to ordinal mRS categories 
    based on the calculated statistical thresholds.
    """
    return np.digitize(continuous_scores, thresholds)

def calculate_sample_size():
    # Target statistical parameters
    alpha = 0.05              # One-sided significance level
    target_power = 0.80       # Required statistical power
    margin = 0.05             # Non-inferiority margin (delta)

    # Estimated probabilities for mRS 0 to 6
    mrs_probs = np.array([0.35, 0.15, 0.15, 0.10, 0.10, 0.05, 0.10])
    min_class_prob = np.min(mrs_probs) 
    min_samples_required = 2
    
    # Build the latent variable model
    # Convert probabilities into thresholds on a standard normal distribution
    cum_probs = np.cumsum(mrs_probs)
    thresholds = norm.ppf(cum_probs[:-1])  # inner boundaries for the 7 classes
    
    # The 'noise_std' controls raters performance
    # A value of ~0.33 typically yields a baseline QWK around 0.90
    noise_std_student = 0.33
    noise_std_llm = 0.33
    
    n_simulations = 1000
    sample_sizes_to_test = range(20, 500, 10) 
    z_alpha = norm.ppf(1 - alpha)
    
    # Calculate power
    print("Starting simulation to determine sample size...\n")
    best_N_stats = None
    for N in sample_sizes_to_test:
        qwks_llm = []
        qwks_student = []
        diffs = []
        
        for _ in range(n_simulations):
            # True underlying patient severity (Standard Normal)
            true_severity = np.random.randn(N)
            
            # Raters observe the true severity but with some random noise
            student_obs = true_severity + np.random.randn(N) * noise_std_student
            llm_obs = true_severity + np.random.randn(N) * noise_std_llm
            
            # Discretize observations into the 7 mRS categories
            y_expert = continuous_to_ordinal(true_severity, thresholds)
            y_student = continuous_to_ordinal(student_obs, thresholds)
            y_llm = continuous_to_ordinal(llm_obs, thresholds)
            
            # Calculate QWK
            qwk_student = cohen_kappa_score(y_expert, y_student, weights='quadratic')
            qwk_llm = cohen_kappa_score(y_expert, y_llm, weights='quadratic')
            
            qwks_llm.append(qwk_llm)
            qwks_student.append(qwk_student)
            diffs.append(qwk_llm - qwk_student)
            
        se_diff = np.std(diffs)
        print(f"QWK students: {np.mean(qwks_student)}")
        print(f"QWK LLM: {np.mean(qwks_llm)}")
        
        # Power calculation for non-inferiority
        power = norm.cdf((margin / se_diff) - z_alpha)
        print(f"N = {N:<4} | SE = {se_diff:.4f} | Power = {power:.1%}")
        
        # Break the loop once target power is reached
        if power >= target_power:
            best_N_stats = N
            print(f"--> Target power reached! Stopping power simulation.\n")
            break

    # Secondary requirement
    best_N_secondary = None
    for N in range(20, 1000, 10):
        prob_achieved = binom.sf(min_samples_required - 1, N, min_class_prob)
        if prob_achieved >= 0.95:
            best_N_secondary = N
            break

    # Combine both requirements
    final_N = max(best_N_stats, best_N_secondary)
    
    print("--- Final Recommendation ---")
    print(f"N required for statistical power (>={target_power}%): {best_N_stats}")
    print(f"N required for secondary condition (>={min_samples_required} samples in rarest class): {best_N_secondary}")
    print(f"-> Recommended total sample size: {final_N}")

if __name__ == "__main__":
    calculate_sample_size()