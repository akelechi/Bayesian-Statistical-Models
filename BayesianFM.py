import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm, poisson
import time

def gibbs_sampler_optimized(n_iter=10000):
    np.random.seed(42) # For reproducibility
    
    # Create the data components
    # Cluster 1: N(0,1), N(2,1), Pois(10)
    xn1 = np.random.normal(0, 1, 1251)
    yn1 = np.random.normal(2, 1, 1251)
    zn1 = np.random.poisson(10, 1251)
    
    # Cluster 2: N(5,1), N(7,1), Pois(20) (Values shifted for distinct clusters)
    xn2 = np.random.normal(5, 1, 500)
    yn2 = np.random.normal(7, 1, 500)
    zn2 = np.random.poisson(20, 500)
    
    # Cluster 3: N(-5,1), N(-2,1), Pois(5)
    xn3 = np.random.normal(-5, 1, 500)
    yn3 = np.random.normal(-2, 1, 500)
    zn3 = np.random.poisson(5, 500)
    
    # Concatenate to form the full dataset (N x 1 arrays)
    xn = np.concatenate([xn1, xn2, xn3])
    yn = np.concatenate([yn1, yn2, yn3])
    zn = np.concatenate([zn1, zn2, zn3])
    
    N = len(xn)
    K = 3 # Number of clusters
    
    # --- 2. Initialization ---
    
    # Initial Assignments (Random)
    s = np.random.choice(K, N) # 0, 1, 2 instead of 1, 2, 3 for Python indexing
    
    # Hyperparameters
    beta = np.array([1/3, 1/3, 1/3])
    
    # Initial Parameters
    # Means for X and Y, Rates for Z
    mu_x = np.random.normal(0, 2, K)
    mu_y = np.random.normal(2, 2, K)
    lam_z = np.random.gamma(shape=10, scale=2, size=K) # shape=k, scale=theta
    
    # Weights
    pi = np.array([1/3, 1/3, 1/3])
    
    # Precision Tau (Gamma(shape=2, scale=2))
    tau = np.random.gamma(2, 2)
    
    # Storage for Trace Plots
    trace_mu_x = np.zeros((K, n_iter))
    trace_mu_y = np.zeros((K, n_iter))
    trace_lam_z = np.zeros((K, n_iter))
    
    # Metric tracking (Equality check)
    equality_count = 0
    
    print(f"Starting Gibbs Sampler for {n_iter} iterations...")
    start_time = time.time()

    # --- 3. The Gibbs Loop ---
    for t in range(n_iter):
        
        # --- A. Update Weights (Pi) ---
        # Count members in each cluster
        # np.bincount is extremely fast for integer counting
        counts = np.bincount(s, minlength=K)
        
        # Dirichlet update
        pi = np.random.dirichlet(beta + counts)
        
        # --- B. Update Assignments (s) - Vectorized ---
        
        # We calculate Log-Likelihoods for stability.
        # Shape of result will be (N, K)
        
        # 1. Log Prob of X (Normal)
        # Using broadcasting: xn is (N,1), mu_x is (1,K) -> Result (N,K)
        lp_x = -0.5 * np.log(2 * np.pi) - 0.5 * (xn[:, None] - mu_x[None, :])**2
        
        # 2. Log Prob of Y (Normal with precision tau)
        # Standard Deviation = 1 / sqrt(tau)
        sigma_y = 1.0 / np.sqrt(tau)
        lp_y = -0.5 * np.log(2 * np.pi * sigma_y**2) - \
               0.5 * ((yn[:, None] - mu_y[None, :])**2) / (sigma_y**2)

        # 3. Log Prob of Z (Poisson)
        # log(Pois(k|lam)) = k*log(lam) - lam - log(k!)
        # We can ignore log(k!) as it cancels out during normalization
        # Add epsilon to lam_z to prevent log(0)
        lp_z = zn[:, None] * np.log(lam_z[None, :] + 1e-10) - lam_z[None, :]
        
        # Total Log Likelihood + Log Prior (Weights)
        log_prob_unnorm = lp_x + lp_y + lp_z + np.log(pi[None, :])
        
        # Normalize in Log-Space (Log-Sum-Exp Trick)
        # p = exp(log_p - max_log_p)
        max_lp = np.max(log_prob_unnorm, axis=1, keepdims=True)
        prob_unnorm = np.exp(log_prob_unnorm - max_lp)
        prob_norm = prob_unnorm / np.sum(prob_unnorm, axis=1, keepdims=True)
        
        # Vectorized Random Sampling
        # This draws an index (0,1,2) for every row based on prob_norm
        cumsum = np.cumsum(prob_norm, axis=1)
        rand_vals = np.random.rand(N, 1)
        s = (rand_vals > cumsum).sum(axis=1) # Returns 0, 1, or 2
        
        # Track equality for specific indices
        if s[10] == s[981]:
            equality_count += 1

        # --- C. Update Parameters ---
        # Recalculate counts after new assignments
        counts = np.bincount(s, minlength=K)
        
        for k in range(K):
            # Boolean mask for current cluster
            mask = (s == k)
            N_k = counts[k]
            
            # Sums needed for updates
            sum_x = np.sum(xn[mask]) if N_k > 0 else 0
            sum_y = np.sum(yn[mask]) if N_k > 0 else 0
            sum_z = np.sum(zn[mask]) if N_k > 0 else 0
            
            # 1. Update Mu_X (Normal-Normal)
            # Prior: N(0,1) -> Variance 1. Likelihood Variance 1.
            # Posterior Precision = Prior Prec + N * Data Prec = 1 + N_k
            # Posterior Variance = 1 / (N_k + 1)
            # Posterior Mean = (Prior_Prec*Prior_Mean + Data_Prec*Sum_X) * Post_Var
            #                = (0 + 1*sum_x) / (N_k + 1)
            post_var_x = 1.0 / (N_k + 1)
            post_mean_x = sum_x / (N_k + 1)
            mu_x[k] = np.random.normal(post_mean_x, np.sqrt(post_var_x))
            
            # 2. Update Mu_Y (Normal-Normal with precision tau)
            post_prec_y = (tau * N_k) + 0.5
            post_var_y = 1.0 / post_prec_y
            # From your code: mean = (tau*sum_y + 1) / precision
            post_mean_y = (tau * sum_y + 1) / post_prec_y
            mu_y[k] = np.random.normal(post_mean_y, np.sqrt(post_var_y))
            
            # 3. Update Lambda_Z (Gamma-Poisson)
            # Posterior Shape = sum(z) + prior_shape (10)
            # Posterior Scale = 1 / (n + 1/prior_scale) 
            shape_z = sum_z + 10
            scale_z = 1.0 / (N_k + 1)
            lam_z[k] = np.random.gamma(shape_z, scale_z)

        # --- D. Update Tau ---
        # Calculate Sum of Squared Errors for Y
        # Efficient numpy indexing: mu_y[s] creates a vector where each y is matched to its cluster mean
        sq_diff = (yn - mu_y[s])**2
        sum_sq_err = np.sum(sq_diff)
        
        # From your code: sum_ga = 0.5 * sum_sq + 1
        # From your code: tau = gamrnd(2, 1/sum_ga) -> scale = 1/sum_ga
        rate_param = 0.5 * sum_sq_err + 1
        tau = np.random.gamma(shape=2, scale=(1.0 / rate_param))

        # Store traces
        trace_mu_x[:, t] = mu_x
        trace_mu_y[:, t] = mu_y
        trace_lam_z[:, t] = lam_z

    end_time = time.time()
    print(f"Done! Time elapsed: {end_time - start_time:.2f} seconds")
    print(f"Equality Probability: {equality_count / n_iter:.4f}")
    
    return trace_mu_x, trace_mu_y, trace_lam_z

# Run the sampler
if __name__ == "__main__":
    tx, ty, tz = gibbs_sampler_optimized(n_iter=5000)
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']
    colors = ['r', 'g', 'b']
    
    for k in range(3):
        axes[0].plot(tx[k, :], color=colors[k], alpha=0.6, label=f'Mu X (C{k+1})')
        axes[1].plot(ty[k, :], color=colors[k], alpha=0.6, label=f'Mu Y (C{k+1})')
        axes[2].plot(tz[k, :], color=colors[k], alpha=0.6, label=f'Lambda Z (C{k+1})')
    
    axes[0].set_title('Trace Plot: X Means')
    axes[0].set_ylabel('Mean Value')
    axes[0].legend()
    
    axes[1].set_title('Trace Plot: Y Means')
    axes[1].set_ylabel('Mean Value')
    
    axes[2].set_title('Trace Plot: Z Rates')
    axes[2].set_ylabel('Rate Value')
    axes[2].set_xlabel('Iteration')
    
    plt.tight_layout()
    plt.show()