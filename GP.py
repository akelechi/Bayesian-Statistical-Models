import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky, cho_solve, solve_triangular

def gp_regression_optimized(X_train, y_train, X_test, noise_var, sigma_f=6.0, len_scale=4.0):
    """
    Optimized Gaussian Process Regression using Cholesky Decomposition.
    
    Args:
        X_train (np.array): Training inputs (N x d).
        y_train (np.array): Training targets (N x 1).
        X_test (np.array): Test inputs (M x d).
        noise_var (float/np.array): Noise variance (scalar or vector).
        sigma_f (float): Signal standard deviation (hyperparameter).
        len_scale (float): Length scale (hyperparameter).
        
    Returns:
        mu (np.array): Posterior mean (M x 1).
        cov_posterior (np.array): Posterior covariance (M x M).
        f_sample (np.array): A sample drawn from the posterior.
    """
    
    # 1. Efficient Input Handling
    # Ensure inputs are 2D arrays (N x 1) for cdist to work correctly
    X_train = np.atleast_2d(X_train).T if X_train.ndim == 1 else np.atleast_2d(X_train)
    y_train = np.atleast_2d(y_train).T if y_train.ndim == 1 else np.atleast_2d(y_train)
    X_test  = np.atleast_2d(X_test).T  if X_test.ndim == 1  else np.atleast_2d(X_test)
    
    N = X_train.shape[0]
    
    # 2. Vectorized Kernel Computation
    # cdist uses optimized C-code to calculate pairwise distances.
    # It replaces the O(N^2) nested loops completely.
    K_train = sigma_f**2 * np.exp(-cdist(X_train, X_train, 'sqeuclidean') / (2 * len_scale**2))
    K_cross = sigma_f**2 * np.exp(-cdist(X_train, X_test,  'sqeuclidean') / (2 * len_scale**2))
    K_test  = sigma_f**2 * np.exp(-cdist(X_test,  X_test,  'sqeuclidean') / (2 * len_scale**2))
    
    
    # Add small epsilon to diagonal to prevent non-positive definite errors
    jitter = 1e-8 * np.eye(N)
    
    # Handle noise matrix efficiently 
    if np.isscalar(noise_var):
        V = noise_var * np.eye(N)
    else:
        V = np.diag(np.ravel(noise_var))
        
    K_y = K_train + V + jitter

    # 4. Cholesky Decomposition
    # L is Lower Triangular such that L * L.T = K_y
    try:
        L = cholesky(K_y, lower=True)
    except np.linalg.LinAlgError:
        # If matrix is ill-conditioned: add more jitter
        L = cholesky(K_y + 1e-5 * np.eye(N), lower=True)

    # 5. Compute Mean (mu)
    # Solve (K + V) * alpha = y  -->  alpha = K_y^-1 * y
    # Using cho_solve is faster than standard solve when L is known
    alpha = cho_solve((L, True), y_train)
    mu = K_cross.T @ alpha

    # 6. Compute Posterior Covariance
    # v = L^-1 * K_cross
    # Cov = K_test - v.T * v
    # solve_triangular is highly optimized for triangular matrices
    v = solve_triangular(L, K_cross, lower=True)
    cov_posterior = K_test - v.T @ v
    
    # 7. Sample from Posterior
    # L_post * u ~ N(0, Cov)
    # We re-decompose the posterior covariance to sample
    cov_post_stable = cov_posterior + 1e-8 * np.eye(cov_posterior.shape[0])
    L_post = cholesky(cov_post_stable, lower=True)
    f_sample = mu + L_post @ np.random.normal(size=(cov_posterior.shape[0], 1))

    return mu, cov_posterior, f_sample

# --- Example Usage ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate dummy data
    np.random.seed(42)
    N_train = 20
    X_tr = np.random.uniform(-5, 5, N_train)
    # True function: sin(x) + noise
    y_tr = np.sin(X_tr) + np.random.normal(0, 0.1, N_train) 
    noise_vals = 0.01 * np.ones(N_train) # Low variance

    # Test points
    X_te = np.linspace(-6, 6, 200)

    # Run Optimized GP
    mean, cov, sample = gp_regression_optimized(X_tr, y_tr, X_te, noise_vals)

    # Visualization
    variance = np.diag(cov)
    std_dev = np.sqrt(variance)

    plt.figure(figsize=(10, 6))
    plt.fill_between(X_te, 
                     mean.flatten() - 2*std_dev, 
                     mean.flatten() + 2*std_dev, 
                     color='gray', alpha=0.3, label='95% Confidence')
    plt.plot(X_te, mean, 'b-', lw=2, label='Mean Prediction')
    plt.plot(X_te, sample, 'g--', lw=1, label='Posterior Sample')
    plt.scatter(X_tr, y_tr, c='r', marker='x', zorder=10, label='Observations')
    plt.title("Gaussian Process Regression (Vectorized Python)")
    plt.legend()
    plt.show()