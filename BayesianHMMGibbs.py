import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import numba

# Use Numba to speed up the explicit loops in FFBS
@numba.jit(nopython=True)
def run_ffbs_numba(N, K, w, Phi, P):
    # --- Forward Pass ---
    A = np.zeros((N, K))
    
    # Precompute Likelihoods (N x K)
    # Manual normpdf for Numba speed: exp(-0.5 * ((x-mu)/sig)^2)
    Likelihood = np.zeros((N, K))
    for k in range(K):
        Likelihood[:, k] = np.exp(-0.5 * (w - Phi[k])**2) # assuming sigma=1
    
    # t=0
    A[0, :] = Likelihood[0, :] * (1.0/K)
    A[0, :] /= np.sum(A[0, :])
    
    # t=1 to N-1
    for t in range(1, N):
        # Forward Step: sum(A[t-1] * P_ij) * Likelihood
        # Vectorized dot product
        pred = np.dot(A[t-1, :], P) 
        A[t, :] = Likelihood[t, :] * pred
        
        # Normalize
        s_sum = np.sum(A[t, :])
        if s_sum > 0:
            A[t, :] /= s_sum
        else:
            A[t, :] = 1.0/K
            
    # --- Backward Sampling ---
    s = np.zeros(N, dtype=np.int64)
    
    # Sample last state
    # Custom sampling for Numba
    cumsum = np.cumsum(A[N-1, :])
    r = np.random.rand()
    s[N-1] = 0
    for k in range(K):
        if r < cumsum[k]:
            s[N-1] = k
            break
            
    # Sample backwards
    for t in range(N-2, -1, -1):
        # Calculate P(s_t | s_t+1)
        # numer = A[t] * P[:, s[t+1]]
        next_state = s[t+1]
        numer = A[t, :] * P[:, next_state]
        
        # Normalize
        prob = numer / np.sum(numer)
        
        # Sample
        cumsum = np.cumsum(prob)
        r = np.random.rand()
        s[t] = K-1 # Default
        for k in range(K):
            if r < cumsum[k]:
                s[t] = k
                break
                
    return s

def hmm_gibbs_sampler(w_data, n_iter=2000):
    N = len(w_data)
    K = 3
    
    # Init
    Phi = np.array([-1.0, 3.0, 6.0]) # Means
    P = np.array([[0.5, 0.25, 0.25],
                  [0.25, 0.5, 0.25],
                  [0.25, 0.25, 0.5]])
    
    trace_Phi = np.zeros((K, n_iter))
    
    print("Starting Sampling...")
    
    for i in range(n_iter):
        
        # 1. & 2. Forward Filtering Backward Sampling
        s = run_ffbs_numba(N, K, w_data, Phi, P)
        
        # 3. Update Means (Phi)
        for k in range(K):
            data_k = w_data[s == k]
            n_k = len(data_k)
            if n_k > 0:
                # Post Mean calculation (assuming prior N(0, 10))
                # Simplification: assuming flat prior for this demo
                # Post Prec = n_k + small_prior
                post_var = 1.0 / (n_k + 0.1)
                post_mean = np.sum(data_k) * post_var
                Phi[k] = np.random.normal(post_mean, np.sqrt(post_var))
            else:
                Phi[k] = np.random.normal(0, 5)
        
        trace_Phi[:, i] = Phi
        
        # 4. Update Transition Matrix (P)
        # Count transitions
        trans_counts = np.zeros((K, K))
        for t in range(N-1):
            trans_counts[s[t], s[t+1]] += 1
            
        # Dirichlet Update
        for k in range(K):
            P[k, :] = np.random.dirichlet(1 + trans_counts[k, :])
            
    return trace_Phi

# --- Usage ---
# Generate dummy data
np.random.seed(42)
true_s = np.concatenate([np.zeros(200), np.ones(200), np.full(200, 2)])
w_dummy = np.concatenate([np.random.normal(-1, 1, 200), 
                          np.random.normal(3, 1, 200), 
                          np.random.normal(6, 1, 200)])

trace = hmm_gibbs_sampler(w_dummy, n_iter=2000)

plt.plot(trace.T)
plt.title("Trace Plot of Emission Means (Phi)")
plt.show()