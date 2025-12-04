# Bayesian Statistical Models: Optimized Python Implementations

This repository contains highly optimized Python implementations of three fundamental Bayesian statistical models. Each implementation focuses on computational efficiency, numerical stability, and vectorization, replacing standard iterative loops with linear algebra operations (NumPy/SciPy) and JIT compilation (Numba) where appropriate.

## 📂 Repository Structure

| File | Model | Key Techniques |
| :--- | :--- | :--- |
| **`GP.py`** | Gaussian Process Regression | Cholesky Decomposition, SciPy `cdist` |
| **`BayesianFM.py`** | 3D Bayesian Finite Mixture | Vectorized Gibbs Sampling, Log-Sum-Exp Trick |
| **`BayesianHMMGibbs.py`** | Bayesian Hidden Markov Model | FFBS Algorithm, Numba JIT Compilation |

---

## 🚀 Installation & Dependencies

To run these scripts, you will need a standard scientific Python environment.

```bash
pip install numpy scipy matplotlib numba
