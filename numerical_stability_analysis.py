import numpy as np
import matplotlib.pyplot as plt
import time

# =====================================================================
# 1. GENERATE AN ILL-CONDITIONED SYNTHETIC DATASET
# =====================================================================
# To simulate real-world numerical instability, we create multicollinearity
np.random.seed(42)
n_samples = 500
x1 = np.random.rand(n_samples, 1)
# x2 is nearly identical to x1, forcing a massive condition number
x2 = x1 + np.random.normal(0, 1e-7, size=(n_samples, 1)) 

# Design Matrix X_b (including bias column)
X_b = np.c_[np.ones((n_samples, 1)), x1, x2]
y = 4 + 3*x1 + 2*x2 + np.random.normal(0, 0.1, size=(n_samples, 1))

# Calculate and log the condition number
cond_number = np.linalg.cond(X_b.T @ X_b)
print(f"Condition Number of (X^T * X): {cond_number:.2e}\n")


# =====================================================================
# 2. NUMERICALLY STABLE DIRECT SOLVERS
# =====================================================================

def solve_via_naive_inverse(X, y):
    """The original approach: highly unstable for ill-conditioned matrices."""
    start = time.time()
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta, time.time() - start

def solve_via_qr(X, y):
    """
    Solves X*theta = y using QR Decomposition.
    Avoids computing (X^T * X) directly, preserving the condition number kappa(X).
    """
    start = time.time()
    Q, R = np.linalg.qr(X)
    # R * theta = Q^T * y -> Solve via back-substitution
    theta = np.linalg.solve(R, Q.T @ y)
    return theta, time.time() - start

def solve_via_svd(X, y):
    """
    Solves using Singular Value Decomposition (Pseudo-inverse).
    The most robust method for near-singular matrices.
    """
    start = time.time()
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    # S_inverse scales down small singular values safely
    S_inv = np.diag(1.0 / s)
    theta = Vt.T @ S_inv @ U.T @ y
    return theta, time.time() - start


# =====================================================================
# 3. ADVANCED ITERATIVE OPTIMIZATION WITH ADAPTIVE STEP-SIZE
# =====================================================================

def compute_cost(X, y, theta):
    m = len(y)
    return (1 / (2 * m)) * np.sum((X @ theta - y) ** 2)

def batch_gradient_descent_adaptive(X, y, max_iters=2000, tol=1e-6):
    """
    Gradient Descent utilizing Armijo-Goldstein Backtracking Line Search
    to mathematically optimize the learning rate (alpha) at each step.
    """
    start_time = time.time()
    m, p = X.shape
    theta = np.random.randn(p, 1)
    cost_history = []
    
    for iteration in range(max_iters):
        gradients = (1 / m) * X.T @ (X @ theta - y)
        
        # Backtracking Line Search (Armijo Condition)
        alpha = 1.0  # Initial bold step size
        c = 0.1      # Control parameter
        tau = 0.5    # Shrinkage factor
        
        current_cost = compute_cost(X, y, theta)
        while compute_cost(X, y, theta - alpha * gradients) > current_cost - c * alpha * np.sum(gradients**2):
            alpha *= tau
            if alpha < 1e-12:  # Prevent infinite loop if gradient is flat
                break
                
        # Gradient Update
        theta_new = theta - alpha * gradients
        cost = compute_cost(X, y, theta_new)
        cost_history.append(cost)
        
        # Convergence criteria check (L2 norm of the step variation)
        if np.linalg.norm(theta_new - theta) < tol:
            theta = theta_new
            break
            
        theta = theta_new
        
    return theta, cost_history, time.time() - start_time

# =====================================================================
# 4. EXECUTION AND BENCHMARKING EVALUATION
# =====================================================================

theta_naive, t_naive = solve_via_naive_inverse(X_b, y)
theta_qr, t_qr       = solve_via_qr(X_b, y)
theta_svd, t_svd     = solve_via_svd(X_b, y)
theta_gd, cost_hist, t_gd = batch_gradient_descent_adaptive(X_b, y)

print("--- NUMERICAL COMPARISON ---")
print(f"Naive Inverse Theta: {theta_naive.ravel()} | Time: {t_naive:.6f}s")
print(f"QR Solver Theta:     {theta_qr.ravel()} | Time: {t_qr:.6f}s")
print(f"SVD Solver Theta:    {theta_svd.ravel()} | Time: {t_svd:.6f}s")
print(f"Adaptive GD Theta:   {theta_gd.ravel()} | Time: {t_gd:.6f}s\n")

# Calculate Residual Norms ||X*theta - y||_2 to evaluate precision
print("--- L2 RESIDUAL NORMS (Precision Verification) ---")
print(f"Naive Residual: {np.linalg.norm(X_b @ theta_naive - y):.8f}")
print(f"QR Residual:    {np.linalg.norm(X_b @ theta_qr - y):.8f}")
print(f"SVD Residual:   {np.linalg.norm(X_b @ theta_svd - y):.8f}")
print(f"GD Residual:    {np.linalg.norm(X_b @ theta_gd - y):.8f}")