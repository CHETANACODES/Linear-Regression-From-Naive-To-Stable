# Comparative Analysis of Direct Numerical Solvers Under Ill-Conditioned Vector Spaces

## Project Overview
This repository investigates the numerical stability and error propagation of direct matrix inversion methods versus orthogonal decomposition techniques when solving the linear least-squares problem:

$$\min_{\theta} \|X\theta - y\|_2^2$$

While algebraically equivalent to the classical Normal Equation, direct matrix inversion is notoriously unstable in floating-point arithmetic when the design matrix $X$ exhibits near-multicollinearity. This project acts as a computational stress-test comparing a Naive Inverse, QR Decomposition, and Singular Value Decomposition (SVD).

## Mathematical Framework

### 1. The Condition Number Problem
The sensitivity of the linear system is governed by the condition number $\kappa(X)$:
$$\kappa(X) = \frac{\sigma_{\max}(X)}{\sigma_{\min}(X)}$$

When forming the Normal Equation $(X^T X)\theta = X^T y$, the condition number is squared:
$$\kappa(X^T X) = \kappa(X)^2$$
This artificial amplification severely degrades numerical precision, making standard operations vulnerable to machine epsilon round-off errors.

### 2. Algorithmic Comparison
* **Naive Inverse:** Computes $\theta = (X^T X)^{-1} X^T y$ directly via standard LU decomposition strategies. High sensitivity to $\kappa(X)^2$.
* **QR Decomposition:** Factors $X = QR$, where $Q$ is orthogonal and $R$ is upper triangular. Solves $R\theta = Q^T y$ via backward substitution, preserving the base condition number $\kappa(X)$.
* **Singular Value Decomposition (SVD):** Decomposes $X = U \Sigma V^T$. Computes the Moore-Penrose pseudoinverse, offering robust stabilization even when $\sigma_{\min} \to 0$.

## Key Findings
Our simulation applies an intentional multicollinearity factor of $10^{-7}$, producing a massive matrix condition number ($\approx 3.10 \times 10^{14}$). The experimental results demonstrate that while the Naive Inverse suffers from structural variance and shifts parameter estimations away from true benchmarks, the QR and SVD implementations isolate the true underlying continuous parameters smoothly with minimal $L_2$ residual error ($2.2119$).
### 3. Theoretical Convergence and the Hessian Spectrum
The mathematical convergence of Gradient Descent on a quadratic cost function is strictly governed by the eigenvalues of the Hessian matrix $H = \frac{1}{m} X^T X$. 



* **The Step-Size Bound:** For a fixed learning rate $\alpha$, the system will converge if and only if:
  $$0 < \alpha < \frac{2}{\lambda_{\max}(H)}$$
  where $\lambda_{\max}(H)$ is the largest eigenvalue of the Hessian. If $\alpha$ exceeds this upper bound, the optimization steps will grow exponentially, causing the algorithm to violently diverge.

* **The Ill-Conditioned Geometry:** Because our design matrix has a massive condition number ($\kappa \approx 3.10 \times 10^{14}$), the ratio of the maximum eigenvalue to the minimum eigenvalue ($\lambda_{\max}/\lambda_{\min}$) is enormous. Geometrically, this warps the cost function's error landscape into an extremely narrow, steep, elongated parabolic valley. 

* **The Role of Backtracking Line Search:** Standard fixed step-sizes oscillate back and forth aggressively across the steep walls of this valley rather than moving down the floor. Our implementation of **Armijo-Goldstein Backtracking Line Search** dynamically computes the local Lipschitz constant at each iteration, shrinking $\alpha$ whenever an overshoot is detected. This mathematically guarantees that every single step satisfies the sufficient decrease condition, forcing convergence even in a highly hostile geometric landscape.