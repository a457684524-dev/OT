"""
Comprehensive Comparison of 4 OT Methods:
1. Quadratic Programming (QP)
2. Frank-Wolfe (FW)
3. ADMM
4. Sinkhorn

This script runs all methods under the same settings and generates comparison plots.
"""

from __future__ import annotations
import time
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from dataclasses import dataclass
import math


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class OTResult:
    """Stores results for each method."""
    method: str
    transport_cost: float
    entropy: float
    loss: float
    runtime: float
    gamma: np.ndarray
    n_iter: int = 0
    convergence_history: dict = None  # Stores iteration-wise metrics


# ============================================================================
# DISTRIBUTION GENERATION (Same as in both files)
# ============================================================================

def gaussian_profile(n: int, mean: float, std: float) -> np.ndarray:
    x = np.arange(n, dtype=np.float64)
    return np.exp(-0.5 * ((x - mean) / std) ** 2)


def build_distributions(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(n, dtype=np.float64)
    a = 0.6 * gaussian_profile(n, mean=15.0, std=4.0)
    a += 0.4 * gaussian_profile(n, mean=35.0, std=6.0)
    a /= a.sum()

    b = 0.4 * gaussian_profile(n, mean=50.0, std=5.0)
    b += 0.35 * gaussian_profile(n, mean=70.0, std=8.0)
    b += 0.25 * gaussian_profile(n, mean=85.0, std=4.0)
    b /= b.sum()

    return x, a, b


def build_cost(x: np.ndarray) -> np.ndarray:
    x_col = x[:, None]
    dist = (x_col - x_col.T) ** 2
    return dist / dist.max()


# ============================================================================
# METHOD 1: QUADRATIC PROGRAMMING (QP)
# ============================================================================

def solve_qp(a: np.ndarray, b: np.ndarray, cost: np.ndarray, reg_qp: float = 1e-4) -> OTResult:
    """Quadratic programming OT solver with L2 regularization."""
    n = len(a)
    gamma = cp.Variable((n, n), nonneg=True)

    objective = cp.Minimize(
        cp.sum(cp.multiply(cost, gamma))
        + reg_qp * cp.sum_squares(gamma)
    )

    constraints = [
        cp.sum(gamma, axis=1) == a,
        cp.sum(gamma, axis=0) == b
    ]

    prob = cp.Problem(objective, constraints)

    start = time.time()
    prob.solve(solver=cp.OSQP, verbose=False)
    runtime = time.time() - start

    gamma_val = gamma.value
    transport_cost = float(np.sum(gamma_val * cost))
    entropy = -float(np.sum(gamma_val * np.log(gamma_val + 1e-12)))
    loss = transport_cost + reg_qp * float(np.sum(gamma_val**2))

    return OTResult("QP", transport_cost, entropy, loss, runtime, gamma_val)


# ============================================================================
# METHOD 2: SINKHORN
# ============================================================================

def solve_sinkhorn(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    reg: float,
    num_iter_max: int = 5000,
    stop_thresh: float = 1e-9,
) -> OTResult:
    """Sinkhorn algorithm for entropy-regularized OT."""
    
    if not math.isclose(a.sum(), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Source histogram must sum to 1.")
    if not math.isclose(b.sum(), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Target histogram must sum to 1.")

    K = np.exp(-cost / reg)
    u = np.ones_like(a)
    v = np.ones_like(b)

    loss_hist = []
    marginal_hist = []
    transport_cost_hist = []
    entropy_hist = []

    start = time.time()
    
    for iteration in range(num_iter_max):
        u_prev = u.copy()
        v_prev = v.copy()

        u = a / (K @ v)
        v = b / (K.T @ u)

        gamma = np.diag(u) @ K @ np.diag(v)

        transport_cost = float(np.sum(gamma * cost))
        entropy = -float(np.sum(gamma * np.log(gamma + 1e-300)))
        loss = transport_cost - reg * entropy

        transport_cost_hist.append(transport_cost)
        entropy_hist.append(entropy)
        loss_hist.append(loss)

        err = max(np.max(np.abs(u - u_prev)), np.max(np.abs(v - v_prev)))
        marginal_violation = max(
            np.max(np.abs(gamma.sum(axis=1) - a)),
            np.max(np.abs(gamma.sum(axis=0) - b)),
        )
        marginal_hist.append(marginal_violation)

        if err < stop_thresh and marginal_violation < stop_thresh:
            break
    
    runtime = time.time() - start
    gamma = np.diag(u) @ K @ np.diag(v)
    
    transport_cost = float(np.sum(gamma * cost))
    entropy = -float(np.sum(gamma * np.log(gamma + 1e-300)))
    loss = transport_cost - reg * entropy

    convergence = {
        'loss': np.array(loss_hist),
        'transport_cost': np.array(transport_cost_hist),
        'entropy': np.array(entropy_hist),
        'marginal_err': np.array(marginal_hist)
    }

    return OTResult("Sinkhorn", transport_cost, entropy, loss, runtime, gamma, 
                   len(loss_hist), convergence)


# ============================================================================
# METHOD 3: ADMM
# ============================================================================

def project_simplex(y: np.ndarray, z: float = 1.0) -> np.ndarray:
    """Projects vector y onto the simplex of sum z."""
    n_features = y.shape[0]
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u)
    ind = np.arange(1, n_features + 1)
    cond = u - (cssv - z) / ind > 0
    
    if np.sum(cond) > 0:
        rho = ind[cond][-1]
        theta = (cssv[rho - 1] - z) / rho
        w = np.maximum(y - theta, 0)
    else:
        w = np.zeros_like(y)
        w[np.argmax(y)] = z
    return w


def solve_admm(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    rho: float,
    num_iter_max: int = 5000,
    stop_thresh: float = 1e-9,
) -> OTResult:
    """ADMM solver for unregularized OT."""
    
    if not math.isclose(a.sum(), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Source histogram must sum to 1.")
    if not math.isclose(b.sum(), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Target histogram must sum to 1.")

    n_a, n_b = cost.shape

    gamma_1 = np.outer(a, b)
    gamma_2 = gamma_1.copy()
    lambda_ = np.zeros_like(cost)

    primal_hist = []
    loss_hist = []
    marginal_hist = []

    C_rho = cost / (2 * rho)

    start = time.time()
    
    for iteration in range(num_iter_max):
        # Update gamma_1
        V1 = gamma_2 - lambda_ / rho
        P1_arg = V1 - C_rho
        gamma_1 = np.zeros_like(cost)
        for i in range(n_a):
            gamma_1[i, :] = project_simplex(P1_arg[i, :], a[i])

        # Update gamma_2
        V2 = gamma_1 + lambda_ / rho
        P2_arg = V2 - C_rho
        gamma_2 = np.zeros_like(cost)
        for j in range(n_b):
            gamma_2[:, j] = project_simplex(P2_arg[:, j], b[j])

        # Dual update
        lambda_ = lambda_ + rho * (gamma_1 - gamma_2)

        # Diagnostics
        primal_res = np.linalg.norm(gamma_1 - gamma_2, "fro")
        loss = float(np.sum(gamma_1 * cost))
        
        marg1_err = np.max(np.abs(gamma_1.sum(axis=0) - b))
        marg2_err = np.max(np.abs(gamma_2.sum(axis=1) - a))
        marginal_err = max(marg1_err, marg2_err)

        primal_hist.append(primal_res)
        loss_hist.append(loss)
        marginal_hist.append(marginal_err)

        if (primal_res < stop_thresh and marginal_err < stop_thresh):
            break

    runtime = time.time() - start
    
    transport_cost = float(np.sum(gamma_1 * cost))
    entropy = -float(np.sum(gamma_1 * np.log(gamma_1 + 1e-12)))
    loss = transport_cost

    convergence = {
        'loss': np.array(loss_hist),
        'primal_res': np.array(primal_hist),
        'marginal_err': np.array(marginal_hist)
    }

    return OTResult("ADMM", transport_cost, entropy, loss, runtime, gamma_1,
                   len(loss_hist), convergence)


# ============================================================================
# METHOD 4: FRANK-WOLFE
# ============================================================================

def solve_frank_wolfe(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    num_iter_max: int = 5000,
    stop_thresh: float = 1e-9,
) -> OTResult:
    """Frank-Wolfe algorithm for unregularized OT."""
    
    if not math.isclose(a.sum(), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Source histogram must sum to 1.")
    if not math.isclose(b.sum(), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Target histogram must sum to 1.")

    gamma = np.outer(a, b)

    loss_hist = []
    marginal_hist = []
    gap_hist = []

    start = time.time()
    
    for iteration in range(num_iter_max):
        loss = float(np.sum(gamma * cost))
        loss_hist.append(loss)

        marginal_violation = max(
            np.max(np.abs(gamma.sum(axis=1) - a)),
            np.max(np.abs(gamma.sum(axis=0) - b)),
        )
        marginal_hist.append(marginal_violation)

        # LMO: Optimal 1D coupling
        s = np.zeros_like(cost)
        a_cumsum = np.cumsum(a)
        b_cumsum = np.cumsum(b)
        
        for i in range(len(a)):
            if i == 0:
                q_start = 0
            else:
                q_start = a_cumsum[i-1]
            q_end = a_cumsum[i]
            
            for j in range(len(b)):
                if j == 0:
                    q_b_start = 0
                else:
                    q_b_start = b_cumsum[j-1]
                q_b_end = b_cumsum[j]
                
                overlap_start = max(q_start, q_b_start)
                overlap_end = min(q_end, q_b_end)
                
                if overlap_end > overlap_start:
                    s[i, j] = overlap_end - overlap_start

        # Frank-Wolfe gap
        fw_gap = np.sum(cost * (gamma - s))
        gap_hist.append(fw_gap)
        
        # Adaptive step size
        cost_s = float(np.sum(s * cost))
        if cost_s < loss:
            alpha = 2.0 / (iteration + 2.0)
        else:
            alpha = 0.0

        # Update
        if alpha > 0:
            gamma = (1 - alpha) * gamma + alpha * s

        # Stopping criterion
        if iteration > 10:
            if abs(fw_gap) < stop_thresh:
                break
            if iteration > 50 and len(loss_hist) > 10:
                recent_improvement = loss_hist[-10] - loss_hist[-1]
                if recent_improvement < stop_thresh:
                    break

    runtime = time.time() - start
    
    transport_cost = float(np.sum(gamma * cost))
    entropy = -float(np.sum(gamma * np.log(gamma + 1e-12)))
    loss = transport_cost

    convergence = {
        'loss': np.array(loss_hist),
        'marginal_err': np.array(marginal_hist),
        'duality_gap': np.array(gap_hist)
    }

    return OTResult("Frank-Wolfe", transport_cost, entropy, loss, runtime, gamma,
                   len(loss_hist), convergence)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comparison_results(x, a, b, results: list[OTResult]):
    """Create comprehensive comparison plots for all methods."""
    
    methods = [r.method for r in results]
    colors = {'QP': 'tab:blue', 'Sinkhorn': 'tab:orange', 
              'ADMM': 'tab:green', 'Frank-Wolfe': 'tab:red'}
    
    # 1. Transport Plans Comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, result in enumerate(results):
        ax = axes[idx // 2, idx % 2]
        im = ax.imshow(result.gamma, cmap='hot', aspect='auto')
        ax.set_title(f'{result.method}: Transport Plan')
        ax.set_xlabel('Target bin')
        ax.set_ylabel('Source bin')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('comparison_transport_plans.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Transport Cost Comparison (Bar Chart)
    fig, ax = plt.subplots(figsize=(8, 5))
    costs = [r.transport_cost for r in results]
    bars = ax.bar(methods, costs, color=[colors[m] for m in methods])
    ax.set_ylabel('Transport Cost')
    ax.set_title('Transport Cost Comparison')
    ax.grid(axis='y', alpha=0.3)
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{cost:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('comparison_transport_cost.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Runtime Comparison (Bar Chart)
    fig, ax = plt.subplots(figsize=(8, 5))
    runtimes = [r.runtime for r in results]
    bars = ax.bar(methods, runtimes, color=[colors[m] for m in methods])
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime Comparison')
    ax.grid(axis='y', alpha=0.3)
    for bar, runtime in zip(bars, runtimes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{runtime:.3f}s', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('comparison_runtime.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Entropy Comparison (Bar Chart)
    fig, ax = plt.subplots(figsize=(8, 5))
    entropies = [r.entropy for r in results]
    bars = ax.bar(methods, entropies, color=[colors[m] for m in methods])
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy Comparison')
    ax.grid(axis='y', alpha=0.3)
    for bar, entropy in zip(bars, entropies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{entropy:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('comparison_entropy.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Histograms with all transport costs
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, a, label='Source', linewidth=2, color='black', linestyle='--')
    ax.plot(x, b, label='Target', linewidth=2, color='gray', linestyle='--')
    ax.set_xlabel('Bin index')
    ax.set_ylabel('Probability')
    ax.set_title('Input Distributions')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('comparison_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. Summary Table (as image)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [['Method', 'Transport Cost', 'Entropy', 'Runtime (s)', 'Iterations']]
    for result in results:
        table_data.append([
            result.method,
            f'{result.transport_cost:.6f}',
            f'{result.entropy:.6f}',
            f'{result.runtime:.4f}',
            f'{result.n_iter}' if result.n_iter > 0 else 'N/A'
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.savefig('comparison_summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ All comparison plots generated successfully!")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("=" * 80)
    print("COMPREHENSIVE COMPARISON: QP, Frank-Wolfe, ADMM, Sinkhorn")
    print("=" * 80)
    
    # Common settings
    n_bins = 100
    max_iter = 5000
    stop_threshold = 1e-9
    reg = 2e-3  # Regularization parameter for Sinkhorn and QP
    reg_qp = 1e-4  # L2 regularization for QP
    rho = 5.0  # ADMM penalty parameter
    
    # Build problem
    x, a, b = build_distributions(n_bins)
    cost = build_cost(x)
    
    print(f"\nProblem size: {n_bins} bins")
    print(f"Regularization (Sinkhorn): {reg}")
    print(f"Regularization (QP L2): {reg_qp}")
    print(f"ADMM rho: {rho}")
    print(f"Max iterations: {max_iter}")
    print(f"Stop threshold: {stop_threshold}")
    
    results = []
    
    # Run QP
    print("\n" + "-" * 80)
    print("Running Quadratic Programming (QP)...")
    print("-" * 80)
    result_qp = solve_qp(a, b, cost, reg_qp)
    results.append(result_qp)
    print(f"✓ QP completed in {result_qp.runtime:.4f}s")
    print(f"  Transport cost: {result_qp.transport_cost:.6f}")
    print(f"  Entropy: {result_qp.entropy:.6f}")
    print(f"  Objective (with L2 reg): {result_qp.loss:.6f}")
    
    # Run Frank-Wolfe
    print("\n" + "-" * 80)
    print("Running Frank-Wolfe...")
    print("-" * 80)
    result_fw = solve_frank_wolfe(a, b, cost, max_iter, stop_threshold)
    results.append(result_fw)
    print(f"✓ Frank-Wolfe completed in {result_fw.runtime:.4f}s")
    print(f"  Iterations: {result_fw.n_iter}")
    print(f"  Transport cost: {result_fw.transport_cost:.6f}")
    print(f"  Entropy: {result_fw.entropy:.6f}")
    
    # Run ADMM
    print("\n" + "-" * 80)
    print("Running ADMM...")
    print("-" * 80)
    result_admm = solve_admm(a, b, cost, rho, max_iter, stop_threshold)
    results.append(result_admm)
    print(f"✓ ADMM completed in {result_admm.runtime:.4f}s")
    print(f"  Iterations: {result_admm.n_iter}")
    print(f"  Transport cost: {result_admm.transport_cost:.6f}")
    print(f"  Entropy: {result_admm.entropy:.6f}")
    
    # Run Sinkhorn
    print("\n" + "-" * 80)
    print("Running Sinkhorn...")
    print("-" * 80)
    result_sinkhorn = solve_sinkhorn(a, b, cost, reg, max_iter, stop_threshold)
    results.append(result_sinkhorn)
    print(f"✓ Sinkhorn completed in {result_sinkhorn.runtime:.4f}s")
    print(f"  Iterations: {result_sinkhorn.n_iter}")
    print(f"  Transport cost: {result_sinkhorn.transport_cost:.6f}")
    print(f"  Entropy: {result_sinkhorn.entropy:.6f}")
    print(f"  Regularized loss: {result_sinkhorn.loss:.6f}")
    
    # Generate comparison plots
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON PLOTS...")
    print("=" * 80)
    plot_comparison_results(x, a, b, results)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\n{'Method':<20} {'Transport Cost':<18} {'Entropy':<15} {'Runtime (s)':<15} {'Iterations':<12}")
    print("-" * 80)
    for result in results:
        iter_str = str(result.n_iter) if result.n_iter > 0 else 'N/A'
        print(f"{result.method:<20} {result.transport_cost:<18.6f} {result.entropy:<15.6f} "
              f"{result.runtime:<15.4f} {iter_str:<12}")
    print("=" * 80)
    
    print("\n✓ Experiment completed successfully!")
    print("\nGenerated files:")
    print("  - comparison_transport_plans.png")
    print("  - comparison_transport_cost.png")
    print("  - comparison_runtime.png")
    print("  - comparison_entropy.png")
    print("  - comparison_histograms.png")
    print("  - comparison_summary_table.png")


if __name__ == "__main__":
    main()
