# Comparative Experimental Results of Four Optimal Transport (OT) Methods

## Experimental Setup

### Problem Parameters
- **Problem Size**: 100 bins (1D optimal transport)
- **Maximum Iterations**: 5000
- **Convergence Threshold**: 1e-9
- **Sinkhorn Regularization Parameter**: 0.002
- **QP L2 Regularization Parameter**: 0.0001
- **ADMM Penalty Parameter ρ**: 5.0

### Test Data
- **Source Distribution**: Bimodal Gaussian mixture
  - 60% @ mean=15, std=4
  - 40% @ mean=35, std=6
  
- **Target Distribution**: Trimodal Gaussian mixture
  - 40% @ mean=50, std=5
  - 35% @ mean=70, std=8
  - 25% @ mean=85, std=4

- **Cost Function**: Normalized squared Euclidean distance

## Experimental Results Summary

### Key Metrics Comparison

| Method | Transport Cost | Entropy | Runtime (s) | Iterations |
|--------|---------------|---------|-------------|------------|
| **QP** | 0.170085 | 4.320597 | 1.4180 | N/A |
| **Frank-Wolfe** | 0.170066 | 4.318784 | 0.0318 | 12 |
| **ADMM** | 0.169593 | 4.328645 | 9.3908 | 5000 |
| **Sinkhorn** | 0.170999 | 6.308112 | 0.9900 | 837 |

## Detailed Analysis

### 1. Transport Cost

**Lowest**: ADMM (0.169593)
**Ranking**: ADMM < QP < Frank-Wolfe < Sinkhorn

- **ADMM** achieves the lowest transport cost because it directly optimizes the unregularized OT problem
- **Sinkhorn** yields a slightly higher transport cost as a trade-off due to entropy regularization
- **QP** and **Frank-Wolfe** results are very close

### 2. Runtime

**Fastest**: Frank-Wolfe (0.0318s)
**Ranking**: Frank-Wolfe << Sinkhorn < QP << ADMM

- **Frank-Wolfe** is significantly fastest, converging in only 12 iterations
- **ADMM** is slowest, requiring 5000 iterations without reaching the convergence threshold
- **Sinkhorn** runtime is approximately 1 second, **QP** approximately 1.4 seconds

### 3. Entropy

**Highest**: Sinkhorn (6.308112)
**Ranking**: Sinkhorn >> ADMM ≈ QP ≈ Frank-Wolfe

- **Sinkhorn** achieves the maximum entropy value due to entropy regularization, resulting in a more dispersed transport plan
- The other three methods have similar entropy values, indicating they produce similar sparse transport plans

### 4. Convergence Characteristics

#### Sinkhorn
- ✅ **Advantage**: Smooth and fast convergence (837 iterations)
- ✅ Excellent marginal constraint satisfaction (< 1e-15)
- 📊 Suitable for entropy-regularized problems

#### ADMM
- ⚠️ **Disadvantage**: Slower convergence (reaches 5000 maximum iterations)
- ⚠️ Large marginal constraint error (~1e-4)
- 📊 Optimal transport cost but high computational cost

#### Frank-Wolfe
- ✅ **Advantage**: Extremely fast convergence (only 12 iterations)
- ✅ Very high computational efficiency
- ⚠️ For 1D linear OT problems, may not reach the global optimal solution
- 📊 Suitable for fast approximate solutions

#### QP
- ✅ Direct solving, no iteration required
- ✅ Reliable results
- ⚠️ Requires commercial solvers (OSQP)
- 📊 Suitable for medium-scale problems

## Transport Plan Characteristics

### Sparsity Comparison
- **Sinkhorn**: Dense transport plan (entropy regularization effect)
- **QP, ADMM, Frank-Wolfe**: Sparse transport plan (distributed along diagonal)

### Visualization Observations
From the transport plan visualizations:
1. **Sinkhorn**'s transport plan shows a wider "hot band", indicating dispersed mass transport
2. **QP/ADMM/Frank-Wolfe** show narrower diagonal structures with concentrated mass transport
3. All methods correctly capture the main transport directions from source to target

## Method Selection Recommendations

### Choose **Sinkhorn** when:
- ✅ Entropy regularization is needed
- ✅ Differentiable solutions are required (for deep learning)
- ✅ Slightly higher transport cost is acceptable in exchange for stability

### Choose **ADMM** when:
- ✅ Exact unregularized OT solution is needed
- ✅ Computation time is not a primary concern
- ✅ Slower convergence speed is acceptable

### Choose **Frank-Wolfe** when:
- ✅ Fast approximate solution is needed
- ✅ Real-time applications or large-scale problems
- ✅ Potentially suboptimal solutions are acceptable

### Choose **QP** when:
- ✅ Reliable baseline solution is needed
- ✅ Problem size is medium
- ✅ Commercial solvers are available

## Convergence Curve Analysis

### Transport Cost Convergence
- **Sinkhorn**: Exponentially fast convergence, completes main optimization in the first 100 iterations
- **ADMM**: Rapid initial descent, slow convergence in later stages
- **Frank-Wolfe**: Extremely fast to reach stable value

### Marginal Constraint Violation
- **Sinkhorn**: Rapidly decreases from 1e-2 to below 1e-15
- **ADMM**: Remains at 1e-4 magnitude, not fully converged
- **Frank-Wolfe**: Immediately satisfies constraints (special property for 1D problems)

## Conclusions

1. **Optimal Transport Cost**: ADMM has a slight edge, but all methods are very close
2. **Computational Efficiency**: Frank-Wolfe is significantly fastest
3. **Practical Trade-offs**: 
   - Research/Benchmarking: Choose QP or ADMM
   - Practical Applications: Choose Sinkhorn or Frank-Wolfe
4. **Regularization Effect**: Sinkhorn's entropy regularization produces distinctly different transport plan characteristics

## Generated Plot Files

**Note**: All plots are displayed in the order **QP → Frank-Wolfe → ADMM → Sinkhorn**

1. `comparison_transport_plans.png` - Comparison of transport plans from four methods (2×2 grid)
2. `comparison_transport_cost.png` - Transport cost bar chart
3. `comparison_runtime.png` - Runtime comparison
4. `comparison_entropy.png` - Entropy comparison
5. `comparison_histograms.png` - Input distribution visualization
6. `comparison_summary_table.png` - Results summary table

---

**Experiment Date**: 2025
**Code File**: `comparison_experiment.py`
