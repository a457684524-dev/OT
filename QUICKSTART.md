# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Comparison

```bash
python comparison_experiment.py
```

### Step 3: View Results

Check the `results/` folder for all generated comparison plots!

## 📊 What You'll Get

The script will automatically:
1. ✅ Run all 4 OT algorithms (QP, Frank-Wolfe, ADMM, Sinkhorn)
2. ✅ Generate 8 comparison plots
3. ✅ Print detailed performance metrics

## ⏱️ Expected Runtime

- Total runtime: ~12 seconds
- Breakdown:
  - QP: ~1.4s
  - Frank-Wolfe: ~0.03s ⚡
  - ADMM: ~9.4s 🐌
  - Sinkhorn: ~1.0s

## 📈 Example Output

```
================================================================================
FINAL SUMMARY
================================================================================

Method               Transport Cost     Entropy         Runtime (s)     Iterations
--------------------------------------------------------------------------------
QP                   0.170085           4.320597        1.4180          N/A
Frank-Wolfe          0.170066           4.318784        0.0318          12
ADMM                 0.169593           4.328645        9.3908          5000
Sinkhorn             0.170999           6.308112        0.9900          837
================================================================================
```

## 🔧 Customization

You can modify the experimental settings in `comparison_experiment.py`:

```python
# Problem size
n_bins = 100

# Regularization parameters
reg = 2e-3         # Sinkhorn entropy regularization
reg_qp = 1e-4      # QP L2 regularization
rho = 5.0          # ADMM penalty parameter

# Convergence settings
max_iter = 5000
stop_threshold = 1e-9
```

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed algorithm explanations
- Check [COMPARISON_RESULTS.md](COMPARISON_RESULTS.md) for in-depth analysis
- Explore the generated plots in the `results/` folder

## ❓ Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'cvxpy'`

**Solution**: Install CVXPY with solvers:
```bash
pip install cvxpy
```

**Issue**: Slow performance on first run

**Solution**: CVXPY may need to install solvers on first run. Subsequent runs will be faster.

## 💡 Tips

- Use `matplotlib` backend: The script uses 'Agg' backend for non-interactive plotting
- Parallel processing: The methods run sequentially for accurate timing measurements
- Memory usage: For larger problems (n > 500), consider reducing `max_iter`

---

Happy experimenting! 🎉
