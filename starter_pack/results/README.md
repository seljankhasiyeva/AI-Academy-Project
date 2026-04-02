#  Math4AI: Capstone Project Experiment Results

This folder contains the quantitative outputs of our machine learning experiments. The results demonstrate the performance transition from a baseline Softmax model to an optimized Neural Network.

---

##  Core Experiment Findings

### 1. Model Comparison & Baseline
- **Softmax Regression:** Achieved a solid baseline of **94.02%** test accuracy.
- **Neural Network (32 hidden units):** Outperformed the baseline with **94.08%** (stable) and reached up to **95.11%** with advanced optimizers.
- **Observation:** While Softmax is strong for digits, the NN provides lower Cross-Entropy loss (**0.1890** vs **0.2694**), indicating higher confidence in predictions.

### 2. Optimizer Study (The Impact of Momentum & Adam)
| Optimizer | Test Accuracy | Test Loss |
| :--- | :--- | :--- |
| **SGD** | 94.02% | 0.1890 |
| **Momentum** | 95.11% | 0.1654 |
| **Adam** | **95.11%** | **0.1571** |
*The Adam optimizer proved to be the most efficient, achieving the lowest loss and highest accuracy.*

### 3. Capacity Ablation (Network Depth)
We tested the NN's ability to learn with different hidden layer sizes:
- **Small (2-8 units):** Significant underfitting (**~85%** accuracy).
- **Optimal (32 units):** Sharp performance jump to **97.50%** on the validation set.
*Conclusion: A minimum of 32 hidden units is required to capture the complexity of the digits dataset.*

### 4. Robustness & Stability (Seed Statistics)
We ran 5 independent trials with different random seeds to ensure reliability:
- **Softmax Std Dev:** 0.0015
- **Neural Network Std Dev:** 0.0012
*The extremely low standard deviation proves that our results are statistically significant and reproducible.*

---

## Data Index

| File | Content |
| :--- | :--- |
| `softmax_results.txt` | Baseline training logs for Softmax Regression. |
| `nn_training_results.txt` | Detailed epoch-by-epoch logs for the Neural Network. |
| `optimizer_study.txt` | Comparative metrics for SGD, Momentum, and Adam. |
| `capacity_ablation.txt` | Accuracy metrics for different hidden layer sizes. |
| `seed_statistics.txt` | 95% Confidence Interval and Variance analysis. |
| `model_comparison.txt` | Final side-by-side comparison of all architectures. |
| `synthetic_results.txt` | Performance on Moons and Linear Gaussian datasets. |

---
**Note:** All corresponding visual plots  are located in the `/figures` directory.