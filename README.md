# Math4AI Capstone — Softmax Regression & Neural Network from Scratch

A from-scratch implementation of Softmax Regression and a two-layer Neural Network in NumPy,
submitted as the final capstone for the Math4AI course.

## Project Structure
```
starter_pack/
├── data/                  # Fixed datasets and split indices (.npz)
├── figures/               # Output plots (decision boundaries, training curves, etc.)
├── models/                # Saved model checkpoints (.pkl)
├── report/                # Written report
├── results/               # Numerical results and experiment outputs
├── scripts/               # Training, evaluation, and experiment scripts
│   ├── capacity_ablation.py
│   ├── compare_digits.py
│   ├── confidence_reliability.py
│   ├── explore_data.py
│   ├── optimizer_study.py
│   ├── plot_decision_boundary.py
│   ├── plot_reliability.py
│   ├── repeated_seed.py
│   ├── train_nn_digits.py
│   ├── train_softmax_digits.py
│   ├── train_synthetic.py
│   ├── verify_nn.py
│   └── verify_softmax.py
├── slides/                # Presentation slides
└── src/                   # Model and data loading source code
    ├── neural_network.py
    ├── softmax_regression.py
    └── data_loading.py
```

## What's Implemented

**Models (`src/`)** — both built from scratch using NumPy only:
- `SoftmaxRegression`: multinomial logistic regression with L2 regularization and mini-batch SGD
- `NeuralNetwork`: two-layer network (Linear → Tanh → Linear → Softmax) with vectorized backpropagation

**Experiments (`scripts/`):**
- Gradient verification (numerical vs analytic) for both models
- Training on three datasets: Linear Gaussian, Moons, and Digits (sklearn)
- Softmax vs NN comparison on digits benchmark
- Capacity ablation: NN hidden width ∈ {2, 8, 32} on Moons
- Optimizer study: SGD vs Momentum vs Adam
- Repeated-seed evaluation (5 seeds, 95% CI)
- Confidence and calibration analysis (reliability diagrams, entropy boxplots)

## Datasets

| Dataset | Type | Classes |
|---|---|---|
| Linear Gaussian | 2D synthetic | 2 |
| Moons | 2D synthetic, non-linear | 2 |
| Digits | 8×8 images (sklearn) | 10 |

All splits are fixed and deterministic (seed=7, 60/20/20 stratified).

## Dependencies
```
numpy
matplotlib
scikit-learn
```

## Reproducing Results
```bash
# 1. Generate datasets
python scripts/generate_synthetic.py
python scripts/make_digits_split.py

# 2. Verify implementations
python scripts/verify_softmax.py
python scripts/verify_nn.py

# 3. Train models
python scripts/train_softmax_digits.py
python scripts/train_nn_digits.py

# 4. Run experiments
python scripts/compare_digits.py
python scripts/plot_decision_boundary.py
python scripts/capacity_ablation.py
python scripts/optimizer_study.py
python scripts/repeated_seed.py
python scripts/plot_reliability.py
```