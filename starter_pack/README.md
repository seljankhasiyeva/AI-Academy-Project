# Math4AI Starter Pack

From-scratch implementation of Softmax Regression and a two-layer Neural Network in NumPy,
as the final capstone for the Math4AI course.

## What Was Provided

- `data/digits_data.npz` — fixed digits feature matrix and label vector
- `data/digits_split_indices.npz` — fixed train/validation/test indices
- `data/linear_gaussian.npz` — linear synthetic dataset
- `data/moons.npz` — nonlinear synthetic dataset
- `scripts/make_digits_split.py` — deterministic split-generation script
- `scripts/generate_synthetic.py` — regenerates both synthetic datasets
- `report/template.tex` — optional LaTeX report template

## What We Implemented

**Models (`src/`):**
- `softmax_regression.py` — multinomial logistic regression with L2 regularization and mini-batch SGD
- `neural_network.py` — two-layer network (Linear → Tanh → Linear → Softmax) with vectorized backpropagation
- `data_loading.py` — shared data utilities

**Experiments (`scripts/`):**
- Gradient verification (numerical vs analytic) for both models
- Training on Linear Gaussian, Moons, and Digits
- Softmax vs NN comparison on the digits benchmark
- Capacity ablation: hidden width ∈ {2, 8, 32} on Moons
- Optimizer study: SGD vs Momentum vs Adam
- Repeated-seed evaluation (5 seeds, 95% CI)
- Confidence and calibration analysis (reliability diagrams, entropy boxplots)

## Repository Layout

- `data/` — provided datasets and fixed split indices
- `scripts/` — all training, evaluation, and experiment scripts
- `src/` — model and data loading source code
- `figures/` — output plots and decision boundaries
- `results/` — saved experiment outputs and summary tables
- `models/` — saved model checkpoints (.pkl)
- `report/` — final report and LaTeX source
- `slides/` — presentation materials

## Team

- Seljan – Core experiments, ablations, optimizer study, presentation.
- Jeyhuna – One-hidden-layer network implementation, Track B analysis.
- Suleyman – Softmax regression implementation, final report writing.