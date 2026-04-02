# `scripts/`

Runnable experiment and utility scripts. All scripts import from `src/` and save outputs to `figures/` or `models/`.

## Data Preparation
| Script | Description |
|---|---|
| `generate_synthetic.py` | Generates `data/linear_gaussian.npz` and `data/moons.npz` |
| `make_digits_split.py` | Generates `data/digits_data.npz` and `data/digits_split_indices.npz` |
| `explore_data.py` | Prints dataset shapes and saves a visual overview to `figures/data_overview.png` |

## Training
| Script | Description |
|---|---|
| `train_softmax_digits.py` | Trains softmax regression on digits, saves model to `models/softmax_model.pkl` |
| `train_nn_digits.py` | Trains neural network on digits, saves model to `models/nn_model.pkl` |
| `train_synthetic.py` | Trains both models on Gaussian and Moons datasets, prints results |

## Verification
| Script | Description |
|---|---|
| `verify_softmax.py` | Sanity checks: probability sums, gradient check, loss decrease, overfit test |
| `verify_nn.py` | Numerical gradient check for all NN parameters (W1, b1, W2, b2) |

## Experiments & Analysis
| Script | Description |
|---|---|
| `compare_digits.py` | Trains both models on digits and plots loss/accuracy curves |
| `plot_decision_boundary.py` | Plots decision boundaries for both models on Gaussian and Moons |
| `capacity_ablation.py` | Sweeps NN hidden width `{2, 8, 32}` on Moons, plots decision boundaries |
| `optimizer_study.py` | Compares SGD, Momentum, and Adam on digits benchmark |
| `repeated_seed.py` | Runs both models over 5 seeds, reports mean ± 95% CI |
| `confidence_reliability.py` | Computes reliability bins, confidence, and entropy stats |
| `plot_reliability.py` | Plots reliability diagrams and confidence/entropy histograms (requires saved models) |