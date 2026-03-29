import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'starter_pack', 'src'))

from neural_network import NeuralNetwork
from softmax_regression import SoftmaxRegression
from data_loading import load_digits


def run_seeds(model_class, model_kwargs, fit_kwargs, X_train, y_train,
              X_val, y_val, X_test, y_test, seeds):
    accs, losses = [], []
    for seed in seeds:
        model_kwargs_s = {**model_kwargs, "seed": seed}
        fit_kwargs_s   = {**fit_kwargs,   "seed": seed} if "seed" in fit_kwargs else fit_kwargs

        model = model_class(**model_kwargs_s)

        if model_class == SoftmaxRegression:
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val,
                      verbose=False, **fit_kwargs_s)
        else:
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val, **fit_kwargs_s)

        accs.append(model.accuracy(X_test, y_test))
        losses.append(model.cross_entropy(X_test, y_test))
        print(f"  seed={seed} | acc={accs[-1]:.4f} | loss={losses[-1]:.4f}")

    return np.array(accs), np.array(losses)


def summarize(name, accs, losses):
    # 95% CI: x_bar ± 2.776 * s / sqrt(5)   (t-distribution, 4 df)
    t_crit = 2.776
    n = len(accs)

    acc_mean  = np.mean(accs)
    acc_std   = np.std(accs, ddof=1)
    acc_ci    = t_crit * acc_std / np.sqrt(n)

    loss_mean = np.mean(losses)
    loss_std  = np.std(losses, ddof=1)
    loss_ci   = t_crit * loss_std / np.sqrt(n)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc_mean:.4f} ± {acc_ci:.4f}  (std={acc_std:.4f})")
    print(f"  CE Loss  : {loss_mean:.4f} ± {loss_ci:.4f}  (std={loss_std:.4f})")
    print(f"  Raw accs : {[round(a,4) for a in accs]}")
    print(f"  Raw losses: {[round(l,4) for l in losses]}")

    return {
        "acc_mean": acc_mean, "acc_ci": acc_ci,
        "loss_mean": loss_mean, "loss_ci": loss_ci,
    }


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()
    num_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]

    seeds = [42, 0, 1, 7, 99]

    # --- Softmax Regression ---
    print("\nRunning Softmax Regression (5 seeds)...")
    sm_kwargs  = {"input_dim": input_dim, "num_classes": num_classes}
    sm_fit     = {"lr": 0.05, "reg": 1e-4, "epochs": 200, "batch_size": 64, "seed": 42}
    sm_accs, sm_losses = run_seeds(
        SoftmaxRegression, sm_kwargs, sm_fit,
        X_train, y_train, X_val, y_val, X_test, y_test, seeds
    )
    sm_summary = summarize("Softmax Regression", sm_accs, sm_losses)

    # --- Neural Network ---
    print("\nRunning Neural Network (5 seeds)...")
    nn_kwargs = {"input_dim": input_dim, "hidden_dim": 32, "num_classes": num_classes}
    nn_fit    = {"lr": 0.05, "reg": 1e-4, "epochs": 200, "batch_size": 64}
    nn_accs, nn_losses = run_seeds(
        NeuralNetwork, nn_kwargs, nn_fit,
        X_train, y_train, X_val, y_val, X_test, y_test, seeds
    )
    nn_summary = summarize("Neural Network (hidden=32)", nn_accs, nn_losses)

    # --- Final comparison table ---
    print("\n" + "="*65)
    print("  FINAL REPEATED-SEED RESULTS (5 seeds, 95% CI)")
    print("="*65)
    print(f"{'Model':>25} | {'Test Acc':>18} | {'Test Loss':>18}")
    print("-"*65)
    print(f"{'Softmax Regression':>25} | "
          f"{sm_summary['acc_mean']:.4f} ± {sm_summary['acc_ci']:.4f} | "
          f"{sm_summary['loss_mean']:.4f} ± {sm_summary['loss_ci']:.4f}")
    print(f"{'Neural Network':>25} | "
          f"{nn_summary['acc_mean']:.4f} ± {nn_summary['acc_ci']:.4f} | "
          f"{nn_summary['loss_mean']:.4f} ± {nn_summary['loss_ci']:.4f}")
    print("="*65)
    print("\nNote: CI computed as x_bar ± 2.776 * s / sqrt(5)  [t-dist, 4 df]")
