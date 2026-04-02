import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'starter_pack', 'src'))

from neural_network import NeuralNetwork
from data_loading import load_digits


class NeuralNetworkWithOptimizer(NeuralNetwork):
    """Extends NeuralNetwork with Momentum and Adam optimizers."""

    def fit_with_optimizer(self, X_train, y_train, X_val=None, y_val=None,
                           optimizer='sgd', lr=0.05, reg=1e-4,
                           epochs=200, batch_size=64,
                           momentum=0.9, beta1=0.9, beta2=0.999, eps=1e-8):

        rng = np.random.default_rng(42)
        n = X_train.shape[0]
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        best_val_loss = np.inf
        best_W1, best_b1 = self.W1.copy(), self.b1.copy()
        best_W2, best_b2 = self.W2.copy(), self.b2.copy()

        # Momentum buffers
        vW1 = np.zeros_like(self.W1)
        vb1 = np.zeros_like(self.b1)
        vW2 = np.zeros_like(self.W2)
        vb2 = np.zeros_like(self.b2)

        # Adam buffers
        mW1 = np.zeros_like(self.W1); mW2 = np.zeros_like(self.W2)
        mb1 = np.zeros_like(self.b1); mb2 = np.zeros_like(self.b2)
        vvW1 = np.zeros_like(self.W1); vvW2 = np.zeros_like(self.W2)
        vvb1 = np.zeros_like(self.b1); vvb2 = np.zeros_like(self.b2)
        t = 0

        for epoch in range(epochs):
            indices = rng.permutation(n)
            X_s, y_s = X_train[indices], y_train[indices]

            for i in range(0, n, batch_size):
                X_batch = X_s[i:i+batch_size]
                y_batch = y_s[i:i+batch_size]
                _, grads = self.loss_and_grads(X_batch, y_batch, reg=reg)

                if optimizer == 'sgd':
                    self.W1 -= lr * grads["dW1"]
                    self.b1 -= lr * grads["db1"]
                    self.W2 -= lr * grads["dW2"]
                    self.b2 -= lr * grads["db2"]

                elif optimizer == 'momentum':
                    vW1 = momentum * vW1 + lr * grads["dW1"]
                    vb1 = momentum * vb1 + lr * grads["db1"]
                    vW2 = momentum * vW2 + lr * grads["dW2"]
                    vb2 = momentum * vb2 + lr * grads["db2"]
                    self.W1 -= vW1
                    self.b1 -= vb1
                    self.W2 -= vW2
                    self.b2 -= vb2

                elif optimizer == 'adam':
                    t += 1
                    for param, grad, m, v in [
                        (self.W1, grads["dW1"], mW1, vvW1),
                        (self.b1, grads["db1"], mb1, vvb1),
                        (self.W2, grads["dW2"], mW2, vvW2),
                        (self.b2, grads["db2"], mb2, vvb2),
                    ]:
                        m[:] = beta1 * m + (1 - beta1) * grad
                        v[:] = beta2 * v + (1 - beta2) * grad**2
                        m_hat = m / (1 - beta1**t)
                        v_hat = v / (1 - beta2**t)
                        param -= lr * m_hat / (np.sqrt(v_hat) + eps)

            train_loss, _ = self.loss_and_grads(X_train, y_train, reg)
            train_acc = self.accuracy(X_train, y_train)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if X_val is not None:
                val_loss, _ = self.loss_and_grads(X_val, y_val, reg)
                val_acc = self.accuracy(X_val, y_val)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_W1, best_b1 = self.W1.copy(), self.b1.copy()
                    best_W2, best_b2 = self.W2.copy(), self.b2.copy()

        if X_val is not None:
            self.W1, self.b1 = best_W1, best_b1
            self.W2, self.b2 = best_W2, best_b2

        return history


if __name__ == "__main__":
    os.makedirs("starter_pack/figures", exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()
    num_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]

    # PDF-də dəqiq parametrlər:
    # SGD:      lr=0.05
    # Momentum: lr=0.05, momentum=0.9
    # Adam:     lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8

    configs = [
        {"name": "SGD",      "optimizer": "sgd",      "lr": 0.05},
        {"name": "Momentum", "optimizer": "momentum",  "lr": 0.05,  "momentum": 0.9},
        {"name": "Adam",     "optimizer": "adam",      "lr": 0.001, "beta1": 0.9,
                                                        "beta2": 0.999, "eps": 1e-8},
    ]

    histories = {}
    results = {}
    colors = {"SGD": "steelblue", "Momentum": "tomato", "Adam": "seagreen"}

    for cfg in configs:
        print(f"Training with {cfg['name']}...")
        model = NeuralNetworkWithOptimizer(
            input_dim=input_dim, hidden_dim=32,
            num_classes=num_classes, seed=42
        )
        kwargs = {k: v for k, v in cfg.items() if k not in ("name",)}
        history = model.fit_with_optimizer(
            X_train, y_train, X_val=X_val, y_val=y_val,
            reg=1e-4, epochs=200, batch_size=64, **kwargs
        )
        histories[cfg["name"]] = history
        results[cfg["name"]] = {
            "test_acc":  model.accuracy(X_test, y_test),
            "test_loss": model.cross_entropy(X_test, y_test),
        }
        print(f"  {cfg['name']:10s} | test_acc={results[cfg['name']]['test_acc']:.4f} | "
              f"test_loss={results[cfg['name']]['test_loss']:.4f}")

    # Plots
    epochs = range(1, 201)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Optimizer Study — Digits Benchmark (NN, hidden_dim=32)",
                 fontsize=13, fontweight='bold')

    for name, hist in histories.items():
        c = colors[name]
        axes[0].plot(epochs, hist["train_loss"], label=f"{name} Train", color=c, linewidth=1.5)
        axes[0].plot(epochs, hist["val_loss"],   label=f"{name} Val",   color=c, linestyle='--', linewidth=1.5)
        axes[1].plot(epochs, hist["train_acc"],  label=f"{name} Train", color=c, linewidth=1.5)
        axes[1].plot(epochs, hist["val_acc"],    label=f"{name} Val",   color=c, linestyle='--', linewidth=1.5)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Loss over Epochs"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy over Epochs"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("starter_pack/figures/optimizer_study.png", dpi=150, bbox_inches='tight')
    print("\nSaved: starter_pack/figures/optimizer_study.png")
    plt.show()

    # Summary
    print("\n=== Optimizer Study Summary ===")
    print(f"{'Optimizer':>10} | {'Test Acc':>9} | {'Test Loss':>10}")
    print("-" * 35)
    for name, r in results.items():
        print(f"{name:>10} | {r['test_acc']:>9.4f} | {r['test_loss']:>10.4f}")
