import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'starter_pack', 'src'))

from neural_network import NeuralNetwork
from data_loading import load_moons


def plot_decision_boundary(ax, model, X, y, title, h=0.02):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    ax.contour(xx, yy, Z, colors='k', linewidths=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k',
               linewidths=0.4, s=25, alpha=0.8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


if __name__ == "__main__":
    os.makedirs("starter_pack/figures", exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_moons()
    num_classes = len(np.unique(y_train))

    hidden_widths = [2, 8, 32]
    models = {}
    results = {}

    for width in hidden_widths:
        print(f"Training NN with hidden_dim={width}...")
        nn = NeuralNetwork(input_dim=2, hidden_dim=width,
                           num_classes=num_classes, seed=42, weight_scale=0.1)
        nn.fit(X_train, y_train, X_val=X_val, y_val=y_val,
               lr=0.1, reg=1e-4, epochs=500, batch_size=32)

        train_acc = nn.accuracy(X_train, y_train)
        val_acc   = nn.accuracy(X_val, y_val)
        test_acc  = nn.accuracy(X_test, y_test)
        test_loss = nn.cross_entropy(X_test, y_test)

        models[width] = nn
        results[width] = {
            "train_acc": train_acc,
            "val_acc":   val_acc,
            "test_acc":  test_acc,
            "test_loss": test_loss,
        }

        print(f"  hidden={width:2d} | train={train_acc:.4f} | val={val_acc:.4f} | "
              f"test={test_acc:.4f} | loss={test_loss:.4f}")

    # Decision boundary plots
    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Capacity Ablation — Moons Dataset\nHidden Width: {2, 8, 32}",
                 fontsize=13, fontweight='bold')

    for ax, width in zip(axes, hidden_widths):
        r = results[width]
        plot_decision_boundary(
            ax, models[width], X_all, y_all,
            f"hidden_dim = {width}\nTest acc = {r['test_acc']:.3f} | Loss = {r['test_loss']:.3f}"
        )

    plt.tight_layout()
    plt.savefig("starter_pack/figures/capacity_ablation.png", dpi=150, bbox_inches='tight')
    print("\nSaved: starter_pack/figures/capacity_ablation.png")
    plt.show()

    # Summary table
    print("\n=== Capacity Ablation Summary ===")
    print(f"{'Hidden':>8} | {'Train Acc':>10} | {'Val Acc':>8} | {'Test Acc':>9} | {'Test Loss':>10}")
    print("-" * 55)
    for width in hidden_widths:
        r = results[width]
        print(f"{width:>8} | {r['train_acc']:>10.4f} | {r['val_acc']:>8.4f} | "
              f"{r['test_acc']:>9.4f} | {r['test_loss']:>10.4f}")
