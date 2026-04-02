import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'starter_pack', 'src'))

from neural_network import NeuralNetwork
from softmax_regression import SoftmaxRegression
from data_loading import load_gaussian, load_moons


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
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


if __name__ == "__main__":
    os.makedirs("starter_pack/figures", exist_ok=True)

    Xg_tr, yg_tr, Xg_v, yg_v, Xg_te, yg_te = load_gaussian()
    Xm_tr, ym_tr, Xm_v, ym_v, Xm_te, ym_te = load_moons()

    num_classes_g = len(np.unique(yg_tr))
    num_classes_m = len(np.unique(ym_tr))

    # --- Gaussian: standard params ---
    print("Training Softmax on Gaussian...")
    sm_g = SoftmaxRegression(input_dim=2, num_classes=num_classes_g, seed=42)
    sm_g.fit(Xg_tr, yg_tr, X_val=Xg_v, y_val=yg_v,
             lr=0.05, reg=1e-4, epochs=200, batch_size=64, verbose=False)

    print("Training NN on Gaussian...")
    nn_g = NeuralNetwork(input_dim=2, hidden_dim=32, num_classes=num_classes_g, seed=42)
    nn_g.fit(Xg_tr, yg_tr, X_val=Xg_v, y_val=yg_v,
             lr=0.05, reg=1e-4, epochs=200, batch_size=64)

    # --- Moons: higher lr + more epochs so NN can learn the curve ---
    print("Training Softmax on Moons...")
    sm_m = SoftmaxRegression(input_dim=2, num_classes=num_classes_m, seed=42)
    sm_m.fit(Xm_tr, ym_tr, X_val=Xm_v, y_val=ym_v,
             lr=0.05, reg=1e-4, epochs=200, batch_size=64, verbose=False)

    print("Training NN on Moons...")
    nn_m = NeuralNetwork(input_dim=2, hidden_dim=32, num_classes=num_classes_m, seed=42, weight_scale=0.1)
    nn_m.fit(Xm_tr, ym_tr, X_val=Xm_v, y_val=ym_v, lr=0.1, reg=1e-4, epochs=500, batch_size=32)

    # Results
    print("\n=== Test Results ===")
    print(f"Gaussian | Softmax -> acc={sm_g.accuracy(Xg_te, yg_te):.4f}, loss={sm_g.cross_entropy(Xg_te, yg_te):.4f}")
    print(f"Gaussian | NN      -> acc={nn_g.accuracy(Xg_te, yg_te):.4f}, loss={nn_g.cross_entropy(Xg_te, yg_te):.4f}")
    print(f"Moons    | Softmax -> acc={sm_m.accuracy(Xm_te, ym_te):.4f}, loss={sm_m.cross_entropy(Xm_te, ym_te):.4f}")
    print(f"Moons    | NN      -> acc={nn_m.accuracy(Xm_te, ym_te):.4f}, loss={nn_m.cross_entropy(Xm_te, ym_te):.4f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Decision Boundaries: Softmax vs Neural Network", fontsize=14, fontweight='bold')

    X_g_all = np.vstack([Xg_tr, Xg_v, Xg_te])
    y_g_all = np.concatenate([yg_tr, yg_v, yg_te])
    X_m_all = np.vstack([Xm_tr, Xm_v, Xm_te])
    y_m_all = np.concatenate([ym_tr, ym_v, ym_te])

    plot_decision_boundary(axes[0, 0], sm_g, X_g_all, y_g_all,
        f"Gaussian — Softmax\nTest acc={sm_g.accuracy(Xg_te, yg_te):.3f}")
    plot_decision_boundary(axes[0, 1], nn_g, X_g_all, y_g_all,
        f"Gaussian — Neural Network\nTest acc={nn_g.accuracy(Xg_te, yg_te):.3f}")
    plot_decision_boundary(axes[1, 0], sm_m, X_m_all, y_m_all,
        f"Moons — Softmax\nTest acc={sm_m.accuracy(Xm_te, ym_te):.3f}")
    plot_decision_boundary(axes[1, 1], nn_m, X_m_all, y_m_all,
        f"Moons — Neural Network\nTest acc={nn_m.accuracy(Xm_te, ym_te):.3f}")

    plt.tight_layout()
    plt.savefig("starter_pack/figures/decision_boundaries.png", dpi=150, bbox_inches='tight')
    print("\nSaved: starter_pack/figures/decision_boundaries.png")
    plt.show()
