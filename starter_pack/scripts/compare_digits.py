import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'starter_pack', 'src'))

from neural_network import NeuralNetwork
from softmax_regression import SoftmaxRegression
from data_loading import load_digits


if __name__ == "__main__":
    os.makedirs("starter_pack/figures", exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()
    num_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]

    print("Training Softmax on Digits...")
    softmax = SoftmaxRegression(input_dim=input_dim, num_classes=num_classes, seed=42)
    hist_sm = softmax.fit(
        X_train, y_train, X_val=X_val, y_val=y_val,
        lr=0.05, reg=1e-4, epochs=200, batch_size=64, verbose=False
    )

    print("Training Neural Network on Digits...")
    nn = NeuralNetwork(input_dim=input_dim, hidden_dim=32, num_classes=num_classes, seed=42)
    hist_nn = nn.fit(
        X_train, y_train, X_val=X_val, y_val=y_val,
        lr=0.05, reg=1e-4, epochs=200, batch_size=64
    )

    # Final results
    print("\n=== Final Test Results ===")
    print(f"Softmax | Test Acc={softmax.accuracy(X_test, y_test):.4f} | Test Loss={softmax.cross_entropy(X_test, y_test):.4f}")
    print(f"NN      | Test Acc={nn.accuracy(X_test, y_test):.4f}      | Test Loss={nn.cross_entropy(X_test, y_test):.4f}")

    epochs = range(1, 201)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Digits Benchmark — Training Dynamics", fontsize=14, fontweight='bold')

    # Loss plot
    axes[0].plot(epochs, hist_sm['train_loss'], label='Softmax Train', color='steelblue', linewidth=1.5)
    axes[0].plot(epochs, hist_sm['val_loss'],   label='Softmax Val',   color='steelblue', linestyle='--', linewidth=1.5)
    axes[0].plot(epochs, hist_nn['train_loss'], label='NN Train',      color='tomato', linewidth=1.5)
    axes[0].plot(epochs, hist_nn['val_loss'],   label='NN Val',        color='tomato', linestyle='--', linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, hist_sm['train_acc'], label='Softmax Train', color='steelblue', linewidth=1.5)
    axes[1].plot(epochs, hist_sm['val_acc'],   label='Softmax Val',   color='steelblue', linestyle='--', linewidth=1.5)
    axes[1].plot(epochs, hist_nn['train_acc'], label='NN Train',      color='tomato', linewidth=1.5)
    axes[1].plot(epochs, hist_nn['val_acc'],   label='NN Val',        color='tomato', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("starter_pack/figures/digits_training_dynamics.png", dpi=150, bbox_inches='tight')
    print("\nSaved: starter_pack/figures/digits_training_dynamics.png")
    plt.show()
