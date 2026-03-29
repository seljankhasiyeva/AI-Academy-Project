import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'starter_pack', 'src'))

from neural_network import NeuralNetwork
from softmax_regression import SoftmaxRegression
from data_loading import load_gaussian, load_moons


def train_and_evaluate(dataset_name, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*50}")

    num_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]

    # --- Softmax Regression ---
    softmax = SoftmaxRegression(input_dim=input_dim, num_classes=num_classes, seed=42)
    history_sm = softmax.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        lr=0.05, reg=1e-4, epochs=200, batch_size=64,
        verbose=False
    )

    print("\n[Softmax Regression]")
    print(f"  Train Accuracy : {softmax.accuracy(X_train, y_train):.4f}")
    print(f"  Val   Accuracy : {softmax.accuracy(X_val, y_val):.4f}")
    print(f"  Test  Accuracy : {softmax.accuracy(X_test, y_test):.4f}")
    print(f"  Test  Loss     : {softmax.cross_entropy(X_test, y_test):.4f}")

    # --- Neural Network ---
    nn = NeuralNetwork(input_dim=input_dim, hidden_dim=32, num_classes=num_classes, seed=42)
    history_nn = nn.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        lr=0.05, reg=1e-4, epochs=200, batch_size=64
    )

    print("\n[Neural Network]")
    print(f"  Train Accuracy : {nn.accuracy(X_train, y_train):.4f}")
    print(f"  Val   Accuracy : {nn.accuracy(X_val, y_val):.4f}")
    print(f"  Test  Accuracy : {nn.accuracy(X_test, y_test):.4f}")
    print(f"  Test  Loss     : {nn.cross_entropy(X_test, y_test):.4f}")

    return softmax, nn, history_sm, history_nn


if __name__ == "__main__":
    # Gaussian
    X_tr, y_tr, X_v, y_v, X_te, y_te = load_gaussian()
    sm_g, nn_g, hist_sm_g, hist_nn_g = train_and_evaluate(
        "Linear Gaussian", X_tr, y_tr, X_v, y_v, X_te, y_te
    )

    # Moons
    X_tr, y_tr, X_v, y_v, X_te, y_te = load_moons()
    sm_m, nn_m, hist_sm_m, hist_nn_m = train_and_evaluate(
        "Moons", X_tr, y_tr, X_v, y_v, X_te, y_te
    )
