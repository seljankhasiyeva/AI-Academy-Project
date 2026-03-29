import numpy as np
from softmax_regression import SoftmaxRegression

from data_loading import load_digits
X_train, y_train, X_val, y_val, X_test, y_test = load_digits()


def check_probabilities_sum_to_one(model, X):
    probs = model.predict_proba(X)
    row_sums = probs.sum(axis=1)
    max_deviation = np.max(np.abs(row_sums - 1.0))
    print("\n[Check 1] Probability sums")
    print("Max deviation from 1:", max_deviation)


def numerical_gradient_check(model, X, y, reg=1e-4, h=1e-5, num_checks=5, seed=0):
    rng = np.random.default_rng(seed)
    loss, grads, _ = model.loss_and_grads(X, y, reg=reg)

    print("\n[Check 2] Gradient check for W")
    for _ in range(num_checks):
        i = rng.integers(model.W.shape[0])
        j = rng.integers(model.W.shape[1])

        old_val = model.W[i, j]

        model.W[i, j] = old_val + h
        loss_plus, _, _ = model.loss_and_grads(X, y, reg=reg)

        model.W[i, j] = old_val - h
        loss_minus, _, _ = model.loss_and_grads(X, y, reg=reg)

        model.W[i, j] = old_val

        grad_num = (loss_plus - loss_minus) / (2 * h)
        grad_an = grads["dW"][i, j]
        rel_error = abs(grad_num - grad_an) / max(1e-8, abs(grad_num) + abs(grad_an))

        print(f"W[{i},{j}] -> numeric={grad_num:.8f}, analytic={grad_an:.8f}, rel_error={rel_error:.2e}")

    print("\n[Check 2] Gradient check for b")
    for _ in range(min(num_checks, model.b.shape[0])):
        i = rng.integers(model.b.shape[0])

        old_val = model.b[i]

        model.b[i] = old_val + h
        loss_plus, _, _ = model.loss_and_grads(X, y, reg=reg)

        model.b[i] = old_val - h
        loss_minus, _, _ = model.loss_and_grads(X, y, reg=reg)

        model.b[i] = old_val

        grad_num = (loss_plus - loss_minus) / (2 * h)
        grad_an = grads["db"][i]
        rel_error = abs(grad_num - grad_an) / max(1e-8, abs(grad_num) + abs(grad_an))

        print(f"b[{i}] -> numeric={grad_num:.8f}, analytic={grad_an:.8f}, rel_error={rel_error:.2e}")


def check_loss_decreases():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    y = rng.integers(0, 3, size=20)

    model = SoftmaxRegression(input_dim=5, num_classes=3, seed=0)

    losses = []
    for step in range(50):
        loss, grads, _ = model.loss_and_grads(X, y, reg=0.0)
        losses.append(loss)
        model.W -= 0.1 * grads["dW"]
        model.b -= 0.1 * grads["db"]

    print("\n[Check 3] Loss decreases")
    print("Initial loss:", losses[0])
    print("Final loss:", losses[-1])


def check_overfit_small_subset(X_train, y_train):
    X_small = X_train[:16]
    y_small = y_train[:16]

    model = SoftmaxRegression(
        input_dim=X_small.shape[1],
        num_classes=len(np.unique(y_train)),
        seed=1
    )

    model.fit(
        X_small,
        y_small,
        lr=0.1,
        reg=0.0,
        epochs=300,
        batch_size=16,
        verbose=False
    )

    train_acc = model.accuracy(X_small, y_small)
    train_loss = model.cross_entropy(X_small, y_small)

    print("\n[Check 4] Tiny subset overfit")
    print("Tiny subset train acc:", train_acc)
    print("Tiny subset train loss:", train_loss)


def check_no_nan_inf(model):
    print("\n[Check 5] Finite parameter check")
    print("W finite:", np.all(np.isfinite(model.W)))
    print("b finite:", np.all(np.isfinite(model.b)))


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()

    model = SoftmaxRegression(
        input_dim=X_train.shape[1],
        num_classes=len(np.unique(y_train)),
        seed=42
    )

    X_batch = X_train[:8]
    y_batch = y_train[:8]

    check_probabilities_sum_to_one(model, X_batch)
    numerical_gradient_check(model, X_batch, y_batch)
    check_loss_decreases()
    check_overfit_small_subset(X_train, y_train)
    check_no_nan_inf(model)


if __name__ == "__main__":
    main()