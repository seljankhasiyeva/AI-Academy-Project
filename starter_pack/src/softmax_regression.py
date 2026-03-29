
import numpy as np


class SoftmaxRegression:
    def __init__(self, input_dim, num_classes, weight_scale=1e-2, seed=42):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, weight_scale, size=(num_classes, input_dim))
        self.b = np.zeros(num_classes)

    def forward_logits(self, X):
        return X @ self.W.T + self.b

    @staticmethod
    def softmax(logits):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    @staticmethod
    def one_hot(y, num_classes):
        Y = np.zeros((y.shape[0], num_classes))
        Y[np.arange(y.shape[0]), y] = 1.0
        return Y

    def loss_and_grads(self, X, y, reg=1e-4):
        n = X.shape[0]
        num_classes = self.W.shape[0]

        logits = self.forward_logits(X)
        probs = self.softmax(logits)
        Y = self.one_hot(y, num_classes)

        eps = 1e-12
        correct_log_probs = -np.log(probs[np.arange(n), y] + eps)
        data_loss = np.mean(correct_log_probs)

        reg_loss = 0.5 * reg * np.sum(self.W * self.W)
        loss = data_loss + reg_loss

        dS = (probs - Y) / n
        dW = dS.T @ X + reg * self.W
        db = np.sum(dS, axis=0)

        grads = {"dW": dW, "db": db}
        return loss, grads, probs

    def predict_proba(self, X):
        logits = self.forward_logits(X)
        return self.softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def cross_entropy(self, X, y):
        probs = self.predict_proba(X)
        eps = 1e-12
        return np.mean(-np.log(probs[np.arange(X.shape[0]), y] + eps))

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        lr=0.05,
        reg=1e-4,
        epochs=200,
        batch_size=64,
        seed=42,
        verbose=True
    ):
        rng = np.random.default_rng(seed)
        n = X_train.shape[0]

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        best_val_loss = np.inf
        best_W = self.W.copy()
        best_b = self.b.copy()

        for epoch in range(epochs):
            indices = rng.permutation(n)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start in range(0, n, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                loss, grads, _ = self.loss_and_grads(X_batch, y_batch, reg=reg)

                self.W -= lr * grads["dW"]
                self.b -= lr * grads["db"]

            train_loss = self.cross_entropy(X_train, y_train) + 0.5 * reg * np.sum(self.W * self.W)
            train_acc = self.accuracy(X_train, y_train)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if X_val is not None and y_val is not None:
                val_loss = self.cross_entropy(X_val, y_val) + 0.5 * reg * np.sum(self.W * self.W)
                val_acc = self.accuracy(X_val, y_val)

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_W = self.W.copy()
                    best_b = self.b.copy()

                if verbose:
                    print(
                        f"Epoch {epoch+1:03d} | "
                        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                    )
            else:
                if verbose:
                    print(
                        f"Epoch {epoch+1:03d} | "
                        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
                    )

        if X_val is not None and y_val is not None:
            self.W = best_W
            self.b = best_b

        return history