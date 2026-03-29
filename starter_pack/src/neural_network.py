import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, num_classes, weight_scale=1e-2, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, weight_scale, size=(hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0.0, weight_scale, size=(num_classes, hidden_dim))
        self.b2 = np.zeros(num_classes)

    @staticmethod
    def softmax(logits):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def one_hot(y, num_classes):
        Y = np.zeros((y.shape[0], num_classes))
        Y[np.arange(y.shape[0]), y] = 1.0
        return Y

    def forward(self, X):
        Z1 = X @ self.W1.T + self.b1
        H = self.tanh(Z1)
        S = H @ self.W2.T + self.b2
        return S, H, Z1
    
    def loss_and_grads(self, X, y, reg=1e-4):
        n = X.shape[0]
        num_classes = self.W2.shape[0]
        
        # Forward Pass
        logits, H, Z1 = self.forward(X)
        probs = self.softmax(logits)
        Y = self.one_hot(y, num_classes)

        # Compute Loss (Cross-Entropy + L2 Regularization)
        eps = 1e-12
        data_loss = np.mean(-np.log(probs[np.arange(n), y] + eps))
        reg_loss = 0.5 * reg * (np.sum(self.W1**2) + np.sum(self.W2**2))
        loss = data_loss + reg_loss

        # -- Vectorized Backpropagation--
        dS = (probs - Y) / n  
        
        dW2 = dS.T @ H + reg * self.W2
        db2 = np.sum(dS, axis=0)

        dH = dS @ self.W2  
        dZ1 = dH * (1 - H**2)
        
        dW1 = dZ1.T @ X + reg * self.W1
        db1 = np.sum(dZ1, axis=0)

        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return loss, grads
    
    def predict(self, X):
        logits, _, _ = self.forward(X)
        return np.argmax(self.softmax(logits), axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

    def fit(self, X_train, y_train, X_val=None, y_val=None, lr=0.1, reg=1e-4, epochs=100, batch_size=64):
        rng = np.random.default_rng(42)
        n = X_train.shape[0]
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        best_val_loss = np.inf
        best_W1, best_b1 = self.W1.copy(), self.b1.copy()
        best_W2, best_b2 = self.W2.copy(), self.b2.copy()

        for epoch in range(epochs):
            indices = rng.permutation(n)
            X_s, y_s = X_train[indices], y_train[indices]

            for i in range(0, n, batch_size):
                X_batch, y_batch = X_s[i:i+batch_size], y_s[i:i+batch_size]
                loss, grads = self.loss_and_grads(X_batch, y_batch, reg=reg)
                
                self.W1 -= lr * grads["dW1"]
                self.b1 -= lr * grads["db1"]
                self.W2 -= lr * grads["dW2"]
                self.b2 -= lr * grads["db2"]

            train_acc = self.accuracy(X_train, y_train)
            train_loss, _ = self.loss_and_grads(X_train, y_train, reg)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            if X_val is not None:
                val_loss, _ = self.loss_and_grads(X_val, y_val, reg)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(self.accuracy(X_val, y_val))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_W1, best_b1 = self.W1.copy(), self.b1.copy()
                    best_W2, best_b2 = self.W2.copy(), self.b2.copy()
        if X_val is not None:
            self.W1, self.b1 = best_W1, best_b1
            self.W2, self.b2 = best_W2, best_b2
        return history
    
    def cross_entropy(self, X, y):
        logits, _, _ = self.forward(X)
        probs = self.softmax(logits)
        eps = 1e-12
        return np.mean(-np.log(probs[np.arange(X.shape[0]), y] + eps))
