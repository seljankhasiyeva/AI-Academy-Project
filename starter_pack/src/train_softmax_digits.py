import numpy as np
from softmax_regression import SoftmaxRegression
from data_loading import load_digits
X_train, y_train, X_val, y_val, X_test, y_test = load_digits()

model = SoftmaxRegression(
    input_dim=X_train.shape[1],
    num_classes=len(np.unique(y_train)),
    seed=42
)

history = model.fit(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    lr=0.05,
    reg=1e-4,
    epochs=200,
    batch_size=64,
    seed=42,
    verbose=True
)

print("\nFinal Results:")
print("Train Accuracy:", model.accuracy(X_train, y_train))
print("Validation Accuracy:", model.accuracy(X_val, y_val))
print("Test Accuracy:", model.accuracy(X_test, y_test))
print("Test Cross-Entropy:", model.cross_entropy(X_test, y_test))