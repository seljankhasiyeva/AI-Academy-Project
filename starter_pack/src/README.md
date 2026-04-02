# src/

Source code for a from-scratch neural network and softmax regression implementation using NumPy.

## Files

### `neural_network.py`
A two-layer fully-connected neural network with tanh activations.

- **Architecture**: Linear → Tanh → Linear → Softmax
- **Training**: Mini-batch SGD with L2 regularization and best-val-loss checkpointing
- **Key methods**: `fit()`, `predict()`, `accuracy()`, `loss_and_grads()`

### `softmax_regression.py`
A single-layer softmax (multinomial logistic) regression model.

- Serves as a linear baseline comparable to the neural network
- Same training interface as `NeuralNetwork` (`fit`, `predict`, `accuracy`)
- Includes `predict_proba()` for raw probability outputs

### `data_loading.py`
Convenience loaders for the three datasets used in experiments.

| Function | Dataset |
|---|---|
| `load_digits()` | Handwritten digits (image classification) |
| `load_gaussian()` | Linearly separable Gaussian blobs |
| `load_moons()` | Non-linearly separable two-moon dataset |

Each loader returns `(X_train, y_train, X_val, y_val, X_test, y_test)`.

## Dependencies
```
numpy
```

## Quick Start
```python
from data_loading import load_moons
from neural_network import NeuralNetwork

X_train, y_train, X_val, y_val, X_test, y_test = load_moons()

model = NeuralNetwork(input_dim=2, hidden_dim=64, num_classes=2)
history = model.fit(X_train, y_train, X_val, y_val, lr=0.1, epochs=100)

print(f"Test accuracy: {model.accuracy(X_test, y_test):.4f}")
```