import numpy as np
import pickle
import os
from neural_network import NeuralNetwork
from data_loading import load_digits

X_train, y_train, X_val, y_val, X_test, y_test = load_digits()

model = NeuralNetwork(
    input_dim=X_train.shape[1],
    hidden_dim=32,
    num_classes=len(np.unique(y_train)),
    seed=42
)

history = model.fit(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    lr=0.05,
    reg=1e-4,
    epochs=200,
    batch_size=64
)

model_path = os.path.join('starter_pack/models', 'nn_model.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print("\nFinal Results:")
print("Train Accuracy:", model.accuracy(X_train, y_train))
print("Val Accuracy  :", model.accuracy(X_val, y_val))
print("Test Accuracy :", model.accuracy(X_test, y_test))
print("Test Loss     :", model.cross_entropy(X_test, y_test))
print(f"Success: Model saved to {model_path}")