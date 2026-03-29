import numpy as np
from neural_network import NeuralNetwork

def gradient_check():
    np.random.seed(0)
    input_dim, hidden_dim, num_classes = 4, 5, 3
    X = np.random.randn(10, input_dim)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    reg = 1e-3
    h = 1e-5

    model = NeuralNetwork(input_dim, hidden_dim, num_classes, seed=0)
    loss, grads = model.loss_and_grads(X, y, reg=reg)

    print("Checking Gradients (Relative Error):")
    for param_name in ['W1', 'W2', 'b1', 'b2']:
        param = getattr(model, param_name)
        grad_analytic = grads['d' + param_name]
        grad_numeric = np.zeros_like(param)

        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            old_val = param[ix]
            
            param[ix] = old_val + h
            loss_plus, _ = model.loss_and_grads(X, y, reg=reg)
            
            param[ix] = old_val - h
            loss_minus, _ = model.loss_and_grads(X, y, reg=reg)
            
            param[ix] = old_val
            grad_numeric[ix] = (loss_plus - loss_minus) / (2 * h)
            it.iternext()

        rel_error = np.abs(grad_analytic - grad_numeric) / (np.maximum(1e-8, np.abs(grad_analytic) + np.abs(grad_numeric)))
        print(f"{param_name} max relative error: {np.max(rel_error):.2e}")

if __name__ == "__main__":
    gradient_check()