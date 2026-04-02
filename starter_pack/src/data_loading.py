import numpy as np

def load_digits():
    digits = np.load('starter_pack/data/digits_data.npz')
    splits = np.load('starter_pack/data/digits_split_indices.npz')
    X, y = digits['X'], digits['y']
    train_idx = splits['train_idx']
    val_idx   = splits['val_idx']
    test_idx  = splits['test_idx']
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]

def load_gaussian():
    data = np.load('starter_pack/data/linear_gaussian.npz')
    return data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test']

def load_moons():
    data = np.load('starter_pack/data/moons.npz')
    return data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test']