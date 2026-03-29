import numpy as np
import matplotlib.pyplot as plt

gauss = np.load('starter_pack/data/linear_gaussian.npz')
print("=== Linear Gaussian ===")
print("Keys:", gauss.files)
for k in gauss.files:
    print(f"  {k}: shape={gauss[k].shape}")

moons = np.load('starter_pack/data/moons.npz')
print("\n=== Moons ===")
print("Keys:", moons.files)
for k in moons.files:
    print(f"  {k}: shape={moons[k].shape}")

digits = np.load('starter_pack/data/digits_data.npz')
splits = np.load('starter_pack/data/digits_split_indices.npz')
print("\n=== Digits ===")
print("Data keys:", digits.files)
print("Split keys:", splits.files)
for k in digits.files:
    print(f"  {k}: shape={digits[k].shape}")
for k in splits.files:
    print(f"  {k}: shape={splits[k].shape}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

X_g = gauss[gauss.files[0]]
y_g = gauss[gauss.files[1]]
axes[0].scatter(X_g[:, 0], X_g[:, 1], c=y_g, cmap='bwr', alpha=0.6)
axes[0].set_title('Linear Gaussian')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

X_m = moons[moons.files[0]]
y_m = moons[moons.files[1]]
axes[1].scatter(X_m[:, 0], X_m[:, 1], c=y_m, cmap='bwr', alpha=0.6)
axes[1].set_title('Moons')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')

X_d = digits[digits.files[0]]
y_d = digits[digits.files[1]]
axes[2].imshow(X_d[0].reshape(8, 8), cmap='gray')
axes[2].set_title(f'Digits - label: {y_d[0]}')

plt.tight_layout()
plt.savefig('starter_pack/figures/data_overview.png')
plt.show()

print("\n=== Summary ===")
print(f"Gaussian : {X_g.shape[0]} samples, {X_g.shape[1]} features, classes: {np.unique(y_g)}")
print(f"Moons    : {X_m.shape[0]} samples, {X_m.shape[1]} features, classes: {np.unique(y_m)}")
print(f"Digits   : {X_d.shape[0]} samples, {X_d.shape[1]} features, classes: {np.unique(y_d)}")