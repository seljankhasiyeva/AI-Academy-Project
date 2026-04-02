import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from data_loading import load_digits
from confidence_reliability import run_track_b

from softmax_regression import SoftmaxRegression
from neural_network import NeuralNetwork


def save(name):
    plt.tight_layout()
    plt.savefig(f"starter_pack/figures/{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: figures/{name}.png")

def plot_reliability_diagram(results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, label, color in zip(axes, ["softmax", "nn"], ["steelblue", "tomato"]):
        bins = results[label]["bins"]
        xs = [b["mean_conf"]     for b in bins]
        ys = [b["empirical_acc"] for b in bins]
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfect calibration")
        ax.plot(xs, ys, "o-", color=color, linewidth=1.8, markersize=6, label=label)
        for b in bins:
            ax.text(b["mean_conf"], b["empirical_acc"] + 0.02, f"n={b['count']}", ha="center", fontsize=7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.1)
        ax.set_xlabel("Mean confidence"); ax.set_ylabel("Empirical accuracy")
        ax.set_title(f"Reliability — {label}", fontweight="bold")
        ax.legend(fontsize=8)
    save("reliability_diagram")

def plot_confidence_hist(results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, label, color in zip(axes, ["softmax", "nn"], ["steelblue", "tomato"]):
        conf    = results[label]["conf"]
        correct = results[label]["correct"].astype(bool)
        ax.hist(conf[correct],  bins=25, alpha=0.6, color="green",  label=f"Correct (n={correct.sum()})")
        ax.hist(conf[~correct], bins=25, alpha=0.6, color="orange", label=f"Incorrect (n={(~correct).sum()})")
        ax.set_xlabel("Confidence (max prob)")
        ax.set_ylabel("Count")
        ax.set_title(f"Confidence — {label}", fontweight="bold")
        ax.legend(fontsize=8)
    save("confidence_hist")

def plot_entropy_boxplot(results):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, label, color in zip(axes, ["softmax", "nn"], ["steelblue", "tomato"]):
        entr    = results[label]["entr"]
        correct = results[label]["correct"].astype(bool)
        bp = ax.boxplot(
            [entr[correct], entr[~correct]],
            tick_labels=["Correct", "Incorrect"],
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
        )
        bp["boxes"][0].set_facecolor("green");  bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("orange"); bp["boxes"][1].set_alpha(0.6)
        ax.set_ylabel("Predictive entropy")
        ax.set_title(f"Entropy — {label}", fontweight="bold")
    save("entropy_boxplot")

def main():
    _, _, _, _, X_test, y_test = load_digits()
    
    try:
        with open('starter_pack/models/softmax_model.pkl', 'rb') as f:
            sm_model = pickle.load(f)
        with open('starter_pack/models/nn_model.pkl', 'rb') as f:
            nn_model = pickle.load(f)
        
        print("Models loaded successfully.")

        results = run_track_b(sm_model, nn_model, X_test, y_test)
        plot_reliability_diagram(results)
        plot_confidence_hist(results)
        plot_entropy_boxplot(results)

    except FileNotFoundError as e:
        print(f"Error: {e}. Check if pkl files exist in src/models/")

if __name__ == "__main__":
    main()