import numpy as np
import os

def get_probs_softmax(model, X):
    return model.predict_proba(X)

def get_probs_nn(model, X):
    logits, _, _ = model.forward(X)
    return model.softmax(logits)

def compute_confidence(probs):
    return np.max(probs, axis=1)

def compute_entropy(probs):
    eps = 1e-12
    return -np.sum(probs * np.log(probs + eps), axis=1)

def reliability_bins(confidence, correct, n_bins=5):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)

        count = mask.sum()

        if count > 0:
            mean_conf = float(np.mean(confidence[mask]))
            emp_acc = float(np.mean(correct[mask]))
        else:
            mean_conf = (lo + hi) / 2
            emp_acc = 0.0

        bins.append({
            "bin": f"[{lo:.1f}, {hi:.1f}]",
            "mean_conf": mean_conf,
            "empirical_acc": emp_acc,
            "count": int(count),
        })

    return bins

def correct_incorrect_stats(confidence, entropy, correct):
    c = correct.astype(bool)
    w = ~c

    stats = {}
    stats["correct_mean_conf"] = float(np.mean(confidence[c])) if np.any(c) else 0.0
    stats["incorrect_mean_conf"] = float(np.mean(confidence[w])) if np.any(w) else 0.0
    stats["correct_mean_entropy"] = float(np.mean(entropy[c])) if np.any(c) else 0.0
    stats["incorrect_mean_entropy"] = float(np.mean(entropy[w])) if np.any(w) else 0.0

    return stats

def run_track_b(sm_model, nn_model, X_test, y_test):
    results = {"softmax": {}, "nn": {}}

    for label, model, get_probs in [
        ("softmax", sm_model, get_probs_softmax),
        ("nn", nn_model, get_probs_nn),
    ]:
        probs = get_probs(model, X_test)
        preds = np.argmax(probs, axis=1)
        correct = (preds == y_test)
        conf = compute_confidence(probs)
        entr = compute_entropy(probs)

        results[label] = {
            "bins": reliability_bins(conf, correct),
            "ci_stats": correct_incorrect_stats(conf, entr, correct),
            "conf": conf,
            "entr": entr,
            "correct": correct,
        }

    print("\n=== Track B Summary ===")

    for label in ["softmax", "nn"]:
        r = results[label]

        print(f"\n[{label.upper()}]")
        print(f"{'Bin Range':<12} | {'Count':<6} | {'Mean Conf':<10} | {'Emp Accuracy':<12}")
        print("-" * 50)

        for b in r["bins"]:
            print(
                f"{b['bin']:<12} | {b['count']:<6} | "
                f"{b['mean_conf']:<10.3f} | {b['empirical_acc']:<12.3f}"
            )

        s = r["ci_stats"]

        print("\n  Average Entropy (Confusion):")
        print(
            f"    Correct: {s['correct_mean_entropy']:.4f} "
            f"vs Incorrect: {s['incorrect_mean_entropy']:.4f}"
        )

    return results
