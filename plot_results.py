import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_LABELS = {
    "baseline": "ResNet50",
    "cutmix": "ResNet50 + CutMix",
    "se": "SE-ResNet50",
    "se_cutmix": "SE-ResNet50 + CutMix",
}

COLORS = {
    "baseline": "#1f77b4",
    "cutmix": "#ff7f0e",
    "se": "#2ca02c",
    "se_cutmix": "#d62728",
}


def load_metrics(csv_path):
    epochs, val_top1, val_top5, train_loss, val_loss = [], [], [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            val_top1.append(float(row["val_top1"]))
            val_top5.append(float(row["val_top5"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
    return {
        "epochs": epochs,
        "val_top1": val_top1,
        "val_top5": val_top5,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }


def find_experiment_dirs(results_dir):
    experiments = defaultdict(list)
    for entry in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, entry)
        if not os.path.isdir(path):
            continue
        csv_path = os.path.join(path, "metrics.csv")
        summary_path = os.path.join(path, "summary.json")
        if not os.path.isfile(csv_path) or not os.path.isfile(summary_path):
            continue

        if entry.startswith("se_cutmix_"):
            experiments["se_cutmix"].append(path)
        elif entry.startswith("se_"):
            experiments["se"].append(path)
        elif entry.startswith("cutmix_"):
            experiments["cutmix"].append(path)
        elif entry.startswith("baseline_"):
            experiments["baseline"].append(path)

    return experiments


def aggregate_metrics(dirs):
    all_val_top1 = []
    all_val_top5 = []
    all_train_loss = []
    all_val_loss = []

    for d in dirs:
        m = load_metrics(os.path.join(d, "metrics.csv"))
        all_val_top1.append(m["val_top1"])
        all_val_top5.append(m["val_top5"])
        all_train_loss.append(m["train_loss"])
        all_val_loss.append(m["val_loss"])

    epochs = load_metrics(os.path.join(dirs[0], "metrics.csv"))["epochs"]
    min_len = min(len(v) for v in all_val_top1)
    epochs = epochs[:min_len]

    def stats(arr_list):
        trimmed = [a[:min_len] for a in arr_list]
        mean = np.mean(trimmed, axis=0)
        std = np.std(trimmed, axis=0)
        return mean, std

    return {
        "epochs": epochs,
        "val_top1_mean": stats(all_val_top1)[0],
        "val_top1_std": stats(all_val_top1)[1],
        "val_top5_mean": stats(all_val_top5)[0],
        "val_top5_std": stats(all_val_top5)[1],
        "train_loss_mean": stats(all_train_loss)[0],
        "train_loss_std": stats(all_train_loss)[1],
        "val_loss_mean": stats(all_val_loss)[0],
        "val_loss_std": stats(all_val_loss)[1],
    }


def plot_comparison(experiments, results_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for key in ["baseline", "cutmix", "se", "se_cutmix"]:
        if key not in experiments:
            continue
        agg = aggregate_metrics(experiments[key])
        epochs = agg["epochs"]
        label = EXPERIMENT_LABELS[key]
        color = COLORS[key]

        ax = axes[0]
        ax.plot(epochs, agg["val_top1_mean"], label=label, color=color)
        ax.fill_between(epochs,
                        agg["val_top1_mean"] - agg["val_top1_std"],
                        agg["val_top1_mean"] + agg["val_top1_std"],
                        alpha=0.15, color=color)

        ax = axes[1]
        ax.plot(epochs, agg["val_top5_mean"], label=label, color=color)
        ax.fill_between(epochs,
                        agg["val_top5_mean"] - agg["val_top5_std"],
                        agg["val_top5_mean"] + agg["val_top5_std"],
                        alpha=0.15, color=color)

        ax = axes[2]
        ax.plot(epochs, agg["val_loss_mean"], label=label, color=color)
        ax.fill_between(epochs,
                        agg["val_loss_mean"] - agg["val_loss_std"],
                        agg["val_loss_mean"] + agg["val_loss_std"],
                        alpha=0.15, color=color)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Top-1 Accuracy (%)")
    axes[0].set_title("Top-1 Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Top-5 Accuracy (%)")
    axes[1].set_title("Top-5 Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Validation Loss")
    axes[2].set_title("Validation Loss")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(results_dir, "comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def print_summary_table(experiments):
    print(f"\n{'Model':<25} {'Val Top-1':>12} {'Test Top-1':>12} {'Test Top-5':>12} {'Params':>10}")
    print("-" * 75)

    for key in ["baseline", "cutmix", "se", "se_cutmix"]:
        if key not in experiments:
            continue
        dirs = experiments[key]
        test_top1s = []
        test_top5s = []
        val_top1s = []
        num_params = 0

        for d in dirs:
            with open(os.path.join(d, "summary.json"), "r") as f:
                s = json.load(f)
            test_top1s.append(s["test_top1_best"])
            test_top5s.append(s["test_top5_best"])
            val_top1s.append(s["best_val_top1"])
            num_params = s["num_params"]

        label = EXPERIMENT_LABELS[key]
        val_str = f"{np.mean(val_top1s):.2f} +/- {np.std(val_top1s):.2f}"
        test1_str = f"{np.mean(test_top1s):.2f} +/- {np.std(test_top1s):.2f}"
        test5_str = f"{np.mean(test_top5s):.2f} +/- {np.std(test_top5s):.2f}"
        print(f"{label:<25} {val_str:>12} {test1_str:>12} {test5_str:>12} {num_params/1e6:>8.1f}M")


def main():
    parser = argparse.ArgumentParser(description="Plot and summarize experiment results")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        return

    experiments = find_experiment_dirs(args.results_dir)
    if not experiments:
        print("No experiment results found.")
        return

    print(f"Found experiments: {list(experiments.keys())}")
    print(f"Runs per experiment: {[(k, len(v)) for k, v in experiments.items()]}")

    plot_comparison(experiments, args.results_dir)
    print_summary_table(experiments)


if __name__ == "__main__":
    main()
