#!/usr/bin/env python3
"""Create four publication-oriented figures from Paramancer summary.csv.

Usage:
    python plot_meta_learning.py summary.csv --out figures

Dependencies: pandas, numpy, matplotlib.
All aggregates are arithmetic means over recorded batches (and any seeds present).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"cold": "#1763a6", "forward_warm": "#d2641b"}
MARKERS = {"cold": "o", "forward_warm": "s"}
NAMES = {"cold": "Cold start", "forward_warm": "Forward-warm start"}
ALGORITHMS = ("gd", "hb")
INITIALIZATIONS = ("cold", "forward_warm")
TRUNCATIONS = (0, 25, 50, 75)
EPOCH = 5

def load(path):
    df = pd.read_csv(path)
    required = {
        "backend", "algorithm", "initialization", "truncation",
        "seed", "epoch", "batch", "objective_error_initial",
        "objective_error_idle", "objective_error_final", "reverse_error"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    # Individual batch errors may be zero; plotted group means are positive.
    return df

def aggregate(df, keys):
    return df.groupby(keys, as_index=False).agg(
        objective_error_initial=("objective_error_initial", "median"),
        objective_error_idle=("objective_error_idle", "median"),
        objective_error_initial_lo=(
            "objective_error_initial", lambda x: x.quantile(0.10)
        ),
        objective_error_initial_hi=(
            "objective_error_initial", lambda x: x.quantile(0.90)
        ),
        objective_error_idle_lo=(
            "objective_error_idle", lambda x: x.quantile(0.10)
        ),
        objective_error_idle_hi=(
            "objective_error_idle", lambda x: x.quantile(0.90)
        ),
        objective_error_final=("objective_error_final", "median"),
        reverse_error=("reverse_error", "median"),
        objective_error_final_lo=(
            "objective_error_final", lambda x: x.quantile(0.10)
        ),
        objective_error_final_hi=(
            "objective_error_final", lambda x: x.quantile(0.90)
        ),
        reverse_error_lo=("reverse_error", lambda x: x.quantile(0.10)),
        reverse_error_hi=("reverse_error", lambda x: x.quantile(0.90)),
    )

def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

def truncation_figure(df, out):
    """Primary figure: fixed-budget truncation T vs errors, ID baseline."""
    d = aggregate(df[df.epoch == EPOCH],
                  ["backend", "algorithm", "initialization", "truncation"])
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.2), sharex="col")
    for j, algorithm in enumerate(ALGORITHMS):
        for i, metric in enumerate(("objective_error_final", "reverse_error")):
            ax = axes[i, j]
            for initialization in INITIALIZATIONS:
                s = d[(d.backend == "unrolled") &
                      (d.algorithm == algorithm) &
                      (d.initialization == initialization)].sort_values("truncation")
                ax.plot(
                    s.truncation, s[metric], "-o", lw=2, ms=5,
                    color=COLORS[initialization], label=NAMES[initialization]
                )
                ax.fill_between(
                    s.truncation,
                    s[f"{metric}_lo"],
                    s[f"{metric}_hi"],
                    color=COLORS[initialization],
                    alpha=0.15,
                    linewidth=0,
                    zorder=1,
                )
                baseline = d[(d.backend == "implicit") &
                             (d.algorithm == algorithm) &
                             (d.initialization == initialization)]
                if not baseline.empty:
                    ax.axhspan(float(baseline[f"{metric}_lo"].iloc[0]),
                               float(baseline[f"{metric}_hi"].iloc[0]),
                               color=COLORS[initialization], alpha=0.10,
                               linewidth=0, zorder=1)
                    ax.axhline(float(baseline[metric].iloc[0]),
                               color=COLORS[initialization], ls="--", lw=1.5,
                               alpha=0.8)
            # Clip zero bounds at the plot edge without changing the statistics.
            ax.set_yscale("log", nonpositive="clip")
            ax.grid(axis="y", alpha=0.22)
            ax.set_xticks(TRUNCATIONS)
            if i == 1:
                ax.set_xlabel(r"Truncation index $T$")
            if j == 0:
                ax.set_ylabel("Final query-objective error" if i == 0
                              else "Relative hypergradient error")
            if i == 0:
                ax.set_title(algorithm.upper())
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="#555555", ls="--", label="ID baseline"))
    labels.append("ID baseline")
    fig.legend(handles, labels, loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 1.045))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out, "effect_of_truncation")

def appendix_truncation_figure(df, out):
    """Appendix figure: truncation T vs all four errors, ID baselines."""
    d = aggregate(df[df.epoch == EPOCH],
                  ["backend", "algorithm", "initialization", "truncation"])
    metrics = (
        ("objective_error_initial", "Initial query-objective error"),
        ("objective_error_idle", "Idle query-objective error"),
        ("objective_error_final", "Final query-objective error"),
        ("reverse_error", "Relative hypergradient error"),
    )
    fig, axes = plt.subplots(4, 2, figsize=(10.0, 12.4), sharex="col")
    for j, algorithm in enumerate(ALGORITHMS):
        for i, (metric, ylabel) in enumerate(metrics):
            ax = axes[i, j]
            for initialization in INITIALIZATIONS:
                s = d[(d.backend == "unrolled") &
                      (d.algorithm == algorithm) &
                      (d.initialization == initialization)].sort_values("truncation")
                ax.plot(
                    s.truncation, s[metric], "-o", lw=2, ms=5,
                    color=COLORS[initialization], label=NAMES[initialization]
                )
                ax.fill_between(
                    s.truncation,
                    s[f"{metric}_lo"],
                    s[f"{metric}_hi"],
                    color=COLORS[initialization],
                    alpha=0.15,
                    linewidth=0,
                    zorder=1,
                )
                baseline = d[(d.backend == "implicit") &
                             (d.algorithm == algorithm) &
                             (d.initialization == initialization)]
                if not baseline.empty:
                    ax.axhspan(float(baseline[f"{metric}_lo"].iloc[0]),
                               float(baseline[f"{metric}_hi"].iloc[0]),
                               color=COLORS[initialization], alpha=0.10,
                               linewidth=0, zorder=1)
                    ax.axhline(float(baseline[metric].iloc[0]),
                               color=COLORS[initialization], ls="--", lw=1.5,
                               alpha=0.8)
            # Clip zero bounds at the plot edge without changing the statistics.
            ax.set_yscale("log", nonpositive="clip")
            ax.grid(axis="y", alpha=0.22)
            ax.set_xticks(TRUNCATIONS)
            if i == 3:
                ax.set_xlabel(r"Truncation index $T$")
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(algorithm.upper())
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="#555555", ls="--", label="ID baseline"))
    labels.append("ID baseline")
    fig.legend(handles, labels, loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 1.045))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out, "appendix_effect_of_truncation")

def evolution_figure(df, out):
    """Mean reverse error over epochs, split by algorithm and initialization."""
    d = aggregate(df, ["backend", "algorithm", "initialization",
                       "truncation", "epoch"])
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.3), sharex=True)
    palette = plt.get_cmap("tab10")
    for i, initialization in enumerate(INITIALIZATIONS):
        for j, algorithm in enumerate(ALGORITHMS):
            ax = axes[i, j]
            for k, truncation in enumerate(TRUNCATIONS):
                s = d[(d.backend == "unrolled") &
                      (d.algorithm == algorithm) &
                      (d.initialization == initialization) &
                      (d.truncation == truncation)].sort_values("epoch")
                ax.plot(s.epoch, s.reverse_error, marker="o", ms=3.5,
                        lw=1.7, color=palette(k), label=f"AD–T{truncation}")
                ax.fill_between(
                    s.epoch,
                    s.reverse_error_lo,
                    s.reverse_error_hi,
                    color=palette(k),
                    alpha=0.12,
                    linewidth=0,
                    zorder=1,
                )
            s = d[(d.backend == "implicit") &
                  (d.algorithm == algorithm) &
                  (d.initialization == initialization)].sort_values("epoch")
            if not s.empty:
                ax.plot(s.epoch, s.reverse_error, "--", lw=1.8,
                        color="#333333", label="ID")
                ax.fill_between(
                    s.epoch,
                    s.reverse_error_lo,
                    s.reverse_error_hi,
                    color="#333333",
                    alpha=0.12,
                    linewidth=0,
                    zorder=1,
                )
            ax.set_title(f"{algorithm.upper()} — {NAMES[initialization].lower()}")
            # Clip zero bounds at the plot edge without changing the statistics.
            ax.set_yscale("log", nonpositive="clip")
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.grid(axis="y", alpha=0.22)
            if i == 1:
                ax.set_xlabel("Training epoch")
            if j == 0:
                ax.set_ylabel("Relative hypergradient error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center",
               frameon=False, bbox_to_anchor=(0.5, 1.045))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, out, "evolution_during_training")

def accuracy_figure(df, out):
    """Trade-off scatter: forward error vs hypergradient error at epoch 5."""
    d = aggregate(df[df.epoch == EPOCH],
                  ["backend", "algorithm", "initialization", "truncation"])
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    for j, algorithm in enumerate(ALGORITHMS):
        ax = axes[j]
        for initialization in INITIALIZATIONS:
            s = d[(d.backend == "unrolled") &
                  (d.algorithm == algorithm) &
                  (d.initialization == initialization)].sort_values("truncation")
            ax.plot(s.objective_error_final, s.reverse_error, "-o",
                    lw=1.9, ms=5, color=COLORS[initialization],
                    label=NAMES[initialization])
            for _, row in s.iterrows():
                ax.annotate(f"T{int(row.truncation)}",
                            (row.objective_error_final, row.reverse_error),
                            xytext=(4, 5), textcoords="offset points",
                            fontsize=8, color=COLORS[initialization])
            baseline = d[(d.backend == "implicit") &
                         (d.algorithm == algorithm) &
                         (d.initialization == initialization)]
            if not baseline.empty:
                ax.scatter(baseline.objective_error_final,
                           baseline.reverse_error, marker="D", s=75,
                           facecolors="none", edgecolors=COLORS[initialization],
                           linewidths=1.8, zorder=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(algorithm.upper())
        ax.grid(alpha=0.2)
        ax.set_xlabel("Final query-objective error")
        if j == 0:
            ax.set_ylabel("Relative hypergradient error")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="D", linestyle="None",
                              markerfacecolor="none", markeredgecolor="gray",
                              label="ID"))
    labels.append("ID")
    fig.legend(handles, labels, loc="upper center", frameon=False,
               ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out, "forward_vs_gradient_accuracy")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="Path to summary.csv")
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    data = load(args.csv)
    truncation_figure(data, args.out)
    evolution_figure(data, args.out)
    accuracy_figure(data, args.out)
    appendix_truncation_figure(data, args.out)
    print(f"Saved 4 figures as PDF and PNG in {args.out.resolve()}")

if __name__ == "__main__":
    main()
