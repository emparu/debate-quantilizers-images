#!/usr/bin/env python3
"""Rebuild the bit-channel comparison figures from the notebooks' training logs.

The per-epoch metrics are recovered from the executed notebooks in `notebooks/`,
so this needs no GPU, no datasets and no re-training.

    python scripts/make_bitchannel_figures.py [--out DIR]

Writes `bitchannel_comparison_{tag}.png` for both runs. Default output is
`plots/regenerated/`; pass `--out post/figures` to update the copies the post links to.
"""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUNS = [
    ("fashionmnist_normalized", "07_bit_channel_fashionmnist_whitened.ipynb"),
    ("cifar10", "08_bit_channel_cifar10.ipynb"),
]

# colour, linestyle, marker, linewidth, alpha -- 2-turn solid, 4-turn dotted,
# so the two depths are told apart by style and not only by opacity.
STYLE = {
    "consultancy_1296":     ("C2", "-.", "d", 1.6, 1.0),
    "consultancy_324":      ("C8", "-.", "X", 1.6, 1.0),
    "consultancy_144":      ("C8", "-.", "X", 1.6, 1.0),
    "consultancy_bits_324": ("C1", "-",  "P", 1.6, 1.0),
    "consultancy_bits_144": ("C1", "-",  "P", 1.6, 1.0),
    "2turn_bits":           ("C3", "-",  "o", 2.0, 1.0),
    "2turn_control":        ("C0", "--", "s", 2.0, 1.0),
    "4turn_bits":           ("C3", ":",  "^", 1.4, 0.75),
    "4turn_control":        ("C0", ":",  "v", 1.4, 0.75),
}

EPOCH_RE = re.compile(
    r"\[([A-Za-z0-9_]+)\]\s*epoch\s*(\d+)/(\d+).*?sel=\s*([\d.]+)%.*?judge_acc=\s*([\d.]+)%", re.S)
BASE_RE = {
    "gold": re.compile(r"gold \(full image\)\s+--\s+--\s+--\s+([\d.]+)%"),
    "single": re.compile(r"random mask \(1\)\s+1\s+--\s+--\s+([\d.]+)%"),
    "ensemble": re.compile(r"random mask \(ens\. ALL\)\s+\d+\s+--\s+--\s+([\d.]+)%"),
}


def outputs_of(path):
    nb = json.load(open(path))
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            text = "".join(out.get("text", []))
            if text:
                yield text


def parse(path):
    """-> ({arm: [(sel, judge) per epoch]}, {baseline: value})"""
    history, baselines = {}, {}
    for text in outputs_of(path):
        for m in EPOCH_RE.finditer(text):
            history.setdefault(m.group(1), []).append((float(m.group(4)), float(m.group(5))))
        for key, rx in BASE_RE.items():
            hit = rx.search(text)
            if hit:
                baselines[key] = float(hit.group(1))
    return history, baselines


def figure(tag, history, baselines, dest):
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.4), sharex=True)
    handles = {}

    for row, (ax, metric, title) in enumerate(
            zip(axes, (0, 1), ("Selection accuracy", "Judge accuracy"))):
        for arm, hist in history.items():
            colour, ls, mk, lw, alpha = STYLE.get(arm, ("C7", ":", "^", 1.2, 0.7))
            ys = [h[metric] for h in hist]
            line, = ax.plot(np.arange(1, len(ys) + 1), ys, color=colour, ls=ls, marker=mk,
                            lw=lw, alpha=alpha, ms=4, label=arm)
            handles.setdefault(arm, line)

        if metric == 1:  # baselines are judge-accuracy quantities only
            for key, style, label in (
                    ("gold", dict(color="0.4", lw=1.4, ls="-"), "gold (full image)"),
                    ("single", dict(color="k", lw=1.0, ls=":"), "random mask (1)"),
                    ("ensemble", dict(color="k", lw=1.4, ls="--"),
                     "random-mask ensemble (all masks)")):
                if key in baselines:
                    line = ax.axhline(baselines[key], label=label, **style)
                    handles.setdefault(label, line)

        ax.set_title(title, fontsize=11)
        ax.set_ylabel("test accuracy (%)")
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel("epoch")
    fig.suptitle(f"bit-argument debate vs control and baselines -- {tag}", y=0.98)
    # one shared legend below both panels, outside the axes
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=4,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("plots", "regenerated"))
    args = ap.parse_args()
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)

    for tag, notebook in RUNS:
        history, baselines = parse(os.path.join(ROOT, "notebooks", notebook))
        if not history:
            print(f"no training logs found in {notebook}, skipping")
            continue
        dest = os.path.join(out, f"bitchannel_comparison_{tag}.png")
        figure(tag, history, baselines, dest)
        print(f"wrote {os.path.relpath(dest, ROOT)}  "
              f"({len(history)} arms, {len(next(iter(history.values())))} epochs, "
              f"{len(baselines)} baselines)")


if __name__ == "__main__":
    main()
