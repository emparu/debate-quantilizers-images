#!/usr/bin/env python3
"""Rebuild the capability-curve figures used in the post from `results/quantilizer/`.

Produces the six side-by-side panels (raw dataset on the left, per-class whitened
on the right) that appear in the Results section:

    pair_selection_{mnist,fashionmnist,cifar10}.png
    pair_judge_{mnist,fashionmnist,cifar10}.png

    python scripts/make_post_figures.py [--out DIR]

The default output directory is `plots/regenerated/`, so the copies committed
under `post/figures/` (the ones the published post links to) are left untouched.
Pass `--out post/figures` to overwrite them.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QRES = os.path.join(ROOT, "results", "quantilizer")

N_MASKS = 58905          # C(36,4): the consultant's option count
QMAX_2T = 630            # C(36,2): A's cell-pairs in the 2-turn game
QMAX_4T = 36             # cells available to A at its first turn in the 4-turn game

PAIRS = [
    ("mnist", "mnist_normalized", "mnist"),
    ("fashionmnist", "fashionmnist_normalized", "fashionmnist"),
    ("cifar10", "cifar10_normalized", "cifar10"),
]


def load(name, tag):
    return np.load(os.path.join(QRES, f"{name}_{tag}.npy"))


def logspaced_q(q_max, n=36):
    """The q grid used by the sweep (mnistdebate3part3.ipynb, `_logspaced_q`)."""
    raw = np.unique(np.round(np.logspace(0, np.log10(q_max), n)).astype(int))
    raw = np.clip(raw, 1, q_max)
    if raw[0] != 1:
        raw = np.concatenate([[1], raw])
    if raw[-1] != q_max:
        raw = np.concatenate([raw, [q_max]])
    return np.unique(raw)


def panel(ax, tag, metric):
    """One dataset's capability curves: consultancy, 2-turn and 4-turn debate."""
    gold = float(load("perfect_selection", tag)[0])

    if metric == "selection":
        cons = load("quantilizer_consultancy_selection", tag)
        d2, d4 = load("quantilizer_diag_2_turn", tag), load("quantilizer_diag_4_turn", tag)
        title, cons_label = "Selection accuracy (A's claim correct)", "consultancy quantilizer (claim)"
        suffix = "(A claim)"
    else:
        cons = load("quantilizer_consultancy_judge", tag)
        d2, d4 = load("quantilizer_diag_judge_2t", tag), load("quantilizer_diag_judge_4t", tag)
        title, cons_label = "Judge accuracy", "consultancy quantilizer (judge)"
        suffix = "(judge)"

    ax.axhline(gold, color="C2", ls="--", lw=1.5,
               label=f"perfect / gold-classifier ({gold * 100:.2f}%)")
    ax.plot(np.arange(1, cons.size + 1) / N_MASKS, cons, lw=1.2, color="C0", label=cons_label)
    ax.plot(logspaced_q(QMAX_2T, 60)[: d2.size] / QMAX_2T, d2, lw=1.5, color="C1",
            label=f"2_turn  {suffix}")
    ax.plot(np.arange(1, d4.size + 1) / QMAX_4T, d4, lw=1.5, color="C3",
            label=f"4_turn  {suffix}")

    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"quantile probability $q/q_{\max}$  (decreasing $\rightarrow$ increased capability)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title} -- {tag}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.text(0.0, -0.16, "uniform random / low capability", transform=ax.transAxes,
            fontsize=9, color="0.45")
    ax.text(1.0, -0.16, "greedy / minimax / high capability", transform=ax.transAxes,
            fontsize=9, color="0.45", ha="right")


def single(tag, metric, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    panel(ax, tag, metric)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("plots", "regenerated"),
                    help="output directory (relative to the repo root)")
    ap.add_argument("--gap", type=int, default=24, help="white gap between the two panels, in px")
    args = ap.parse_args()

    out = os.path.join(ROOT, args.out)
    tmp = os.path.join(out, "_panels")
    os.makedirs(tmp, exist_ok=True)

    for metric in ("selection", "judge"):
        for raw, whitened, name in PAIRS:
            paths = []
            for tag in (raw, whitened):
                p = os.path.join(tmp, f"{metric}_{tag}.png")
                single(tag, metric, p)
                paths.append(p)

            ims = [Image.open(p).convert("RGB") for p in paths]
            h = max(i.height for i in ims)
            canvas = Image.new("RGB", (sum(i.width for i in ims) + args.gap, h), "white")
            x = 0
            for i in ims:
                canvas.paste(i, (x, 0))
                x += i.width + args.gap
            dest = os.path.join(out, f"pair_{metric}_{name}.png")
            canvas.save(dest, optimize=True)
            print(f"wrote {os.path.relpath(dest, ROOT)}  ({canvas.width}x{canvas.height})")

    print(f"\nindividual panels kept in {os.path.relpath(tmp, ROOT)}/")


if __name__ == "__main__":
    main()
