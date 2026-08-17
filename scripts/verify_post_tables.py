#!/usr/bin/env python3
"""Recompute every table in the post from the saved result arrays.

Reads only `results/` -- no GPU, no datasets, no notebook re-runs -- and checks
the recomputed values against the numbers printed in `post/post.md`.

    python scripts/verify_post_tables.py

Exit status is 0 if every value matches to the last published digit, 1 otherwise.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

DATASETS = [
    ("mnist", "MNIST"),
    ("mnist_normalized", "MNIST (whitened)"),
    ("fashionmnist", "FashionMNIST"),
    ("fashionmnist_normalized", "FashionMNIST (whitened)"),
    ("cifar10", "CIFAR10"),
    ("cifar10_normalized", "CIFAR10 (whitened)"),
]

failures = []


def check(label, got, want, tol=0.005):
    """Compare a recomputed percentage against the published one."""
    ok = abs(got - want) < tol
    if not ok:
        failures.append(f"{label}: recomputed {got:.4f}, post says {want:.4f}")
    return ok


def q(name, tag):
    return np.load(os.path.join(RES, "quantilizer", f"{name}_{tag}.npy"))


# --------------------------------------------------------------------------
# 1. Setup table: gold / perfect-mask oracle / sparse judge on a random mask
# --------------------------------------------------------------------------
SETUP = {  # gold, perfect, sparse
    "mnist": (99.24, 99.24, 60.59),
    "mnist_normalized": (99.74, 99.74, 57.99),
    "fashionmnist": (90.51, 90.52, 66.66),
    "fashionmnist_normalized": (97.00, 96.83, 55.61),
    "cifar10": (70.88, 70.15, 39.30),
    "cifar10_normalized": (69.62, 67.36, 31.04),
}

# --------------------------------------------------------------------------
# 2. Appendix table 1: exact frozen-judge minimax at q = 1
# --------------------------------------------------------------------------
EXACT = {  # ensemble, consultancy, debate 2t, debate 4t, judge 2t, judge 4t
    "mnist": (93.53, 90.44, 95.25, 95.30, 88.81, 89.86),
    "mnist_normalized": (94.73, 71.95, 93.00, 93.32, 72.82, 75.76),
    "fashionmnist": (79.41, 72.59, 79.50, 79.61, 76.22, 77.40),
    "fashionmnist_normalized": (72.50, 50.62, 73.30, 73.56, 62.14, 64.33),
    "cifar10": (49.09, 40.86, 47.84, 47.83, 42.44, 42.88),
    "cifar10_normalized": (44.35, 31.53, 40.34, 40.33, 32.16, 33.34),
}

# --------------------------------------------------------------------------
# 3. Appendix table 2: each protocol at its best capability
#    (consultancy over its whole curve, debate along the diagonal qA = qB)
# --------------------------------------------------------------------------
BEST = {  # cons sel, debate 2t sel, debate 4t sel, cons judge, debate 2t judge, debate 4t judge
    "mnist": (93.58, 95.32, 95.43, 90.44, 88.81, 89.86),
    "mnist_normalized": (94.73, 95.44, 95.65, 79.46, 72.82, 75.76),
    "fashionmnist": (79.58, 80.02, 80.06, 76.71, 76.42, 77.40),
    "fashionmnist_normalized": (72.50, 74.95, 75.37, 63.17, 62.14, 64.33),
    "cifar10": (49.11, 49.49, 49.40, 44.35, 42.44, 42.88),
    "cifar10_normalized": (44.39, 44.67, 44.92, 35.93, 32.42, 33.34),
}


def frozen_judge_tables():
    print("\n" + "=" * 78)
    print("FROZEN JUDGE / QUANTILIZERS  (results/quantilizer/*.npy)")
    print("=" * 78)

    hdr = f"{'dataset':26} {'gold':>6} {'perf':>6} {'sparse':>6} {'ens':>6} {'cons':>6} {'d2t':>6} {'d4t':>6} {'j2t':>6} {'j4t':>6}"
    print("\nsetup table + appendix table 1 (all at q = 1, exact minimax)")
    print(hdr)
    for tag, name in DATASETS:
        gold = float(q("perfect_selection", tag)[0]) * 100
        perfect = float(q("quantilizer_perfect", tag)[0]) * 100
        cj, cs = q("quantilizer_consultancy_judge", tag), q("quantilizer_consultancy_selection", tag)
        sparse, ens, cons = float(cj[-1]) * 100, float(cs[-1]) * 100, float(cs[0]) * 100
        d2 = float(q("quantilizer_2_turn", tag)[0, 0]) * 100
        d4 = float(q("quantilizer_4_turn", tag)[0, 0]) * 100
        j2 = float(q("quantilizer_judge_2t", tag)[0, 0]) * 100
        j4 = float(q("quantilizer_judge_4t", tag)[0, 0]) * 100

        g, p, s = SETUP[tag]
        check(f"setup/{tag}/gold", gold, g)
        check(f"setup/{tag}/perfect", perfect, p)
        check(f"setup/{tag}/sparse", sparse, s)
        for got, want, what in zip(
            (ens, cons, d2, d4, j2, j4), EXACT[tag],
            ("ensemble", "consultancy", "debate2t", "debate4t", "judge2t", "judge4t"),
        ):
            check(f"exact/{tag}/{what}", got, want)

        print(f"{name:26} {gold:6.2f} {perfect:6.2f} {sparse:6.2f} {ens:6.2f} "
              f"{cons:6.2f} {d2:6.2f} {d4:6.2f} {j2:6.2f} {j4:6.2f}")

    print("\nappendix table 2 (best capability; debate read on the diagonal qA = qB)")
    print(f"{'dataset':26} {'c-sel':>6} {'d2-sel':>6} {'d4-sel':>6} {'c-jud':>6} {'d2-jud':>6} {'d4-jud':>6}")
    for tag, name in DATASETS:
        vals = (
            float(q("quantilizer_consultancy_selection", tag).max()) * 100,
            float(q("quantilizer_diag_2_turn", tag).max()) * 100,
            float(q("quantilizer_diag_4_turn", tag).max()) * 100,
            float(q("quantilizer_consultancy_judge", tag).max()) * 100,
            float(q("quantilizer_diag_judge_2t", tag).max()) * 100,
            float(q("quantilizer_diag_judge_4t", tag).max()) * 100,
        )
        for got, want, what in zip(vals, BEST[tag],
                                   ("cons_sel", "d2_sel", "d4_sel", "cons_jud", "d2_jud", "d4_jud")):
            check(f"best/{tag}/{what}", got, want)
        print(f"{name:26} " + " ".join(f"{v:6.2f}" for v in vals))


# --------------------------------------------------------------------------
# 4. D1 table (judge never sees the claim), MNIST
# --------------------------------------------------------------------------
D1 = {  # (judge acc, selection acc); consultancy has no selection metric
    ("-", "consultancy"): (95.79, None),
    ("asym", "debate 2-turn"): (92.30, 92.60),
    ("asym", "debate 4-turn"): (85.19, 86.81),
    ("asym", "control 2-turn"): (86.98, 91.43),
    ("asym", "control 4-turn"): (73.67, 87.11),
    ("sym", "debate 2-turn"): (58.41, 93.80),
    ("sym", "debate 4-turn"): (58.55, 88.62),
    ("sym", "control 2-turn"): (57.85, 94.26),
    ("sym", "control 4-turn"): (57.77, 88.19),
    ("lastdig", "debate 2-turn"): (87.83, 88.08),
    ("lastdig", "debate 4-turn"): (81.35, 84.61),
    ("lastdig", "control 2-turn"): (86.80, 92.11),
    ("lastdig", "control 4-turn"): (72.48, 86.43),
}

D1_FILES = {  # (row label) -> (judge file stem, selection file stem)
    "debate 2-turn": ("debate2_epoch_judge_{r}_mnist", "debate2_epoch_selection_{r}_mnist"),
    "debate 4-turn": ("debate4_epoch_judge_{r}_mnist", "debate4_epoch_selection_{r}_mnist"),
    "control 2-turn": ("control2_epoch_judge_{r}_mnist", "control2_epoch_selection_{r}_mnist"),
    "control 4-turn": ("control4_epoch_judge_{r}_mnist", "control4_epoch_selection_{r}_mnist"),
}


def d1_table():
    print("\n" + "=" * 78)
    print("D1: JUDGE NEVER SEES THE CLAIM -- MNIST  (results/trained_judges_D1_mnist/)")
    print("=" * 78)
    d = os.path.join(RES, "trained_judges_D1_mnist")
    load = lambda stem: float(np.load(os.path.join(d, stem + ".npy"))[-1]) * 100

    print(f"\n{'rule':10} {'judge':16} {'judge acc':>10} {'sel acc':>9}")
    got = load("consultancy_epoch_judge_mnist")
    check("D1/consultancy/judge", got, D1[("-", "consultancy")][0])
    print(f"{'-':10} {'consultancy':16} {got:10.2f} {'-':>9}")

    for rule in ("asym", "sym", "lastdig"):
        for row, (jf, sf) in D1_FILES.items():
            j, s = load(jf.format(r=rule)), load(sf.format(r=rule))
            wj, ws = D1[(rule, row)]
            check(f"D1/{rule}/{row}/judge", j, wj)
            check(f"D1/{rule}/{row}/sel", s, ws)
            print(f"{rule:10} {row:16} {j:10.2f} {s:9.2f}")


# --------------------------------------------------------------------------
# 5. D3 table (judge sees the claim), MNIST and FashionMNIST whitened
# --------------------------------------------------------------------------
D3 = {  # rule, arm -> (mnist judge, mnist sel, fmnist-wh judge, fmnist-wh sel)
    ("-", "consultancy"): (94.42, 94.42, 75.46, 75.46),
    ("asym", "debate_2turn"): (93.34, 93.37, 75.26, 75.36),
    ("asym", "debate_4turn"): (88.91, 88.91, 65.47, 65.49),
    ("asym", "control_2turn"): (94.07, 94.07, 75.87, 75.85),
    ("asym", "control_4turn"): (87.90, 87.90, 67.85, 67.83),
    ("sym", "debate_2turn"): (94.02, 94.02, 76.10, 76.23),
    ("sym", "debate_4turn"): (88.29, 88.29, 67.54, 67.50),
    ("sym", "control_2turn"): (93.32, 93.32, 74.30, 74.29),
    ("sym", "control_4turn"): (88.12, 88.12, 65.96, 66.01),
    ("lastdig", "debate_2turn"): (94.07, 94.05, 75.38, 75.50),
    ("lastdig", "debate_4turn"): (88.78, 88.78, 66.79, 66.75),
    ("lastdig", "control_2turn"): (93.81, 93.81, 75.36, 75.34),
    ("lastdig", "control_4turn"): (87.62, 87.62, 65.98, 66.03),
}


def d3_table():
    print("\n" + "=" * 78)
    print("D3: JUDGE SEES THE CLAIM -- MNIST and FashionMNIST whitened")
    print("=" * 78)
    dirs = {
        "mnist": os.path.join(RES, "trained_judges_D3_mnist"),
        "fashionmnist_normalized": os.path.join(RES, "trained_judges_D3_fashionmnist_whitened"),
    }

    def load(tag, arm, metric, rule=None):
        stem = f"{arm}_epoch_{metric}_cc_" + (f"{rule}_{tag}" if rule else tag)
        return float(np.load(os.path.join(dirs[tag], stem + ".npy"))[-1]) * 100

    print(f"\n{'rule':10} {'judge':16} {'MN jud':>8} {'MN sel':>8} {'FMw jud':>8} {'FMw sel':>8}")
    vals = tuple(load(t, "consultancy", m) for t in dirs for m in ("judge_acc", "sel"))
    for got, want, what in zip(vals, D3[("-", "consultancy")], ("mn_j", "mn_s", "fm_j", "fm_s")):
        check(f"D3/consultancy/{what}", got, want)
    print(f"{'-':10} {'consultancy':16} " + " ".join(f"{v:8.2f}" for v in vals))

    for rule in ("asym", "sym", "lastdig"):
        for arm in ("debate_2turn", "debate_4turn", "control_2turn", "control_4turn"):
            vals = tuple(load(t, arm, m, rule) for t in dirs for m in ("judge_acc", "sel"))
            for got, want, what in zip(vals, D3[(rule, arm)], ("mn_j", "mn_s", "fm_j", "fm_s")):
                check(f"D3/{rule}/{arm}/{what}", got, want)
            print(f"{rule:10} {arm:16} " + " ".join(f"{v:8.2f}" for v in vals))


# --------------------------------------------------------------------------
# 6. D2 (verifier head) -- the unreported experiment cited in the advice section
# --------------------------------------------------------------------------
def d2_summary():
    print("\n" + "=" * 78)
    print("D2: VERIFIER HEAD -- MNIST  (the failure described in the advice section)")
    print("=" * 78)
    d = os.path.join(RES, "verifier_head_D2_mnist")
    sels = sorted(f for f in os.listdir(d) if "_epoch_sel_" in f)
    finals = [float(np.load(os.path.join(d, f))[-1]) * 100 for f in sels]
    losses = [f for f in os.listdir(d) if "step_loss" in f]
    print(f"\n{len(finals)} judges, final selection accuracy: "
          f"min {min(finals):.2f}%  median {sorted(finals)[len(finals)//2]:.2f}%  max {max(finals):.2f}%")
    at_chance = sum(1 for v in finals if v < 10.5)
    print(f"{at_chance} of {len(finals)} sit below 10.5% (chance is 10%).")
    ex = np.load(os.path.join(d, sorted(losses)[0]))
    print(f"meanwhile the training loss falls: e.g. {sorted(losses)[0]} "
          f"{ex[:50].mean():.3f} (first 50 steps) -> {ex[-50:].mean():.3f} (last 50)")
    if not (at_chance == 10 and 20.0 < max(finals) < 21.0):
        failures.append("D2: summary no longer matches the description in the post")


if __name__ == "__main__":
    frozen_judge_tables()
    d1_table()
    d3_table()
    d2_summary()

    print("\n" + "=" * 78)
    if failures:
        print(f"FAIL -- {len(failures)} value(s) disagree with the post:")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("PASS -- every value in the post's tables reproduces from results/.")
    sys.exit(0)
