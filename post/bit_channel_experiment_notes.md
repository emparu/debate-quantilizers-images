# Minimal arguments: a 1-bit-per-cell evidence channel

*Methodology and results dump for the two runs in `bit-argument-debate.ipynb`
(FashionMNIST, per-class whitened) and `bit-argument-debate-cifar10.ipynb`
(CIFAR-10, raw).*

## What this tests

The future-work section proposes a minimal version of arguments: let the agents
select, along with each cell, a number from a small alphabet, and check under the
same search budget, judge architecture and training steps whether this beats
plain cell selection. Unlike text arguments, the symbols carry no prior meaning
or human bias — whatever semantics they end up with is learned in equilibrium —
so if the symbol version wins, the judge is learning to offload computation onto
the agents, who can encode the result of their unbounded search into the extra
bits.

We ran exactly that, on the **last-$d_B$** scoring rule with the
**claim-conditioned judge** (design D3), which is the configuration where the
judge sees $A$'s commitment and therefore where an extra channel has to earn its
place *on top of* the claim.

## Setup

Everything not mentioned here is unchanged from the claim-conditioned sweep:
$18\times18$ images, $6\times6$ grid of $3\times3$ cells, $K=4$ cells revealed,
Adam at $10^{-3}$ with $0.85$ exponential decay, batch 128, cross-entropy on the
true label at the selected play, judges trained on-policy (the agents
best-respond exactly to the current judge, a play is realised, the judge takes
one gradient step, repeat). 15 epochs.

### The channel

A play carries $K \times 1 = 4$ bits — **one bit per revealed cell** — in every
protocol, so all three are compared at equal channel width. What differs is who
chooses which bits and when:

| protocol | decisions | alphabet per decision | total |
|---|---|---|---|
| 4-turn debate | $A,B,A,B$, one bit each | $V=2$ | 4 bits |
| 2-turn debate | $A$ takes bits 1–2, $B$ takes 3–4 | $V=4$ | 4 bits |
| consultancy | $A$ takes all four at once | $V=16$ | 4 bits |

A *move* is therefore the pair (cell, symbol), and the branching factor at a turn
is (cell candidates) $\times\, V$. The symbol is chosen jointly with the cells by
the same exact minimax, so it is subject to the same optimisation as the
evidence.

The judge always sees the same object: 4 binary features in reveal order,
appended to its head input alongside the projected CNN features,
$\text{onehot}(d_A)$ and $\text{onehot}(d_B)$. Position in the vector identifies
the emitter, so in the debate arms the judge can tell $A$'s bits from $B$'s. The
bits never enter the convolutional stack — they have no spatial structure, and
keeping them in the head means the CNN runs once per *mask* rather than once per
(mask, symbol), so enumerating the $2^4=16$ symbol sequences costs $16\times$ on
the head only.

**Deeper head.** The MLP head is 3 hidden layers of 64 with dropout 0.1, against
1 hidden layer of 64 in the base sweep — 5 linear layers from CNN features to
logits rather than 3. All arms share the architecture; the no-channel arms differ
only in the 4 missing input dims.

### The control, and why it is the right one

Each bits arm has one control: the *same* protocol with the *same* mask search,
minus the channel. This control is **exactly nested** inside the treatment. If
the judge's output does not depend on the bit inputs, then the leaf value $L$ is
constant in the symbol, so $\min$ and $\max$ over (cell, symbol) equal $\min$ and
$\max$ over cell alone, and the two arms produce the same claim, the same $d_B$
and the same reveal on the same sampled masks. A null result is therefore a real
null: there is no "did the search change?" confound to unpick.

This is worth being explicit about because the tempting alternative — matching
*judge evaluations* by giving the control $16\times$ more masks — is not a
control. It changes the mask search, which is not the treatment. The extra
evaluations the bits arm spends are the cost of searching the symbol dimension,
which is what the treatment *is*. ("Is a channel a better buy than more search?"
is a legitimate but different, resource-allocation question.)

The nesting is verified on the implementation rather than assumed: zero the head
weights on the bit inputs, build the control judge computing the identical
function, and check that the (cell, symbol) minimax reproduces the plain cell
minimax play for play. It does, for every arm, on both datasets.

### Budget

| | FashionMNIST-wh | CIFAR-10 |
|---|---|---|
| debate arms, mask search | 256 cell paths | 144 |
| debate `bits`, judge evals | 4096 | 2304 |
| debate `control`, judge evals | 256 | 144 |
| consultancy, plain | 1296 and 324 masks | 1296 and 144 |
| consultancy, with channel | 324 masks (5184 evals) | 144 masks (2304 evals) |

CIFAR is 3-channel with the CNN widened $2\times$ (c1 = c2 = 64), making each
path about $4\times$ dearer through the conv stack, hence the smaller search. On
CIFAR this has the useful side effect that `consultancy_144` does double duty: it
is the nested control for `consultancy_bits_144` *and* consultancy at the debate
arms' exact mask search, so debate-vs-consultancy is search-matched there.

Both runs are below the base sweep's 1296-mask search, so absolute numbers are
not directly comparable to the D3 table; every comparison that matters is
internal.

### Baselines (same architecture)

* **gold** — full image, no mask.
* **random mask (1)** — one uniform 4/36 mask, no agent, no claim input: the
  sparse-judge floor.
* **random mask (ensemble)** — $\arg\max_c \frac{1}{N}\sum_m p[m,c]$, reported at
  every mask budget an arm searches *and* exactly over all
  $\binom{36}{4} = 58905$ masks. The matched-budget rows are the sharp
  comparison — a protocol that searches $N$ masks against simply averaging $N$
  random ones — and the all-masks row is the zero-capability limit the whole
  protocol family collapses to.
* **consultancy** — $A$ commits a class and shows the single mask that best
  supports it; judge sees mask and $d_A$, no adversary.

### Metrics and diagnostics

Selection accuracy $\Pr[\hat c = y]$, judge accuracy, and belief (the judge's
softmax mass on $A$'s claim), all on the test set at the final epoch. For the
bits arms, additionally:

* **codeword entropy** — realised entropy of the 4-bit word, out of 4. Says how
  much of the channel is used at all.
* **randomised-bit ablation** — re-score the *same* selected plays with a random
  codeword. Says whether the judge conditions on the channel.
* **$I(b;\text{claim})$ and $I(b;y \mid \text{claim})$** per bit — what the bit
  says about $A$'s commitment, versus what it says about the truth *beyond* that
  commitment. Normalising both by $H(b)$ makes them comparable across bits with
  different usage rates.

## Results

### FashionMNIST (per-class whitened)

| run | masks | judge evals | sel acc | judge acc | belief |
|---|---|---|---|---|---|
| gold (full image) | — | — | — | **95.36** | — |
| random mask (1) | 1 | — | — | 51.35 | — |
| random mask (ens.) | 256 | — | — | 64.36 | — |
| random mask (ens.) | 324 | — | — | 64.46 | — |
| random mask (ens.) | 1296 | — | — | 64.60 | — |
| random mask (ens. **all**) | 58905 | — | — | **64.57** | — |
| consultancy | 1296 | 1296 | 76.58 | **76.58** | 77.76 |
| consultancy | 324 | 324 | 73.57 | 73.54 | 74.65 |
| consultancy + 4 bits | 324 | 5184 | 72.48 | 72.50 | 75.57 |
| debate 2-turn + 4 bits | 256 | 4096 | 71.98 | 71.93 | 72.40 |
| debate 2-turn control | 256 | 256 | 71.56 | 71.65 | 73.11 |
| debate 4-turn + 4 bits | 256 | 4096 | 63.77 | 63.56 | 64.04 |
| debate 4-turn control | 256 | 256 | 63.84 | 63.76 | 63.71 |

### CIFAR-10 (raw)

| run | masks | judge evals | sel acc | judge acc | belief |
|---|---|---|---|---|---|
| gold (full image) | — | — | — | **67.42** | — |
| random mask (1) | 1 | — | — | 37.41 | — |
| random mask (ens.) | 144 | — | — | 45.75 | — |
| random mask (ens.) | 1296 | — | — | 46.14 | — |
| random mask (ens. **all**) | 58905 | — | — | **46.12** | — |
| consultancy | 1296 | 1296 | 46.98 | **47.03** | 46.58 |
| consultancy | 144 | 144 | 44.94 | 45.03 | 44.38 |
| consultancy + 4 bits | 144 | 2304 | 44.67 | 44.70 | 43.99 |
| debate 2-turn + 4 bits | 144 | 2304 | 45.66 | 45.39 | 45.18 |
| debate 2-turn control | 144 | 144 | 45.99 | 45.94 | 45.20 |
| debate 4-turn + 4 bits | 144 | 2304 | 44.33 | 44.03 | 43.02 |
| debate 4-turn control | 144 | 144 | 43.98 | 43.78 | 43.71 |

### The channel's effect (bits minus its nested control, percentage points)

| protocol | FashionMNIST-wh | CIFAR-10 |
|---|---|---|
| consultancy | $-1.04$ | $-0.33$ |
| debate 2-turn | $+0.28$ | $-0.55$ |
| debate 4-turn | $-0.20$ | $+0.25$ |

Six comparisons, mixed signs, all $\le 1.04$ pp. For scale, the epoch-to-epoch
spread of judge accuracy within a single run's last three epochs is 0.2–0.8 pp,
and re-evaluating *identical weights* with a fresh mask draw moves the number by
0.29–1.07 pp. Every delta is inside that.

### What the bits encode

| dataset / arm | codeword entropy | randomise → judge acc | randomise → belief | $A$'s bits: $I(b;\text{claim})/H(b)$ | all bits: $I(b;y\mid\text{claim})/H(b)$ |
|---|---|---|---|---|---|
| FMNIST-wh, 2-turn | 3.27 / 4 | $-3.50$ | $-8.67$ | 70–87 % | 0.6–1.8 % |
| FMNIST-wh, 4-turn | 3.71 / 4 | $-2.56$ | $-7.39$ | 71–85 % | 0.6–1.9 % |
| FMNIST-wh, consultancy | 1.31 / 4 | $-0.11$ | $-4.46$ | 22–36 % | 1.5–3.6 % |
| CIFAR-10, 2-turn | 3.55 / 4 | $-1.43$ | $-6.68$ | 83–85 % | 0.8–1.4 % |
| CIFAR-10, 4-turn | 3.49 / 4 | $-1.06$ | $-5.74$ | 73–83 % | 0.8–1.3 % |
| CIFAR-10, consultancy | 2.08 / 4 | $-0.01$ | $-2.05$ | 16–52 % | 1.1–2.8 % |

Per bit, $I(b;\text{claim})$ exceeds $I(b;y\mid\text{claim})$ by 4× to 152×, and
by 37–152× on $A$'s bits in the debate arms. $B$'s bits carry far less about the
claim than $A$'s (5–35 % of their entropy against 70–87 %), which is what one
expects from a player that is only trying to suppress a belief rather than
advocate a specific answer.

## Findings

**1. The channel is used, and it says nothing about the world.** This is not a
case of the agents ignoring a channel they were handed. In the debate arms they
fill 82–93 % of it, and the judge demonstrably conditions on it — randomising the
codeword on an otherwise identical play costs 1.1–3.5 pp of accuracy and 5.7–8.7
pp of belief. But 70–87 % of the entropy of each of $A$'s bits is accounted for
by *which class $A$ committed to*, against about 1 % for anything about $y$
beyond that commitment. The judge is already told the claim as a one-hot. So the
agents spent the whole channel on a redundant re-transmission of something the
judge had, and accuracy lands on the nested control.

**2. The channel became cheap talk.** $A$'s payoff is $p[\cdot, d_A]$ whatever
$y$ is, so its preference over messages is state-independent, and the same holds
for $B$ given $d_A$. That is the classical condition under which no informative
signalling equilibrium exists and the message degenerates into one that merely
identifies the advocated answer — which is exactly what
$I(b;\text{claim}) \gg I(b;y\mid\text{claim})$ measures. The writeup speculates
that arguments could act as "a sort of hash that proves the selection is honest".
This is that hash, freely chosen, and it proves nothing.

**3. Corroborability is not the binding constraint.** CIFAR-10 raw was chosen
against a specific hypothesis: whitening equalises each class's mean and
covariance, destroying the low-order global structure that an assertion about the
image would have to be checked against, so on a whitened dataset nothing a symbol
could say is falsifiable by four cells. Raw natural images keep colour, texture
and scene statistics shared across cells, so a global assertion is weakly
checkable against *any* revealed cell — and $B$ chooses half of them. If that had
been what was blocking informative signalling, CIFAR should have shown an effect.
It did not: same null, same claim-encoding equilibrium, if anything *more*
extreme ($I(b;\text{claim})/H(b) = 83{-}85\%$ on $A$'s 2-turn bits). The
obstruction is the incentive structure, not the corroborability of the medium.

**4. Whether evidence selection beats the ensemble is dataset-dependent.** On
FashionMNIST-whitened the on-policy consultancy judge reaches 76.58 against an
exact all-masks ensemble of 64.57 — it closes 39 % of the gap between the
ensemble and gold. On CIFAR-10 the same protocol reaches 47.03 against an
ensemble of 46.12, closing 4 %, and *every other arm is at or below the floor*.
The quantilizer sweep's conclusion that no protocol noticeably surpasses the
ensemble reproduces on CIFAR and fails on FashionMNIST-whitened, so it is a
property of the environment rather than a general result. (The sweep's judges
were frozen; these are trained on-policy, which is where the FashionMNIST gain
comes from.)

**5. 4-turn debate is below the zero-capability floor on both datasets.**
63.6–63.8 against 64.57 on FashionMNIST-wh, 43.8–44.0 against 46.12 on CIFAR-10.
Averaging random masks beats it. The control shows the same, so this is the turn
structure — half the cells chosen by a minimiser, spread over four alternating
commitments — and not the channel.

**6. The channel moves confidence more than decisions.** Randomising the bits
costs 1.5–2.5× more belief than accuracy in every arm. In the FashionMNIST
consultancy arm it costs 4.46 pp of belief and 0.11 pp of accuracy: the bits
were doing essentially nothing except making the judge more confident.

## Caveats

* **One seed per arm.** The bits-vs-control deltas support "no effect" but cannot
  support any particular sign.
* **Evaluation noise.** Mask sampling at evaluation moves judge accuracy by
  0.29–1.07 pp on identical weights — larger than every delta in the table.
  Averaging the evaluation over several mask draws would be nearly free and is
  the first thing to fix on a rerun.
* **Per-bit, not joint, information.** $I(b;y\mid\text{claim})$ was measured per
  bit, so a purely synergistic encoding (parity-style) would be invisible to it.
  The accuracy-equals-control result does not depend on the MI estimates and
  rules out synergistic encodings being *exploited*; it does not formally rule
  out their being *present*.
* **Architecture is slightly weaker than the sweep's.** Gold 95.36 against 97.00
  on FashionMNIST-wh and 67.42 against 70.88 on CIFAR-10; the sparse judge 51.35
  against 55.61 and 37.41 against 39.30. Deeper-but-narrower head, and a smaller
  mask search. Internal comparisons are unaffected.
* **Smaller search than the base sweep** (256 / 144 cell paths against 1296),
  which is the price of enumerating a 16-way symbol space with a deeper head.

## What would have to change

For symbols to carry evidence rather than advocacy, the sender's preference over
messages has to depend on the state — which in this environment means either a
judge that is *compute*-limited on the revealed evidence (so there is something
to offload that the judge cannot recompute), or a scoring rule under which
misdescribing the evidence is punished by more than the belief it buys. Neither
is a dataset property, which is why the second dataset behaved like the first.

The environment as built gives the agents no private information beyond the full
image and the results of their own mask search, and the search result is already
summarised by the claim. There is, quite literally, nothing else for the bits to
say that the judge can check.

## Reproduction

Two notebooks, run on Kaggle 2×T4:

| | FashionMNIST-wh | CIFAR-10 |
|---|---|---|
| judges trained | 9 (7 search arms + gold + random-mask) | 9 |
| projected / actual | 5.6 h projected, 127 min slowest judge | 3.7 h projected, 64 min slowest judge |

Pre-flight checks that ran green in both notebooks before training:

* every sampled play reveals exactly 4 **distinct** cells, for every arm's
  branching (no agent may re-pick a revealed cell, which would silently reduce
  the evidence budget);
* every control is **exactly nested** inside its bits arm;
* the multi-GPU split of the selection forward matches the single-GPU result to
  $1.6\times10^{-5}$ (chunk-size reassociation in fp16).

Offline, against synthetic data: the (cell, symbol) minimax was checked against
an independent loop-based brute force for all six arm types, with the channel
encoding re-derived by hand rather than reusing the notebook's table; and the
exact all-masks ensemble was checked against an independent enumeration on a
reduced $3\times3$ grid.

Each judge's weights are saved every epoch, together with per-step losses,
per-epoch metrics, plots, and a summary JSON.
