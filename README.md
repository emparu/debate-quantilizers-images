# Debate, consultancy and quantilizers in an image-based scalable-oversight toy

Code, saved results, trained judge weights and figures for the post
**"Quantilized debate and consultancy in image environments: protocol design lessons
for scalable oversight experiments"** ([`post/post.md`](post/post.md)).

The environment: images are downscaled to 18×18 and cut into a 6×6 grid of 3×3-pixel
cells. A weak CNN **judge** classifies an image from a *mask* that reveals only
$K=4$ of the 36 cells, so there are $\binom{36}{4} = 58{,}905$ possible masks. An
agent $A$ commits to a class and reveals cells to support it; in the debate protocols
an adversary $B$ reveals cells to minimise $A$'s score. The experiments compare
**consultancy** (no adversary) against **debate** at 2 and 4 turns, under three
scoring rules, against both a frozen judge and judges trained on-policy.

## Layout

```
post/          the post itself, plus the figures it links to
notebooks/     the eight notebooks behind the numbers
results/       every array the post's tables and curves are computed from (.npy)
plots/         all plots the notebooks produced, including ones not used in the post
weights/       the 52 trained judges (13 per experiment), as .pt state dicts
scripts/       recompute the tables and rebuild the figures from results/
```

## Checking the numbers without a GPU

Every table in the post is recomputed from `results/` and compared against the
published values by:

```bash
pip install -r requirements.txt
python scripts/verify_post_tables.py     # exits non-zero if anything disagrees
```

This touches no datasets and no GPU: it reads the saved arrays only. To rebuild the
capability-curve figures from the same arrays:

```bash
python scripts/make_post_figures.py        # capability curves, from results/
python scripts/make_bitchannel_figures.py  # bit-channel panels, from the notebook logs
```

Both write to `plots/regenerated/` by default; pass `--out post/figures` to update the copies the post links to.

## The notebooks

| notebook | what it produces | in the post | original filename |
|---|---|---|---|
| `01_frozen_judge_exact_minimax.ipynb` | gold + sparse classifiers; exact minimax debate/consultancy against the frozen judge; the worked-example figure; dumps per-image judge probabilities over all 58,905 masks | setup table, example figure | `mnistdebate3.ipynb` |
| `02_quantilizer_sweep.ipynb` | the quantilizer sweep over $(q_A, q_B)$ from those dumps | "Frozen judges and quantilized agents", both appendix tables | `mnistdebate3part3.ipynb` |
| `03_trained_judges_D1_mnist.ipynb` | 13 on-policy judges that never see $A$'s claim | D1 table | `combined-judges-sweep-noshow-mnist.ipynb` |
| `04_trained_judges_D3_mnist.ipynb` | 13 claim-conditioned judges, MNIST | D3 table | `claim-conditioned-judges-sweepmnist.ipynb` |
| `05_trained_judges_D3_fashionmnist_whitened.ipynb` | the same, FashionMNIST per-class whitened | D3 table | `claim-conditioned-judges-sweepfashionmnist-norm.ipynb` |
| `06_verifier_head_D2_mnist.ipynb` | the failed verifier-head design: judge outputs $P(\text{claim correct})$ | the D2 lesson in the advice section | `combined-judges-sweep-yesshow-mnist.ipynb` |
| `07_bit_channel_fashionmnist_whitened.ipynb` | the 1-bit-per-cell evidence channel, with nested controls | minimal-arguments appendix | `bit-argument-debate.ipynb` |
| `08_bit_channel_cifar10.ipynb` | the same on CIFAR-10 raw | minimal-arguments appendix | `bit-argument-debate-cifar10.ipynb` |

Notebooks 03–08 are the executed copies, with their outputs and training logs intact.
Notebooks 01 and 02 were run with their outputs cleared; their results are the arrays
in `results/quantilizer/`.

## Results files

| directory | contents |
|---|---|
| `results/quantilizer/` | 114 arrays. `quantilizer_consultancy_{judge,selection}_{tag}.npy` are curves indexed by $q = 1 \dots 58905$; `quantilizer_{2_turn,4_turn}_{tag}.npy` and `quantilizer_judge_{2t,4t}_{tag}.npy` are $(q_A, q_B)$ grids; `quantilizer_diag_*` are the diagonal $q_A = q_B$ curves; `perfect_selection_{tag}.npy` is the gold accuracy. |
| `results/trained_judges_D1_mnist/` | per-epoch judge accuracy, selection accuracy and per-step loss for each of the 13 D1 judges |
| `results/trained_judges_D3_{mnist,fashionmnist_whitened}/` | the same for the 13 claim-conditioned judges, plus `belief` (the judge's softmax mass on $A$'s claim) |
| `results/verifier_head_D2_mnist/` | the same for the 13 verifier-head judges (no judge-accuracy metric: that head is binary) |

Each `*_epoch_*.npy` is one value per epoch, so `[-1]` is the final-epoch number quoted
in the post. The six dataset tags are `mnist`, `fashionmnist`, `cifar10` and their
`_normalized` (per-class ZCA whitened) variants.

## Reproducibility

- **No Kaggle datasets.** Every notebook builds its data from `torchvision.datasets`
  (MNIST, FashionMNIST, CIFAR-10), downloaded at runtime. There are no `/kaggle/input`
  paths anywhere.
- **Kaggle output paths.** The notebooks write to `/kaggle/working/...`. Off Kaggle,
  edit the `plot_dir` / `data_dir` / `weights_dir` entries in each notebook's `CONFIG`.
- **Notebook 02 needs notebook 01's dumps.** Notebook 01 saves, for each of the six
  dataset variants, a `(n_images, 58905, 10)` tensor of judge probabilities — about
  2.4 GB each — and notebook 02 does the whole quantilizer sweep offline from those.
  They are on the Hugging Face Hub at
  [`eruzak/pixeldebatev1-dump`](https://huggingface.co/datasets/eruzak/pixeldebatev1-dump)
  (public, no token needed). `_resolve_hf_token` tries Kaggle Secrets first and falls
  back to the `HF_TOKEN` environment variable, so it works off Kaggle too; a token is
  only needed to *upload* a fresh dump.
- **Hardware.** Everything ran on Kaggle T4s. As a rough guide, one trained-judge sweep
  (13 judges × 10 epochs) took about 45 minutes; the bit-channel runs (7 judges ×
  15 epochs) took 5.6 h on FashionMNIST and 3.7 h on CIFAR-10 across 2×T4.

## Related files not included here

The repository this was assembled from also contains earlier single-rule versions of
the trained-judge sweep (`lastdigit_debate_sweep`, `symmetric_debate_sweep`,
`perfect_judges_sweep`, `perfect_consultancy_judge`) and an intermediate offline
analysis notebook (`mnistdebate3part2`). They are superseded by the notebooks above and
nothing in the post depends on them.
