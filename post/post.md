# Quantilized debate and consultancy in image environments: protocol design lessons for scalable oversight experiments

## Consultancy vs Debate: motivations and definitions

Powerful AIs today are trained by optimizing the mostly uninterpretable weights of a neural network in order to maximize a reward function; the hope is that a system that is trained to maximize this human-designed reward function will be useful and beneficial to humans. For example, reasoning LLMs are trained to maximize first the negative cross-entropy on text prediction, and then reward functions in environments. Some of these are math or algorithmic problems, where the reward function can be computed algorithmically, others are "softer" domains, such as their answers satisfying another LLM prompted with a constitution, or optimizing a reward model trained with human preferences. So the process looks like a human designing a reward function $R$ and a distribution of tasks $t \sim D$, and optimizing the parameters $\theta$ of a model $M$ such that

$$\mathbb{E}_{t\sim D}\big[\,R(M_\theta(t),\, t)\,\big]$$

is maximized; the obtained function $M_\theta$ should then be useful in some sense. We call this protocol **consultancy**.

However, there can be other protocols, such as 2-turn debate. The general form would be finding $\theta_1$ and $\theta_2$ in an equilibrium of the game with score

$$\mathbb{E}_{t\sim D}\Big[\,R\big(\,M^2_{\theta_2}(M^1_{\theta_1}(t),\, t),\;\, M^1_{\theta_1}(t),\;\, t\,\big)\Big],$$

where $M^1$ plays to maximize and $M^2$ to minimize. If the reward function doesn't depend on the $M^2$ term, this is equivalent to consultancy. But adding $M^2$ (and assuming our models are expressive enough and have unlimited compute) allows a polynomial-complexity reward function to, during this process, shape a model that solves tasks requiring complexity $\Sigma_2^P$, in a way that consultancy cannot, as in that case the reward can only verify certificates for NP tasks. Roughly, a task is in NP when a solution can be proposed together with a certificate that a polynomial-time verifier accepts (a single $\exists$ over what the verifier checks), and a task is in $\Sigma_2^P$ when there exists a solution such that, for every rebuttal, the verifier accepts it, meaning that we add $\forall$ on top, in the style of an adversarial game, with the verifier checking the solution-rebuttal pair in polynomial time. [*AI Safety via Debate*](https://arxiv.org/abs/1805.00899) (Irving, Christiano & Amodei, 2018) shows that extending this protocol to more turns lets a polynomial-time judge supervise all tasks in PSPACE. Further work attempts to make Debate *doubly efficient* and avoid the problem of obfuscated arguments, meaning that, in some sense, an honest model should be able to win without needing more compute than a dishonest one.

## Empirical experiments with images

In algorithmic environments, even though the insights can be translated to practical protocols, scalable oversight is mostly not relevant: for example, the model might be asked to produce an algorithm and a formalized proof of its correctness, which can be done with consultancy, instead of running the $\Sigma_2^P$ algorithm by itself and having it checked by an adversary and a verifier.

We want to set up a toy problem of scalable oversight that acts as an analogue to something humans might actually intend to solve with it. One such setup is the one prototyped in the original debate paper, which consists of a classifier that can only see some pixels from an image, and a powerful model selecting pixels from the image; the goal of the classifier is to, in some sense, oversee the large model, such that it is trained in order to help the weak classifier increase its accuracy. We not only want to measure whether debate performs better than random selection, as in *AI Safety via Debate* (2018), but also whether it outperforms consultancy, and why. In our case, we assume the powerful model is able to express every function, so we use a perfect (sampled) minimax to produce the pixel selection.

So, we have two goals: to make a better version of this image experiment, and to develop better methodology and conceptualization for debate experiments.

### Contributions

On the first goal, improving over the 2018 experiment:

- We compare debate against consultancy, not only against random selection.
- We sweep the agents' optimization pressure against the untrained judge. The consultant's selection accuracy declines as its capability grows on all six dataset variants (between 3.1 pp and 22.8 pp), while debate stays close to the "zero-selection-capability" consultancy throughout. At full capability debate's selection accuracy is 4.8 to 22.9 pp above consultancy's on all six, but debate's score at the optimal capability level never exceeds the zero-capability consultancy by more than 2.9 pp. We believe this is caused by having a dataset where the random selection is informative and a judge that was only trained on random masks, thus making the "zero-capability" consultancy selection accuracy equivalent to that of an ensemble of judges, and thus apparently hard to surpass without the judge being aware of the adversarial dynamics of debate.
- We use trained judges that are aware of the adversarial dynamics, and thus can leverage the debate process, as opposed to judges that treat the pixels as random evidence.
- We run judges that can access the agent's proposed class, where the 2018 judge only ever saw the revealed pixels.

On the second goal, new methodology and concepts for debate experiments, of which the last 4 seem to be novel.

- Protocol-aware judges trained on-policy, which brings the judge closer to the theoretical assumption that its only limitation is its compute, and whose learned judging policy is itself a result. (the [Anthropic Fall 2023 debate progress update](https://www.lesswrong.com/posts/QtqysYdJRenWFeWc4/anthropic-fall-2023-debate-progress-update) (Radhakrishnan, 2023) did something similar: its judge is trained by supervised learning to answer the question given the debaters' arguments, in parallel with the debaters' RL, though on QuALITY rather than images)
- A sequential framing in which $A$ proposes a solution first and then $B$ proposes another solution after seeing $A$'s moves, rather than a simultaneous game, making the exact minimax analysis tractable.
- Quantilized debates and consultancy, allowing us to dial the agents' optimization power and providing a concrete implementation of the intuition of "not optimizing too hard to avoid reward hacking" for debate.
- A scoring rule in which $A$ commits its class first and $B$ commits last, after the full reveal, breaking the game's symmetry in a way that avoids uninformative evidence and might make training easier.
- Matched controls that keep the selected pixels but shuffle from which debater they come from, in order to measure how much the adversarial structure matters, in this case a toy-specific tool, as in more complex environments the judge could presumably tell who is who anyway.

### Setup

We take images from 10-class image datasets, downscale them to $18\times 18$ pixels, and partition each into a $6\times 6$ grid of $3\times 3$-pixel cells. A *mask* is an image that reveals $K=4$ of the 36 cells and sets the pixel values of the rest of them to zero. There are in total $N = \binom{36}{4} = 58{,}905$ possible masks for each image.

Every experiment involves a weak **judge** that classifies the image from a mask; an agent $A$ (the consultant, or the first debater) that commits to an image class and reveals cells to support it; and, in the debate protocols, an adversary $B$ that plays to minimize $A$'s score. Note that $B$ is $A$'s adversary, not the judge's, and doesn't necessarily advocate for a rival answer: in some scoring rules $B$ commits a class of its own, which can even coincide with $A$'s.

![Example plays of the 4-turn debate on MNIST](figures/debate_example_4turn_mnist.png)

*Example plays of the 4-turn debate on MNIST. Numbers are the turn order; red cells were chosen by $A$ (turns 1 and 3) and blue cells by $B$ (turns 2 and 4). Each title shows the true class and the class $A$ committed to; below each image, the judge's top probabilities on the realized reveal. Top row: $A$'s class is correct; bottom row: it isn't. These plays are exact minimax against the frozen sparse judge under the asymmetric rule, and that judge does not see $A$'s claim, which is why its top class can differ from the class $A$ committed to.*

The datasets are MNIST, FashionMNIST and CIFAR10, each in two versions: *raw*, which is the original dataset, and *per-class whitened*, where a per-class ZCA transform $x' = \Sigma_k^{-1/2}(x-\mu_k) + \bar{x}$ gives every class identical mean and identity covariance, such that a linear classifier gives a chance accuracy (10%) by construction and only higher-order structure carries signal.

Two classifiers are trained on each dataset variant. The judge is the **sparse classifier** $p[m, c]$: a small CNN which outputs the probability of the image belonging to each of the 10 classes given a masked input $m$, and is trained on uniformly random masks. The other classifier is a **gold classifier**, which is trained on the full images in order to provide a reference accuracy ceiling.

| Dataset (variant) | Gold | Perfect mask selection | Sparse judge (random 4/36) |
|---|---|---|---|
| MNIST | 99.24% | 99.24% | 60.59% |
| MNIST (whitened) | 99.74% | 99.74% | 57.99% |
| FashionMNIST | 90.51% | 90.52% | 66.66% |
| FashionMNIST (whitened) | 97.00% | 96.83% | 55.61% |
| CIFAR10 | 70.88% | 70.15% | 39.30% |
| CIFAR10 (whitened) | 69.62% | 67.36% | 31.04% |

The gap between *Perfect mask selection* and *Sparse judge* is the room that evidence selection has to work in. *Perfect mask selection* is an oracle agent that picks, for every image, the mask under which the sparse judge is most confident in the **gold classifier's** predicted class, note it is not allowed to look at the true label, so it is roughly the best judge-score of weak-to-strong generalization assuming the agent doesn't have access to the labels, the column reports the judge's accuracy on that mask. It recovers essentially the gold accuracy: within 0.2 pp on the four MNIST and FashionMNIST variants, 0.7 pp on CIFAR-10 and 2.3 pp on whitened CIFAR-10. So a weak judge shown four cells out of 36 can in principle reach its strong counterpart; everything the rest of this post measures is how much of that room a protocol actually recovers. Two scalable oversight metrics are used throughout the experiments: **selection accuracy**, which measures how often the class the agent commits to equals the true label $y$, and **judge accuracy**, which measures how often the judge's argmax class equals $y$ given the information that it sees, which depending on the protocol might be only a 4-cell selection (the certificate), or can also include the class committed by A, and possibly B (the solution). **Selection accuracy** should be read as the main metric, as it measures how good the solutions that the agent picks during deployment are, regardless of what the judge used for training believes of them.

### Experiment design

Around this fixed environment, the experiments vary three things.

The first is the judge's **policy**: In one regime, the judge is trained on uniformly random masks and then frozen while the agent optimizes against it: its policy is near-optimal for random evidence, but we show that it is systematically exploitable on selected evidence. In the other regime the judge is trained on-policy, meaning that it is iteratively trained on the data that the protocol produces under the judge itself, such that it ends up approximately playing the optimal judging policy allowed by its capacity.

The second is the judge's **access to the agent's proposal**. In the complexity analogy, $A$ proposes a solution together with a certificate for it; here the committed class is the solution and the revealed cells are the certificate. In the original 2018 experiment the judge only saw the revealed pixels. While that might work in this specific toy scenario, the point of scalable oversight is that the agent computes something that is hard to compute by the judge, and then gives evidence for its correctness, so hiding the proposal isn't the default option if we wanted to test a more realistic case. So we run both variants: a judge that sees only the revealed cells, as in the original experiment, and must recover the answer from the evidence alone; and a judge that also sees the committed class, but thus is able to slide into simply believing the agent. These are judge designs D1 and D3. (there was also an experiment D2, a judge trained to score whether $A$'s claim was correct rather than to classify; it collapsed to chance and we dropped it, the lesson is in the advice section at the end)

The third is the agent's **optimization pressure**, which matters most against the frozen judge: if the judge's biases can't be trained away, the remaining lever is bounding how hard the agent optimizes, and quantilization is that dial: it can dial the selection of the agent from argmax down to random, so we can measure how much of the judge's bias gets exploited at each capability level, and how much adding an adversary suppresses that exploitation compared to consultancy.

## Why train the judge

The theoretical arguments assume that the verifier's only limitation is compute, so in some sense it is not biased and cannot be hacked. Translated to our environment, they would predict that the sparse classifier performs better under debate than under consultancy when having the optimal parameters for each of those respectively, or in some sense, that allowing the judge to use the best $\Sigma_2^P$ verifier (which also includes $NP$ verifiers with 2 cells if the adversary is ignored) that fits in its parameters will make it perform better than only allowing it to use the best $NP$ verifier instead. These arguments also require the judge to be aware in some sense that the certificates in front of it were chosen in a specific adversarial way, rather than acting as a classifier for randomly chosen evidence from an arbitrary distribution.

Humans are not perfect unbiased judges. But if we only measure debate with untrained judges, we can't disentangle whether it works due to the original computational-complexity argument or due to other reasons, such as simpler regularization effects where adversarial dynamics play less of a role. Training the judge lets us check whether the mechanism works due to the expected reason or depends on less fundamental properties that might not always be present in more complex environments. This doesn't simulate a realistic protocol, as real judging ultimately depends on human judgment, directly or through a human-imitating model, which can only be trained in a limited way without introducing new alignment problems, but it does simulate a judge that knows how the pixels are chosen.

Beyond the accuracy comparison: the learned judging policy is an empirical answer to what the optimal way of judging debates looks like for this judge, which is a result by itself. (As we'll see, when the judge can read $A$'s claim, the optimal policy turns out to be mostly believing it.) Patterns like this could inform the design of debate protocols and the instructions of judges in more realistic experiments.

## The protocols

In **consultancy**, the agent commits to a class and then reveals the mask that best supports it, so $A$ plays the class–mask pair with the highest judge score. This acts as an analogy to a verifier for an $NP$ problem, or to RLHF: a trained judge can in some sense "ask" for the best mask, but can only judge it by itself and doesn't have access to an adversary that can see the whole image.

In **debate**, $A$ commits to a class and then the two agents alternately choose cells until four are revealed: in the *2-turn* version $A$ picks two cells and then $B$ picks two, in the *4-turn* version they alternate single cells ($A, B, A, B$). Both play minimax on the value defined by the scoring rule, and $A$ commits to the class with the highest game value, $\widehat{c} = \arg\max_c V_0(c)$. This minimax is *exact* over the complete tree in the frozen-judge and quantilizer experiments; in the trained-judge experiments it is over a sampled candidate tree, since the judge changes on every step and it is too slow to re-enumerate the full tree on each batch.

Another design choice is the scoring rule, which determines what $B$ is exactly playing for.

**Asymmetric.** $B$ has no class of its own and simply minimizes the judge's score on $A$'s class: $V_0(d_A) = \max_a \min_b \, p[s_4, d_A]$. This makes $B$ a pure critic, criticism is cheap in some sense as $B$ never has to defend an alternative, this can cause the judge to be underconfident in A's proposal.

**Symmetric.** To make criticism not free, $B$ also commits a class $d_B$ after seeing $A$'s, and the payoff is $p[s_4, d_A] - p[s_4, d_B]$: to attack, $B$ has to provide a solution too. But this opens a degenerate case: when $A$'s class is strong enough that any rival would lose, $B$'s best commitment is the *same* class as $A$, thus the payoff becomes identically zero and neither player cares about which cells (evidences) are revealed to the judge. This would be bad in real applications, as we lose a useful signal that should preferably be there.

**Last-$d_B$.** $B$ commits its class only at the end, after the full reveal, so it plays $d_B = \arg\max_d p[s_4, d]$ and the leaf value is $p[s_4, d_A] - \max_d p[s_4, d]$. Criticism still isn't free as $B$ still names a solution, but $B$ can no longer make the payoff zero in advance: the payoff is zero only if $A$'s class actually wins on the revealed play, so $A$ keeps its incentive to reveal evidence under which its class wins, while $B$'s cells try to make some rival class overtake it, thus preventing uninformative evidence.

## The training loop

At every step, the current judge defines the game: the agents best-respond exactly (the agents have no parameters of their own, the agent's answers are simply the sampled minimax under the current judge), a play is realized, and the judge takes one gradient step on the cross-entropy between its output on that play and the true label. This moves the reward landscape the agents face, and a new game begins.

Formally, write $p_\phi$ for the judge with parameters $\phi$, and $s_4(\phi; x)$ for the play the protocol selects on image $x$ under that judge: the four revealed cells, plus the committed classes where the judge sees them. The training objective is

$$L(\phi) \;=\; \mathbb{E}_{(x,y)}\Big[\,\ell\big(\,p_\phi[\,s_4(\phi;x)\,,\,\cdot\,]\,,\; y\,\big)\Big],$$

where $\ell$ is the cross-entropy. Note that $\phi$ enters twice: in the evaluation $p_\phi$ and in the selection $s_4(\phi; x)$. Each training step differentiates only the evaluation, holding the play fixed, but since the selection is an argmax over a finite set of plays, it is piecewise constant in $\phi$, so away from the measure-zero boundaries where the minimax ranking reorders, this partial gradient *is* the gradient of $L$. The loop is therefore just gradient descent (almost everywhere) on the single function $L$: the reward is policy-dependent, but there is no second learner whose updates could create cycles or spirals.

In particular, the optimal judge is a fixed point of the process: if $\phi^\star$ minimizes $L$, meaning that the judge plays the best judging policy its capacity allows for this protocol, then $\nabla L(\phi^\star) = 0$ away from selection boundaries and thus the optimal policy is not trained away once reached. Empirically, we didn't have stability issues when training the judges.

## Results with trained judges

For each dataset we train 13 judges: one consultancy judge, plus, for each of the three scoring rules, a debate judge and a shuffled control at both 2 and 4 turns. All protocols get the same selection budget of 1296 judge evaluations per image: consultancy scores 1296 random masks; 2-turn scores $36$ candidate cell-pairs for $A$, each against $36$ responses for $B$; 4-turn scores $6$ candidate cells at each of the four turns, so $36\times36 = 6^4 = 1296$ in both cases.

Cells never overlap within a play: each turn samples only from the cells still unrevealed, so all 1296 candidates reveal exactly four distinct cells. However, the implementation can sometimes search the same play multiple times, so the effective search is slightly below the nominal budget: 99.0% of it for consultancy, 96.8% for 2-turn and 91.1% for 4-turn (accounting in the appendix).

Debate judges see each of the $2$ or $4$ reveals as separate input channels, meanwhile **control** judges see the same four selected cells, but shuffled across four channels in a random order. The controls (partially) isolate the value of the adversarial structure, as the accuracy gap between a debate judge and its control comes from knowing who revealed what, and when.

### D1: the judge never sees the claim

Here the judge is a plain 10-way classifier on the masked reveal, trained with cross-entropy on the true label; it never sees which class $A$ committed to. Final-epoch test accuracies on MNIST (%):

| Rule | Judge | Judge acc | Selection acc |
|---|---|---|---|
| — | consultancy | 95.79 | — |
| asym | debate 2-turn | 92.30 | 92.60 |
| asym | debate 4-turn | 85.19 | 86.81 |
| asym | control 2-turn | 86.98 | 91.43 |
| asym | control 4-turn | 73.67 | 87.11 |
| sym | debate 2-turn | 58.41 | 93.80 |
| sym | debate 4-turn | 58.55 | 88.62 |
| sym | control 2-turn | 57.85 | 94.26 |
| sym | control 4-turn | 57.77 | 88.19 |
| last-$d_B$ | debate 2-turn | 87.83 | 88.08 |
| last-$d_B$ | debate 4-turn | 81.35 | 84.61 |
| last-$d_B$ | control 2-turn | 86.80 | 92.11 |
| last-$d_B$ | control 4-turn | 72.48 | 86.43 |

Three things stand out:

- Every judge under the asym and last-$d_B$ rules surpassed the 60.6% random-mask classifier baseline; the four sym judges did not (57.8-58.6%), for the reason in the third point below.
- On judge accuracy, debate judges beat their shuffled controls consistently in the asym and last-$d_B$ rules: 92.3 vs 87.0 at 2 turns, and 85.2 vs 73.7 at 4 turns for asym, so knowing who revealed each cell carries relevant information. This empirically hints at the "verifier must know the certificates are adversarial" point: protocol awareness is worth several accuracy points in this case. However, it has to be taken into account that here, the judges don't know the precommitment, so they might not be exploiting the adversarial structure in an interesting way, but rather they might simply have to use it to determine which digit was selected by $A$, or at least the fact that they don't know what $A$ picked introduces a confounder in our experiment.
- The sym rule produces the issue mentioned previously in the protocols section: selection accuracy stays around 88–94% while judge accuracy collapses to ~58%, which is around the random-mask baseline. This happens because $A$ and $B$ often commit the same digit and neither is then incentivized to reveal informative evidence, thus the judge ends up reading pixels that are effectively random. We also observe selection accuracy consistently exceeding judge accuracy across this whole table, although less prominently than in sym. The judges cannot see A's claim and have to classify the digit only from the evidence, meanwhile A has to likely commit a digit that will work across a wide range of evidence, thus ends up having a high selection accuracy.

### D3: the judge sees the claim

Now the judge receives $A$'s committed class as an additional input (and $B$'s where applicable) before producing its classification.

| Rule | Judge | MNIST judge | MNIST sel. | FMNIST-wh. judge | FMNIST-wh. sel. |
|---|---|---|---|---|---|
| — | consultancy | 94.42 | 94.42 | 75.46 | 75.46 |
| asym | debate 2-turn | 93.34 | 93.37 | 75.26 | 75.36 |
| asym | debate 4-turn | 88.91 | 88.91 | 65.47 | 65.49 |
| asym | control 2-turn | 94.07 | 94.07 | 75.87 | 75.85 |
| asym | control 4-turn | 87.90 | 87.90 | 67.85 | 67.83 |
| sym | debate 2-turn | 94.02 | 94.02 | 76.10 | 76.23 |
| sym | debate 4-turn | 88.29 | 88.29 | 67.54 | 67.50 |
| sym | control 2-turn | 93.32 | 93.32 | 74.30 | 74.29 |
| sym | control 4-turn | 88.12 | 88.12 | 65.96 | 66.01 |
| last-$d_B$ | debate 2-turn | 94.07 | 94.05 | 75.38 | 75.50 |
| last-$d_B$ | debate 4-turn | 88.78 | 88.78 | 66.79 | 66.75 |
| last-$d_B$ | control 2-turn | 93.81 | 93.81 | 75.36 | 75.34 |
| last-$d_B$ | control 4-turn | 87.62 | 87.62 | 65.98 | 66.03 |

Now, the judge accuracy equals selection accuracy to within noise in every single cell: the judge has essentially learned to adopt $A$'s claim, but that doesn't destroy accuracy. The (presumably close to) optimal learned judging policy here was mostly to trust the agent, but also in some sense be watchful and penalize when it lies. If this penalty didn't exist, there would be no reason for the agent to pick the correct class.

We can also see in both tables that 2-turn beats 4-turn essentially everywhere, for debates and controls alike, which we didn't find a clear explanation for; the single exception is D1 sym debate (58.41 against 58.55), a 0.14 pp inversion inside the rule that has already collapsed to the random baseline. And no debate configuration meaningfully beat the plain consultancy judge, although there are a few cases where it edges past, specifically sym debate on FashionMNIST-whitened selection accuracy which scored 0.77 pp higher, and the CIFAR-10 run reported in the minimal-arguments appendix, where at a matched 144-mask search the 2-turn debate scored 45.99 against consultancy's 44.94 on selection accuracy. In the CIFAR-10 run we estimate that the advantage is not statistically significant. Note that this ordering is specific to trained judges: against the frozen judge at full capability, exact debate beats exact consultancy on **selection accuracy** by 4.8 to 22.9 pp on all six variants (see the appendix tables), however not necessarily due to the mechanism for why we expect debate to work as we will see later.

This time, the controls are very close, meaning that here the judge isn't leveraging the adversarial structure much unlike the previous case, where it didn't know what $A$'s claim was.

## Frozen judges and quantilized agents

These experiments go back to the frozen judge: the sparse classifier trained on random masks, now exploited by agents that optimize against it. To measure how the dynamics change with the agents' strength, we use quantilization ([Taylor, 2016](https://intelligence.org/files/QuantilizersSaferAlternative.pdf)) to vary the capability of the agents, in a way that formalizes the tradeoff between the increase in capability and reward hacking. A $q$-quantilizer takes a trusted *base distribution* over actions: the motivating example being the distribution of actions a human would take, and returns a single random action from its top $q$-quantile as sorted by a value (typically the conditional expected utility of the agent). At one end it reproduces the base distribution, at the other it becomes the maximizer, and in between the tradeoff is quantified: the quantilizer cannot put more than $1/q$ times the base distribution's probability on any action, so the expected cost of its actions, under any cost measure, is at most $1/q$ times that of the base distribution.

Limiting for example the depth or rollouts of an MCTS would also produce agents of intermediate strength, but the idea is that quantilizers are more principled, although we haven't proven anything about them. We just used quantilizers as an attempt of a more principled version of "a game played badly, or in a more human-like way", but they have some problems at representing that, in the sense that they have to recursively model the game in a way that in some sense is too far from human capabilities.

We use the *uniform* base distribution as our "less intelligent, but safer strategy", under which the quantilizer of capability $q$ simply ranks the finite set of options by some value $v$ and samples uniformly from the top $q$ of them:

$$\pi_q^v(s) \;=\; \tfrac{1}{q}\,\mathbf{1}\big[s \in \mathrm{Top}_q^v\big], \qquad \mathbb{E}_{q,v}[f] \;=\; \tfrac{1}{q}\sum_{s\in\mathrm{Top}_q^v} f(s).$$

At $q=1$ this is the argmax (the "infinitely capable" agent), at the maximum $q$ it is uniform random (the "zero-capability" agent), and in between it acts as a budget on optimization power. (From here on $q$ counts options rather than naming a quantile fraction, as in the definition above; the quantile probability is $q/q_{\max}$, which is what the plots' $x$-axis shows.)

**Quantilizer consultancy.** The consultant has a single capability $q \in \{1, \dots, N\}$ and ranks all masks by the judge's confidence on them, $\mathrm{maxP}(m) = \max_c p[m, c]$. It then plays in two steps:

1. It commits, *without* quantilization, to the class that scores best in expectation over its top-$q$ evidence: $c^\star(q) = \arg\max_c \, \mathbb{E}_{q,\,\mathrm{maxP}}\big[\,p[m, c]\,\big]$.
2. It samples the mask it actually shows to the judge from that top-$q$: $m \sim \pi_q^{\mathrm{maxP}}$.

Selection accuracy is $\Pr[c^\star(q) = y]$, and judge accuracy is averaged over the sampled evidence, as before.

**Quantilizer debate.** $A$ commits to a class $c$ and the cell game is played with quantilized moves: at each of its turns, instead of taking its best move, each player ranks its available moves by the value of the resulting position (best for itself first) and samples uniformly among its top $q_A$ or $q_B$. The game values are computed backwards as before, with each max and min replaced by the corresponding top-$q$ average:

$$V_0(c) = \max\mathbb{E}_{q_A}\,\min\mathbb{E}_{q_B}\;\, p[s_4, c] \;\; \text{(2-turn)}, \qquad V_0(c) = \max\mathbb{E}_{q_A}\,\min\mathbb{E}_{q_B}\,\max\mathbb{E}_{q_A}\,\min\mathbb{E}_{q_B}\;\, p[s_4, c] \;\; \text{(4-turn)},$$

where $\max\mathbb{E}_{q}$ denotes the uniform average over the $q$ best children for the maximizing player, and $\min\mathbb{E}_{q}$ the same for the minimizer. Note that the payoff here is $p[s_4, c]$ with no class of $B$'s, so all the frozen-judge experiments use the **asymmetric** rule; the frozen judge is a plain classifier and has no input for a second class anyway. The claim is simply an argmax, $\widehat{c} = \arg\max_c V_0(c)$, and the two capabilities $(q_A, q_B)$ can be set independently, so we can play a strong $A$ against a weak $B$ and vice versa.

**Why the class isn't quantilized.** Using the uniform base distribution is a rough approximation of the original quantilizer framing. For the cell choices the approximation seems acceptable, as the way to hack the classifier is presumably through the pixels rather than the class, and we could also say that the baseline is a random choice of masks. For the class it is not, as a uniform-based class quantilizer would mix random labels into the answer and score near chance at low capability. So for this reason we dropped the quantilization on the class, which stays as the argmax of a value that is computed under quantilized evidence. That said, the game formulation extends naturally: one could add an initial quantilizer over the class, sampling the commitment from the top classes ranked by the game value $V_0(c)$, with everything downstream unchanged.

Another flaw of the construction is that each quantilizer is perfectly informed about the quantilizers that come after it: It ranks its moves by the exact expectations over the downstream quantilized play, which are themselves exact expectations over the play after that, and so on. The agent's choices are limited by randomization, but its model of the game is perfect in some sense. So in this sense our agents arguably get further from the human-like agent of the original framing as the number of turns grows.

These choices determine what the zero-capability agent is. At the maximum $q$ every cell choice is uniformly random, so the consultant's choice collapses to

$$c^\star(N) \;=\; \arg\max_c \; \frac{1}{N}\sum_{m} p[m, c],$$

which is the sparse judge averaged over every possible 4-cell mask of the image: an **ensemble classifier** that needs no evidence selection at all. Both consultancy and debate converge to this ensemble at their lowest capability, so it is the shared zero-capability endpoint of this family of protocols.

### Results

**Selection accuracy vs capability.** The $x$-axis is the quantile probability $q/q_{\max}$, *decreasing to the right*, so the left edge is the uniform-random / zero-capability agent and the right edge is the exact maximizer / minimax; raw dataset on the left of each pair, per-class whitened on the right. The green dashed line is the gold classifier.

![Selection accuracy vs capability, MNIST](figures/pair_selection_mnist.png)
![Selection accuracy vs capability, FashionMNIST](figures/pair_selection_fashionmnist.png)
![Selection accuracy vs capability, CIFAR-10](figures/pair_selection_cifar10.png)

The main result of the experiment, on **selection accuracy** (the metric that matters most here, since it is what the agent's answer is when the protocol is deployed) is that randomly picked evidence makes the selection honest. As the consultant is randomized harder its committed class is the true one more and more often, and at the maximum $q$ it plays the ensemble classifier, which needs no evidence selection at all. We observe that **consultancy's selection accuracy falls essentially monotonically from it as capability grows, on all six dataset variants**, by 3.1 pp on raw MNIST (the mildest) up to 22.8 pp on whitened MNIST and 21.9 pp on whitened FashionMNIST. (not exactly monotone step to step, but on no dataset does the curve ever rise more than 0.17 pp above its zero-capability value.) Meanwhile **debate stays close to it**, ending full-capability play between 4.0 pp below and 1.8 pp above the endpoint.

Debate does edge past the ensemble, but barely. Read at matched capability ($q_A = q_B$), the margins over the endpoint are +2.87 pp on whitened FashionMNIST and +1.90 pp on MNIST, and under one point on the other four (+0.40 to +0.92). (Curiously, whitened FashionMNIST is also where sym debate slightly edged past consultancy in the trained-judge experiments.) So against the frozen judge, everything stays far below the gold classifier and neither smarter evidence selection nor the adversary buys much over just averaging the weak judge.

This suggests that against an untrained judge debate is mostly acting as a **regularizer**: it stops the agent from reaching the rare masks that "hack" the classifier, rather than adding anything through the adversarial structure itself. The trained-judge experiments support that reading, there, where the judge is no longer exploitable, no debate configuration meaningfully beat consultancy either, so what debate buys here is presumably not the generalizable "computational complexity" advantage that requires a judge aware of the adversarial dynamics, but rather the simpler fact that a minimizing adversary keeps the maximizer away from the tail of the mask distribution.

This is a property of the environment more than of the protocols. In these image datasets, the class that scores best under random evidence is mostly the true class (although there's still presumably margin for more advanced scalable oversight methods to surpass this, as the ensemble scores well below the perfect agent). This likely doesn't hold in general, as in real applications the performance of this "ensemble" might be only marginally higher than the random baseline, and also there might not be a way to ensure a "non deceptive" random distribution of evidence that is useful in some way. Let's say picking random lines in a book has a very low probability of being helpful to answer a specific question, and an average of human classifiers among every choice of lines might be biased towards the wrong answer.

**Judge accuracy vs capability.** Same axes and layout.

![Judge accuracy vs capability, MNIST](figures/pair_judge_mnist.png)
![Judge accuracy vs capability, FashionMNIST](figures/pair_judge_fashionmnist.png)
![Judge accuracy vs capability, CIFAR-10](figures/pair_judge_cifar10.png)

**Judge accuracy** behaves differently. On five of the six datasets, consultancy's judge accuracy *rises* with capability as the consultant learns to select better evidence and then *falls* as it starts exploiting the judge, peaking 78–95% of the way to full capability; raw MNIST is the only exception, where it climbs all the way to the maximizer.

Capability imbalances give a consistent picture. $A$ almost always ends up committing the honest class, but a stronger $B$ pushes $A$'s selection slightly more honest still (on all six variants at 2 turns and five of six at 4 turns, the exception being whitened CIFAR-10 where the effect is flat), while judge accuracy moves the other way: it increases with a stronger $A$ and decreases with a stronger $B$. The two metrics respond to the balance of power in opposite directions: a stronger adversary makes $A$'s claim more trustworthy, but the revealed play harder to read. This is however less of a problem in the full setups (the ones where we used trained judges) where the judge can see the proposed classes.

A caveat is that comparing debate against consultancy on judge accuracy might not be entirely fair, as a debate reveal in some sense contains fewer relevant cells, half of the evidence is chosen by $B$ only to minimize A's accuracy, and with only four cells this handicap is large. However, the point of scalable oversight is that ultimately a good protocol should score better on the same amount of information, otherwise it would be pointless, so giving a bigger budget to debate would make an unfair comparison.

## Conclusions and future work

For images, with these models and this setup, it doesn't look like there is a useful $\Sigma_2^P$-style certificate, something like "the class is valid if there exist two cells such that, whatever two cells the adversary adds, the judge still accepts" that performs significantly better than consultancy once the judge plays the (approximately) optimal policy its capacity allows; against the frozen judge the comparison depends on the capability and specific dataset, but we find that at their respective optimum capability levels, and with both agents held at the same capability, debate never significantly beats consultancy: it is ahead by under one point on selection accuracy on four of the six datasets (at most 2.87 pp, on whitened FashionMNIST) and behind on judge accuracy on four of the six. We wouldn't rule out that smarter judges could change this: bigger and more trained models, judging more complex images, might be able to leverage the debate structure in ways our small CNNs can't. Another direction is pixels accompanied by text arguments, although arguments make the experiment messier, they externalize computation out of the judge (in the sense that a sufficiently good judge could build the arguments from the revealed masks by itself), and they introduce honesty biases inherited from the base VLMs. This last consideration also suggests keeping an ablation of "evidence-only" debates, with no verbalized arguments when running debate experiments.

There is a more general point behind this: an infinite-compute judge with a limited input channel doesn't care about arguments at all, since given the evidence it can compute the minimax arguments by itself, arguments are what allow a limited-compute judge to gain something from the debaters' computation. This suggests that the tasks where debate gains the most from arguments might not be the evidence-limited ones, but the compute-limited ones. Another intuition in this direction is that arguments can act as a sort of hash that proves the selection is honest, but presumably only while the debaters are weak enough to produce only "natural" arguments in some sense, rather than adversarially crafted ones.

A minimal version of arguments can be tested directly in our setup: let the perfect agents select, along with each cell, a symbol from a small alphabet (one bit per revealed cell), and check against an exactly nested control with the same mask search, judge architecture and training steps, whether the channel helps. Unlike text arguments, these symbols carry no prior meaning or human bias; whatever semantics they end up having is learned in equilibrium, so if the symbol version won, the judge would be learning to offload computation onto the debaters. We ran this on FashionMNIST-whitened and CIFAR-10, on the last-$d_B$ rule with the claim-conditioned judge (methodology and results in the appendix). The result is a clean null: the agents use the channel and the judge conditions on it, but they seem to spend it mostly re-transmitting $A$'s claim (which the judge already sees), and accuracy is almost the same as the control in every arm. However we don't know whether the optimization dynamics in this case could find an interesting way to use the information channel if there was one.

A natural next experiment beyond this setup is debate with VLMs (vision language models) trained with SFT/RL: unlike our CNNs, VLM judges and debaters could contextualize and do computations about the revealed pixels and arguments in some sense.

Beyond measuring whether debate outperforms the other protocols, trained models with trained judges (e.g. in experiments with LLMs) would let us observe what the equilibrium of debate actually looks like, such as what a model does when it is pressured to be honest in a way that RLHF fundamentally can't reach, even if only inside an experimental setup, and, on the other side these debate and consultancy experiments can show what a model does when it has to lie to become more convincing, either testing the limits of a compute-limited judge (in the case of a trained judge) or testing the biases/lack of capability of a judge that wasn't specifically trained for the given task (in the case of an untrained judge).

Another next step can be to test debates with protocols where the debate is an iterated game, as opposed to approaches where both agents propose a solution or emit their arguments simultaneously, as these proposals are harder to model as a minmax and might in certain cases not offer the proposed computational complexity guarantees, and specifically to test the last-$d_B$ protocol.

## Advice for debate experiments with LLMs, VLMs and other toy models

*(Draft — concrete recommendations distilled from the experiments above; to be expanded in a later pass.)*

**Train the judge.** We observed significantly different dynamics with off-distribution ("untrained"), and on-distribution judges specifically trained to judge a given protocol. The theoretical basis of debate is supposed to hold when the judge is perfect but has limited compute, so debate working better than consultancy only on untrained judges would mean that it worked for the wrong reason in some sense. For LLMs, the judge can be trained either with RL or by optimizing its system prompt. It can also be beneficial to make the judge aware of the exact protocol and what it is supposed to achieve (In a more concrete sense, the judge has to reward the game in a way that makes it converge to an answer that benefits it).
To avoid the judge LLM becoming capable enough to not need the help of more powerful agents, the amount of parameters and the CoT length of the judge can be limited.

**Train the judge on exactly the quantity it is supposed to pursue.** Our unreported experiment D2 failed for this reason. There the judge was trained to say whether $A$'s claim was correct: a single $P(\text{claim correct})$ output, and the debaters played to maximize or minimize that confidence. Selection accuracy collapsed to chance: ten of the thirteen judges ended between 9.8% and 10.3%, and the three that rose above it, all under the asym rule, reached at best 20.4%. The judge's own training loss fell steadily throughout. To avoid these sorts of problems, the judge should be trained as a classifier whose classification is exactly the thing the protocol is meant to produce: a probability distribution over the multiple-choice answers. In the case of untrained judges, it might be useful to prompt the judge in a way such as "What is the probability with which you believe A's answer is the single correct answer to the problem? and what about B's?", rather than something like "Who do you think is right?" or "Who won the debate?".

This leaves an open problem whenever the answer space is not a small set of choices: an AIME-style problem where the answer is one of a thousand integers, or worse a free-form question. How to parametrize the judge so that it can still be trained on the protocol's own objective, and so leverage the debaters, is unclear to us, as we would have to ask the judge about a large number of options in order to calculate the cross entropy loss. One possible way could be asking the judge about the probability of each debater's answers being the correct one, and training the judge with RL by comparing the judge's answer to the correct one, as in LLM RLVR training. But, for example, training the judge to maximize this second score by SFT instead of RL might cause an improvement in the judge that is disentangled from its capability at judging debates, as the judge LLM would be updated only on the final answer rather than the debate-judging process that generated it, although RL might solve this problem as it would reward the model more when it emits the correct probability outputs prior to the final answer and thus correctly rewards the debaters. Further theoretical analysis would be beneficial for this case.

**Measure whether the consultancy optimum is deceptive.** Before comparing debate against consultancy, check whether the policy that optimizes directly for the metric that the judge tries to optimize actually scores poorly under the consultancy training metric: if honesty already maximizes the consultancy reward, then the comparison ends up being mostly about training dynamics or generalization rather than about how the optimum of consultancy or debate look like, which makes it weaker.

**Be careful of the failure mode in the sym debate.** If two agents can propose the same solution, then there is the likely possibility that they will pick the same answer and produce an uninformative argument. A way to fix it is by forcing the debaters to pick different answers, however we believe this is not relevant for realistic implementations, as there is no concrete way to measure whether two answers to general problems are really different, and implementing a similarity metric when training production LLMs with debate could cause unexpected issues. Meanwhile, allowing them to pick the same answer still satisfies the theoretical computational complexity arguments, as the one who picks the wrong answer would be penalized. A better option to avoid this problem, without the under-confidence problem of asym debate is to use the last-$d_B$ protocol, or other new protocols that can be discovered by analyzing debate in toy environments like ours.

## Appendix

### Methodology and implementation details

**Models and training.** The gold and sparse classifiers use the same CNN: `Conv(in, c1, 3×3, pad 1) → ReLU → MaxPool(2) → Conv(c1, c2, 3×3, pad 1) → ReLU → MaxPool(2) → Flatten → Linear(·, 128) → ReLU → Dropout(0.3) → Linear(128, 10)`, with $(c_1, c_2) = (32, 64)$ for the 1-channel datasets and $(64, 128)$ for CIFAR-10; Adam at $10^{-3}$ with $0.85$/epoch exponential decay. The baseline classifiers train for 10 epochs at batch size 256; the on-policy judges for 10 epochs at batch size 128. The per-class ZCA whitening uses Tikhonov shrinkage on the class covariances, with the per-class means and covariances estimated on the combined train and test split (the transform is class-conditional by construction, so it needs labels on both splits).

**Trained-judge experiments (D1/D3).**

- **Judges trained.** We train 13 judges per dataset: one consultancy judge, and, for each scoring rule, a debate judge and a shuffled control at both 2 and 4 turns.

- **Selection budget.** Every protocol gets the same 1296 judge evaluations per image. Consultancy scores 1296 random masks, 2-turn scores 36 candidate cell-pairs for $A$ against 36 responses for $B$, and 4-turn scores 6 candidate cells at each of the four turns, so both debates come to $36 \times 36 = 6^4 = 1296$. The branching factor at a turn is the number of *sampled candidates* (36 or 6), independent of how many cells remain unrevealed; the shrinking pool only determines which cells are available to sample from.

- **Duplicate plays.** Candidates are drawn with replacement, so some of the 1296 evaluations repeat work. Distinct judge inputs per image average 1282.7 for consultancy, 1254.3 for 2-turn and 1180.9 for 4-turn.

- **How thin the sample is.** The 1296 plays cover only a small part of the space in every protocol: 2.2% of the $\binom{36}{4} = 58{,}905$ masks for consultancy, 0.37% of the $\binom{36}{2}\binom{34}{2} = 353{,}430$ 2-turn leaves, and 0.09% of the $36\cdot35\cdot34\cdot33 = 1{,}413{,}720$ ordered 4-turn paths. So the duplicates above are a birthday effect inside each node's candidate list — 6 draws from the 36 cells available at turn 1 give 5.59 distinct cells on average — and not exhaustion of the space.

- **Why 4-turn's leaf space equals 2-turn's.** $A$'s two cells are merged into one input channel, so the ordering *within* a player never reaches the judge. The leaf-input space is therefore $\binom{36}{2}\binom{34}{2} = 353{,}430$ for both protocols, and they differ only in the tree built over it: of the 1296 plays, 1184.6 are distinct as *ordered tree paths* but only 1180.9 as *judge inputs*.

- **What the judge sees.** The consultancy judge sees one masked image, the debate judges see the 2 or 4 reveals as separate channel groups, and the control judges see the same four selected cells in a random per-sample channel order. In D3 the committed classes additionally enter as one-hots concatenated to the projected CNN features.

- **The D1 and D3 judges do not share a trunk.** D1 judges use the same CNN as the gold and sparse classifiers ($c_1, c_2 = 32, 64$ into a 128-unit head). D3 judges use a narrower one ($c_1 = c_2 = 32$) whose features are projected to 64 dimensions and then fed, with the class one-hots, to an MLP head with one hidden layer of 64 — this way the convolutional trunk runs once per play and only the head reruns per class hypothesis. Comparisons *within* either table are architecture-matched; comparisons *across* D1 and D3 — such as the observation that the controls are much closer in D3 — carry that confound on top of the change in what the judge sees.

- **Cost per scoring rule.** The budget above equalizes sampled cell-plays, but judge-*head* evaluations per play differ across rules: one conditioning per candidate $d_A$ for asym and consultancy (10 per play), and the full $(d_A, d_B)$ grid for **both** sym and last-$d_B$ (100 per play). In the implementation, last-$d_B$'s inner minimisation over $d_B$ is taken over that same grid rather than as a separate cheaper pass, so it costs the same as sym.

**Quantilizer experiments.** The judge is the frozen random-mask sparse classifier, and nothing here is sampled: its probabilities are precomputed once per image over all $N = 58{,}905$ masks, and every quantilizer expectation is an exact backward induction over the complete tree, on the full 10,000-image test set. Quantilizer consultancy ranks all $N$ masks. For debate, $q$ counts candidates at a player's own move, so the two protocols have different grids: 4-turn ranges over $q_A, q_B \in \{1, \dots, 36\}$ (one cell per turn, clamped to the cells still unrevealed), while 2-turn ranges over $q_A \in \{1, \dots, 630\}$ (the $\binom{36}{2}$ cell pairs) and $q_B \in \{1, \dots, 561\}$ (the $\binom{34}{2}$ replies), log-spaced to 31 points per axis for the heatmaps and 48 for the diagonal curves. So $q/q_{\max}$ is not the same quantity in the 2-turn and 4-turn plots. The diagonal $q_A = q_B$ is also an equality of raw counts rather than of quantile fractions: at 2 turns $A$ picks among 630 cell pairs and $B$ among 561, so at a given $q$ the same number of candidates is a slightly larger share of $B$'s menu than of $A$'s ($1/561$ against $1/630$), making $B$ marginally the more randomized of the two. The metrics are the ones defined in the main text.

**Perfect-mask ablation and exact frozen-judge minimax.** All values below are recomputed directly from the arrays saved by the part-3 quantilizer sweep (`quantilizer_data/*.npy`), the same run that produced the capability curves and matrixplots, so they are consistent with the gold and sparse columns of the setup table (both reproduce it exactly).

*Perfect* is an oracle agent that selects, for each image, the mask under which the sparse judge assigns the highest probability to the *gold classifier's* predicted class; the column reports the judge's accuracy on that mask, and is the upper bound of what evidence selection can achieve for a given judge, referenced in the setup. *Sparse* is the judge's expected accuracy on a uniformly random mask, and *Ensemble* is the zero-capability endpoint $c^\star(N)$ of the protocol family. *Consultancy* and *Debate* are at maximum capability ($q=1$, exact minimax); at $q=1$ consultancy's selection and judge accuracy coincide, since the consultant commits the judge's argmax on the single highest-confidence mask. All values are percentages:

| Dataset (variant) | Gold | Perfect | Sparse | Ensemble | Consultancy | Debate 2t | Debate 4t | Judge 2t | Judge 4t |
|---|---|---|---|---|---|---|---|---|---|
| MNIST | 99.24 | 99.24 | 60.59 | 93.53 | 90.44 | 95.25 | 95.30 | 88.81 | 89.86 |
| MNIST (whitened) | 99.74 | 99.74 | 57.99 | 94.73 | 71.95 | 93.00 | 93.32 | 72.82 | 75.76 |
| FashionMNIST | 90.51 | 90.52 | 66.66 | 79.41 | 72.59 | 79.50 | 79.61 | 76.22 | 77.40 |
| FashionMNIST (whitened) | 97.00 | 96.83 | 55.61 | 72.50 | 50.62 | 73.30 | 73.56 | 62.14 | 64.33 |
| CIFAR10 | 70.88 | 70.15 | 39.30 | 49.09 | 40.86 | 47.84 | 47.83 | 42.44 | 42.88 |
| CIFAR10 (whitened) | 69.62 | 67.36 | 31.04 | 44.35 | 31.53 | 40.34 | 40.33 | 32.16 | 33.34 |

Two things are worth reading off this table. The oracle column is essentially the gold accuracy on every dataset, which is the sense in which the gap between gold and sparse is real room for evidence selection. And the ensemble column is far above the single-random-mask column everywhere (93.53 against 60.59 on MNIST, 44.35 against 31.04 on whitened CIFAR10), which is why the zero-capability endpoint of the quantilizer family is already a strong agent.

**Best-capability comparison.** Because consultancy degrades with capability and debate does not, the $q=1$ columns above are not the fairest debate-vs-consultancy comparison; the table below takes each protocol's *best* point over its capability range. Consultancy has a single dial, so debate is read along the diagonal $q_A = q_B$ (equally capable players); taking debate's maximum over the full two-dimensional grid instead would let it choose a crippled adversary, and indeed its grid-maximum for judge accuracy always sits at a near-maximal $A$ ($q_A \le 5$ everywhere) against a substantially weakened $B$, which is not a debate. All values are percentages:

| Dataset (variant) | Cons. sel | Debate 2t sel | Debate 4t sel | Cons. judge | Debate 2t judge | Debate 4t judge |
|---|---|---|---|---|---|---|
| MNIST | 93.58 | 95.32 | 95.43 | **90.44** | 88.81 | 89.86 |
| MNIST (whitened) | 94.73 | 95.44 | 95.65 | **79.46** | 72.82 | 75.76 |
| FashionMNIST | 79.58 | 80.02 | 80.06 | 76.71 | 76.42 | **77.40** |
| FashionMNIST (whitened) | 72.50 | 74.95 | 75.37 | 63.17 | 62.14 | **64.33** |
| CIFAR10 | 49.11 | 49.49 | 49.40 | **44.35** | 42.44 | 42.88 |
| CIFAR10 (whitened) | 44.39 | 44.67 | 44.92 | **35.93** | 32.42 | 33.34 |

On judge accuracy consultancy is ahead on four of the six datasets, and where debate is ahead it is by at most 1.16 pp. On selection accuracy debate is ahead on all six, but by 0.38–2.87 pp, four of them under one point; the largest margin is FashionMNIST whitened. So at matched capability neither protocol pulls away from the other, which is the same picture the trained-judge tables give.


### Additional figures

**Selection accuracy over the $(q_A, q_B)$ grid.** The four panels are, in order, 2-turn MNIST, 4-turn MNIST, 2-turn FashionMNIST whitened and 4-turn FashionMNIST whitened. The strongest $A$ is the top row and the strongest $B$ is the rightmost column, so the top-right corner is the full minimax and the bottom-left corner is both agents playing at random.

![2-turn selection, MNIST](figures/quantilizer_2_turn_mnist.png)
![4-turn selection, MNIST](figures/quantilizer_4_turn_mnist.png)
![2-turn selection, FashionMNIST whitened](figures/quantilizer_2_turn_fashionmnist_normalized.png)
![4-turn selection, FashionMNIST whitened](figures/quantilizer_4_turn_fashionmnist_normalized.png)

**Winrate ($A$'s expected score for its claim).** The panels are in the same order.

![2-turn winrate, MNIST](figures/quantilizer_winrate_2turn_mnist.png)
![4-turn winrate, MNIST](figures/quantilizer_winrate_4turn_mnist.png)
![2-turn winrate, FashionMNIST whitened](figures/quantilizer_winrate_2turn_fashionmnist_normalized.png)
![4-turn winrate, FashionMNIST whitened](figures/quantilizer_winrate_4turn_fashionmnist_normalized.png)

**Judge accuracy over the $(q_A, q_B)$ grid.** The panels are in the same order.

![2-turn judge accuracy, MNIST](figures/quantilizer_judge_2t_mnist.png)
![4-turn judge accuracy, MNIST](figures/quantilizer_judge_4t_mnist.png)
![2-turn judge accuracy, FashionMNIST whitened](figures/quantilizer_judge_2t_fashionmnist_normalized.png)
![4-turn judge accuracy, FashionMNIST whitened](figures/quantilizer_judge_4t_fashionmnist_normalized.png)


### The minimal-arguments experiment: a 1-bit-per-cell evidence channel

This is the experiment proposed in the conclusions, run on the **last-$d_B$** scoring rule with the **claim-conditioned judge** (design D3) — the configuration where the judge sees $A$'s commitment, and therefore where an extra channel has to earn its place *on top of* the claim. We ran it on two datasets: FashionMNIST per-class whitened, and CIFAR-10 raw.

**Setup.** Everything not mentioned here is unchanged from the claim-conditioned sweep: $18\times18$ images, $6\times6$ grid of $3\times3$ cells, $K=4$ cells revealed, Adam at $10^{-3}$ with $0.85$ exponential decay, batch 128, cross-entropy on the true label at the selected play, and judges trained on-policy. These runs train for 15 epochs rather than the base sweep's 10.

**The channel.** A play carries $K \times 1 = 4$ bits — one bit per revealed cell — in every protocol, so all three are compared at equal channel width. What differs is who chooses which bits and when:

| protocol | decisions | alphabet per decision | total |
|---|---|---|---|
| 4-turn debate | $A,B,A,B$, one bit each | $V=2$ | 4 bits |
| 2-turn debate | $A$ takes bits 1–2, $B$ takes 3–4 | $V=4$ | 4 bits |
| consultancy | $A$ takes all four at once | $V=16$ | 4 bits |

A *move* is therefore the pair (cell, symbol), and the branching factor at a turn is (cell candidates) $\times\, V$. The symbol is chosen jointly with the cells by the same exact minimax, so it is subject to the same optimisation as the evidence.

The judge always sees the same object: 4 binary features in reveal order, appended to its head input alongside the projected CNN features, $\text{onehot}(d_A)$ and $\text{onehot}(d_B)$. Position in the vector identifies the emitter, so in the debate arms the judge can tell $A$'s bits from $B$'s. The bits never enter the convolutional stack — they have no spatial structure, and keeping them in the head means the CNN runs once per *mask* rather than once per (mask, symbol), so enumerating the $2^4=16$ symbol sequences costs $16\times$ on the head only. The MLP head here is 3 hidden layers of 64 with dropout 0.1, against 1 hidden layer of 64 in the base sweep; all arms share the architecture, and the no-channel arms differ only in the 4 missing input dims.

**The control, and why it is the right one.** Every arm that has the channel is paired with the same protocol run without it: the same number of masks searched, the same judge architecture apart from the four extra input features, and the same number of training steps. Each arm trains its own judge from its own initialisation, so a control is a separate training run and not a re-scoring of the channel arm's judge. The only thing that differs is whether the agents get to emit symbols.

That pairing makes the comparison clean, because the no-channel arm is a special case of the channel arm. Suppose the judge ends up ignoring the four bit features. Then a play's score does not depend on which symbols were emitted, so when the agents search over (cell, symbol) pairs the symbol never decides anything: the search settles on exactly the cells the no-channel search would have settled on, and with them the same claim, the same $d_B$ and the same reveal. "The channel is useless" and "the two arms score the same" are therefore the same statement in this design, which is what makes a null result here worth reporting — it cannot be explained away by the two arms having searched different things.

There is a tempting alternative we did not use. Trying all $2^4 = 16$ symbol sequences on each mask costs the channel arm $16\times$ more judge evaluations than its control, and one could even that up by letting the control search $16\times$ more masks. But then the control would be searching a larger set of masks, and any difference between the two arms would mix together two effects: whether the symbols helped, and whether the deeper mask search helped. Those extra evaluations are what searching the symbol dimension costs, and searching the symbol dimension is the thing under test.

We check this in the code rather than only arguing for it: making a judge blind to the bits and re-running the search reproduces its control's play exactly — same cells, same claim, same $d_B$ — for every arm on both datasets (`check_control_nesting` in the notebooks).

**Budget.** Both runs search fewer masks than the base sweep, and CIFAR-10 fewer than FashionMNIST:

| | FashionMNIST-wh | CIFAR-10 |
|---|---|---|
| debate arms, mask search | 256 cell paths | 144 |
| debate `bits`, judge evals | 4096 | 2304 |
| debate `control`, judge evals | 256 | 144 |
| consultancy, plain | 1296 and 324 masks | 1296 and 144 |
| consultancy, with channel | 324 masks (5184 evals) | 144 masks (2304 evals) |

CIFAR is 3-channel with the CNN widened $2\times$ (c1 = c2 = 64), making each path about $4\times$ dearer through the conv stack, hence the smaller search. On CIFAR this has the useful side effect that `consultancy_144` does double duty: it is the nested control for `consultancy_bits_144` *and* consultancy at the debate arms' exact mask search, so debate-vs-consultancy is search-matched there. Both runs are below the base sweep's 1296-mask search, so absolute numbers are not directly comparable to the D3 table; every comparison that matters is internal.

**Baselines.** Alongside the arms, each table carries a few reference points, all on the same judge architecture:

- *gold* — the judge on the full image, with no mask at all: the ceiling.
- *random mask (1)* — the judge on one uniform 4/36 mask, with no agent and no claim input: the floor.
- *consultancy* — $A$ commits a class and shows the single mask that best supports it; the judge sees that mask and $d_A$, and there is no adversary.
- *random mask (ensemble)* — the judge's probabilities averaged over $M$ random masks, $\arg\max_c \frac{1}{M}\sum_m p[m,c]$: no agent and no selection, just averaging.

The arms do not all search the same number of masks — on FashionMNIST-whitened the debates search 256 and the consultancy runs 324 and 1296 — so the ensemble is computed once at each of those numbers. Every arm then has an ensemble row that saw exactly as many masks as it did, and the gap between the two is what choosing the masks was worth against not choosing at all. The final ensemble row averages over all $\binom{36}{4} = 58{,}905$ masks and is the zero-capability endpoint of the protocol family from the main text.

**Metrics and diagnostics.** For every arm we report selection accuracy $\Pr[\hat c = y]$, judge accuracy, and belief (the judge's softmax mass on $A$'s claim), all measured on the test set at the final epoch. The bits arms get three further diagnostics. *Codeword entropy* is the realised entropy of the 4-bit word, out of a possible 4, and says how much of the channel gets used at all. The *randomised-bit ablation* re-scores the *same* selected plays with a random codeword, and says whether the judge conditions on the channel. And $I(b;\text{claim})$ and $I(b;y \mid \text{claim})$, both computed per bit, separate what the bit says about $A$'s commitment from what it says about the truth *beyond* that commitment; we normalise both by $H(b)$ so that bits with different usage rates stay comparable.

**Results — FashionMNIST (per-class whitened).**

![Bit-channel arms vs control and baselines, FashionMNIST whitened](figures/bitchannel_comparison_fashionmnist_normalized.png)

*(The CIFAR-10 version of this figure is below the CIFAR-10 table.)*

| run | masks | judge evals | sel acc | judge acc | belief |
|---|---|---|---|---|---|
| gold (full image) | — | — | — | **95.36** | — |
| random mask (1) | 1 | — | — | 51.35 | — |
| random mask (ens.) | 256 | — | — | 64.36 | — |
| random mask (ens.) | 324 | — | — | 64.46 | — |
| random mask (ens.) | 1296 | — | — | 64.60 | — |
| random mask (ens. **all**) | 58,905 | — | — | **64.57** | — |
| consultancy | 1296 | 1296 | 76.58 | **76.58** | 77.76 |
| consultancy | 324 | 324 | 73.57 | 73.54 | 74.65 |
| consultancy + 4 bits | 324 | 5184 | 72.48 | 72.50 | 75.57 |
| debate 2-turn + 4 bits | 256 | 4096 | 71.98 | 71.93 | 72.40 |
| debate 2-turn control | 256 | 256 | 71.56 | 71.65 | 73.11 |
| debate 4-turn + 4 bits | 256 | 4096 | 63.77 | 63.56 | 64.04 |
| debate 4-turn control | 256 | 256 | 63.84 | 63.76 | 63.71 |

**Results — CIFAR-10 (raw).**

| run | masks | judge evals | sel acc | judge acc | belief |
|---|---|---|---|---|---|
| gold (full image) | — | — | — | **67.42** | — |
| random mask (1) | 1 | — | — | 37.41 | — |
| random mask (ens.) | 144 | — | — | 45.75 | — |
| random mask (ens.) | 1296 | — | — | 46.14 | — |
| random mask (ens. **all**) | 58,905 | — | — | **46.12** | — |
| consultancy | 1296 | 1296 | 46.98 | **47.03** | 46.58 |
| consultancy | 144 | 144 | 44.94 | 45.03 | 44.38 |
| consultancy + 4 bits | 144 | 2304 | 44.67 | 44.70 | 43.99 |
| debate 2-turn + 4 bits | 144 | 2304 | 45.66 | 45.39 | 45.18 |
| debate 2-turn control | 144 | 144 | 45.99 | 45.94 | 45.20 |
| debate 4-turn + 4 bits | 144 | 2304 | 44.33 | 44.03 | 43.02 |
| debate 4-turn control | 144 | 144 | 43.98 | 43.78 | 43.71 |

![Bit-channel arms vs control and baselines, CIFAR-10](figures/bitchannel_comparison_cifar10.png)

**The channel's effect on judge accuracy (bits arm minus its nested control, percentage points).**

| protocol | FashionMNIST-wh | CIFAR-10 |
|---|---|---|
| consultancy | $-1.04$ | $-0.33$ |
| debate 2-turn | $+0.28$ | $-0.55$ |
| debate 4-turn | $-0.20$ | $+0.25$ |


**What the bits encode.** The diagnostics for the six bits arms. Each mutual-information column gives the range across the bits it covers, so the fourth column ranges over $A$'s bits only and the fifth over all four:

| dataset / arm | codeword entropy | randomise → judge acc | randomise → belief | $A$'s bits: $I(b;\text{claim})/H(b)$ | all bits: $I(b;y\mid\text{claim})/H(b)$ |
|---|---|---|---|---|---|
| FMNIST-wh, 2-turn | 3.27 / 4 | $-3.50$ | $-8.67$ | 70–87 % | 0.6–1.8 % |
| FMNIST-wh, 4-turn | 3.71 / 4 | $-2.56$ | $-7.39$ | 71–85 % | 0.6–1.9 % |
| FMNIST-wh, consultancy | 1.31 / 4 | $-0.11$ | $-4.46$ | 22–36 % | 1.5–3.6 % |
| CIFAR-10, 2-turn | 3.55 / 4 | $-1.43$ | $-6.68$ | 83–85 % | 0.8–1.4 % |
| CIFAR-10, 4-turn | 3.49 / 4 | $-1.06$ | $-5.74$ | 73–83 % | 0.8–1.3 % |
| CIFAR-10, consultancy | 2.08 / 4 | $-0.01$ | $-2.05$ | 16–52 % | 1.1–2.8 % |

Per bit, $I(b;\text{claim})$ exceeds $I(b;y\mid\text{claim})$ by 4× to 152×, and by 37–152× on $A$'s bits in the debate arms. $B$'s bits carry far less about the claim than $A$'s (5–35 % of their entropy against 70–87 %).

**Reproduction.** The notebooks, the saved result arrays, the trained judge weights and the figures are at [github.com/emparu/debate-quantilizers-images](https://github.com/emparu/debate-quantilizers-images), together with a script that recomputes every table in this post from the arrays. The judge-probability dumps that the quantilizer sweep runs on are on the Hugging Face Hub at [`eruzak/pixeldebatev1-dump`](https://huggingface.co/datasets/eruzak/pixeldebatev1-dump).
