# Populations and interpretation

| Population | Size | Used for |
|---|---|---|
| Fresh saved MATH-500 generation | 500 prompts × 8 completions | Offline reward/advantage audit |
| Mixed groups from that sample | 185 groups; 1,480 completions | Complete model-gradient replay |
| Groups permitting within-incorrect variation | 132 of 500 | Conditional residual-energy distribution |
| Single-incorrect mixed groups | 53 of 185 | Structural zero-residual sanity check |
| Oracle-conditioned diagnostic | 1,460 scorable completions | Pooled AUROC 0.711 |
| Within-prompt diagnostic pairs | 1,944 pairs in 165 prompts | Pair-weighted AUROC 0.618 |
| Historical training coverage | 2,000 groups of 16 | 163 all-wrong, 1,570 mixed, 267 all-correct |
| Earlier gradient pilot | 32 groups | Labelled subset analysis only |

The fresh generation sample has 47 all-wrong, 185 mixed and 268 all-correct
groups. Its 26.4% opportunity count is a property of this final-checkpoint test
sample at group size eight, not a training-time statistic.

Distances use correct in-group references. Shaping is unavailable in all-wrong
groups, and a single incorrect completion has no within-incorrect variation.
Only incorrect completions receive direct penalties, but normalization can
change every completion's advantage in an eligible group.

The historical task comparison uses one training seed. The nearest-expert
comparator changes the reference pool and the optimization route. Its outcome
cannot isolate an objective effect. The recovered learned-head code also has
the documented gradient-clearing/clipping defect.

The original evaluation had very little **parsed-answer** diversity: 484 of
500 prompts had one parsed answer across eight samples. Missing raw completions
prevent a claim about identical token sequences or a verified cause of the
evaluation discrepancy.

The replay uses recovered, pinned weights, reconstructed prompt tokens and saved
completion tokens. It computes fixed-checkpoint, sequence-mean LoRA score
gradients. It is not an unbiased on-policy estimate or a replay of the original
optimizer step. No policy is updated.

The exact permutation expectation conditions on those gradients and the fixed
multiset of penalties. An observed/expected ratio is descriptive. The 32 draws
reassign learned penalties only. No sampled length-shuffle distribution,
independent performance test, or corrected matched-seed study is supplied.

The work separates correctness discrimination, surviving advantage variation,
and the consequences of assigning that variation to particular completions.
It does not identify which assignment improves reasoning.
