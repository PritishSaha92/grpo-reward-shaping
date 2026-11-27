| Rule | Std cosine | Std norm ratio | Mean-only cosine | Mean-only norm ratio | Median group energy |
|---|---:|---:|---:|---:|---:|
| Learned-distance shaping | 0.9786 | 1.000092 | 0.9593 | 1.673407 | 1.35% |
| Group-constant penalty | 1.0000 | 1.000088 | 0.9837 | 1.632011 | 0.00% |
| Token-length shaping | 0.9794 | 1.000088 | 0.9325 | 1.741786 | 0.56% |

Fixed post-hoc sample: 500 prompts, eight rollouts each; per-group metrics on 185 mixed groups. Length uses one coefficient calibrated to the eligible-rollout mean learned penalty. Global metrics include prompt weighting; per-group angle is invariant to positive normalization. No retraining.
