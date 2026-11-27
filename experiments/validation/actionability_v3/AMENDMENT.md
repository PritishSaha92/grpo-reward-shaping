# Third post-result amendment: assignment invariance and alignment floors

Recorded 26 November 2026, before running this amendment. The original CPU comparison and the two earlier validation amendments have already been inspected. This document does not amend their immutable seal or claim preregistration.

## Scope

1. Validate that the within-group cosine equals the Pearson correlation between nonconstant shaped reward and binary correctness, and that its squared complement equals the within-class residual sum of squares divided by total centered shaped-reward energy.
2. Validate invariance under permutations of fixed realized shaping values within each prompt and correctness class. Check per-group cosines and pooled cosine, norm ratio, relative difference, global energy, within-group energy and prompt-reweighting energy. The fixed binary vector and its group normalization are preserved. This says nothing about permuting input text and recomputing a scorer, or about model-gradient invariance.
3. Validate the residual gradient contrast identity and the full difference including aligned rescaling, with fixed synthetic score-gradient vectors. Include a deterministic example where a within-class permutation preserves all geometry but changes the residual gradient. This is algebra, not a model-gradient replay or an estimate of task benefit.
4. Add a new, explicitly post-hoc descriptive summary of the coefficient-independent cosine floor for learned distances and stored token length. This is a new summary of existing data, not merely validation of an already reported result.

## Data and calculations

Use all 4,000 rows of runs/cpu_v1/joined_rollouts.jsonl, preserving group and sample IDs. Expected ledger SHA-256: c74f8705d3df940ea7f697d36747d2525ee187152ed709a5ca3e54a6a223a4a0. Use all 185 mixed groups; exclude the 315 homogeneous groups because their binary-centered vector is zero. Do not filter by correlation, angle, group size beyond mixedness, or score magnitude.

For d equal to deployed_distance or stored_generated_token_count on incorrect completions, compute population mean mu0, population SD sigma0, CV0 = sigma0/mu0 and limiting cosine F = sqrt(p/(p+CV0^2)). A nonnegative all-zero score has cosine one at every coefficient: record CV as undefined and F = 1. A single incorrect completion has zero SD and F = 1. Negative/nonfinite values fail the run.

Save every group's values and summarize count, minimum, quartiles (NumPy linear convention), median and maximum of F, plus zero-mean and single-incorrect counts. These are finite-sample descriptive summaries, with no confidence intervals, significance tests, selected coefficient or training claim. The median F is not a bound on concatenated-vector cosine.

Verify the finite-coefficient formula and bound at beta = 0, 0.1, 1, 10, 1000000. Verify the beta-to-infinity vector limit for positive-mean scores. These coefficients are identity-check fixtures, not an experimental sweep or a performance search.

## Validation conventions

Use float64; random generator seed 20261126; absolute tolerance 1e-10 and relative tolerance 1e-8. Check synthetic mixed groups for sizes 2 through 32 and all possible nonhomogeneous class counts, with signed, incorrect-only and correct-only shaping, as well as explicit negative-s and zero-s cases. Exclude zero-norm shaped vectors from cosine checks and record those exclusions. Check mean-only centering and population-SD standardization with epsilon 0 and 1e-4. Use one within-class permutation per synthetic case and per saved group/rule with the fixed generator. The original three primary rules plus the per-group length sensitivity are included for saved-ledger invariance checks.

Reports record input, amendment and code hashes and check counts. No original input, original result, protocol or seal may be edited. No GPU, network, retraining or new decoder samples are required. Do not interpret a passing identity as evidence of useful behavior change.
