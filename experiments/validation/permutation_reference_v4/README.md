# Post-hoc permutation-reference and presentation amendment

Recorded 27 November 2026, after the original CPU comparison, assignment extension,
and completed 32-group gradient replay were inspected. This amendment adds a
standard permutation-covariance identity and descriptive summaries of already
observed draws. It is neither preregistered new model evidence nor a permutation
significance test. No sample, checkpoint, coefficient, or permutation is selected.

Scope:

1. Recover residual magnitude/alignment and finite-draw rank counts from the
   completed replay's direction Gram, independently of its formatted metrics.
2. Check that full and residual shuffle differences have the same numerator.
3. Exhaustively enumerate small synthetic groups to test the second-moment and
   squared-difference identities, including independent groups and batch scaling.
4. Retain the counterexample to a zero expected-cosine claim.

From the paper directory, run:

```
python validation/permutation_reference_v4/validate_reference.py
```

The portable evidence archive uses:

```
python validation/permutation_reference_v4/validate_reference.py --result gradient_replay/results/result.json --output new_reference_check.json
```

The exact **model** permutation expectation cannot be recovered from the aggregate
Gram. Its per-group completion-gradient scatter is missing. The prepared 185-group
GPU extension will collect that scatter; it has not been executed and supplies no
results here. The reported 0.7648 uses the empirical mean across the 32 stored draws,
not the exact uniform-permutation expectation. A smaller residual norm or closer
gradient is not a measure of better learning.
