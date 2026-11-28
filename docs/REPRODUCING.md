# Reproducing the saved analyses

Run the commands in the root README from a Python 3.12 environment. Only NumPy
and SciPy are needed for the standard path. Dependency installation uses the
network; the reproduction itself does not.

`scripts/reproduce.py` first verifies the repository manifest, then copies the
experiment files into a fresh output directory. All computations run on that
copy. It refuses to overwrite an existing output directory and writes a log for
each stage. A failure stops the workflow immediately.

## Stages

1. Run synthetic tests of normalization, coverage and the pooled decomposition.
2. Recompute the learned, group-constant and globally calibrated length rules
   under population-SD normalization and mean-only centering; retain the
   prespecified sensitivity calculations.
3. Independently validate all 4,000 saved token records, rollout outputs,
   per-group metrics and rank associations.
4. Recompute learned–length comparisons and the within-outcome reassignment
   controls under both normalizations.
5. Reconstruct the full replay comparisons from the saved basis/direction Gram
   matrices, and recompute each rule's exact expectation from group scatter.
   Reproduce the float32 coefficient-cast diagnostic and run the complete
   reporting pipeline on synthetic gradients.
6. Verify every completion's token accounting, all 210 reference comparisons,
   the group contributions and the labelled pilot/full-sample comparison.
7. Recompute the earlier pilot arithmetic separately.
8. Check the general decomposition, coefficient-independent floor and exact
   permutation identity on synthetic fixtures and saved groups.
9. Verify the motivating AUROC/geometry, training coverage, task means and
   fixed-seed paired-bootstrap/McNemar comparisons.

The final `verification.json` reports structured output comparisons and
explicitly states the scope. Original training and model backpropagation are
not part of this CPU command. The paired comparison repeats the saved
estimator's 20,000 resamples and seed zero; it does not introduce a new
significance test or training-seed evidence.

## Comparison policy

Input integrity uses exact SHA-256 hashes. Regenerated JSON, JSONL and CSV are
parsed before comparison, accepting Windows or Unix line endings. Keys, array
order, row counts, integer counts, identifiers, strings, booleans and nulls must
match exactly. Finite floating-point outputs use relative tolerance `1e-9` and
absolute tolerance `1e-12`; the latter permits numerical roundoff near zero.
This tolerance checks computational agreement, not statistical uncertainty.
Each file's report includes maximum absolute/relative error and a separate
byte-identity flag. No committed input, result or historical seal is normalized
or rehashed to make a failed calculation pass.

Run `python scripts/test_compare_outputs.py` to check that line-ending changes
and small numerical roundoff pass, while changed identifiers, counts, structure
and material numerical differences fail. The GitHub Actions workflow runs the
complete CPU path on Windows, Ubuntu and macOS after the repository is pushed;
adding that workflow is not evidence that all three runs have already passed.

The tested execution time and output count are recorded in
[VERIFICATION.md](VERIFICATION.md). Allow extra time on slower disks: input
hashing and copying contribute to runtime.

## Figures

The supplied PDF figures are vector exports; SVG sources are in
`figures/editable/`. To regenerate all nine figures in a fresh directory:

```bash
python -m pip install -r requirements-figures.txt
python scripts/render_figures.py --output regenerated-figures
```

The generator uses repository-relative inputs and does not modify the supplied
figures. PDF metadata and SVG identifiers may differ across plotting-library
versions; the numerical data, axes and plotted marks should agree.

## Integrity and export provenance

`checksums.json` covers all release files except itself. Git is configured to
preserve bytes across operating systems. The original historical seals are
retained under `provenance/original_seals/`; export seals and original seals have
different roles. Do not interpret a newly generated hash as proof that a result
was unknown at the time of the original experiment.
