# Post-result theory validation amendment

Recorded 26 November 2026 before running this validation. This is an identity and implementation check prompted by manuscript review, not a new empirical experiment or preregistration.

Preserve the original sealed CPU experiment, all cohorts, coefficients, comparisons, and outputs. Validate the generalized additive-shaping decomposition for mixed groups of sizes 2–32, arbitrary signed shaping, negative/zero/positive outcome coefficients, and the zero-shaped-vector exception. Check incorrect-only penalties, correct-only bonuses, their closed-form angles, and both population standardization (with and without the stabilizer) and mean-only centering.

Check the fixed-feature gradient identity algebraically and by central finite differences of an explicit clipped surrogate at ratio one. This is a synthetic surrogate check, not a model-gradient replay or evidence about historical optimizer steps. Check that group-constant incorrect penalties give identical gradients with ideal standardization, while their mean-only magnitudes change.

Use deterministic seed 20261126, float64, absolute tolerance 1e-10 and relative tolerance 1e-8 for identities; the finite-difference check uses a predeclared absolute tolerance 1e-7, relative tolerance 1e-6 and step 1e-6. Save input/source hashes and all failures. Also count strictly positive learned penalties in the unchanged ledger. Do not fit or test additional empirical hypotheses.
