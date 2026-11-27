# Assignment-sensitive extension: protocol v1

Decision date: 26 November 2026 (Asia/Kolkata).

## Status and purpose

This is a **post-hoc extension**, specified after reading the existing audit and
review suggestions. It is not part of the original preregistration. Its new
numerical comparisons will be inspected only after this protocol, the source,
configuration and input hashes have been sealed. The existing `cpu_v1` run and
paper are read-only inputs. No claim of improved learning follows from this work.

The questions are:

1. How similar are the learned and length rules to each other, rather than only
   to binary advantages?
2. How much within-outcome variation remains when groups with no structural
   opportunity are separated out?
3. With a fixed model and completions, does assigning the same penalties to
   different incorrect completions change the measured score-function gradient?

## A. Complete now: CPU analysis of the existing ledger

Use all 500 groups / 4,000 completions in the sealed `cpu_v1` joined ledger.
Verify every file in the original seal before and after the analysis. The ledger
hash is `c74f8705d3df940ea7f697d36747d2525ee187152ed709a5ca3e54a6a223a4a0`.
Use sample indices 0–7 in each group; refuse missing or duplicate records.

Reuse the original learned penalty, and the original globally calibrated length
coefficient, without refitting. The group-constant rule uses the mean learned
penalty among that group's incorrect completions. All three rules are inactive
outside mixed groups. Preserve the original token-length definition, including
saved stop tokens; termination status and effective generation caps remain
unknown. This extension does not reinterpret those lengths as uncensored.

Primary normalization is population standard deviation plus epsilon=1e-4.
Mean-only centering is a prespecified companion analysis. Do not add comparisons
based on whichever normalization gives a more appealing result.

For each group j and rule m, form binary A_j and shaped B_j^m, then

    alpha_j^m = <A_j, B_j^m> / ||A_j||^2
    W_j^m = B_j^m - alpha_j^m A_j.

W is the **within-group** residual, not the residual of a single projection of
the concatenated batch. For numerical stability also compute W by subtracting
the mean shaped advantage within each correctness class, and check agreement.
Use float64. Residual norms at most 1e-12 times max(1, ||B||) count as numerical
zero; report undefined comparisons explicitly rather than setting cosine to 0.

### Prespecified outputs

- Opportunity counts with both denominators: all 500 groups, 185 mixed groups,
  and mixed groups with at least two incorrect completions. These are structural
  opportunities, not evidence that the score changes the gradient.
- Direct cosine and relative difference of concatenated learned and length
  shaped advantages, over all mixed groups.
- Direct cosine of concatenated W_learned and W_length. Include their norms and
  norm ratio; global shaped-advantage similarity can hide residual differences.
- Per-group direct shaped-advantage cosine, residual cosine, and signed energy
  difference E_learned - E_length. Retain every group and undefined-case reason.
- Minimum, Q1, median, Q3 and maximum (NumPy linear quantiles) of these values,
  both across all mixed groups and conditional on at least two incorrect
  completions. Keep opportunity groups with constant scores; do not exclude
  them from the energy distribution.
- Conditional and unconditional energy distributions for each rule, shown by
  empirical CDFs generated from the records. No fitted or schematic density.
- A compact table of existing c, rho, D and the within-group/prompt-weighting
  energy split, regenerated as a compatibility check rather than a new result.

No p-values, significance tests, bootstrap intervals, post-hoc threshold search,
selection of favorable groups, or performance claims will be added. These are
descriptive measurements on one fixed audit sample.

### Within-incorrect permutations

Generate 32 independent uniform permutations for each mixed group's incorrect
sample indices. Use NumPy PCG64, master seed 20261126, sorted group IDs and
shuffle index order. Save the actual destination-to-source mappings. Single-
incorrect groups retain the identity; identity draws in other groups are valid
and are not redrawn. No correct/incorrect label, token, or completion is moved.

Check all of c, rho, D, E, E_within and E_weight for invariance under every
permutation, for both normalization conventions. Verify the local residual is
permuted in the same way. This validates an identity; it is not 32 independent
replications of training or evidence that an assignment is useful.

## B. Prepare now, execute only on an authorized GPU: model-gradient replay

The user selected preparation only on 26 November. Do not start paid GPU jobs.
Do not use a quantized approximation to fit the local 4 GB GPU and describe it
as the same experiment.

### Fixed sample and provenance

Use the same 32 IDs already recorded in `evidence/recorded/gradient_replay.json`,
in that recorded order, retaining all eight completions. The selection is fixed
before inspecting new model gradients. Report how many of these groups permit
nontrivial within-incorrect permutations. Use all 32 saved shuffle mappings,
including identity mappings; do not optimize or select them using gradients.

The old ~0.9813 replay is a pilot, not a new verified baseline. Its checkpoint
and base revision were not hashed. Restoring a repository now pins a new replay;
it cannot retroactively establish which files generated the historical data.

Candidate adapter: `REPLAY_ADAPTER`. Before execution:

1. Authenticate locally if needed; never copy tokens into this directory or logs.
2. Resolve adapter and base model (`Qwen/Qwen2.5-Math-7B`) to immutable commits.
3. Locate the intended final adapter within the repository and inspect its
   config. Do not silently substitute another checkpoint or the contrastive arm.
4. Hash the actual downloaded config, weights and tokenizer files. Pin the
   runtime/container and record all library versions and CUDA/GPU information.
5. Use saved completion token IDs. Historical prompt token IDs are absent.
   Reconstruct prompts from the saved problem text using the documented cot-
   4shot template and the pinned tokenizer, save their exact token IDs, and
   label them **reconstructed**, not recovered generation-time inputs.
6. Preserve max context 4096: if necessary drop the minimum number of prompt
   tokens from the left, preserving at least one prompt token and every stored
   completion token. Record the count and each altered prompt. Never truncate
   completion tokens silently.
7. Freeze a replay input manifest before the first model-gradient run. If the
   adapter, prompt construction or runtime cannot be pinned, stop preflight.

This replay uses stored distances; it does not retrain the head or reproduce
their extraction. A frozen VAE is unnecessary to differentiate the policy with
fixed scores. Keep the score ledger provenance separate from model provenance.

### Estimator

At fixed parameters theta, compute h_i = grad_theta mean_t log pi_theta(y_it |
prompt, previous completion tokens), once per completion, only for LoRA
parameters. Use untempered model logits, completion-token mean, no padding
positions, no policy update, no optimizer, no dropout, and no KL or head loss.
Use the identical h_i for every weighting rule. The historical sampler used
temperature 0.6; these stored samples and reconstructed prompts do not produce
an unbiased on-policy estimator. Call it a **fixed-sample completion-mean
score-function gradient replay**.

Primary GPU replay uses standardized advantages with epsilon=1e-4. The same
frozen h_i can support mean-only coefficients later, but that is not necessary
for the primary assignment comparison and must not double GPU work now.

Compare binary, learned, group-constant, globally matched length, and all 32
within-incorrect shuffles. Also report the residual-gradient contributions and
the aligned components. All gradients use a common factor 1/256, so reported
norms are means over the fixed completions, not unnormalized sums.

For each rule report cosine, norm ratio and relative difference versus binary;
for shuffled rules additionally compare with original learned shaping and its
residual gradient. Report all draws and descriptive min/Q1/median/Q3/max. No
permutation p-value: there is no prespecified external utility target here.

Implementation: accumulate the shared outcome-aligned learned component once,
and the residual contributions for learned, length and the 32 shuffles. This
avoids recomputing model backward passes or storing every completion gradient.
Use float32 accumulators, float64 final inner products in chunks, and report
numerical checks. Group-constant shaping uses the actual stabilized advantages;
its tiny magnitude differences are not treated as exact ideal invariance.

### Validation and resource gates

- CPU synthetic linear-map tests must agree with direct matrix multiplication;
  test a permutation that preserves geometry but changes a synthetic gradient.
- Before a GPU run, verify all hashes, indices, weight reconstruction and finite
  coefficients. Refuse a mismatch or missing pin.
- GPU canary uses the first fixed group, checks finite nonzero gradients, LoRA
  parameter membership, token alignment and peak memory. It is an infrastructure
  check, not a basis to change the sample or rules. Save it separately.
- Save progress and full accumulators after each complete group. Resume only
  with the same inputs, source, runtime and parameters. Avoid losing progress.
- A successful short run is monitored to completion; a job expected to exceed
  30 minutes is handed back after several minutes of healthy progress, with its
  exact job ID/log and next command. No automatic expensive training campaign.

## C. Interpretation and later studies

A gradient difference establishes assignment sensitivity at this checkpoint and
sample. It does not establish that the learned assignment is better. Testing
that requires independent evaluation after controlled updates. Corrected
multi-seed training, other checkpoints, binary-trained policies and k=16 groups
are separate prospective extensions requiring compute and new specifications.
Do not combine different prompts to imitate k=16; do not reuse parent-group
nearest-reference distances after changing a group's reference set.

## Deliverables

- This protocol, input/source seal, execution log and reproducible commands.
- CPU summary, per-group records, permutation maps, validation report and plots.
- A portable, preflighted GPU runner, fixed weights/sample plan and a runbook
  documenting unresolved authentication, pinning or resource requirements.
- A concise results report distinguishing completed CPU evidence, synthetic
  software validation and model-gradient results that have not yet been run.
- An original-input integrity check at completion. No manuscript changes yet.
