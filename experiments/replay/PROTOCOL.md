# Complete mixed-group replay and exact permutation reference

Decision date: 26 November 2026, Asia/Kolkata. This post-hoc extension follows
inspection of the 32-group replay and its review. It is not a new preregistration
of the historical experiment. Freeze this protocol, source, inputs and runtime
before obtaining any new 7B model gradients.

## Scope and stopping rule

Use all 185 mixed groups (1,480 saved completions) from the unchanged 500-prompt
ledger. Exclude the 47 all-wrong and 268 all-correct groups because their task
advantages under this particular gate are zero. Retain all 53 single-incorrect
groups and any realized constant scores. No training, decoding, retuning, new
dataset or checkpoint selection. The paper is a read-only input for this task.

Run only on authorized Thor GPU 0, after the user confirms it is available.
Other users' jobs must remain untouched. Reuse the existing isolated environment
and hashed base/adapter files. A failed scientific check stops the run; do not
silently change samples, weights, masks, precision or estimator to obtain a result.

## Frozen inputs

- Use the original ledger SHA-256
  `c74f8705d3df940ea7f697d36747d2525ee187152ed709a5ca3e54a6a223a4a0`.
- Use its calibrated length coefficient, 0.0007621985617683552, without refitting.
- Reuse all 32 original CPU within-incorrect permutation maps for every group;
  do not draw fresh controls or select them using model outcomes.
- Groups run in sorted group-ID order; all eight sample indices stay in order.
- Reuse the exact previously frozen prompt-template source and tokenizer. Encode
  the saved problem text, without adding special tokens. Original prompt IDs
  remain unavailable. Preserve every saved completion token. Drop only the
  minimum left prompt prefix needed to fit the same 4,096-token context, recording
  every alteration. Reject missing, duplicate or mismatched records.
- For the existing 32 groups, require token records and weight plans to match the
  previous replay exactly. This is an input check, not selection by outcome.
- Adapter revision: `9a7fa0c034c1884f141563611bb52c795e1e466f`.
  Base revision: `b101308fe89651ea5ce025f25317fea6fc07e96e`.
  Preserve model, tokenizer and source hashes and verify all weights on the server.

## Estimator and controls

For each completion compute h_i, the gradient of the untempered mean stored-token
log probability with respect to the same 161,480,704 LoRA parameters. Use identical
causal masks, SDPA attention, BF16 base computation, FP32 LoRA parameters, disabled
dropout, no KL/head loss and no optimizer. This is a fixed-sample replay with
reconstructed prompts, not an on-policy estimate or historical-update reproduction.

Every rule uses the same h_i, with common divisor N=1480 (conditional mixed-group
completion average). The all-500-group average would be 1480/4000 times these
gradients, since the omitted task-advantage vectors are zero. Cosines, relative
norm ratios and observed/expected squared-norm ratios are unchanged by that factor.
The earlier 32-group absolute norms used N=256 and are not directly comparable.

Primary advantages use population SD plus 1e-4. Compare binary, learned distance,
within-group constant penalty, globally calibrated length and all 32 fixed learned
penalty permutations. Define the aligned and within-outcome residual components
separately in each prompt. Accumulate the same 38 basis directions as before,
reusing each backward pass for all 70 derived full/residual directions.

## Exact random-assignment reference

For m_j incorrect completions in group j, W_j denotes the incorrect entries of
its class-centered, normalized residual. Their sum is zero. Define

    S(pi) = (1/N) sum_j sum_i W_j[pi_j(i)] h_ji,
    scatter_j = sum_i ||h_ji - mean_wrong(h_j)||^2.

For independent uniform within-incorrect permutations,

    E ||S(pi)||^2 = (1/N^2) sum_{j:m_j>=2}
                       ||W_j||^2 / (m_j-1) * scatter_j.

Compute the reference for learned AND length residual weights, using the same
gradient scatter. A single-incorrect or zero-residual group contributes exactly
zero by definition. Collect the per-group count, sum of individual squared
gradient norms and squared norm of the gradient sum. Use a float64 streaming
Welford vector mean/M2 calculation for stable centered scatter, and independently
check the subtraction identity. No full per-completion gradient matrix is needed.

The expectation is over the finite permutation population, conditional on the
fixed computed gradients and weights; numerical evaluation still has finite
precision. It is not a population learning claim. E[S]=0 gives zero expected dot
product with a fixed direction, NOT generally zero expected cosine. Also report
E||S(pi)-S_learned||^2 = E||S(pi)||^2 + ||S_learned||^2. Full shuffled-minus-learned
and residual shuffled-minus-learned differences have the same numerator.

## Prespecified reports

1. All full/residual gradient cosines, norm ratios and relative differences from
   binary and from learned, with every permutation retained and undefined cases
   explicit. Report min/Q1/median/Q3/max, without p-values or usefulness rankings.
2. Learned and length residual norms relative to binary; their cosine with binary
   and with each other. Describe norm ratios as norms, not energy or learning gains.
3. Exact expected squared residual norms, square-root reference scales and each
   observed squared norm divided by its own rule's exact expectation. Undefined
   zero-denominator cases remain explicit.
4. All 32 shuffled residual squared norms and their descriptive mean relative to
   the exact expectation; no requirement that a finite Monte Carlo mean equal it.
5. Exact expected shuffled-minus-learned squared difference alongside the finite
   draw mean. The sample distribution is retained, not replaced by the expectation.
6. Counts of shuffles whose residual norm exceeds the learned norm and whose
   learned-reference difference exceeds the length-rule difference. These are
   post-hoc descriptive outputs specified before this extension's gradients.
7. Every group's scatter and exact-reference contribution, including all zero
   structural cases, plus all prompt-token trims and source/runtime identities.

## Validation and numerical implementation

- Exhaustive small-group permutation enumeration verifies the second moment,
  independent-group summation, m=1/constant cases, common scaling and the expected
  squared-difference identity. Retain a counterexample to zero expected cosine.
- Test streaming scatter against explicit centered matrices in float64, including
  highly aligned vectors, and synthetic full-cohort weighting against direct sums.
- Reuse tiny real Qwen/LoRA checks for causal token masks and weighted backward
  linearity. Test the new statistics and checkpoint serialization together.
- FP32 basis accumulation matches the earlier estimator; float64 chunked inner
  products produce the Gram matrix. The reference uses original float64 residual
  weights; quantify FP32 cast drift rather than describing arithmetic as symbolic.
- Canary: the first sorted group with >=2 incorrect samples plus the first with
  exactly one, in sorted order. This deterministic pair exercises both branches.
  It is separate from the primary run and cannot change the chosen cohort.
- Save atomic alternating bank checkpoints every eight complete groups, including
  a snapshot of all completed group statistics. Publish the progress pointer only
  after both bank and statistics files are hashed and durable. Exact input/source/
  runtime identity is mandatory for resume. No deletion of old replay artifacts.
- Verify a completed result independently from the stored Gram and group-statistic
  records after local retrieval; verify transfer and all sealed-file hashes.

## Resource and communication plan

Use the already authorized GPU UUID and eight CPU threads. Bind the device by UUID
and refuse launch when it has any compute process. Use a project lock shared with
the previous replay. The 38-direction CPU bank is about 22.86 GiB, plus one float64
mean vector (~1.20 GiB) and temporary chunks. Reuse models and environment without
copying their weights. Reserve space for alternating bank states and one temporary
state, plus headroom. Small results are fetched locally; large banks stay on Thor.

No GPU work starts while GPU 0 is occupied. The user will notify availability.
After launch, monitor the canary and several primary groups. If the primary needs
over 30 minutes, provide PID, log, progress and resume instructions for handoff,
then inspect and fetch the result when the user reports completion. Do not call a
launched or merely prepared run complete. Maintain EXECUTION_LOG.md throughout.
