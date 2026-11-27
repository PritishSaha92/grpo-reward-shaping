# MATH4AI 2027: CPU experiment protocol for the credit-assignment audit

Prepared: 26 November 2026, Asia/Kolkata.  
Authors of the planned paper: [research contributor] and [research contributor].  
Status: **analysis specification; the new comparisons have not been run.**  
Scope: one offline analysis package for the proposed four-page workshop paper. No new model training, generation, or GPU replay is required.

This document records the agreed experiments, exact estimands, checks, outputs, and interpretation boundaries. It accompanies paper writing; it is not a reason to postpone the draft until another experimental campaign is complete.

## 1. Scientific question and scope

The question is:

> On a fixed set of sampled solutions, where does a correctness-gated penalty introduce additional within-group credit, and how do centering and standardization alter its direction, magnitude, and weighting across prompts?

The study is an **offline counterfactual advantage analysis**. Correctness labels, generated solutions, and learned distances are fixed. Only the reward construction and normalization are changed. It does not estimate the task performance of a policy trained with the counterfactual rewards.

The existing dataset and earlier audit results have already been examined. Timestamping this protocol makes the new comparison auditable; it does not make the historical dataset an unseen confirmatory sample. Describe the work as a prospectively specified extension of a post-hoc audit, not a newly preregistered training experiment.

### Required analysis package

| Component | Required work | Main purpose |
|---|---|---|
| Reward comparison | Learned-distance penalty; group-constant counterpart; globally calibrated token-length penalty | Compare additional credit under three precisely defined constructions |
| Normalization comparison | Population-standard-deviation normalization with the recorded stabilizer; mean-only centering | Separate direction, magnitude, and prompt weighting |
| Structural opportunity | Count mixed groups with exactly one incorrect rollout | Quantify a case where no within-incorrect variation is possible |
| Length association | Within-prompt Spearman correlation between deployed distance and token count among incorrect rollouts | Check association with a simple observable feature |
| Length provenance and censoring | Verify stored-token semantics, limits, missing termination reasons, and ties | Make the length comparison interpretable |

These are components of one dataset analysis, not five training experiments. No success threshold, acceptance prediction, or task-performance hypothesis is attached to the new metrics.

### Explicitly outside this submission's experiment scope

- New training seeds, models, encoders, benchmarks, or hyperparameter searches.
- Inference-time reranking or selected-top-1 evaluation.
- New extraction of model latents or new model-gradient replay.
- A claim that length shaping or learned-distance shaping improves learning.
- A claim that normalization caused the historical performance result.
- A claim that all learned shaping fails, or that all continuous rewards need within-incorrect variation under every possible objective.

The later empirical study remains three matched training seeds each for binary-reward GRPO, corrected learned-distance shaping, and route-matched fixed-L2 shaping. The nearest-expert pool question is separate. These are not prerequisites for executing this CPU protocol and must not be presented as completed evidence.

## 2. Reader-facing terminology

Use the following names in the paper and tables. Keep internal identifiers only in artifact mappings.

| Paper name | Artifact identifier / meaning |
|---|---|
| Binary-reward baseline | Correctness rewards, evaluated on the same fixed samples; not a newly trained baseline |
| Learned-distance shaping | Historical `G7` deployed-distance rule reconstructed on its post-hoc re-decode |
| Group-constant penalty | Within each mixed group, replace incorrect-rollout penalties by their mean |
| Token-length shaping | Incorrect-only, mixed-group-gated penalty using a single globally calibrated coefficient |
| Per-group matched length sensitivity | Secondary length construction using each group's learned mean penalty |

Do not describe the primary token-length rule as an ordinary, ungated length penalty. It deliberately preserves the historical gate. Its absence of all-wrong coverage is imposed by this comparison and is not an inherent limitation of length-based rewards.

## 3. Source artifacts and provenance

Workspace root: `<workspace>`.

Paths below are relative to that root unless otherwise stated.

| Role | Existing file |
|---|---|
| Full re-decode, including completions and token IDs | `LaViDA/eval_results/oracle_phase3_seed0_n8/g7_principled_cr_seed0_math500_n8_redecode/math500_pass8_results.json` |
| Recorded decoder configuration | `LaViDA/eval_results/oracle_phase3_seed0_n8/g7_principled_cr_seed0_math500_n8_redecode/config.json` |
| Recorded evaluation summary | `LaViDA/eval_results/oracle_phase3_seed0_n8/g7_principled_cr_seed0_math500_n8_redecode/summary.json` |
| Deployed and diagnostic distance records | `grpo-mechanism-audit/evidence/inputs/geometry_records.json` |
| Blog's correctness/prediction projection | `grpo-mechanism-audit/evidence/inputs/G7_redecode.json` |
| Existing input provenance | `grpo-mechanism-audit/evidence/input_hashes.json` |
| Existing reconstructed advantages | `grpo-mechanism-audit/evidence/derived/groups.json` |
| Existing audit aggregates | `grpo-mechanism-audit/evidence/derived/coverage_advantage.json` |
| Existing implementation snapshot | `grpo-mechanism-audit/evidence/estimators/analyze_coverage_advantage_novelty.py` |
| Existing generic CPU companion | `grpo-mechanism-audit/companion/audit_rewards.py` |

### Observed hashes at preparation time

Recalculate these before execution. A mismatch is a provenance failure to investigate, not a reason to silently replace the expected value.

| File | SHA-256 |
|---|---|
| Full re-decode JSON | `d5a29873bede905e4501a365677793c65325e032aec6f426762053a003cc27bd` |
| Geometry records JSON | `ea070a0281d6dab2d9f3a08745c2d0df19eccc4cea5f6506ee6e14af14e4742e` |
| Decoder config JSON | `da2377b9e8d4b3bdd0d2a52405dc676aff5454be9c01a07fff572eb9436dc966` |

The geometry metadata identifies the same raw-results hash. It also records missing local checkpoint/VAE hashes and an unpinned base-model revision. This protocol does not repair that extraction provenance. It reproduces transformations of saved distances and token records, not the end-to-end model extraction.

### Existing, already inspected cohort facts

- 500 prompts, eight sampled solutions per prompt: 4,000 rollouts.
- 185 mixed groups, 47 all-wrong groups, 268 all-correct groups.
- The earlier audit identifies 650 incorrect rollouts within mixed groups; recompute and verify this count during preflight.
- All 4,000 raw samples have nonempty integer token-ID lists in the inspected file.
- Prompt identities, correctness, and parsed predictions matched the blog projection in the prior read-only check.
- The decoder config records `max_tokens = 3072`, `max_model_len = 4096`, and saving of completions/token IDs. These are configured limits, not proof of each sample's actual stopping cause.
- The existing sample objects do not contain a `finish_reason` field.

These facts are not new results of the planned comparison. Do not confuse the 185-group geometry cohort with the smaller, differently selected AUROC cohort, or with the 2,000 historical training groups of size 16.

## 4. Freeze the specification before new outcome inspection

Before producing new rule-comparison metrics or distance-length correlations:

1. Read this protocol and resolve only source-schema, token-semantics, and execution issues.
2. Write a machine-readable configuration with the fixed definitions in Sections 6-12.
3. Preserve a dated copy of this protocol, configuration, input manifest, and analysis source in a new run directory.
4. Hash each of them with SHA-256 and record UTC and local timestamps, Python/dependency versions, and the applicable Git revision plus dirty-file hashes. A parent-repository commit hash alone does not identify uncommitted analysis code.
5. Run synthetic correctness checks before applying the comparison to the real records.
6. Mark the snapshot sealed, then execute the actual analysis.

Do not claim a local hash is an external preregistration timestamp. It is an integrity record for this analysis. Do not amend a sealed snapshot in place. Corrections require a new version, an explanation, and retention of superseded outputs.

If new metrics have already been inspected by the time execution begins, record exactly which ones and when. Do not backdate the protocol or describe those choices as made before seeing results.

## 5. Input validation and canonical join

### 5.1 Stable sample identity

Join by `(group_id, sample_idx)`:

- Raw JSON: `per_problem[*].unique_id`, with `sample_idx` taken from the original order of its eight `samples` entries.
- Geometry JSON: `records[*].group_id` and `records[*].sample_idx`.
- Never join by the array position of prompts, prediction text, or completion text. Identical predictions and duplicate completions are legitimate separate samples.

For each group, require unique indices `0..7`, eight records, matching correctness, and consistent group composition. Require all 500 identities to match. Check the raw file against its projected correctness/prediction records as an additional provenance check.

Use deterministic lexical group-ID order and ascending sample index for all concatenations and output files. Retain the original identity in every derived row.

### 5.2 Required fields

Canonical rollout rows should retain at least:

```text
group_id, sample_idx, correctness, group_type, n_correct,
stored_generated_token_count, token_sequence_sha256,
deployed_distance, learned_penalty,
termination_status, nominal_cap_reached, effective_cap_reached,
effective_cap, cap_provenance
```

`termination_status` must distinguish known length termination, known non-length termination, and unknown. Boolean cap flags need a separate unknown state when their limit is not established.

Hash token sequences using a documented canonical representation, for example UTF-8 bytes of compact JSON integer arrays. Do not hash a Python object representation or platform-dependent text.

### 5.3 Hard validation requirements

- Correctness is genuinely binary; do not coerce strings such as `"False"` with `bool(...)`.
- Eligible deployed distances are present, finite, and nonnegative.
- Use `deployed_distance`, **not** `diagnostic_distance`. Diagnostic leave-one-out scores answer a different question and have a different missingness pattern.
- Noneligible deployed distances are expected to be zero in this source. If a nonzero value appears, investigate rather than silently hiding it with a new gate.
- Token lists are lists of integer IDs, not booleans, and are nonempty. Record validation failures explicitly.
- A missing eligible distance, missing token record, duplicate key, inconsistent sample count, or identity mismatch stops the full paired comparison. Do not silently analyze the remaining subset.
- Preserve the raw files and existing blog outputs; use new output paths.

## 6. Token length and stopping-limit protocol

### 6.1 Primary length definition

Use

\[
L_{pi}=\operatorname{len}(\text{stored generated token IDs}_{pi}).
\]

Use raw counts as stored. Do not retokenize decoded text, subtract a prompt length, normalize within a group, winsorize, log-transform, or remove end tokens in the primary comparison. The coefficient absorbs the token-count unit.

Verify from the exporter or traceable source that the saved IDs represent generated completion tokens. Cite the inspected source/version in the provenance report. Document whether terminal special tokens are present, excluded, or not recoverably specified. The primary operation remains the count of the stored completion-token list.

If the source cannot establish that these are completion-only IDs, stop the length comparison until that ambiguity is resolved. Do not substitute word counts or a guessed tokenizer revision.

### 6.2 Cap and termination investigation

Record these separately:

1. Configured requested generation limit: 3,072 tokens in the saved config.
2. Configured model context limit: 4,096 tokens.
3. Any verifiable per-prompt effective generation limit used by the evaluator.
4. Any saved termination reason or decoder log that can be joined unambiguously.

Do not assume the effective limit is `4096 - prompt_length` without checking the generation code. A framework may reject, truncate, or otherwise handle long prompts differently.

A length equal to a verified cap is **cap-reaching**, not automatically confirmed truncation. A length below the nominal cap is not proof that no smaller effective cap or other stopping rule applied. A final token resembling an end marker is not a reliable substitute for a missing termination reason without verifying exporter semantics.

If stopping reasons or effective caps cannot be recovered, retain explicit `unknown` values. Report the observable nominal-cap counts and the limits of inference. This missing metadata does not invalidate counting stored tokens; it limits claims about censoring.

### 6.3 Required counts

Report numerator and denominator for:

- Known length-terminated, known otherwise-terminated, and unknown samples.
- Nominal-cap-reaching and verified-effective-cap-reaching samples.
- The same counts restricted to eligible incorrect rollouts and to mixed groups with at least one affected sample.
- Within each mixed group, the number of distinct incorrect-token lengths and the number of tied incorrect-length pairs. For aggregate tie fraction, divide summed tied pairs by summed possible incorrect pairs; exclude zero-pair groups from that denominator.

Do not confuse generation termination with the geometry metadata's teacher-forcing prompt-context truncation. That metadata describes dropped prompt context during later distance extraction while preserving completions; it is not a response-generation truncation count. The separate gradient replay has its own scope and truncation notes.

### 6.4 Inclusion and fixed sensitivity rule

Keep every valid sample in the primary comparison, including cap-reaching samples.

If any samples are confirmed length-terminated or reach a verified nominal/effective cap, run one secondary analysis excluding **whole prompt groups** containing such a sample. This preserves eight-rollout groups and avoids redefining correctness composition by deleting individual rollouts.

- Keep the primary globally fitted length coefficient fixed in this sensitivity analysis.
- Compare all rules on exactly the same retained groups.
- Report removed and retained counts, plus remaining unknown termination status.
- Do not call the retained sample "untruncated" when unknown stopping causes remain.
- If the exclusion removes no groups, report that it produces no distinct sensitivity result; do not invent a percentile-based cutoff instead.
- If no mixed groups remain, mark the sensitivity unavailable rather than inventing a metric.

## 7. Mathematical definitions and assumptions

For prompt \(p\), let \(k=8\), \(R_{pi}\in\{0,1\}\), \(\pi_p=k^{-1}\sum_i R_{pi}\), and

\[
g_p=\mathbf 1\{0<\textstyle\sum_i R_{pi}<k\}.
\]

Let \(q_{pi}\) denote the **actual nonnegative subtracted penalty**, including its coefficient, with zero penalty outside incorrect members of mixed groups. Each rule uses

\[
r'_{pi}=R_{pi}-q_{pi}.
\]

For a mixed group, let \(m_p=\sum_i(1-R_{pi})\),

\[
\bar q_{0p}=\frac{1}{m_p}\sum_{i:R_{pi}=0}q_{pi},\qquad
u_{pi}=(1-R_{pi})(q_{pi}-\bar q_{0p}),\qquad s_p=1+\bar q_{0p}.
\]

Then, with \(x_{pi}=R_{pi}-\pi_p\),

\[
r'_{pi}-\bar r'_p=s_p x_{pi}-u_{pi},\qquad x_p^\top u_p=0.
\]

Writing \(v_p=k^{-1}\sum_i u_{pi}^2\),

\[
\cos^2(x_p,r'_p-\bar r'_p)=
\frac{s_p^2\pi_p(1-\pi_p)}{s_p^2\pi_p(1-\pi_p)+v_p}.
\]

The decomposition holds for real-valued incorrect-only penalties. Nonnegativity in this study ensures \(s_p\geq1\), so the aligned component does not reverse sign. With signed penalties, the sign of the cosine follows \(s_p\); if both components vanish, the shaped vector is zero and the angle is undefined. Do not use the squared-cosine formula alone to infer a positive direction in that broader signed case.

If all incorrect penalties are equal, \(u_p=0\). In particular, \(m_p=1\) forces \(u_p=0\). These are algebraic statements about within-group geometry, not claims about the eventual policy update.

Use this penalty-inclusive notation consistently. If the paper instead writes distances \(d\) and a coefficient \(\beta\), then \(q=\beta d\); the residual variance term changes by \(\beta^2\). Do not multiply by the coefficient twice.

## 8. Reward rules: primary comparison

The baseline is \(r_{pi}=R_{pi}\). All rules use the same saved group, correctness, eligibility gate, and sample identities.

### 8.1 Learned-distance shaping

\[
q^{\mathrm{learned}}_{pi}=0.1\,g_p(1-R_{pi})d^{\mathrm{deployed}}_{pi}.
\]

Use the saved deployed distances without rescaling, clipping, rank transformation, or refitting. Verify that the input metadata has the recorded coefficient 0.1.

### 8.2 Group-constant counterpart

For each mixed group, set

\[
q^{\mathrm{constant}}_{pi}=g_p(1-R_{pi})\bar q^{\mathrm{learned}}_{0p}.
\]

This preserves each group's average incorrect penalty, average shaped reward, and correct-incorrect mean gap. It removes the within-incorrect residual exactly. It is a diagnostic construction using the learned score, not an independently deployable learned method.

### 8.3 Globally calibrated token-length shaping

Define the eligible rollout set

\[
E=\{(p,i):g_p=1,\ R_{pi}=0\}.
\]

Fit one coefficient:

\[
\lambda=\frac{\sum_{(p,i)\in E}q^{\mathrm{learned}}_{pi}}
{\sum_{(p,i)\in E}L_{pi}},\qquad
q^{\mathrm{length}}_{pi}=g_p(1-R_{pi})\lambda L_{pi}.
\]

This matches total penalty and therefore mean penalty **over eligible rollouts**. It is not an equal-prompt-weighted calibration. Record both the numerator and denominator, eligible count, resulting coefficient, and unit (reward penalty per stored generated token).

The coefficient is computed once on the full primary cohort and reused under both normalization schemes. Do not calibrate separately per group, subject, correctness fraction, normalization, or selected subset in the primary analysis. Do not select the coefficient to match cosine or maximize novelty.

If the denominator is zero, stop: it conflicts with the validated nonempty token lists. If the learned penalty sum is zero, the coefficient is zero and the comparison is a degenerate no-penalty case; report it rather than choosing a nonzero replacement.

The calculation intentionally uses the audit data to match scale. It does not provide an independently validated training hyperparameter. Matching one global mean does not match per-group mean penalties; report that difference rather than claiming all remaining differences reflect only spread.

### 8.4 Secondary per-group matched length construction

Run this prespecified appendix sensitivity to separate global-calibration effects from within-group spread:

\[
q^{\mathrm{length,group}}_{pi}
=g_p(1-R_{pi})\bar q^{\mathrm{learned}}_{0p}\,
\frac{L_{pi}}{\bar L_{0p}}.
\]

Compute it only for mixed groups; use zero penalties for homogeneous groups. Positive stored lengths make \(\bar L_{0p}>0\). A zero learned mean gives zero penalties.

Label it explicitly as a learned-score-dependent sensitivity construction. It does not replace the global-coefficient row, regardless of how attractive its results look. No coefficient sweep is planned.

## 9. Normalization comparison

For any group reward vector \(r_p\), use float64 and define:

\[
N_{\mathrm{std}}(r_p)=\frac{r_p-\bar r_p}{\sigma_0(r_p)+10^{-4}},
\qquad N_{\mathrm{mean}}(r_p)=r_p-\bar r_p,
\]

where \(\sigma_0\) is the population standard deviation (`ddof=0`). The stabilizer is added **after** the square root, not inside the variance.

For each normalization \(N\), compare \(A_p=N(R_p)\) with \(B_p=N(r'_p)\). Reconstruct the binary baseline separately for each convention. Comparing mean-centered shaped rewards against standardized binary rewards is not the intended experiment.

Homogeneous groups return zero task-reward advantages for every primary rule. Their angle is undefined. No KL or other auxiliary loss is represented here.

### Essential interpretation

- Within a nondegenerate group, positive scalar normalization cannot change the angle. Per-group cosine and orthogonal-energy fraction must agree between the two conventions.
- For the constant counterpart, mean-only centering gives \(B_p=s_p A_p\): unchanged direction, changed magnitude.
- With ideal standardization and zero stabilizer, the constant counterpart gives \(B_p=A_p\).
- With the recorded stabilizer, its per-group norm ratio is

\[
\rho_p=\frac{s_p(\sigma_0(R_p)+\epsilon)}{s_p\sigma_0(R_p)+\epsilon},
\qquad \epsilon=10^{-4}.
\]

- Across prompts, different scaling factors can change the concatenated-vector angle. This is especially important for mean-only centering, but the stabilizer can introduce small relative-weight changes under standardization too.
- Do not conclude that the constant counterpart must have exactly zero **global** orthogonal energy. It must have zero **within-group** orthogonal energy, up to numerical precision.

Ideal zero-stabilizer standardization is a mathematical/unit-check condition, not a third primary normalization column.

## 10. Metrics, denominators, and numerical conventions

### 10.1 Per-group geometry

For each mixed group and rule, with nonzero vectors \(A_p,B_p\), compute:

\[
c_p=\frac{A_p^\top B_p}{\|A_p\|\|B_p\|},\quad
e_p=1-c_p^2,\quad
\rho_p=\frac{\|B_p\|}{\|A_p\|},\quad
D_p=\frac{\|B_p-A_p\|}{\|A_p\|}.
\]

Check the identity

\[
D_p^2=1+\rho_p^2-2\rho_pc_p.
\]

Compute the projection residual directly as an independent check on \(e_p\). The angle in degrees is optional, derived from the same cosine; it is not independent evidence.

Report unweighted prompt-level median, quartiles, minimum, maximum, and defined/excluded counts. Use an explicitly recorded linear-interpolation quantile convention. Each mixed prompt contributes once; do not weight the median by its number of wrong answers.

Per-group directional statistics need only be tabulated once because they are invariant to the two positive normalization scalars. Retain both calculations in the machine-readable output and verify agreement.

### 10.2 Global concatenated-vector geometry

Concatenate all 500 groups in canonical order and calculate the same four metrics, using each normalization's own baseline. Homogeneous groups contribute zero blocks; including or removing them must not change these aggregate metrics.

Do not compute the global cosine by averaging per-group cosines. Do not call global orthogonal energy a mean, median, or percentage of useful learning.

### 10.3 Separate within-group residual from prompt reweighting

Because mean-only centering can alter weights across prompts, include this decomposition as a validation and appendix diagnostic. For every mixed group define

\[
\alpha_p=\frac{A_p^\top B_p}{\|A_p\|^2},\qquad W_p=B_p-\alpha_p A_p,
\qquad \alpha=\frac{\sum_p A_p^\top B_p}{\sum_p\|A_p\|^2}.
\]

Then compute

\[
E_{\mathrm{within}}=\frac{\sum_p\|W_p\|^2}{\sum_p\|B_p\|^2},\qquad
E_{\mathrm{weight}}=\frac{\sum_p(\alpha_p-\alpha)^2\|A_p\|^2}{\sum_p\|B_p\|^2}.
\]

Check

\[
E_{\mathrm{global}}=E_{\mathrm{within}}+E_{\mathrm{weight}}.
\]

This distinguishes changes within groups from variation in outcome-direction coefficients across groups. It is an accounting identity, not a new theorem or an additional training experiment. The constant counterpart must have \(E_{\mathrm{within}}=0\), even when \(E_{\mathrm{global}}>0\).

### 10.4 Coverage and structural opportunity

Report separately:

- All-wrong, mixed, and all-correct group counts over all 500 groups.
- Gate-eligible incorrect rollouts over all 4,000 rollouts.
- Strictly positive penalties among eligible rollouts; do not equate this with eligibility.
- The full histogram of incorrect count \(m_p\in\{1,\ldots,7\}\) among mixed groups.
- The count and percentage of mixed groups with \(m_p=1\), using 185 as the expected denominator after verification.
- The fraction with \(m_p\geq2\), described as an opportunity for within-incorrect variation, not a guarantee of novelty or usefulness.

No incorrect-only nonnegative penalty can introduce within-group orthogonal variation when \(m_p=1\). This does not imply that no magnitude change or learning effect is possible.

### 10.5 Numerical handling

- Compute using float64; keep unrounded values in JSON/CSV.
- For numerical identity checks, use `atol=1e-10, rtol=1e-8`; report the largest discrepancy. Do not loosen tolerance to conceal a conceptual mismatch.
- Clip cosine to `[-1,1]` and energy to `[0,1]` only for floating-point excursions within the numerical tolerance; larger excursions are failures.
- Represent undefined metrics as JSON `null` with an explicit reason and a count, never as 0, 1, or nonstandard JSON `NaN`.
- Treat zero norms explicitly before division. With valid nonnegative primary penalties, every mixed-group comparison should be defined.
- Use exact stored values for positivity and rank ties. Separately flag any values whose interpretation depends on numerical resolution; do not silently round them into ties.
- Percentages in prose use an explicit decimal rounding convention. Do not edit historical source artifacts to change their display rounding.

## 11. Within-prompt distance-length association

For each mixed group, restrict to incorrect rollouts and compare **positive deployed distance** with stored generated-token length. Do not use negative distance, diagnostic AUROC scores, or correct-rollout zero distances.

Define Spearman correlation as Pearson correlation of average ranks:

1. Require at least two incorrect rollouts.
2. Assign average ranks to exactly tied values, without rounding distances first.
3. If either variable is constant, mark the correlation undefined with the relevant reason.
4. Otherwise compute the correlation and save it with group ID, incorrect count, distinct-value counts, and tie counts.

Report:

- Defined group count and all exclusion reasons.
- Unweighted median and interquartile range of the defined per-group correlations.
- A distribution plot or complete per-group downloadable table.
- The number of defined groups containing exactly two incorrect rollouts. When both variables vary, their Spearman correlation is necessarily +1 or -1, so this subset can make a pooled median look more decisive than it is.
- A prespecified supplementary summary restricted to groups with at least three incorrect rollouts, again with defined/excluded counts. Report both populations; do not select the more appealing one.

A positive correlation means longer incorrect responses tend to be farther from the reference and receive larger penalties. It does not establish that length causes the learned distance, that the encoder encodes only length, or that either ordering is useful for learning.

Do not replace this with a pooled across-prompt correlation as the headline. Do not add unplanned significance tests or choose correlation weighting after seeing the distribution. The primary outputs are descriptive, with no new p-value or pass/fail scientific threshold.

## 12. Statistical status and scope of uncertainty

The new comparison is deterministic conditional on the saved records and the scale-matching procedure. Report empirical distributions and counts; no additional bootstrap or significance test is required by this protocol.

Existing bootstrap intervals in the historical audit retain their original scope and provenance. Do not attach them to the new length rule or treat them as training-seed uncertainty.

If uncertainty intervals are added later, make that a documented amendment before computing them. Specify prompt-level resampling, paired rule comparisons, whether calibration is refitted in every resample, the number of resamples, seed, and undefined-resample handling. A rollout-level bootstrap that treats the eight solutions as independent is not acceptable.

Do not treat matching the algebra on the dataset as independent empirical confirmation of the identity. The empirical contribution is the observed distribution of penalties and resulting credit under clearly defined rules; the identity explains and validates those computations.

## 13. Implementation and execution order

### Step 1 — Preserve scope and begin writing

- Start the four-page draft using existing verified results and the qualified proposition.
- Keep placeholders for new results; do not invent expected numerical values.
- Do not edit the blog or its released evidence as a side effect of this analysis.

### Step 2 — Create isolated analysis files

Suggested new working directory, relative to the blog folder:

```text
analysis/math4ai_2027/
  src/prepare_inputs.py
  src/run_comparison.py
  src/validate_results.py
  src/render_tables.py
  src/test_synthetic_cases.py
  inputs/
  runs/<run-id>/
```

These are **proposed files, not existing executable commands**. The implementation may consolidate the scripts if that is simpler, but must preserve separate preflight, validation, and outcome-reporting stages. Do not claim they have been created or run merely because they are listed here.

Use the existing estimators/companion as references, not as proof that a newly added rule is correct. They do not automatically provide the raw token lengths, correct joining, or the full normalization comparison.

### Step 3 — Validate and project the input data

- Verify hashes, join identities, and correctness; resolve token semantics.
- Produce a minimal joined numeric ledger with source hashes and provenance.
- Keep source records unchanged.
- Write cap/termination metadata and unknown-status definitions before inspecting comparative metrics.
- Reconfirm the known cohort counts and source settings.

### Step 4 — Validate the implementation with synthetic fixtures

Run the cases in Section 14 using hand-checkable inputs. Fix implementation errors before processing the real comparison. Record test output and code hashes.

### Step 5 — Seal the analysis snapshot

Save protocol/configuration/code/input hashes and timestamp. Record the exact command and environment. After sealing, any change affecting results requires a versioned amendment.

### Step 6 — Reproduce the historical learned-distance row

Before analyzing the new alternatives, independently reconstruct the existing standard-normalized learned-distance geometry from the joined records.

Expected approximate anchors from the prior audit:

| Quantity | Existing value / scope |
|---|---|
| Global cosine | 0.978646 |
| Global orthogonal energy | 0.042252, about 4.23% |
| Global norm ratio | About 1.000092 |
| Global relative difference | About 0.206669, or 20.7% |
| Median mixed-group energy | About 0.013460, or 1.35% |

Compare against full-precision existing JSON values, not these rounded anchors, using a documented tolerance. A mismatch must be investigated before reporting new alternatives. Do not recalibrate the learned row to force agreement.

These are linked descriptions of geometry, not five independent findings. In particular, relative difference depends on both cosine and norm ratio.

### Step 7 — Execute the three-rule, two-normalization comparison

- Calculate the single global length coefficient once.
- Generate all primary penalties and advantages for all 500 groups.
- Calculate per-group/global metrics and the within-group/prompt-weight accounting.
- Run the fixed per-group matched-length sensitivity.
- Run the whole-group cap-exclusion sensitivity when its documented trigger applies.
- Save every defined/undefined result and validation discrepancy.

### Step 8 — Calculate structural opportunity and length association

- Produce the incorrect-count histogram and exactly-one-incorrect fraction.
- Calculate the rank-correlation summaries with fixed tie and exclusion handling.
- Include censoring counts and any unknown metadata in the report.

### Step 9 — Validate and render the paper assets

- Recalculate metric identities from saved output, not only from in-memory variables.
- Compare all metrics on the same sample population where a paired comparison is intended.
- Create tables and plots from machine-readable outputs, never manual transcription.
- Run one deterministic repeat and compare numerical outputs and ordered rows. Exclude timestamps and elapsed times from byte-equivalence checks.
- Record whether the repeat is identical or differs within an explained numerical tolerance.

### Step 10 — Integrate and stop expanding

- Insert the compact comparison and structural observation into the paper.
- Put association distributions, sensitivity analyses, and implementation details in the appendix.
- Update a new paper claim map linking every new number to file, key, cohort, script, and hash.
- Stop adding experiments after the specified analysis passes validation. Negative, null, awkward, and redundant outcomes are still reported accurately.
- Revisit the public blog only as a separate, explicit revision after the paper's results and claims are settled.

## 14. Required correctness fixtures

These checks test implementation, not scientific effectiveness.

| Fixture | Required behavior |
|---|---|
| Mixed group `[1,1,0,0]`, wrong penalties `[2,2]` | Shaped rewards `[1,1,-2,-2]`; ideal standardized advantages unchanged; mean-only norm ratio 3 |
| Mixed group `[1,1,0,0]`, wrong penalties `[1,3]` | Shaped rewards `[1,1,-1,-3]`; centered rewards `[1.5,1.5,-0.5,-2.5]`; population variance 2.75 |
| Exactly one incorrect rollout | Residual zero for every nonnegative penalty on that sole wrong entry; cosine 1; magnitude follows normalization |
| Equal nonzero incorrect distances | Group-constant and learned rules coincide |
| Zero penalties | Shaped and binary rewards/advantages coincide |
| All wrong and all correct | Gate off; all task-reward advantages zero; angle undefined |
| Two mixed groups with different constant penalties | Every per-group cosine 1, but global cosine may be below 1; within-group energy zero; weighting term explains global residual |
| Both normalization conventions | Per-group cosines agree; norm ratios need not |
| Stabilizer on/off | Exact equality for affine rescaling only in the zero-stabilizer idealization; recorded-stabilizer ratio matches Section 9 |
| Global length calibration | Summed eligible length penalties equal summed eligible learned penalties |
| Group-constant construction | Each mixed group's penalty mean and shaped-reward mean match the learned rule |
| Group-matched length sensitivity | Each mixed group's penalty mean matches the learned rule; zero means remain zero |
| Geometry identities | Direct residual agrees with `1-cos^2`; relative-distance identity and global energy partition agree |
| Rank ties | Average ranks; constant variable produces undefined correlation; two distinct entries produce +1 or -1 |
| Shuffled input prompt order | Identical outputs after identity-based joining and canonical sorting |
| Duplicate/missing identity or missing eligible distance | Fail loudly; no silent deletion |
| Whole-group cap exclusion | Removes entire groups and uses the original global coefficient |
| Signed-penalty theory check | If included, handles negative aligned component and zero-shaped-vector cases without inferring sign from squared cosine |

Tests should also independently verify that adding zero homogeneous blocks does not alter global metrics, and that all primary mixed-group comparisons are finite under validated nonnegative penalties.

## 15. Required outputs and file contracts

Each run must produce a self-contained directory containing:

| File | Contents |
|---|---|
| `protocol.md` | Sealed copy of this specification or its explicitly amended version |
| `config.json` | Rule definitions, coefficient policy, normalization, tolerances, rank/censoring choices |
| `input_manifest.json` | Original paths, hashes, schema, token/exporter provenance, transformation notes |
| `environment.json` | Python/dependency/platform versions, code revision and hashes |
| `joined_rollouts.jsonl` | Canonical numerical records, lengths, source identity and cap metadata |
| `calibration.json` | Global coefficient, numerator, denominator, unit and eligible population |
| `rollout_results.jsonl` | Rewards and advantages for each primary rule and normalization, keyed by rollout |
| `group_metrics.csv` | Counts, penalty summaries, metric values and undefined reasons for each group/rule/normalization |
| `summary.json` | Primary aggregate/per-group summaries with denominators and definitions |
| `length_association.csv` | Per-group rank correlations, wrong counts, tie counts, exclusion reasons |
| `censoring_report.json` | Known/unknown termination and cap information with counts by cohort |
| `sensitivity_results.json` | Group-matched length and triggered cap-exclusion results, or explicit unavailable reasons |
| `validation.json` | Input, numerical, fixture and repeatability checks; maximum discrepancies |
| `claim_map.json` | New paper claims mapped to exact result keys, cohorts, scripts and hashes |
| `analysis_report.md` | Readable results, limitations and deviations; no causal training claims |
| `tables/` and `figures/` | Data-generated LaTeX/CSV tables and vector figures with source data |
| `SHA256SUMS.txt` | Hashes of final artifacts; do not attempt to include the file's own hash in itself |

Use strict JSON with `null` for unavailable values. Include a schema/version field and clearly distinguish primary from sensitivity results.

A public minimal ledger can reproduce geometry and association from supplied distances and token counts. Independently verifying token counts additionally requires the token records or an accessible source plus the extraction script. If those are not distributed, state that limitation rather than claiming full raw-input reproduction. Do not automatically distribute the entire parent workspace or unrelated artifacts.

## 16. Paper presentation plan

### Main comparison table

Use three primary rows and two normalization column groups. Each normalization should show at least global cosine and norm ratio; energy/relative difference can be adjacent columns or a linked appendix table if space is limited.

Include a per-group directional summary once, not duplicated as independent evidence in both normalization columns. Caption requirements:

- Fixed post-hoc re-decode sample; 500 prompts, eight rollouts each; defined per-group metrics on mixed groups.
- Global versus per-group quantities explicitly labeled.
- Global length coefficient matches mean penalty over eligible incorrect rollouts.
- Mean-only results include prompt-reweighting effects.
- No retraining or task-performance estimate.

### Other main-text items

- One precisely stated proposition with assumptions and a short explanation; proof may be in the appendix.
- One worked example, if not already incorporated into a figure.
- One sentence reporting the exactly-one-incorrect fraction and its structural implication.
- One clear paragraph explaining the historical head-lifecycle finding and limits on causal interpretation.
- A concise limitations paragraph retaining single-seed scope, decoder discrepancy, and comparator confound.

### Appendix

- Full formulas, zero-vector handling, signed-penalty qualification, stabilizer behavior, and global energy partition.
- Length-correlation distribution, defined/excluded counts and the at-least-three-incorrect sensitivity.
- Censoring provenance and applicable sensitivity analyses.
- Per-group matched-length comparison.
- Artifact mapping, execution record, validation and extraction limitations.

### Claims to avoid

- "The update changed by 4.23%."
- "The cosine proves the optimizer update barely changed."
- "The audit predicted the training result."
- "Constant penalties change nothing" without specifying ideal standardized advantages.
- "Any global orthogonal energy is within-incorrect novelty."
- "The length rule repairs all-wrong starvation" when the matched gate disables it there.
- "The learned score is just length" based only on correlation.
- "No truncation occurred" based only on an observed maximum below the configured nominal cap.
- "Three independent geometry findings" for algebraically related metrics on the same vectors.
- "The new comparison fixes the historical implementation or seed uncertainty."

The core claim is a reproducible diagnostic of coverage, credit geometry and intended optimization paths. A diagnostic changing or detecting geometry does not establish that the geometry is useful.

## 17. Stop conditions, amendments, and completion criteria

### Stop the affected analysis if

- Source hashes or identities conflict with the recorded provenance.
- Completion-token semantics cannot be established for the length comparison.
- Any eligible learned distance is missing, negative, or nonfinite.
- The historical row cannot be reproduced from the same inputs and definitions.
- The decomposition, scalar-normalization invariance, calibration, or energy-accounting checks fail beyond tolerance.
- An unexplained discrepancy appears between direct and saved-output recomputation.

Do not treat unavailable termination causes alone as a reason to invent data or silently drop the length rule. Report the unknown status and restrict censoring claims, as specified above.

### Amendments

Record the old and new specification, reason, timestamp, whether results had been seen, and the affected outputs. Retain prior versions. A result that is uninteresting is not a reason to change the protocol.

### Completion checklist

- [ ] Input hashes and complete identity join verified.
- [ ] Token-count semantics and stopping-limit uncertainty documented.
- [ ] Protocol, configuration and code sealed before new comparison metrics.
- [ ] Required synthetic checks pass.
- [ ] Historical learned-distance row reproduced.
- [ ] All three primary rules evaluated under both normalization conventions.
- [ ] Per-group and global quantities distinguished; energy partition checked.
- [ ] Global length calibration recorded and checked.
- [ ] Exactly-one-incorrect count and full incorrect-count histogram reported.
- [ ] Within-prompt rank association, ties and exclusions reported.
- [ ] Per-group matched-length sensitivity reported.
- [ ] Cap-exclusion sensitivity performed when triggered, or its absence explained.
- [ ] Tables/figures generated from saved outputs with matching denominators.
- [ ] Deterministic repeat checked and outputs hashed.
- [ ] New claim map and limitations completed.
- [ ] No new training-performance claim inferred from the offline comparison.
- [ ] Paper updated; public blog revision handled separately.

At document creation, these boxes describe future execution. Only source inspection, existing metadata checks, and the stated input hashes have been performed for preparing this guide. The new experiments, scripts, results, and paper assets have not been created by this document.
